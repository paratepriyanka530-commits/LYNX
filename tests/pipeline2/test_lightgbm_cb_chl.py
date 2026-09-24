"""Focused tests for the opt-in, binary CB-CHL LightGBM mode."""

import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline2_prediction.models import lightgbm
from pipelines.pipeline2_prediction.models.lightgbm.cb_chl import (
    CBCHLConfigurationError,
    cb_chl_grad_hess,
    cb_chl_loss,
)
from pipelines.pipeline2_prediction.models.lightgbm.lightgbm_config import (
    LIGHTGBM_CONFIG,
)


# Neutral numeric values solely for mathematical/unit-test coverage. They are
# not an anneal or MIC research cost matrix and must never be persisted as one.
TEST_ONLY_COST_MATRIX = np.array(
    [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]], dtype=float
)


class CBCHLLightGBMTests(unittest.TestCase):
    def test_standard_is_default_and_configuration_is_unchanged(self):
        model = lightgbm.build_model("classification", random_state=17)
        self.assertEqual(model.training_mode, "standard")
        self.assertEqual(model.model.get_params()["objective"], LIGHTGBM_CONFIG["objective"])
        self.assertEqual(model.model.get_params()["n_estimators"], LIGHTGBM_CONFIG["n_estimators"])
        self.assertEqual(model.model.get_params()["random_state"], 17)

    def test_cb_chl_requires_explicit_activation(self):
        model = lightgbm.build_model(
            "classification",
            training_mode="standard",
            cb_chl_eligible=True,
            cost_config=TEST_ONLY_COST_MATRIX,
        )
        self.assertEqual(model.training_mode, "standard")

    def test_cb_chl_rejects_missing_cost_configuration(self):
        with self.assertRaisesRegex(CBCHLConfigurationError, "explicit cost"):
            lightgbm.build_model(
                "classification", training_mode="cb_chl", cb_chl_eligible=True
            )

    def test_noneligible_dataset_cannot_activate_cb_chl(self):
        with self.assertRaisesRegex(CBCHLConfigurationError, "not marked"):
            lightgbm.build_model(
                "classification",
                training_mode="cb_chl",
                cb_chl_eligible=False,
                cost_config=TEST_ONLY_COST_MATRIX,
            )

    def test_objective_outputs_are_finite_and_have_expected_shape(self):
        raw_scores = np.array([-1.2, -0.2, 0.4, 1.1])
        labels = np.array([0, 1, 0, 1])
        gradient, hessian = cb_chl_grad_hess(
            raw_scores, labels, TEST_ONLY_COST_MATRIX
        )
        self.assertEqual(gradient.shape, labels.shape)
        self.assertEqual(hessian.shape, labels.shape)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.isfinite(hessian).all())

    def test_gradient_and_hessian_match_finite_differences(self):
        raw_scores = np.array([-1.2, -0.2, 0.4, 1.1])
        labels = np.array([0, 1, 0, 1])
        gradient, hessian = cb_chl_grad_hess(
            raw_scores, labels, TEST_ONLY_COST_MATRIX
        )
        step = 1e-5
        numerical_gradient = np.empty_like(raw_scores)
        numerical_hessian = np.empty_like(raw_scores)
        for index in range(len(raw_scores)):
            above = raw_scores.copy()
            below = raw_scores.copy()
            above[index] += step
            below[index] -= step
            numerical_gradient[index] = (
                cb_chl_loss(above, labels, TEST_ONLY_COST_MATRIX)[index]
                - cb_chl_loss(below, labels, TEST_ONLY_COST_MATRIX)[index]
            ) / (2.0 * step)
            numerical_hessian[index] = (
                cb_chl_grad_hess(above, labels, TEST_ONLY_COST_MATRIX)[0][index]
                - cb_chl_grad_hess(below, labels, TEST_ONLY_COST_MATRIX)[0][index]
            ) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-5, atol=1e-7)

    def test_training_state_uses_only_labels_passed_to_train(self):
        features = pd.DataFrame(
            {"x": np.arange(12, dtype=float), "z": np.tile([0.0, 1.0], 6)}
        )
        y_train = pd.Series([0] * 9 + [1] * 3)
        costs = np.tile([1.0, 2.0], (len(y_train), 1))
        model = lightgbm.build_model(
            "classification",
            random_state=3,
            training_mode="cb_chl",
            cb_chl_eligible=True,
            cost_config=costs,
        )
        model.model.set_params(n_estimators=3, min_child_samples=1, num_leaves=3)
        lightgbm.train_model(model, features, y_train)
        state = model.cb_chl_training_state
        self.assertEqual((state.positive_count, state.negative_count), (3, 9))
        self.assertAlmostEqual(state.negative_class_balance_factor, 1.0 / 3.0)

        # A held-out label vector is intentionally never supplied to either
        # build_model or train_model and therefore cannot alter training state.
        held_out_labels = pd.Series([1, 1, 1, 1])
        self.assertEqual(len(held_out_labels), 4)
        self.assertEqual(model.cb_chl_training_state, state)
        predictions = np.asarray(lightgbm.predict_model(model, features.iloc[:4]))
        probabilities = np.asarray(model.predict_probability(features.iloc[:4]))
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_cb_chl_rejects_multiclass_labels(self):
        with self.assertRaisesRegex(CBCHLConfigurationError, "binary-only"):
            cb_chl_grad_hess(
                np.zeros(4), np.array([0, 1, 2, 0]), TEST_ONLY_COST_MATRIX
            )


if __name__ == "__main__":
    unittest.main()
