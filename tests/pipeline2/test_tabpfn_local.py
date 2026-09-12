import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from tabpfn.errors import TabPFNLicenseError

from benchmarks.tabarena.pipeline2_groundwork import feasibility
from pipelines.pipeline2_prediction.models import tabpfn
from pipelines.pipeline2_prediction.models.tabpfn.tabpfn_model import (
    LYNXTabPFN,
    TabPFNProvisioningError,
)


class TabPFNLocalConstructionTests(unittest.TestCase):
    def test_model_path_is_forwarded_to_base_local_classifier(self):
        local_specification = object()
        with patch(
            "pipelines.pipeline2_prediction.models.tabpfn.tabpfn_model.TabPFNClassifier"
        ) as classifier:
            LYNXTabPFN(device="cpu", model_path=local_specification)
        classifier.assert_called_once_with(
            device="cpu", model_path=local_specification
        )

    def test_standalone_trainer_uses_package_local_builder(self):
        from pipelines.pipeline2_prediction.models.tabpfn import build_model
        from pipelines.pipeline2_prediction.models.tabpfn import tabpfn_trainer

        self.assertIs(tabpfn_trainer.build_model, build_model)

    def test_common_builder_uses_auto_without_reference_pickle(self):
        with (
            patch(
                "pipelines.pipeline2_prediction.models.tabpfn."
                "tabpfn_model.TabPFNClassifier"
            ) as classifier,
            patch.object(
                tabpfn,
                "load_pickle_artifact",
                create=True,
            ) as legacy_loader,
        ):
            model = tabpfn.build_model(
                "classification",
                random_state=17,
                device="cpu",
            )

        classifier.assert_called_once_with(device="cpu", model_path="auto")
        classifier.return_value.set_params.assert_called_once_with(
            random_state=17
        )
        legacy_loader.assert_not_called()
        self.assertIsInstance(model, LYNXTabPFN)

    def test_auto_loading_path_supports_common_train_predict_lifecycle(self):
        X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
        y_train = pd.Series([0, 1, 0, 1])
        X_test = pd.DataFrame({"feature": [4.0, 5.0]})

        with patch(
            "pipelines.pipeline2_prediction.models.tabpfn."
            "tabpfn_model.TabPFNClassifier"
        ) as classifier:
            classifier.return_value.predict.return_value = np.array([0, 1])
            model = tabpfn.build_model("classification", device="cpu")
            trained = tabpfn.train_model(model, X_train, y_train)
            predictions = tabpfn.predict_model(trained, X_test)

        classifier.assert_called_once_with(device="cpu", model_path="auto")
        classifier.return_value.fit.assert_called_once_with(X_train, y_train)
        classifier.return_value.predict.assert_called_once_with(X_test)
        self.assertIs(trained, model)
        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_missing_provisioning_raises_controlled_lynx_error(self):
        X_train = pd.DataFrame({"feature": [0.0, 1.0]})
        y_train = pd.Series([0, 1])

        with patch(
            "pipelines.pipeline2_prediction.models.tabpfn."
            "tabpfn_model.TabPFNClassifier"
        ) as classifier:
            classifier.return_value.fit.side_effect = TabPFNLicenseError(
                "upstream detail containing a sensitive placeholder"
            )
            model = LYNXTabPFN(model_path="auto")
            with self.assertRaises(TabPFNProvisioningError) as raised:
                tabpfn.train_model(model, X_train, y_train)

        message = str(raised.exception)
        self.assertIn("one-time PriorLabs license acceptance", message)
        self.assertIn("Interactive browser login", message)
        self.assertIn("TABPFN_TOKEN", message)
        self.assertIn("no reference pickle fallback", message)
        self.assertNotIn("sensitive placeholder", message)

    def test_common_four_function_interface_remains_available(self):
        for function_name in (
            "get_model_info",
            "build_model",
            "train_model",
            "predict_model",
        ):
            self.assertTrue(callable(getattr(tabpfn, function_name)))

    def test_tabpfn_feasibility_limits_are_unchanged(self):
        self.assertTrue(
            feasibility("tabpfn", 10_000, 500, "classification", 10)[0]
        )
        self.assertFalse(
            feasibility("tabpfn", 10_001, 500, "classification", 10)[0]
        )
        self.assertFalse(
            feasibility("tabpfn", 10_000, 501, "classification", 10)[0]
        )
        self.assertFalse(
            feasibility("tabpfn", 10_000, 500, "classification", 11)[0]
        )
        self.assertFalse(feasibility("tabpfn", 100, 4, "regression")[0])


if __name__ == "__main__":
    unittest.main()
