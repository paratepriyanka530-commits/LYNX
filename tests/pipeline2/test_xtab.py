import unittest
import warnings
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import torch

from pipelines.pipeline2_prediction.models.xtab import (
    DEFAULT_PRETRAINED_BACKBONE_PATH,
    get_model_info,
    load_trained_model,
)
from pipelines.pipeline2_prediction.models.xtab.xtab_model import (
    PretrainedXTabTransformer,
)
from pipelines.pipeline2_prediction.models.xtab.xtab_config import XTabConfig
from pipelines.pipeline2_prediction.models.xtab.xtab_trainer import XTabTrainer


class XTabTests(unittest.TestCase):
    def test_official_backbone_loads_with_exact_tensor_match(self):
        model = PretrainedXTabTransformer(
            num_numerical_features=4,
            categorical_cardinalities=[],
            output_dim=2,
            task="classification",
        )
        report = model.load_pretrained_backbone(DEFAULT_PRETRAINED_BACKBONE_PATH)
        source = torch.load(
            DEFAULT_PRETRAINED_BACKBONE_PATH,
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
        target = model.transformer.state_dict()
        self.assertTrue(model.pretrained_backbone_loaded)
        self.assertEqual(report["loaded_tensor_count"], 46)
        for source_key, source_value in source.items():
            target_key = model._map_official_backbone_key(source_key)
            torch.testing.assert_close(target[target_key], source_value)

    def test_optimizer_refuses_randomly_initialized_backbone(self):
        trainer = XTabTrainer(XTabConfig(task="classification"))
        trainer.model = PretrainedXTabTransformer(
            num_numerical_features=4,
            categorical_cardinalities=[],
            output_dim=2,
            task="classification",
        )
        with self.assertRaisesRegex(RuntimeError, "before the pretrained backbone"):
            trainer._create_optimizer()

    def test_capabilities_and_capacity_metadata(self):
        info = get_model_info()
        self.assertEqual(info["supported_tasks"], ["classification", "regression"])
        self.assertEqual(info["minimum_training_rows"], 10)
        self.assertIsNone(info["hard_maximum_rows"])
        self.assertIsNone(info["hard_maximum_features"])
        self.assertIsNone(info["hard_maximum_classes"])

    def test_supplied_checkpoint_loads_and_predicts_without_refitting(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor = load_trained_model(device="cpu")
        # Compatibility warnings are asserted separately in the pinned
        # provenance environment; this test also runs in developer environments.
        self.assertEqual(predictor.preprocessor.task, "classification")
        self.assertTrue(hasattr(predictor.preprocessor.scaler, "mean_"))

        means = predictor.preprocessor.scaler.mean_
        frame = pd.DataFrame(
            np.repeat(means.reshape(1, -1), 3, axis=0),
            columns=predictor.preprocessor.numerical_columns,
        )
        scaler_id = id(predictor.preprocessor.scaler)
        predictions = predictor.predict(frame)
        probabilities = predictor.predict_proba(frame)

        self.assertEqual(predictions.shape, (3,))
        self.assertEqual(probabilities.shape, (3, 6))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3), rtol=1e-6)
        self.assertEqual(scaler_id, id(predictor.preprocessor.scaler))

    def test_legacy_aliases_are_automatic_restored_and_repeatable(self):
        legacy_names = ("model", "model.xtab_preprocessor")
        previous = {name: sys.modules.get(name) for name in legacy_names}
        was_present = {name: name in sys.modules for name in legacy_names}
        try:
            for name in legacy_names:
                sys.modules.pop(name, None)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = load_trained_model(device="cpu")
            self.assertEqual(first.preprocessor.task, "classification")
            self.assertTrue(all(name not in sys.modules for name in legacy_names))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                second = load_trained_model(device="cpu")
            self.assertEqual(second.preprocessor.task, "classification")
            self.assertTrue(all(name not in sys.modules for name in legacy_names))

            sentinels = {name: ModuleType(name) for name in legacy_names}
            sys.modules.update(sentinels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                load_trained_model(device="cpu")
            for name, sentinel in sentinels.items():
                self.assertIs(sys.modules[name], sentinel)
        finally:
            for name in legacy_names:
                if was_present[name]:
                    sys.modules[name] = previous[name]
                else:
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
