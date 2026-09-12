import numpy as np
import torch

from .xtab_model import PretrainedXTabTransformer, XTabTransformer
from ..artifact_io import module_aliases


class XTabPredictor:
    """
    Prediction interface for the LYNX XTab model.

    Supports:
        - Classification prediction
        - Regression prediction
        - Classification probabilities
        - Model saving
        - Model loading
    """

    def __init__(self, model, preprocessor, device=None):

        if model is None:
            raise ValueError(
                "A trained XTab model is required."
            )

        if preprocessor is None:
            raise ValueError(
                "A fitted XTab preprocessor is required."
            )

        self.model = model

        self.preprocessor = preprocessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.model.eval()

    def _transform_features(self, dataframe):
        """
        Transform new tabular data using the fitted
        preprocessing state.
        """

        numerical_data, categorical_data = (
            self.preprocessor.transform_features(
                dataframe
            )
        )

        numerical_tensor = torch.tensor(
            numerical_data,
            dtype=torch.float32,
            device=self.device
        )

        categorical_tensor = torch.tensor(
            categorical_data,
            dtype=torch.long,
            device=self.device
        )

        return (
            numerical_tensor,
            categorical_tensor
        )

    def predict(self, dataframe):
        """
        Generate predictions for new tabular rows.

        Classification:
            Returns original class labels.

        Regression:
            Returns predictions in the original
            target scale.
        """

        (
            numerical_tensor,
            categorical_tensor
        ) = self._transform_features(
            dataframe
        )

        with torch.no_grad():

            output = self.model(
                numerical_features=numerical_tensor,
                categorical_features=categorical_tensor
            )

        if self.preprocessor.task == "classification":

            predicted_indices = torch.argmax(
                output,
                dim=1
            )

            predicted_indices = (
                predicted_indices
                .cpu()
                .numpy()
            )

            # Convert encoded class indices back
            # to the original target values.
            reverse_mapping = {
                index: value
                for value, index
                in self.preprocessor.target_mapping.items()
            }

            predictions = np.array([
                reverse_mapping[index]
                for index in predicted_indices
            ])

            return predictions

        # Regression
        predictions = (
            output
            .cpu()
            .numpy()
        )

        predictions = (
            self.preprocessor
            .inverse_transform_target(
                predictions
            )
        )

        return predictions.astype(
            np.float32
        )

    def predict_proba(self, dataframe):
        """
        Return class probabilities.

        Only available for classification.
        """

        if self.preprocessor.task != "classification":

            raise ValueError(
                "predict_proba is only available "
                "for classification."
            )

        (
            numerical_tensor,
            categorical_tensor
        ) = self._transform_features(
            dataframe
        )

        with torch.no_grad():

            output = self.model(
                numerical_features=numerical_tensor,
                categorical_features=categorical_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

        return probabilities.cpu().numpy()

    def save(self, path):
        """
        Save the trained XTab model and
        preprocessing state.
        """

        if getattr(self.model, "architecture", None) == "official_xtab_ft_transformer":
            model_config = {
                "architecture": self.model.architecture,
                "num_numerical_features": len(self.preprocessor.numerical_columns),
                "categorical_cardinalities": self.preprocessor.categorical_cardinalities,
                "embedding_dim": self.model.embedding_dim,
                "num_heads": self.model.num_heads,
                "num_layers": self.model.num_layers,
                "feedforward_dim": self.model.feedforward_dim,
                "dropout": self.model.dropout,
                "task": self.preprocessor.task,
                "output_dim": (
                    self.preprocessor.num_classes
                    if self.preprocessor.task == "classification"
                    else 1
                ),
                "pretrained_backbone_loaded": self.model.pretrained_backbone_loaded,
                "pretrained_tensor_count": self.model.pretrained_tensor_count,
            }
        else:
            model_config = {
                "architecture": "legacy_generic_transformer",
                "num_numerical_features": len(self.preprocessor.numerical_columns),
                "categorical_cardinalities": self.preprocessor.categorical_cardinalities,
                "embedding_dim": self.model.embedding_dim,
                "num_heads": self.model.transformer.layers[0].self_attn.num_heads,
                "num_layers": len(self.model.transformer.layers),
                "feedforward_dim": self.model.transformer.layers[0].linear1.out_features,
                "dropout": self.model.transformer.layers[0].dropout.p,
                "task": self.preprocessor.task,
                "output_dim": (
                    self.preprocessor.num_classes
                    if self.preprocessor.task == "classification"
                    else 1
                ),
            }

        checkpoint = {

            "model_state_dict":
                self.model.state_dict(),

            "model_config": model_config,

            "preprocessor":
                self.preprocessor,
        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"XTab model saved to: {path}"
        )

    @classmethod
    def load(cls, path, device=None):
        """
        Load a saved XTab model and
        preprocessing state.
        """

        # The staged checkpoint pickled its preprocessor under a legacy
        # top-level module name. Keep that alias scoped to deserialization.
        with module_aliases({
            "model": "pipelines.pipeline2_prediction.models.xtab",
            "model.xtab_preprocessor": (
                "pipelines.pipeline2_prediction.models.xtab.xtab_preprocessor"
            ),
        }):
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )

        model_config = (
            checkpoint["model_config"]
        )

        model_class = (
            PretrainedXTabTransformer
            if model_config.get("architecture") == "official_xtab_ft_transformer"
            else XTabTransformer
        )

        model = model_class(

            num_numerical_features=(
                model_config[
                    "num_numerical_features"
                ]
            ),

            categorical_cardinalities=(
                model_config[
                    "categorical_cardinalities"
                ]
            ),

            embedding_dim=(
                model_config[
                    "embedding_dim"
                ]
            ),

            num_heads=(
                model_config[
                    "num_heads"
                ]
            ),

            num_layers=(
                model_config[
                    "num_layers"
                ]
            ),

            feedforward_dim=(
                model_config[
                    "feedforward_dim"
                ]
            ),

            dropout=(
                model_config[
                    "dropout"
                ]
            ),

            output_dim=(
                model_config[
                    "output_dim"
                ]
            ),

            task=(
                model_config[
                    "task"
                ]
            ),
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if isinstance(model, PretrainedXTabTransformer):
            model.pretrained_backbone_loaded = bool(
                model_config.get("pretrained_backbone_loaded", True)
            )
            model.pretrained_tensor_count = int(
                model_config.get("pretrained_tensor_count", 46)
            )

        preprocessor = (
            checkpoint[
                "preprocessor"
            ]
        )

        return cls(
            model=model,
            preprocessor=preprocessor,
            device=device
        )
