import torch
import torch.nn as nn

from pathlib import Path

from pytabkit.models.nn_models.rtdl_resnet import FT_Transformer as RTDLFTTransformer

from .xtab_embeddings import FeatureTokenEmbedding


class XTabTransformer(nn.Module):
    """
    Generic Transformer-based tabular model for LYNX.

    Supports:
        - numerical features
        - categorical features
        - classification
        - regression
    """

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: list,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        output_dim: int = 1,
        task: str = "classification",
    ):
        super().__init__()

        if task not in {"classification", "regression"}:
            raise ValueError(
                "task must be 'classification' or 'regression'."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than 0."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than 0."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0."
            )

        if feedforward_dim <= 0:
            raise ValueError(
                "feedforward_dim must be greater than 0."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be greater than 0."
            )

        self.task = task       
        self.embedding_dim = embedding_dim

        # --------------------------------------------------
        # Feature embedding
        # --------------------------------------------------

        self.feature_embedding = FeatureTokenEmbedding(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=embedding_dim,
        )

        # --------------------------------------------------
        # Feature-type independent positional embeddings
        # --------------------------------------------------

        self.feature_norm = nn.LayerNorm(embedding_dim)

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --------------------------------------------------
        # Representation layer
        # --------------------------------------------------

        self.representation = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Prediction head
        # --------------------------------------------------

        self.output_layer = nn.Linear(
            embedding_dim,
            output_dim,
        )

    def forward(
        self,
        numerical_features=None,
        categorical_features=None,
    ):
        """
        Forward pass.

        Returns:
            Classification:
                logits

            Regression:
                predictions
        """

        # ----------------------------------------------
        # Convert features into tokens
        # ----------------------------------------------

        tokens = self.feature_embedding(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        # ----------------------------------------------
        # Normalize tokens
        # ----------------------------------------------

        tokens = self.feature_norm(tokens)

        # ----------------------------------------------
        # Transformer
        # ----------------------------------------------

        transformed = self.transformer(tokens)

        # ----------------------------------------------
        # Aggregate feature representations
        # ----------------------------------------------

        representation = transformed.mean(dim=1)

        # ----------------------------------------------
        # Final representation
        # ----------------------------------------------

        representation = self.representation(
            representation
        )

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        output = self.output_layer(
            representation
        )

        if self.task == "regression":
            return output.squeeze(-1)

        return output


class PretrainedXTabTransformer(nn.Module):
    """Official XTab FT-Transformer backbone with task-specific outer layers.

    XTab pretrains only the shared transformer blocks. Feature tokenizers and
    the downstream prediction head are intentionally initialized per dataset.
    """

    architecture = "official_xtab_ft_transformer"

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: list,
        output_dim: int,
        task: str,
        embedding_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        expected = (192, 8, 3, 192)
        actual = (embedding_dim, num_heads, num_layers, feedforward_dim)
        if actual != expected:
            raise ValueError(
                "Official XTab pretrained weights require "
                "embedding_dim=192, num_heads=8, num_layers=3, and "
                f"feedforward_dim=192; received {actual}."
            )

        self.task = task
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.num_numerical_features = num_numerical_features
        self.categorical_cardinalities = list(categorical_cardinalities)
        indicator = (
            torch.tensor(
                [False] * num_numerical_features
                + [True] * len(self.categorical_cardinalities),
                dtype=torch.bool,
            )
            if self.categorical_cardinalities
            else None
        )
        self.transformer = RTDLFTTransformer(
            d_in=num_numerical_features,
            categories=self.categorical_cardinalities or None,
            token_bias=True,
            n_layers=num_layers,
            d_token=embedding_dim,
            n_heads=num_heads,
            d_ffn_factor=feedforward_dim / embedding_dim,
            attention_dropout=0.2,
            ffn_dropout=dropout,
            residual_dropout=0.0,
            activation="geglu",
            prenormalization=True,
            initialization="kaiming",
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=output_dim,
            regression=(task == "regression"),
            categorical_indicator=indicator,
        )
        self.pretrained_backbone_loaded = False
        self.pretrained_backbone_path = None
        self.pretrained_tensor_count = 0

    @staticmethod
    def _map_official_backbone_key(key: str) -> str:
        return (
            key.replace("blocks.", "layers.")
            .replace(".ffn.linear_first.", ".linear0.")
            .replace(".ffn.linear_second.", ".linear1.")
            .replace(".attention_normalization.", ".norm0.")
            .replace(".ffn_normalization.", ".norm1.")
        )

    def load_pretrained_backbone(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"XTab pretrained backbone not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        source_state = checkpoint.get("state_dict")
        if not isinstance(source_state, dict) or not source_state:
            raise ValueError("XTab pretrained checkpoint has no non-empty state_dict")
        mapped_state = {
            self._map_official_backbone_key(key): value
            for key, value in source_state.items()
        }
        result = self.transformer.load_state_dict(mapped_state, strict=False)
        expected_missing = {
            "tokenizer.weight",
            "tokenizer.bias",
            "last_normalization.weight",
            "last_normalization.bias",
            "head.weight",
            "head.bias",
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(
                "Official XTab checkpoint is incompatible: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        self.pretrained_backbone_loaded = True
        self.pretrained_backbone_path = str(path.resolve())
        self.pretrained_tensor_count = len(mapped_state)
        return {
            "path": self.pretrained_backbone_path,
            "loaded_tensor_count": self.pretrained_tensor_count,
            "dataset_specific_random_keys": sorted(expected_missing),
        }

    def forward(self, numerical_features=None, categorical_features=None):
        pieces = []
        if numerical_features is not None and numerical_features.shape[1]:
            pieces.append(numerical_features.float())
        if categorical_features is not None and categorical_features.shape[1]:
            pieces.append(categorical_features.float())
        if not pieces:
            raise ValueError("At least one numerical or categorical feature is required.")
        combined = torch.cat(pieces, dim=1)
        if self.transformer.categorical_indicator is not None:
            self.transformer.categorical_indicator = (
                self.transformer.categorical_indicator.to(combined.device)
            )
        output = self.transformer(combined)
        return output.squeeze(-1) if self.task == "regression" else output
