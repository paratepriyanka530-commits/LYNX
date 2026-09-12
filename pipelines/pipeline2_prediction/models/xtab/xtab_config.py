from dataclasses import dataclass


@dataclass
class XTabConfig:
    """
    Configuration for the LYNX XTab model.

    This configuration is dataset-independent.
    """

    # Reproducibility
    random_seed: int = 42

    # Dataset
    task: str = None
    test_size: float = 0.20
    validation_size: float = 0.10

    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # XTab Transformer
    embedding_dim: int = 192
    num_heads: int = 8
    num_layers: int = 3
    feedforward_dim: int = 192
    dropout: float = 0.1

    # Official XTab cross-table FT-Transformer backbone. The shared-interface
    # candidate supplies the repository reference artifact by default.
    pretrained_backbone_path: str = None

    # Training control
    patience: int = 7

    def __post_init__(self):
      if self.task is not None and self.task not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            "task must be 'classification', 'regression', or None."
        )

      if not 0 < self.test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

      if not 0 < self.validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

      if self.test_size + self.validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

      if self.batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

      if self.epochs <= 0:
        raise ValueError(
            "epochs must be greater than 0."
        )

      if self.learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

      if self.weight_decay < 0:
        raise ValueError(
            "weight_decay must be greater than or equal to 0."
        )

      if self.embedding_dim <= 0:
        raise ValueError(
            "embedding_dim must be greater than 0."
        )

      if self.num_heads <= 0:
        raise ValueError(
            "num_heads must be greater than 0."
        )

      if self.embedding_dim % self.num_heads != 0:
        raise ValueError(
            "embedding_dim must be divisible by num_heads."
        )

      if self.num_layers <= 0:
        raise ValueError(
            "num_layers must be greater than 0."
        )

      if self.feedforward_dim <= 0:
        raise ValueError(
            "feedforward_dim must be greater than 0."
        )

      if not 0 <= self.dropout < 1:
        raise ValueError(
            "dropout must be between 0 and 1."
        )

      if self.patience <= 0:
        raise ValueError(
            "patience must be greater than 0."
        )
