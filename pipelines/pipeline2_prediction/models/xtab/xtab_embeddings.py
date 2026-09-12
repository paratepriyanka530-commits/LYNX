import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()

        if num_features <= 0:
            raise ValueError(
                "num_features must be greater than 0."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than 0."
            )

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.randn(num_features, embedding_dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(num_features, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            [batch_size, num_features]

        returns:
            [batch_size, num_features, embedding_dim]
        """

        if x.dim() != 2:
            raise ValueError(
                "Numerical input must have shape "
                "[batch_size, num_features]."
            )

        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} numerical features, "
                f"but received {x.size(1)}."
            )

        x = x.unsqueeze(-1)

        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    def __init__(
        self,
        cardinalities: list,
        embedding_dim: int
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than 0."
            )

        if not cardinalities:
            raise ValueError(
                "At least one categorical feature is required."
            )

        if any(cardinality <= 0 for cardinality in cardinalities):
            raise ValueError(
                "All categorical cardinalities must be greater than 0."
            )

        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=embedding_dim
                )
                for cardinality in cardinalities
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            [batch_size, num_categorical_features]

        returns:
            [batch_size, num_categorical_features, embedding_dim]
        """

        if x.dim() != 2:
            raise ValueError(
                "Categorical input must have shape "
                "[batch_size, num_categorical_features]."
            )

        if x.size(1) != len(self.embeddings):
            raise ValueError(
                f"Expected {len(self.embeddings)} categorical features, "
                f"but received {x.size(1)}."
            )

        tokens = []

        for index, embedding in enumerate(self.embeddings):
            feature = x[:, index].long()

            tokens.append(
                embedding(feature)
            )

        return torch.stack(tokens, dim=1)


class FeatureTokenEmbedding(nn.Module):
    """
    Combines numerical and categorical embeddings
    into a single sequence of tabular tokens.
    """

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: list,
        embedding_dim: int
    ):
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = len(
            categorical_cardinalities
        )

        self.numerical_embedding = None

        if num_numerical_features > 0:
            self.numerical_embedding = NumericalEmbedding(
                num_features=num_numerical_features,
                embedding_dim=embedding_dim
            )

        self.categorical_embedding = None

        if self.num_categorical_features > 0:
            self.categorical_embedding = CategoricalEmbedding(
                cardinalities=categorical_cardinalities,
                embedding_dim=embedding_dim
            )

    def forward(
        self,
        numerical_features: torch.Tensor = None,
        categorical_features: torch.Tensor = None
    ) -> torch.Tensor:

        tokens = []

        if (
            self.numerical_embedding is not None
            and numerical_features is not None
        ):
            numerical_tokens = self.numerical_embedding(
                numerical_features
            )

            tokens.append(numerical_tokens)

        if (
            self.categorical_embedding is not None
            and categorical_features is not None
        ):
            categorical_tokens = self.categorical_embedding(
                categorical_features
            )

            tokens.append(categorical_tokens)

        if not tokens:
            raise ValueError(
                "At least one numerical or categorical "
                "feature is required."
            )

        return torch.cat(tokens, dim=1)
    

