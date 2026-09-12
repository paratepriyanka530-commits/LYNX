import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from .xtab_config import XTabConfig
from .xtab_model import PretrainedXTabTransformer
from .xtab_preprocessor import XTabPreprocessor
from .xtab_evaluator import XTabEvaluator


class XTabTrainer:
    """
    Training pipeline for the LYNX XTab model.

    Pipeline:
        DataFrame
            â†“
        Train / Validation / Test split
            â†“
        Fit preprocessing on training data only
            â†“
        Transform train / validation / test
            â†“
        XTab Transformer
            â†“
        Training
            â†“
        Validation
            â†“
        Early stopping
            â†“
        Restore best model
            â†“
        Test evaluation
    """

    def __init__(self, config=None):

        self.config = config or XTabConfig()

        self.preprocessor = XTabPreprocessor(task=getattr(self.config, "task", None))

        self.model = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.best_model_state = None

        self.best_validation_loss = float("inf")

        self.test_data = None

        self.test_metrics = None

        self._set_seed(
            self.config.random_seed
        )

    # ==================================================
    # REPRODUCIBILITY
    # ==================================================

    @staticmethod
    def _set_seed(seed):
        """
        Set random seeds for reproducibility.
        """

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed)

    # ==================================================
    # CREATE MODEL
    # ==================================================

    def _create_model(self):
        """
        Create the XTab Transformer using information
        learned by the preprocessor.
        """

        self.model = PretrainedXTabTransformer(

            num_numerical_features=len(
                self.preprocessor.numerical_columns
            ),

            categorical_cardinalities=(
                self.preprocessor.categorical_cardinalities
            ),

            embedding_dim=(
                self.config.embedding_dim
            ),

            num_heads=(
                self.config.num_heads
            ),

            num_layers=(
                self.config.num_layers
            ),

            feedforward_dim=(
                self.config.feedforward_dim
            ),

            dropout=(
                self.config.dropout
            ),

            output_dim=(
                self.preprocessor.num_classes
                if self.preprocessor.task
                == "classification"
                else 1
            ),

            task=(
                self.preprocessor.task
            ),
        )

        if not self.config.pretrained_backbone_path:
            raise ValueError(
                "XTab requires an official cross-table pretrained backbone checkpoint."
            )
        self.pretraining_report = self.model.load_pretrained_backbone(
            self.config.pretrained_backbone_path
        )

        self.model.to(
            self.device
        )

    # ==================================================
    # CREATE DATALOADER
    # ==================================================

    def _create_dataloader(
        self,
        numerical_data,
        categorical_data,
        target_data,
        shuffle=True,
    ):
        """
        Convert NumPy arrays into a PyTorch DataLoader.
        """

        numerical_tensor = torch.tensor(
            numerical_data,
            dtype=torch.float32
        )

        categorical_tensor = torch.tensor(
            categorical_data,
            dtype=torch.long
        )

        if (
            self.preprocessor.task
            == "classification"
        ):

            target_tensor = torch.tensor(
                target_data,
                dtype=torch.long
            )

        else:

            target_tensor = torch.tensor(
                target_data,
                dtype=torch.float32
            )

        dataset = TensorDataset(

            numerical_tensor,

            categorical_tensor,

            target_tensor,
        )

        return DataLoader(

            dataset,

            batch_size=(
                self.config.batch_size
            ),

            shuffle=shuffle,
        )

    # ==================================================
    # LOSS FUNCTION
    # ==================================================

    def _create_loss_function(self):
        """
        Select loss function according to the task.
        """

        if (
            self.preprocessor.task
            == "classification"
        ):

            return torch.nn.CrossEntropyLoss()

        return torch.nn.MSELoss()

    # ==================================================
    # OPTIMIZER
    # ==================================================

    def _create_optimizer(self):
        """
        Create AdamW optimizer.
        """

        if not getattr(self.model, "pretrained_backbone_loaded", False):
            raise RuntimeError(
                "XTab optimization cannot begin before the pretrained backbone is loaded."
            )

        return torch.optim.AdamW(

            self.model.parameters(),

            lr=(
                self.config.learning_rate
            ),

            weight_decay=(
                self.config.weight_decay
            ),
        )

    # ==================================================
    # CALCULATE LOSS
    # ==================================================

    def _calculate_loss(
        self,
        loader,
        loss_function,
    ):
        """
        Calculate average loss for a dataset.
        """

        self.model.eval()

        total_loss = 0.0

        total_samples = 0

        with torch.no_grad():

            for (
                numerical_batch,
                categorical_batch,
                target_batch,
            ) in loader:

                numerical_batch = (
                    numerical_batch.to(
                        self.device
                    )
                )

                categorical_batch = (
                    categorical_batch.to(
                        self.device
                    )
                )

                target_batch = (
                    target_batch.to(
                        self.device
                    )
                )

                output = self.model(

                    numerical_features=(
                        numerical_batch
                    ),

                    categorical_features=(
                        categorical_batch
                    ),
                )

                loss = loss_function(

                    output,

                    target_batch,
                )

                batch_size = (
                    target_batch.size(0)
                )

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

        if total_samples == 0:

            raise ValueError(
                "Cannot calculate loss because "
                "the dataset is empty."
            )

        return (
            total_loss
            / total_samples
        )

    # ==================================================
    # TEST EVALUATION
    # ==================================================

    def evaluate_test(self):
        """
        Evaluate the trained XTab model on the
        held-out test dataset.

        Classification:
            accuracy
            precision
            recall
            f1
            confusion_matrix

        Regression:
            MAE
            MSE
            RMSE
            R2
        """

        if self.model is None:

            raise ValueError(
                "Model has not been trained yet."
            )

        if self.test_data is None:

            raise ValueError(
                "Test data is not available."
            )

        numerical_tensor = torch.tensor(

            self.test_data["numerical"],

            dtype=torch.float32,

            device=self.device
        )

        categorical_tensor = torch.tensor(

            self.test_data["categorical"],

            dtype=torch.long,

            device=self.device
        )

        self.model.eval()

        with torch.no_grad():

            output = self.model(

                numerical_features=(
                    numerical_tensor
                ),

                categorical_features=(
                    categorical_tensor
                ),
            )

        # ----------------------------------------------
        # Classification
        # ----------------------------------------------

        if (
            self.preprocessor.task
            == "classification"
):

           predictions = torch.argmax(
              output,
              dim=1
    )

        else:

            predictions = (
               self.preprocessor.inverse_transform_target(
                   output.cpu().numpy()
        )
    )

        if (
             self.preprocessor.task
             == "classification"
):

            predictions = (
                predictions
                .cpu()
                .numpy()
    )

        y_true = self.test_data["target"]

        if (
            self.preprocessor.task
            == "regression"
        ):

            y_true = (
                self.preprocessor.inverse_transform_target(
                  y_true
                )
            )

        evaluator = XTabEvaluator()

        metrics = evaluator.evaluate(

            y_true=y_true,

            y_pred=predictions,

            task=self.preprocessor.task,
        )

        print()

        print(
            "Test Evaluation"
        )

        print(
            "----------------"
        )

        for metric, value in (
            metrics.items()
        ):

            print(
                f"{metric}: {value}"
            )

        return metrics

    # ==================================================
    # TRAIN
    # ==================================================

    def train(
        self,
        dataframe,
        target_column,
    ):
        """
        Train the XTab Transformer.

        Steps:

            1. Validate input
            2. Split raw dataframe
            3. Fit preprocessing on training data only
            4. Transform train/validation/test
            5. Create XTab model
            6. Create DataLoaders
            7. Train model
            8. Validate model
            9. Early stopping
            10. Restore best model
            11. Evaluate test set
        """

        # ==================================================
        # INPUT VALIDATION
        # ==================================================

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if (
            target_column
            not in dataframe.columns
        ):

            raise ValueError(

                f"Target column "
                f"'{target_column}' "
                f"does not exist in the dataset."
            )

        if len(dataframe) < 10:

            raise ValueError(

                "Dataset must contain at least "
                "10 rows for "
                "train/validation/test splitting."
            )

        # ==================================================
        # RESET TRAINING STATE
        # ==================================================

        self.best_model_state = None

        self.best_validation_loss = float(
            "inf"
        )

        self.test_data = None

        self.test_metrics = None

        # ==================================================
        # SPLIT RAW DATA FIRST
        # ==================================================

        indices = np.arange(
           len(dataframe)
)

# Determine task before splitting
        if self.config.task is not None:
           detected_task = self.config.task
        else:
           target = dataframe[target_column]

           if pd.api.types.is_numeric_dtype(target):

              unique_values = target.nunique()
              total_values = target.notna().sum()

              unique_ratio = (
                  unique_values / total_values
                  if total_values > 0
                  else 0
              )

              if (
                  pd.api.types.is_integer_dtype(target)
                  and unique_values <= 10
                  and unique_ratio <= 0.20
              ):
                  detected_task = "classification"
              else:
                  detected_task = "regression"

           else:
                  detected_task = "classification"

        test_indices, remaining_indices = (
            train_test_split(

                indices,

                test_size=(
                    1.0
                    - self.config.test_size
                ),

                random_state=(
                    self.config.random_seed
                ),

            )
        )

        validation_ratio = (

            self.config.validation_size

            /

            (
                1.0
                - self.config.test_size
            )
        )

        train_indices, validation_indices = (
            train_test_split(

                remaining_indices,

                test_size=validation_ratio,

                random_state=(
                    self.config.random_seed
                ),

            )
        )

        train_dataframe = (
            dataframe.iloc[
                train_indices
            ].copy()
        )

        validation_dataframe = (
            dataframe.iloc[
                validation_indices
            ].copy()
        )

        test_dataframe = (
            dataframe.iloc[
                test_indices
            ].copy()
        )

        print(
            f"Train samples: "
            f"{len(train_dataframe)}"
        )

        print(
            f"Validation samples: "
            f"{len(validation_dataframe)}"
        )

        print(
            f"Test samples: "
            f"{len(test_dataframe)}"
        )

        # ==================================================
        # FIT PREPROCESSING ONLY ON TRAINING DATA
        # ==================================================
        self.preprocessor.requested_task = detected_task
        self.preprocessor.fit(

            train_dataframe,

            target_column,
        )

        # ==================================================
        # TRANSFORM TRAIN DATA
        # ==================================================

        (
            numerical_train,
            categorical_train,
        ) = (
            self.preprocessor.transform_features(

                train_dataframe
            )
        )

        target_train = (
            self.preprocessor.transform_target(

                train_dataframe[
                    target_column
                ]
            )
        )

        # ==================================================
        # TRANSFORM VALIDATION DATA
        # ==================================================

        (
            numerical_validation,
            categorical_validation,
        ) = (
            self.preprocessor.transform_features(

                validation_dataframe
            )
        )

        target_validation = (
            self.preprocessor.transform_target(

                validation_dataframe[
                    target_column
                ]
            )
        )

        # ==================================================
        # TRANSFORM TEST DATA
        # ==================================================

        (
            numerical_test,
            categorical_test,
        ) = (
            self.preprocessor.transform_features(

                test_dataframe
            )
        )

        target_test = (
            self.preprocessor.transform_target(

                test_dataframe[
                    target_column
                ]
            )
        )

        # ==================================================
        # STORE TEST DATA
        # ==================================================

        self.test_data = {

            "numerical":
                numerical_test,

            "categorical":
                categorical_test,

            "target":
                target_test,
        }

        # ==================================================
        # CREATE MODEL
        # ==================================================

        self._create_model()

        # ==================================================
        # CREATE DATALOADERS
        # ==================================================

        train_loader = (
            self._create_dataloader(

                numerical_train,

                categorical_train,

                target_train,

                shuffle=True,
            )
        )

        validation_loader = (
            self._create_dataloader(

                numerical_validation,

                categorical_validation,

                target_validation,

                shuffle=False,
            )
        )

        # ==================================================
        # LOSS
        # ==================================================

        loss_function = (
            self._create_loss_function()
        )

        # ==================================================
        # OPTIMIZER
        # ==================================================

        optimizer = (
            self._create_optimizer()
        )

        # ==================================================
        # TRAINING
        # ==================================================

        patience_counter = 0

        for epoch in range(
            self.config.epochs
        ):

            self.model.train()

            total_training_loss = 0.0

            total_samples = 0

            # ------------------------------------------
            # Training batches
            # ------------------------------------------

            for (
                numerical_batch,
                categorical_batch,
                target_batch,
            ) in train_loader:

                numerical_batch = (
                    numerical_batch.to(
                        self.device
                    )
                )

                categorical_batch = (
                    categorical_batch.to(
                        self.device
                    )
                )

                target_batch = (
                    target_batch.to(
                        self.device
                    )
                )

                # Clear gradients

                optimizer.zero_grad()

                # Forward pass

                output = self.model(

                    numerical_features=(
                        numerical_batch
                    ),

                    categorical_features=(
                        categorical_batch
                    ),
                )

                # Loss

                loss = loss_function(

                    output,

                    target_batch,
                )

                # Backpropagation

                loss.backward()

                # Update parameters

                optimizer.step()

                batch_size = (
                    target_batch.size(0)
                )

                total_training_loss += (

                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

            # ------------------------------------------
            # Training loss
            # ------------------------------------------

            if total_samples == 0:

                raise ValueError(
                    "Training dataset is empty."
                )

            training_loss = (

                total_training_loss

                /

                total_samples
            )

            # ------------------------------------------
            # Validation loss
            # ------------------------------------------

            validation_loss = (
                self._calculate_loss(

                    validation_loader,

                    loss_function,
                )
            )

            print(

                f"Epoch "
                f"{epoch + 1}/"
                f"{self.config.epochs} "

                f"- Train Loss: "
                f"{training_loss:.4f} "

                f"- Validation Loss: "
                f"{validation_loss:.4f}"
            )

            # ==================================================
            # BEST MODEL
            # ==================================================

            if (
                validation_loss
                < self.best_validation_loss
            ):

                self.best_validation_loss = (
                    validation_loss
                )

                self.best_model_state = {

                    key:
                        value.detach()
                        .cpu()
                        .clone()

                    for key, value
                    in self.model.state_dict().items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

            # ==================================================
            # EARLY STOPPING
            # ==================================================

            if (
                patience_counter
                >= self.config.patience
            ):

                print(

                    f"Early stopping at epoch "
                    f"{epoch + 1}."
                )

                break

        # ==================================================
        # RESTORE BEST MODEL
        # ==================================================

        if (
            self.best_model_state
            is not None
        ):

            self.model.load_state_dict(

                self.best_model_state
            )

        print(
            "Training completed successfully."
        )

        # ==================================================
        # TEST EVALUATION
        # ==================================================

        self.test_metrics = (
            self.evaluate_test()
        )

        return self.model
