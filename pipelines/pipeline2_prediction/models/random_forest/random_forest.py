"""Random Forest candidate with fitted preprocessing in one persisted pipeline."""

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..artifact_io import reference_artifact_path


DEFAULT_MODEL_PATH = reference_artifact_path("random_forest", "random_forest.pkl")
DEFAULT_METADATA_PATH = reference_artifact_path(
    "random_forest", "random_forest_metadata.pkl"
)


def get_model_info():
    return {
        "name": "random_forest",
        "display_name": "Random Forest",
        "supported_tasks": ["classification", "regression"],
        "preprocessing": "fitted sklearn ColumnTransformer persisted in Pipeline",
    }


def _categorical_columns(frame):
    """Select categorical inputs when the pipeline is fitted."""
    return frame.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


def build_model(task_type, random_state=42, oob_score=False):
    """Build an unfitted pipeline; preprocessing is fitted only by train_model."""
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for Random Forest: {task_type}")

    estimator_class = (
        RandomForestClassifier if task_type == "classification"
        else RandomForestRegressor
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                _categorical_columns,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                estimator_class(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=oob_score,
                ),
            ),
        ]
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    return model.predict(X_test)


def load_trained_model(
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """Load the incoming fitted pipeline and its input-schema metadata."""
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata


def predict_with_artifact(
    frame,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    model, metadata = load_trained_model(model_path, metadata_path)
    feature_columns = metadata["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return model.predict(frame.loc[:, feature_columns])
