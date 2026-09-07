from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


def get_model_info():
    """
    Return basic information about the model.
    """
    return {
        "name": "random_forest",
        "display_name": "Random Forest",
        "supported_tasks": [
            "classification",
            "regression",
        ],
    }


def build_model(task_type, random_state=42):
    """
    Create a Random Forest model for the requested task.
    """

    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    if task_type == "regression":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported task type for Random Forest: {task_type}"
    )


def train_model(model, X_train, y_train):
    """
    Train the supplied Random Forest model.
    """
    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    """
    Generate predictions using the trained model.
    """
    return model.predict(X_test)