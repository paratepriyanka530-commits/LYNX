import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from pipelines.pipeline2_prediction.models.random_forest import (
    get_model_info,
    build_model,
    train_model,
    predict_model,
)


# --------------------------------------------------
# 1. Dataset path
# --------------------------------------------------

DATASET_PATH = "tests/datasets/valid/pizza_sales.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

print("\nAvailable columns:")
for column in df.columns:
    print("-", column)


# --------------------------------------------------
# 3. Ask user for target column
# --------------------------------------------------

target_column = input(
    "\nEnter the target column for prediction: "
).strip()

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' does not exist."
    )


# --------------------------------------------------
# 4. Ask prediction task
# --------------------------------------------------

print("\nPrediction task:")
print("1. Classification")
print("2. Regression")

choice = input("Select task type (1/2): ").strip()

if choice == "1":
    task_type = "classification"

elif choice == "2":
    task_type = "regression"

else:
    raise ValueError("Invalid task selection.")


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = df.drop(columns=[target_column])
y = df[target_column]


# --------------------------------------------------
# 6. Encode categorical feature columns
# --------------------------------------------------

X = pd.get_dummies(
    X,
    drop_first=False,
)


# --------------------------------------------------
# 7. Encode target for classification if necessary
# --------------------------------------------------

target_encoder = None

if task_type == "classification":

    if y.dtype == "object" or str(y.dtype) == "category":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)


# --------------------------------------------------
# 8. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# --------------------------------------------------
# 9. Show model information
# --------------------------------------------------

model_info = get_model_info()

print("\nModel Information")
print("-" * 50)
print("Name:", model_info["display_name"])
print("Task:", task_type)


# --------------------------------------------------
# 10. Build model
# --------------------------------------------------

model = build_model(
    task_type=task_type,
    random_state=42,
)


# --------------------------------------------------
# 11. Train model
# --------------------------------------------------

model = train_model(
    model,
    X_train,
    y_train,
)


# --------------------------------------------------
# 12. Prediction
# --------------------------------------------------

predictions = predict_model(
    model,
    X_test,
)


# --------------------------------------------------
# 13. Evaluation
# --------------------------------------------------

print("\nModel Results")
print("=" * 60)

if task_type == "classification":

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


elif task_type == "regression":

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


print("\nRandom Forest test completed successfully.")