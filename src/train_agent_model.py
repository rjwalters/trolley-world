# src/train_agent_model.py

import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple


def preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the training data

    Args:
        data_path: Path to the training data CSV file

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the action labels
    """
    # Load data
    df = pd.read_csv(data_path)

    # Drop unnecessary columns
    drop_cols = [
        "game_id",
        "turn",
        "agent_id",
        "energy_change",
        "score_change",
        "tied_change",
        "survived",
    ]
    features = df.drop(columns=drop_cols + ["action"])

    # Target variable is the action
    target = df["action"]

    return features, target


def build_model_pipeline() -> Pipeline:
    """
    Build the model pipeline with preprocessing and classifier

    Returns:
        Scikit-learn Pipeline object
    """
    # Define numeric and categorical columns
    numeric_features = [
        "agent_energy",
        "agent_x",
        "agent_y",
        "agent_altruism",
        "agent_risk_aversion",
        "agent_aggression",
        "trolley_x",
        "trolley_y",
        "distance_to_trolley",
        "switch_x",
        "switch_y",
        "distance_to_switch",
        "distance_to_food_edge",
        "num_nearby_agents",
        "num_colocated_agents",
        "colocated_min_energy",
        "colocated_max_energy",
        "colocated_tied_agents",
        "avg_affinity_to_others",
    ]

    categorical_features = [
        "agent_tied",
        "trolley_present",
        "switch_state",
        "is_at_switch",
        "on_track",
    ]

    # Create preprocessors
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create pipeline with preprocessor and model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    return pipeline


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "models",
    hyperparameter_tuning: bool = False,
) -> Dict[str, Any]:
    """
    Train the model and save it to disk

    Args:
        X: Feature DataFrame
        y: Target labels
        output_dir: Directory to save the model
        hyperparameter_tuning: Whether to perform hyperparameter tuning

    Returns:
        Dictionary containing model info and performance metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get model pipeline
    pipeline = build_model_pipeline()

    # Fit the model
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Use best model
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        model = pipeline
        model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    model_path = os.path.join(output_dir, "agent_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Plot feature importance if using RandomForest
    if hasattr(model, "named_steps") and hasattr(
        model.named_steps["classifier"], "feature_importances_"
    ):
        # Get feature names after preprocessing
        preprocessor = model.named_steps["preprocessor"]
        feature_names = []

        # Get numeric feature names (they stay the same)
        numeric_features = preprocessor.transformers_[0][2]
        feature_names.extend(numeric_features)

        # Get one-hot encoded feature names
        categorical_features = preprocessor.transformers_[1][2]
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        categorical_feature_names = []
        for i, feature in enumerate(categorical_features):
            encoded_features = [f"{feature}_{val}" for val in ohe.categories_[i]]
            categorical_feature_names.extend(encoded_features)

        feature_names.extend(categorical_feature_names)

        # Get importance scores
        importances = model.named_steps["classifier"].feature_importances_

        # Plot top 20 features
        indices = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))

    # Return model info and metrics
    return {
        "model_path": model_path,
        "accuracy": report["accuracy"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agent decision model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="analysis/training_data.csv",
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Perform hyperparameter tuning"
    )

    args = parser.parse_args()

    # Preprocess data
    print(f"Loading data from {args.data_path}...")
    X, y = preprocess_data(args.data_path)
    print(f"Data loaded: {X.shape[0]} samples with {X.shape[1]} features")

    # Train model
    print("Training model...")
    model_info = train_model(X, y, args.output_dir, args.tune)
    print(f"Model training complete. Accuracy: {model_info['accuracy']:.4f}")

    # Let the user know where to find the strategy file
    strategy_path = os.path.join("src", "strategies", "ml_strategy.py")
    print(f"\nTo use this model, make sure {strategy_path} is in your project.")
    print(
        "If not, create it by implementing the MLStrategy class that uses the trained model."
    )
