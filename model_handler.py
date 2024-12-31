
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def train_xgboost_regressor_model(params, X_train, y_train):
    """
    Train a XGBRegressor model.
    :param X_train: Features for training.
    :param y_train: Target for training.
    :return: Trained model.
    """
    print(f"Params: {params}")
    model = XGBRegressor(n_estimators=params['n_estimators'], learning_rate=params['lr'], max_depth=params['max_depth'], random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset.
    :param model: Trained model.
    :param X_test: Features for testing.
    :param y_test: Target for testing.
    :return: Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return {"MSE": mse}


def save_model(model, file_path):
    """
    Save the trained model to a file.
    :param model: Trained model.
    :param file_path: Path to save the model.
    """
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    Load a trained model from a file.
    :param file_path: Path to the saved model.
    :return: Loaded model.
    """
    return joblib.load(file_path)


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation Metrics:", metrics)

    # Save model
    #save_model(model, "../models/linear_regression.pkl")

    print("Model training and evaluation completed.")