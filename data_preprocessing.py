import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame
    """
    return pd.read_csv(file_path)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    :param df: Preprocessed DataFrame.
    :param test_size: Proportion of test set.
    :param random_state: Random seed for reproducibility.
    :return: X_train, X_test, y_train, y_test
    """
    # TODO

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    :param X_train: Training data.
    :param X_test: Testing data.
    :return: Scaled X_train and X_test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def create_sliding_window(data, window_size, base_column, additional_columns):
    """
    Create sliding windows for time series data.
    :param data: DataFrame containing the data.
    :param window_size: Size of the sliding window.
    :param base_column:
    :param additional_columns: List of additional column names to include as features.
    :return: Arrays for features (X) and targets (y).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        # Extract window features
        window = data[base_column].iloc[i:i + window_size].values
        additional_features = []
        for col in additional_columns:
            additional_features.append(data[col].iloc[i + window_size])
        features = np.concatenate([window, additional_features])
        X.append(features)

        # Target value
        y.append(data[base_column].iloc[i + window_size])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage
    file_path = "../data/raw/sample.csv"
    target_column = "target"

    # Load and preprocess data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    X_train, X_test = scale_features(X_train, X_test)

    print("Data preprocessing completed.")