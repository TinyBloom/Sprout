from data_preprocessing import load_data, create_sliding_window
from data_postprocessing import filter_data
from model_handler import train_xgboost_regressor_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import Job

def training_temperature_model(job):

    # TODO source file location
    data = load_data("/tmp/sample_new_data.csv")

    data['time'] = pd.to_datetime(data['collect_time'])
    data['hour'] = data['time'].dt.hour
    data['dayofweek'] = data['time'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data['temperature'] = scaler.fit_transform(data[['temperature']])

    window_size = job.params['window_size']
    X, y = create_sliding_window(data, window_size, 'temperature', ['hour','dayofweek','is_weekend'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if job.params['m_name'] == 'xgboost_regression':
        model = train_xgboost_regressor_model(job.params, X_train, y_train)
    else:
        # TODO other
        print("IN PROGRESS")

    future_predictions = []
    last_window = X_test[-1]  # Start with the last window from the test set

    for _ in range(4320):
        next_temp = model.predict(last_window.reshape(1, -1))[0]
        future_predictions.append(next_temp)

        # Slide the window: drop the first value and append the new prediction
        last_window = np.roll(last_window, -1)
        last_window[-1] = next_temp

    # Reverse scaling of predictions
    future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_predictions_original = future_predictions_original.round().astype(int)

    future_timestamps = pd.date_range(start=data['time'].iloc[-1] + pd.Timedelta(seconds=60), periods=4320, freq='1T')

    future_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "Predicted_Temperature": filter_data(job.params, future_predictions_original).flatten(),
    })

    # Optionally, save the future predictions to a CSV
    # future_df.to_csv(f"data/raw/{job.job_id}/future_temperature_predictions.csv", index=False)

if __name__ == "__main__":
    job = Job(
        job_id=1,
        params={ 'm_name':'xgboost_regression', 'n_estimators':700, 'lr':0.8, 'max_depth':6, 'window_size':800, 'post_sigma':0.8 }
    )
    training_temperature_model(job)