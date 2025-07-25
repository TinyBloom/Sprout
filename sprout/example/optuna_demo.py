import optuna
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
from sklearn.datasets import fetch_california_housing  # Example dataset

# 1. Load your data (replace with real data if needed)
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
        "random_state": 42,
        "eval_metric": "rmse"
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=10,
        verbose=False
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds)
    return rmse

# 3. Create study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

# 4. Retrain the model with best params on full training data
best_params = study.best_params
best_params["random_state"] = 42
best_params["eval_metric"] = "rmse"

best_model = XGBRegressor(**best_params)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    #early_stopping_rounds=10,
    verbose=False
)

# 5. Save best model and hyperparameters
dump(best_model, "best_xgb_model.joblib")
with open("best_xgb_hyperparams.json", "w") as f:
    json.dump(best_params, f)

print("✅ Best model and hyperparameters saved.")