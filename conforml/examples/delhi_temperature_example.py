import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from conforml.models.arima import ARIMAModel
from conforml.conformal import CVPlusConformal, AdaptiveConformal
from conforml.metrics import RMSECalculator, MAPECalculator
import warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")

# --- Data Loading ---
def load_delhi_temperature():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, parse_dates=['Date'])
    df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'value'}, inplace=True)
    return df

def preprocess_data(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    test = test.iloc[:50].copy()   # reduce test set for demo
    return train, test

# --- Baseline Forecast (Recursive) ---
def baseline_forecast(train, test):
    y_train = train['value'].values
    y_test = test['value'].values
    history = list(y_train)

    preds = []
    n_steps = min(len(y_test), 50)

    model = ARIMAModel()
    for t in range(n_steps):
        X_hist = np.arange(len(history)).reshape(-1, 1)
        model.fit(X_hist, np.array(history))
        X_pred = np.array([[len(history)]])
        pred = model.predict(X_pred)
        preds.append(pred[0])
        history.append(y_test[t])  # recursive forecast with actuals
    return np.array(preds), y_test[:n_steps]

# --- Multi-step Forecast (No Updating) ---
def multi_step_forecast(train, steps_ahead=[7, 14, 30]):
    y_train = train['value'].values
    history = list(y_train)

    model = ARIMAModel()
    X_hist = np.arange(len(history)).reshape(-1, 1)
    model.fit(X_hist, np.array(history))

    preds = []
    for i in range(1, max(steps_ahead) + 1):
        X_pred = np.array([[len(history)]])
        pred = model.predict(X_pred)
        preds.append(pred[0])
        history.append(pred[0])  # use predicted values, not actuals

    return {h: preds[h-1] for h in steps_ahead}

# --- Conformal Forecast ---
def fit_predict_conformal(train, test, method='adaptive', alpha=0.2):
    y_train = train['value'].values
    y_test = test['value'].values
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    model = ARIMAModel()

    if method == 'cvplus':
        conformal = CVPlusConformal(model, n_folds=5, alpha=alpha)
    else:
        conformal = AdaptiveConformal(model, threshold=0.1, alpha=alpha)

    conformal.fit(X_train, y_train)

    preds, lower, upper = [], [], []
    history = list(y_train)
    n_steps = min(len(y_test), 50)

    for t in range(n_steps):
        X_hist = np.arange(len(history)).reshape(-1, 1)
        conformal.model.fit(X_hist, np.array(history))
        X_pred = np.array([[len(history)]])
        pred, l, u = conformal.model.predict_interval(X_pred, confidence=1-alpha)
        preds.append(pred[0])
        lower.append(l[0])
        upper.append(u[0])
        history.append(y_test[t])

    return np.array(preds), np.array(lower), np.array(upper), y_test[:n_steps]

# --- Visualization ---
def plot_baseline(test, baseline_preds, rmse_b, mape_b):
    plt.figure(figsize=(10, 5))
    plt.plot(test['timestamp'], test['value'], label='Test', color='black')
    plt.plot(test['timestamp'], baseline_preds, label='Forecast', color='orange')
    plt.title("Baseline ARIMA Forecast (No Conformal)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()

    # Add RMSE & MAPE box
    textstr = f"RMSE: {rmse_b:.2f}\nMAPE: {mape_b:.2f}%"
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.tight_layout()
    plt.savefig("delhi_baseline_forecast.png")
    plt.show()

def plot_conformal(test, conformal_preds, lower, upper, rmse_c, mape_c, method):
    plt.figure(figsize=(10, 5))
    plt.plot(test['timestamp'], test['value'], label='Test', color='black')
    plt.plot(test['timestamp'], conformal_preds, label='Forecast', color='orange')
    plt.fill_between(test['timestamp'], lower, upper, color='red', alpha=0.4, label='Conformal Interval')
    plt.title(f"Forecast with {method.capitalize()} Conformal Intervals")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()

    # Add RMSE & MAPE box
    textstr = f"RMSE: {rmse_c:.2f}\nMAPE: {mape_c:.2f}%"
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.tight_layout()
    plt.savefig("delhi_conformal_forecast.png")
    plt.show()

def plot_multi_step(multi_step_preds):
    plt.figure(figsize=(8, 5))
    horizons = list(multi_step_preds.keys())
    values = list(multi_step_preds.values())
    plt.bar(horizons, values, color='orange', width=5)
    plt.title("Multi-step Ahead Forecast (No Updating with Actuals)")
    plt.xlabel("Forecast Horizon (Days Ahead)")
    plt.ylabel("Predicted Temperature (C)")
    plt.xticks(horizons)
    plt.tight_layout()
    plt.savefig("delhi_multistep_forecast.png")
    plt.show()

# --- Main Script ---
def main():
    df = load_delhi_temperature()
    train, test = preprocess_data(df)

    # --- Baseline forecast ---
    baseline_preds, y_test_b = baseline_forecast(train, test)
    rmse_b = RMSECalculator(); mape_b = MAPECalculator()
    rmse_b.update(y_test_b, baseline_preds)
    mape_b.update(y_test_b, baseline_preds)
    rmse_b_val = rmse_b.get_rmse()
    mape_b_val = mape_b.get_mape()

    # --- Conformal forecast ---
    conformal_preds, lower, upper, y_test_c = fit_predict_conformal(train, test, method='adaptive', alpha=0.2)
    rmse_c = RMSECalculator(); mape_c = MAPECalculator()
    rmse_c.update(y_test_c, conformal_preds)
    mape_c.update(y_test_c, conformal_preds)
    rmse_c_val = rmse_c.get_rmse()
    mape_c_val = mape_c.get_mape()

    # --- Multi-step forecast ---
    multi_step_preds = multi_step_forecast(train, steps_ahead=[7, 14, 30])

    # --- Plot all ---
    plot_baseline(test, baseline_preds, rmse_b_val, mape_b_val)
    plot_conformal(test, conformal_preds, lower, upper, rmse_c_val, mape_c_val, method="adaptive")
    plot_multi_step(multi_step_preds)

if __name__ == "__main__":
    main()
