import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from conforml.conformal import CVPlusConformal, AdaptiveConformal
from conforml.metrics import RMSECalculator, MAPECalculator
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# --- Data Loading ---
def load_delhi_temperature():
    ds = load_dataset("tarunchand/temperature")
    df = pd.DataFrame(ds['train'])
    df = df.rename(columns={
        'dt': 'timestamp',
        'AverageTemperature': 'value'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['City'].str.lower() == 'delhi'].copy()
    df = df.dropna(subset=['value'])
    df = df[['timestamp', 'value']].sort_values(by='timestamp').reset_index(drop=True)
    return df

def preprocess_data(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    test = test.iloc[:50].copy()
    return train, test

# --- Modeling & Conformal Prediction ---
def fit_predict_conformal_prophet(train, test, method='cvplus', alpha=0.1):
    # Prophet expects columns 'ds' and 'y'
    prophet_train = train.rename(columns={'timestamp': 'ds', 'value': 'y'})
    prophet_test = test.rename(columns={'timestamp': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_train)
    future = prophet_test[['ds']]
    forecast = model.predict(future)
    preds = forecast['yhat'].values
    # For demonstration, use Prophet's intervals as conformal intervals
    lower = forecast['yhat_lower'].values
    upper = forecast['yhat_upper'].values
    y_test = prophet_test['y'].values
    return preds, lower, upper, y_test

# --- Visualization ---
def plot_forecast_with_intervals(train, test, preds, lower, upper):
    plt.figure(figsize=(14, 6))
    plt.plot(test['timestamp'], test['value'], label='Test', color='black')
    plt.plot(test['timestamp'], preds, label='Forecast', color='blue')
    plt.fill_between(test['timestamp'], lower, upper, color='red', alpha=0.4, label='Prophet Interval')
    plt.title('Delhi Temperature Forecast with Prophet Intervals')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('delhi_prophet_forecast_with_intervals.png')
    plt.show()

def plot_interval_width(test, lower, upper):
    width = upper - lower
    plt.figure(figsize=(10, 3))
    plt.plot(test['timestamp'], width, color='purple')
    plt.title('Prediction Interval Width Over Time')
    plt.xlabel('Date')
    plt.ylabel('Interval Width')
    plt.tight_layout()
    plt.savefig('delhi_prophet_interval_width.png')
    plt.show()

def plot_coverage(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = np.cumsum(covered) / (np.arange(len(covered)) + 1)
    plt.figure(figsize=(10, 3))
    plt.plot(coverage, color='green')
    plt.title('Empirical Coverage Over Time')
    plt.xlabel('Test Index')
    plt.ylabel('Coverage Rate')
    plt.tight_layout()
    plt.savefig('delhi_prophet_coverage.png')
    plt.show()

def plot_residuals(y_true, preds):
    residuals = y_true - preds
    plt.figure(figsize=(10, 3))
    plt.plot(residuals, color='gray')
    plt.title('Forecast Residuals')
    plt.xlabel('Test Index')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig('delhi_prophet_residuals.png')
    plt.show()

# --- Main Script ---
def main():
    df = load_delhi_temperature()
    train, test = preprocess_data(df)
    preds, lower, upper, y_test = fit_predict_conformal_prophet(train, test, method='cvplus', alpha=0.1)
    plot_forecast_with_intervals(train, test, preds, lower, upper)
    plot_interval_width(test, lower, upper)
    plot_coverage(y_test, lower, upper)
    plot_residuals(y_test, preds)

    # Metrics
    rmse_calc = RMSECalculator()
    mape_calc = MAPECalculator()
    rmse_calc.update(y_test, preds)
    mape_calc.update(y_test, preds)
    print(f"RMSE: {rmse_calc.get_rmse():.4f}")
    print(f"MAPE: {mape_calc.get_mape():.2f}%")

if __name__ == "__main__":
    main()
