import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Import ConformL components
try:
    from conforml.models.arima import ARIMAModel
    from conforml.models.sarima import SARIMAModel
    from conforml.models.lstm import LSTMModel
    from conforml.conformal import CVPlusConformal, AdaptiveConformal, IntervalSharpnessConformal
    from conforml.metrics import RMSECalculator, MAPECalculator
except ImportError:
    st.error("ConformL package not found. Please install the conforml package first.")
    st.stop()

# Page config
st.set_page_config(
    page_title="ConformL - Time Series Conformal Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header"> ConformL Time Series Forecasting</h1>', unsafe_allow_html=True)
st.markdown("### Forecasting with Conformal Prediction Intervals")

# Sidebar
st.sidebar.title(" Configuration")
st.sidebar.markdown("---")

# Data source
st.sidebar.subheader(" Data Source")
data_source = st.sidebar.radio(
    "Choose your data source:",
    ["Sample Dataset (Delhi Temperature)", "Upload CSV File"]
)

# Model config
st.sidebar.subheader(" Model Selection")
model_type = st.sidebar.selectbox(
    "Choose a forecasting model:",
    ["ARIMA", "SARIMA", "Linear Regression", "LSTM"]
)

# Model parameters
st.sidebar.subheader(" Model Parameters")
if model_type == "ARIMA":
    arima_p = st.sidebar.slider("AR Order (p)", 0, 5, 1)
    arima_d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
    arima_q = st.sidebar.slider("MA Order (q)", 0, 5, 1)
elif model_type == "SARIMA":
    sarima_p = st.sidebar.slider("AR Order (p)", 0, 5, 1)
    sarima_d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
    sarima_q = st.sidebar.slider("MA Order (q)", 0, 5, 1)
    sarima_P = st.sidebar.slider("Seasonal AR (P)", 0, 2, 0)
    sarima_D = st.sidebar.slider("Seasonal Diff (D)", 0, 2, 0)
    sarima_Q = st.sidebar.slider("Seasonal MA (Q)", 0, 2, 0)
    sarima_s = st.sidebar.slider("Seasonality (s)", 1, 12, 12)
elif model_type == "LSTM":
    lstm_units = st.sidebar.slider("LSTM Units", 10, 200, 50)
    lstm_epochs = st.sidebar.slider("Training Epochs", 5, 100, 20)
    lstm_batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Conformal config
st.sidebar.subheader(" Conformal Prediction")
conformal_method = st.sidebar.selectbox(
    "Choose conformal method:",
    ["CVPlusConformal", "AdaptiveConformal", "ISOC"]
)

alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.5, 0.1, 0.01)
confidence_level = (1 - alpha) * 100

if conformal_method == "CVPlusConformal":
    cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5)
elif conformal_method == "AdaptiveConformal":
    adaptive_threshold = st.sidebar.slider("Threshold", 0.01, 0.2, 0.05, 0.01)
    adaptive_method = st.sidebar.selectbox("Adaptive Method", ["decay", "sliding"])
    if adaptive_method == "sliding":
        window_size = st.sidebar.slider("Window Size", 10, 200, 50)
elif conformal_method == "ISOC":
    sharpness_weight = st.sidebar.slider("Sharpness Weight λ", 0.0, 1.0, 0.5, 0.05)

# Train-test config
st.sidebar.subheader(" Data Split")
train_split = st.sidebar.slider("Training Set Ratio", 0.5, 0.95, 0.8, 0.05)
test_steps = st.sidebar.slider("Test Steps", 10, 100, 50)

# === Data loading ===
def load_sample_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, parse_dates=['Date'])
    df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'value'}, inplace=True)
    return df

def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=[0])
    df.columns = ['timestamp', 'value']
    return df

# === Model creation ===
def create_model(model_type):
    if model_type == "ARIMA":
        return ARIMAModel(order=(arima_p, arima_d, arima_q))
    elif model_type == "SARIMA":
        return SARIMAModel(order=(sarima_p, sarima_d, sarima_q),
                           seasonal_order=(sarima_P, sarima_D, sarima_Q, sarima_s))
    elif model_type == "Linear Regression":
        from conforml.models.linear import LinearRegressionModel
        return LinearRegressionModel()
    elif model_type == "LSTM":
        return LSTMModel(units=lstm_units, epochs=lstm_epochs, batch_size=lstm_batch_size)

# === Conformal creation ===
def create_conformal_predictor(model):
    if conformal_method == "CVPlusConformal":
        return CVPlusConformal(model, alpha=alpha, n_folds=cv_folds)
    elif conformal_method == "AdaptiveConformal":
        kwargs = {'model': model, 'alpha': alpha, 'threshold': adaptive_threshold, 'method': adaptive_method}
        if adaptive_method == "sliding":
            kwargs['window_size'] = window_size
        return AdaptiveConformal(**kwargs)
    elif conformal_method == "ISOC":
        return IntervalSharpnessConformal(model, alpha=alpha, lambda_sharpness=sharpness_weight)

# === Forecast runner ===
def run_forecast(df):
    train_size = int(len(df) * train_split)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:train_size+test_steps].copy()

    y_train = train['value'].values
    y_test = test['value'].values
    X_train = np.arange(len(y_train)).reshape(-1, 1)

    model = create_model(model_type)
    conformal = create_conformal_predictor(model)

    with st.spinner("Training model and fitting conformal predictor..."):
        conformal.fit(X_train, y_train)

    predictions, lower_bounds, upper_bounds = [], [], []
    history = list(y_train)

    with st.spinner("Generating predictions..."):
        for t in range(len(y_test)):
            X_hist = np.arange(len(history)).reshape(-1, 1)
            try:
                conformal.model.fit(X_hist, np.array(history))
            except Exception:
                try:
                    conformal.model.fit(None, np.array(history))
                except Exception:
                    pass

            X_pred = np.array([[len(history)]])
            pred, lower, upper = conformal.predict(X_pred)
            predictions.append(pred[0] if hasattr(pred, "__getitem__") else pred)
            lower_bounds.append(lower[0] if hasattr(lower, "__getitem__") else lower)
            upper_bounds.append(upper[0] if hasattr(upper, "__getitem__") else upper)

            history.append(y_test[t])

    rmse_calc = RMSECalculator()
    mape_calc = MAPECalculator()
    rmse_calc.update(y_test, np.array(predictions))
    mape_calc.update(y_test, np.array(predictions))

    return {
        'test': test,
        'predictions': np.array(predictions),
        'lower_bounds': np.array(lower_bounds),
        'upper_bounds': np.array(upper_bounds),
        'rmse': rmse_calc.get_rmse(),
        'mape': mape_calc.get_mape(),
        'y_test': y_test
    }

# === Plotting results (only test set) ===
def plot_results(results):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Test Forecast with Conformal Intervals', 'Prediction Errors'),
                        vertical_spacing=0.12, row_heights=[0.7, 0.3])

    test_dates = results['test']['timestamp'].values

    # Actual values
    fig.add_trace(go.Scatter(x=test_dates, y=results['y_test'], mode='lines+markers',
                             name='Actual', line=dict(color='black', width=2)), row=1, col=1)

    # Predictions
    fig.add_trace(go.Scatter(x=test_dates, y=results['predictions'], mode='lines+markers',
                             name='Predicted', line=dict(color='orange', width=2)), row=1, col=1)

    # Conformal interval
    fig.add_trace(go.Scatter(x=test_dates, y=results['upper_bounds'], mode='lines',
                             name='Upper Bound', line=dict(color='red', width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=results['lower_bounds'], mode='lines',
                             name=f'{confidence_level:.0f}% Interval', line=dict(color='red', width=0),
                             fill='tonexty', fillcolor='rgba(255, 0, 0, 0.3)'), row=1, col=1)

    # Residuals
    residuals = results['y_test'] - results['predictions']
    fig.add_trace(go.Scatter(x=test_dates, y=residuals, mode='markers', name='Residuals',
                             marker=dict(color='blue', size=6), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=800, title_text=f"{model_type} with {conformal_method}", title_x=0.5)
    return fig

# === Main app ===
if data_source == "Sample Dataset (Delhi Temperature)":
    df = load_sample_data()
elif data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    df = load_uploaded_data(uploaded_file) if uploaded_file else None

if df is not None:
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Records", len(df))
    with col2: st.metric("Training Records", int(len(df) * train_split))
    with col3: st.metric("Test Records", min(test_steps, len(df) - int(len(df) * train_split)))

    st.subheader(" Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button(" Run Forecast", type="primary"):
        results = run_forecast(df)
        if results:
            st.success("✅ Forecasting completed successfully!")
            col1, col2 = st.columns(2)
            with col1: st.metric("RMSE", f"{results['rmse']:.4f}")
            with col2: st.metric("MAPE", f"{results['mape']:.2f}%")
            st.subheader(" Forecast Results (Test Set Only)")
            st.plotly_chart(plot_results(results), use_container_width=True)

            in_interval = ((results['y_test'] >= results['lower_bounds']) &
                           (results['y_test'] <= results['upper_bounds']))
            empirical_coverage = np.mean(in_interval) * 100

            st.subheader(" Conformal Prediction Analysis")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Target Coverage", f"{confidence_level:.0f}%")
            with col2: st.metric("Empirical Coverage", f"{empirical_coverage:.1f}%")
            with col3: st.metric("Avg Interval Width", f"{np.mean(results['upper_bounds'] - results['lower_bounds']):.2f}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ConformL Package</div>", unsafe_allow_html=True)
