import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Import your ConformL package components
try:
    from conforml.models.arima import ARIMAModel
    from conforml.models.sarima import SARIMAModel
    from conforml.models.linear import LinearRegressionModel
    from conforml.models.lstm import LSTMModel
    from conforml.conformal import CVPlusConformal, AdaptiveConformal
    from conforml.metrics import RMSECalculator, MAPECalculator
except ImportError:
    st.error("ConformL package not found. Please ensure it is installed (`pip install conforml`).")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="ConformL - Time Series Conformal Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Configuration")
    st.markdown("---")

    # Data source selection
    st.subheader("Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["Sample Dataset (Delhi Temperature)", "Upload CSV File"],
        key="data_source"
    )

    # Model configuration
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Choose a forecasting model:",
        ["ARIMA", "SARIMA", "Linear Regression", "LSTM"],
        key="model_type"
    )

    # Model-specific parameters
    st.subheader("Model Parameters")
    if model_type == "ARIMA":
        arima_p = st.slider("AR Order (p)", 0, 5, 1)
        arima_d = st.slider("Differencing (d)", 0, 2, 1)
        arima_q = st.slider("MA Order (q)", 0, 5, 1)
    elif model_type == "SARIMA":
        sarima_p = st.slider("AR Order (p)", 0, 5, 1)
        sarima_d = st.slider("Differencing (d)", 0, 2, 1)
        sarima_q = st.slider("MA Order (q)", 0, 5, 1)
        sarima_P = st.slider("Seasonal AR (P)", 0, 2, 0)
        sarima_D = st.slider("Seasonal Diff (D)", 0, 2, 0)
        sarima_Q = st.slider("Seasonal MA (Q)", 0, 2, 0)
        sarima_s = st.slider("Seasonality (s)", 1, 12, 12)
    elif model_type == "LSTM":
        lstm_units = st.slider("LSTM Units", 10, 200, 50)
        lstm_epochs = st.slider("Training Epochs", 5, 100, 20)
        lstm_batch_size = st.slider("Batch Size", 8, 128, 32)

    # Conformal prediction configuration
    st.subheader("Conformal Prediction")
    conformal_method = st.selectbox(
        "Choose conformal method:",
        ["CVPlusConformal", "AdaptiveConformal"],
        key="conformal_method"
    )
    alpha = st.slider("Significance Level (α)", 0.01, 0.5, 0.1, 0.01)
    if conformal_method == "CVPlusConformal":
        cv_folds = st.slider("CV Folds", 2, 10, 5)
    elif conformal_method == "AdaptiveConformal":
        adaptive_threshold = st.slider("Threshold", 0.01, 0.2, 0.05, 0.01)
        adaptive_method = st.selectbox("Adaptive Method", ["decay", "sliding"])
        if adaptive_method == "sliding":
            window_size = st.slider("Window Size", 10, 200, 50)

    # Data split configuration
    st.subheader("Data Split")
    train_split = st.slider("Training Set Ratio", 0.5, 0.95, 0.8, 0.05)
    test_steps = st.slider("Forecast Steps", 10, 100, 50)

    # Sidebar help
    st.markdown("---")
    st.subheader("ℹ️ Help")
    st.markdown("""
    **Model Types:**
    - **ARIMA**: AutoRegressive Integrated Moving Average.
    - **SARIMA**: Seasonal ARIMA for data with seasonality.
    - **Linear Regression**: Simple linear trend model.
    - **LSTM**: Neural network for complex patterns.

    **Conformal Methods:**
    - **CVPlusConformal**: Robust method for stationary series.
    - **AdaptiveConformal**: Adapts to non-stationary series.
    """)

# --- Data Loading Functions ---
@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    try:
        df = pd.read_csv(url, parse_dates=['Date'])
        df.rename(columns={'Date': 'timestamp', 'Temp': 'value'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, parse_dates=[0])
        df.columns = ['timestamp', 'value'] # Assume first two columns
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

# --- Model & Predictor Creation ---
def create_model(model_type):
    if model_type == "ARIMA":
        return ARIMAModel(order=(arima_p, arima_d, arima_q))
    elif model_type == "SARIMA":
        return SARIMAModel(
            order=(sarima_p, sarima_d, sarima_q),
            seasonal_order=(sarima_P, sarima_D, sarima_Q, sarima_s)
        )
    elif model_type == "Linear Regression":
        return LinearRegressionModel()
    elif model_type == "LSTM":
        return LSTMModel(
            input_shape=(1, 1), units=lstm_units,
            epochs=lstm_epochs, batch_size=lstm_batch_size
        )

def create_conformal_predictor(model):
    if conformal_method == "CVPlusConformal":
        return CVPlusConformal(model, alpha=alpha, n_folds=cv_folds)
    elif conformal_method == "AdaptiveConformal":
        kwargs = {
            'model': model, 'alpha': alpha,
            'threshold': adaptive_threshold, 'method': adaptive_method
        }
        if adaptive_method == "sliding":
            kwargs['window_size'] = window_size
        return AdaptiveConformal(**kwargs)

# --- Forecasting Pipeline ---
def run_forecast(df):
    try:
        # 1. Data preparation
        train_size = int(len(df) * train_split)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size : train_size + test_steps]
        
        y_train = train_df['value'].values
        y_test = test_df['value'].values
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

        # 2. Create model and conformal predictor
        model = create_model(model_type)
        conformal_predictor = create_conformal_predictor(model)

        # 3. Fit the conformal predictor ONCE on the training data
        with st.spinner("Training model and calibrating intervals..."):
            conformal_predictor.fit(X_train, y_train)

        # 4. Generate all predictions and intervals in ONE step
        with st.spinner("Generating forecasts..."):
            predictions, lower_bounds, upper_bounds = conformal_predictor.predict(X_test)
        
        # 5. Calculate metrics (CORRECTED)
        rmse_calc = RMSECalculator()
        rmse_calc.update(y_test, predictions)
        rmse = rmse_calc.get_rmse()

        mape_calc = MAPECalculator()
        mape_calc.update(y_test, predictions)
        mape = mape_calc.get_mape()
        
        return {
            'test_df': test_df, 'y_test': y_test,
            'predictions': predictions, 'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds, 'rmse': rmse, 'mape': mape
        }

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        return None

# --- Plotting Function ---
def plot_results(results):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series Forecast with Conformal Intervals', 'Prediction Residuals'),
        vertical_spacing=0.15, row_heights=[0.7, 0.3]
    )
    test_dates = results['test_df']['timestamp']
    confidence_level = (1 - alpha) * 100

    # Main forecast plot traces
    fig.add_trace(go.Scatter(x=test_dates, y=results['y_test'], name='Actual', mode='lines', line=dict(color='#2E86AB')), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=results['predictions'], name='Predicted', mode='lines', line=dict(color='#A23B72')), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_dates, test_dates[::-1]]),
        y=np.concatenate([results['upper_bounds'], results['lower_bounds'][::-1]]),
        fill='toself', fillcolor='rgba(242, 113, 28, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence_level:.0f}% Confidence Interval'
    ), row=1, col=1)

    # Residuals plot
    residuals = results['y_test'] - results['predictions']
    fig.add_trace(go.Scatter(x=test_dates, y=residuals, mode='markers', name='Residuals', marker=dict(color='#2E86AB', opacity=0.7)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)

    # Update layout
    fig.update_layout(height=600, title_text=f"{model_type} Forecast with {conformal_method}", title_x=0.5, plot_bgcolor='white')
    fig.update_xaxes(title_text="Date", showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Value", showgrid=True, gridcolor='#E5E5E5', row=1, col=1)
    fig.update_yaxes(title_text="Residual", showgrid=True, gridcolor='#E5E5E5', row=2, col=1)
    return fig

# --- Main App UI ---
st.markdown('<h1 class="main-header">ConformL Time Series Forecasting</h1>', unsafe_allow_html=True)
st.markdown("### A tool for robust forecasting with statistically guaranteed prediction intervals.")

# Load data based on user selection
df = None
if data_source == "Sample Dataset (Delhi Temperature)":
    df = load_sample_data()
elif data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV with 'timestamp' and 'value' columns", type="csv")
    if uploaded_file:
        df = load_uploaded_data(uploaded_file)

if df is not None:
    # Display data preview and plot
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    fig_orig = px.line(df, x='timestamp', y='value', title="Original Time Series Data", color_discrete_sequence=['#2E86AB'])
    st.plotly_chart(fig_orig, use_container_width=True)

    # Run forecast button
    if st.button("🚀 Run Forecast", type="primary"):
        st.session_state.results = run_forecast(df)

    # --- Display Results ---
    if st.session_state.results:
        results = st.session_state.results
        st.success("Forecasting completed successfully!")

        st.subheader("Performance Metrics")
        # Calculate coverage
        in_interval = (results['y_test'] >= results['lower_bounds']) & (results['y_test'] <= results['upper_bounds'])
        empirical_coverage = np.mean(in_interval) * 100
        avg_width = np.mean(results['upper_bounds'] - results['lower_bounds'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{results['rmse']:.4f}")
        col2.metric("MAPE", f"{results['mape']:.2f}%")
        col3.metric("Actual Coverage", f"{empirical_coverage:.1f}%", f"Target: {(1-alpha)*100:.0f}%")
        col4.metric("Avg. Interval Width", f"{avg_width:.3f}")

        # Plot results
        st.subheader("Forecast Results")
        fig_results = plot_results(results)
        st.plotly_chart(fig_results, use_container_width=True)

        # Prepare and show download button
        st.subheader("Export Results")
        results_df = pd.DataFrame({
            'timestamp': results['test_df']['timestamp'],
            'actual': results['y_test'],
            'predicted': results['predictions'],
            'lower_bound': results['lower_bounds'],
            'upper_bound': results['upper_bounds']
        })
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"forecast_results_{model_type}_{conformal_method}.csv",
            mime="text/csv",
        )
else:
    st.info("Please select a data source to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using the <strong>ConformL</strong> package and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)