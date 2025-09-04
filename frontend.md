import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    st.error("ConformL package not found. Please install the conforml package first.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="ConformL - Time Series Conformal Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 ConformL Time Series Forecasting</h1>', unsafe_allow_html=True)
st.markdown("### A user-friendly interface for time series forecasting with conformal prediction intervals")

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Data source selection
st.sidebar.subheader("📊 Data Source")
data_source = st.sidebar.radio(
    "Choose your data source:",
    ["Sample Dataset (Delhi Temperature)", "Upload CSV File"]
)

# Model configuration
st.sidebar.subheader("🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose a forecasting model:",
    ["ARIMA", "SARIMA", "Linear Regression", "LSTM"]
)

# Model-specific parameters
st.sidebar.subheader("⚙️ Model Parameters")
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

# Conformal prediction configuration
st.sidebar.subheader("🎯 Conformal Prediction")
conformal_method = st.sidebar.selectbox(
    "Choose conformal method:",
    ["CVPlusConformal", "AdaptiveConformal"]
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

# Train-test split configuration
st.sidebar.subheader("📊 Data Split")
train_split = st.sidebar.slider("Training Set Ratio", 0.5, 0.95, 0.8, 0.05)
test_steps = st.sidebar.slider("Test Steps", 10, 100, 50)

# Main content area
def load_sample_data():
    """Load the Delhi temperature dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    try:
        df = pd.read_csv(url, parse_dates=['Date'])
        df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'value'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def load_uploaded_data(uploaded_file):
    """Load uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file, parse_dates=[0])
        # Assume first column is timestamp, second is value
        df.columns = ['timestamp', 'value']
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

def create_model(model_type):
    """Create model instance based on selection"""
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
            input_shape=(1, 1),
            units=lstm_units,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size
        )

def create_conformal_predictor(model):
    """Create conformal predictor based on selection"""
    if conformal_method == "CVPlusConformal":
        return CVPlusConformal(model, alpha=alpha, n_folds=cv_folds)
    elif conformal_method == "AdaptiveConformal":
        kwargs = {
            'model': model,
            'alpha': alpha,
            'threshold': adaptive_threshold,
            'method': adaptive_method
        }
        if adaptive_method == "sliding":
            kwargs['window_size'] = window_size
        return AdaptiveConformal(**kwargs)

def run_forecast(df):
    """Run the complete forecasting pipeline"""
    try:
        # Data preparation
        train_size = int(len(df) * train_split)
        train = df.iloc[:train_size].copy()
        test = df.iloc[train_size:train_size+test_steps].copy()
        
        # Prepare data for modeling
        y_train = train['value'].values
        y_test = test['value'].values
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        
        # Create model and conformal predictor
        model = create_model(model_type)
        conformal = create_conformal_predictor(model)
        
        # Fit conformal predictor
        with st.spinner("Training model and fitting conformal predictor..."):
            conformal.fit(X_train, y_train)
        
        # Generate predictions
        predictions = []
        lower_bounds = []
        upper_bounds = []
        history = list(y_train)
        
        with st.spinner("Generating predictions..."):
            for t in range(len(y_test)):
                X_hist = np.arange(len(history)).reshape(-1, 1)
                conformal.model.fit(X_hist, np.array(history))
                X_pred = np.array([[len(history)]])
                
                if hasattr(conformal.model, 'predict_interval'):
                    pred, lower, upper = conformal.model.predict_interval(X_pred, confidence=1-alpha)
                else:
                    pred, lower, upper = conformal.predict(X_pred)
                
                predictions.append(pred[0] if hasattr(pred, '__getitem__') else pred)
                lower_bounds.append(lower[0] if hasattr(lower, '__getitem__') else lower)
                upper_bounds.append(upper[0] if hasattr(upper, '__getitem__') else upper)
                history.append(y_test[t])
        
        # Calculate metrics
        rmse_calc = RMSECalculator()
        mape_calc = MAPECalculator()
        rmse_calc.update(y_test, np.array(predictions))
        mape_calc.update(y_test, np.array(predictions))
        
        results = {
            'train': train,
            'test': test,
            'predictions': np.array(predictions),
            'lower_bounds': np.array(lower_bounds),
            'upper_bounds': np.array(upper_bounds),
            'rmse': rmse_calc.get_rmse(),
            'mape': mape_calc.get_mape(),
            'y_test': y_test
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        return None

def plot_results(results):
    """Create interactive plots using Plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series Forecast with Conformal Intervals', 'Prediction Errors'),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Main forecast plot
    test_dates = results['test']['timestamp'].values
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=results['y_test'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=results['predictions'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=results['upper_bounds'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='red', width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=results['lower_bounds'],
            mode='lines',
            name=f'{confidence_level:.0f}% Confidence Interval',
            line=dict(color='red', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)'
        ),
        row=1, col=1
    )
    
    # Add residuals plot
    residuals = results['y_test'] - results['predictions']
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=6),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{model_type} Model with {conformal_method}",
        title_x=0.5,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    
    return fig

# Main application logic
if data_source == "Sample Dataset (Delhi Temperature)":
    df = load_sample_data()
elif data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
    else:
        df = None
        st.info("Please upload a CSV file with timestamp and value columns.")

# Display data information
if df is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Training Records", int(len(df) * train_split))
    with col3:
        st.metric("Test Records", min(test_steps, len(df) - int(len(df) * train_split)))
    
    # Show data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Plot original data
    st.subheader("📈 Original Time Series")
    fig_orig = px.line(df, x='timestamp', y='value', title="Complete Time Series")
    fig_orig.update_layout(height=400)
    st.plotly_chart(fig_orig, use_container_width=True)
    
    # Run forecast button
    if st.button("🚀 Run Forecast", type="primary"):
        results = run_forecast(df)
        
        if results is not None:
            # Display results
            st.success("✅ Forecasting completed successfully!")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{results['rmse']:.4f}")
            with col2:
                st.metric("MAPE", f"{results['mape']:.2f}%")
            
            # Plot results
            st.subheader("📊 Forecast Results")
            fig_results = plot_results(results)
            st.plotly_chart(fig_results, use_container_width=True)
            
            # Coverage analysis
            in_interval = (
                (results['y_test'] >= results['lower_bounds']) & 
                (results['y_test'] <= results['upper_bounds'])
            )
            empirical_coverage = np.mean(in_interval) * 100
            
            st.subheader("🎯 Conformal Prediction Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Target Coverage", f"{confidence_level:.0f}%")
            with col2:
                st.metric("Empirical Coverage", f"{empirical_coverage:.1f}%")
            with col3:
                avg_width = np.mean(results['upper_bounds'] - results['lower_bounds'])
                st.metric("Avg Interval Width", f"{avg_width:.2f}")
            
            # Coverage status
            coverage_diff = abs(empirical_coverage - confidence_level)
            if coverage_diff <= 5:
                st.success(f"✅ Good coverage! Difference: {coverage_diff:.1f}%")
            elif coverage_diff <= 10:
                st.warning(f"⚠️ Acceptable coverage. Difference: {coverage_diff:.1f}%")
            else:
                st.error(f"❌ Poor coverage. Difference: {coverage_diff:.1f}%")
            
            # Download results
            if st.button("💾 Download Results"):
                results_df = pd.DataFrame({
                    'timestamp': results['test']['timestamp'],
                    'actual': results['y_test'],
                    'predicted': results['predictions'],
                    'lower_bound': results['lower_bounds'],
                    'upper_bound': results['upper_bounds']
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"forecast_results_{model_type}_{conformal_method}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using ConformL package and Streamlit</p>
        <p>For more information about conformal prediction, visit the ConformL documentation</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar help
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Help")
st.sidebar.markdown("""
**Model Types:**
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA with seasonal components
- **Linear Regression**: Simple linear trend model
- **LSTM**: Long Short-Term Memory neural network

**Conformal Methods:**
- **CVPlusConformal**: Cross-validation based method for stationary series
- **AdaptiveConformal**: Adaptive method for non-stationary series

**Parameters:**
- **α (Alpha)**: Significance level (lower = wider intervals)
- **CV Folds**: Number of cross-validation folds
- **Threshold**: Adaptation rate for adaptive method
""")