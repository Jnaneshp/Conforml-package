import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio  # for static image export (needs kaleido installed)

warnings.filterwarnings("ignore")

# -------------------------------
# Import ConformL components
# -------------------------------
try:
    from conforml.models.arima import ARIMAModel
    from conforml.models.sarima import SARIMAModel
    from conforml.models.prophet import ProphetModel
    from conforml.models.xgboost_model import XGBoostTimeSeriesModel
    from conforml.conformal import CVPlusConformal, AdaptiveConformal, IntervalSharpnessConformal
    from conforml.metrics import RMSECalculator, MAPECalculator
except ImportError as e:
    st.error(f"❌ Error importing ConformL components: {e}")
    st.stop()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Conformal Prediction for Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">ConformL Time Series Forecasting</h1>', unsafe_allow_html=True)
st.markdown("### Forecasting with Conformal Prediction Intervals")

# -------------------------------
# Sidebar Config
# -------------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Data Source
st.sidebar.subheader("📂 Data Source")
data_source = st.sidebar.radio(
    "Choose your data source:",
    ["Sample Dataset (Delhi Temperature)", "Upload CSV File"]
)

# Model Selection (single-mode UI still present for single run)
st.sidebar.subheader("🤖 Model Selection (single run)")
model_type = st.sidebar.selectbox(
    "Choose a forecasting model:",
    ["ARIMA", "SARIMA", "Prophet", "XGBoost"]
)

# Model Compare selection
st.sidebar.subheader("🔁 Model Compare")
models_to_compare = st.sidebar.multiselect(
    "Select models to compare (leaderboard):",
    ["ARIMA", "SARIMA", "Prophet", "XGBoost"],
    default=["ARIMA", "SARIMA", "Prophet"]
)

# Model Parameters
st.sidebar.subheader("📌 Model Parameters")
if model_type == "ARIMA" or "ARIMA" in models_to_compare:
    arima_p = st.sidebar.slider("AR Order (p)", 0, 5, 1)
    arima_d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
    arima_q = st.sidebar.slider("MA Order (q)", 0, 5, 1)
if model_type == "SARIMA" or "SARIMA" in models_to_compare:
    sarima_p = st.sidebar.slider("SARIMA AR (p)", 0, 5, 1)
    sarima_d = st.sidebar.slider("SARIMA Diff (d)", 0, 2, 1)
    sarima_q = st.sidebar.slider("SARIMA MA (q)", 0, 5, 1)
    sarima_P = st.sidebar.slider("Seasonal AR (P)", 0, 2, 0)
    sarima_D = st.sidebar.slider("Seasonal Diff (D)", 0, 2, 0)
    sarima_Q = st.sidebar.slider("Seasonal MA (Q)", 0, 2, 0)
    sarima_s = st.sidebar.slider("Seasonality (s)", 1, 12, 12)
if model_type == "Prophet" or "Prophet" in models_to_compare:
    yearly = st.sidebar.checkbox("Yearly Seasonality", True)
    weekly = st.sidebar.checkbox("Weekly Seasonality", True)
    daily = st.sidebar.checkbox("Daily Seasonality", False)
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"])

# Conformal Config
st.sidebar.subheader("📏 Conformal Prediction")
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

# Data Split
st.sidebar.subheader("🧪 Data Split")
test_steps = st.sidebar.slider("Test Steps", 10, 100, 50)

# Model Compare run button
run_compare = st.sidebar.button("▶️ Run Compare (selected models)")

# -------------------------------
# Data Loading
# -------------------------------
def load_sample_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, parse_dates=['Date'])
    df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'value'}, inplace=True)
    return df

def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=[0])
    df.columns = ['timestamp', 'value']
    return df

# -------------------------------
# Model Creation
# -------------------------------
def create_model(model_type):
    if model_type == "ARIMA":
        return ARIMAModel(order=(arima_p, arima_d, arima_q))
    elif model_type == "SARIMA":
        return SARIMAModel(order=(sarima_p, sarima_d, sarima_q),
                           seasonal_order=(sarima_P, sarima_D, sarima_Q, sarima_s))
    elif model_type == "Prophet":
        return ProphetModel(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            seasonality_mode=seasonality_mode
        )
    elif model_type == "XGBoost":
        return XGBoostTimeSeriesModel()

# -------------------------------
# Conformal Predictor
# -------------------------------
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

# -------------------------------
# Forecast Runner (fresh training)
# -------------------------------
def run_forecast(df, model_type, test_steps):
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:train_size + test_steps].copy()

    y_train = train['value'].values
    y_test = test['value'].values

    model = create_model(model_type)

    # ---- Pre-train model on full training data ----
    if model_type == "Prophet":
        model.fit(train[['timestamp', 'value']].rename(columns={'timestamp': 'ds', 'value': 'y'}))
        conformal = create_conformal_predictor(model)
        conformal.fit(train['timestamp'], y_train)
    else:
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        model.fit(X_train, y_train)
        conformal = create_conformal_predictor(model)
        conformal.fit(X_train, y_train)

    # ---- Forecasting ----
    predictions, lower_bounds, upper_bounds = [], [], []
    history = list(y_train)

    for t in range(len(y_test)):
        if model_type == "Prophet":
            X_pred = [test['timestamp'].iloc[t]]
            pred, lower, upper = conformal.predict(X_pred)
        else:
            X_pred = np.array([[len(history)]])
            pred, lower, upper = conformal.predict(X_pred)

        predictions.append(pred[0] if hasattr(pred, "__getitem__") else pred)
        lower_bounds.append(lower[0] if hasattr(lower, "__getitem__") else lower)
        upper_bounds.append(upper[0] if hasattr(upper, "__getitem__") else upper)

        history.append(y_test[t])
        if model_type != "Prophet":
            model.fit(np.arange(len(history)).reshape(-1, 1), np.array(history))

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

# -------------------------------
# Plot Results (accept title override)
# -------------------------------
def plot_results(results, title_override=None):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Test Forecast with Conformal Intervals', 'Prediction Errors'),
        vertical_spacing=0.12, row_heights=[0.65, 0.35]
    )

    test_dates = results['test']['timestamp'].values

    fig.add_trace(go.Scatter(x=test_dates, y=results['y_test'], mode='lines+markers',
                             name='Actual', line=dict(color='black', width=2),
                             marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=results['predictions'], mode='lines+markers',
                             name='Predicted', line=dict(color='orange', width=2),
                             marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=results['upper_bounds'], mode='lines',
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=results['lower_bounds'], mode='lines',
                             name=f'{confidence_level:.0f}% Interval', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255,0,0,0.20)'), row=1, col=1)

    residuals = results['y_test'] - results['predictions']
    fig.add_trace(go.Scatter(x=test_dates, y=residuals, mode='markers',
                             marker=dict(color='blue', size=6), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    title_text = title_override if title_override is not None else f"{model_type} with {conformal_method}"
    fig.update_layout(
        height=750,
        title_text=title_text,
        title_x=0.5,
        template="plotly_white",
        margin=dict(l=70, r=40, t=80, b=60)
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

# -------------------------------
# Conformal Metrics Plot (combined, improved)
# -------------------------------
def plot_conformal_metrics(results, title_override=None):
    test_dates = results['test']['timestamp'].values
    actual = np.array(results['y_test'])
    pred = np.array(results['predictions'])
    lower = np.array(results['lower_bounds'])
    upper = np.array(results['upper_bounds'])

    in_interval = (actual >= lower) & (actual <= upper)
    interval_width = upper - lower
    abs_error = np.abs(actual - pred)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"Coverage Indicator (Target {confidence_level:.0f}%)",
            "Interval Width Over Time",
            "Absolute Error Over Time"
        ),
        vertical_spacing=0.10,
        row_heights=[0.30, 0.35, 0.35]
    )

    color_actual_in = "#2ca02c"
    color_actual_out = "#d62728"
    color_pred = "#1f77b4"
    color_width = "#ff7f0e"
    color_abs = "#9467bd"

    # Coverage – split inside/outside for clearer legend
    fig.add_trace(
        go.Scatter(
            x=test_dates[in_interval],
            y=actual[in_interval],
            mode="markers+lines",
            line=dict(color=color_actual_in, width=1.8),
            marker=dict(color=color_actual_in, size=7),
            name="Actual (inside interval)"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates[~in_interval],
            y=actual[~in_interval],
            mode="markers",
            marker=dict(color=color_actual_out, size=8, symbol="x"),
            name="Actual (outside interval)"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=pred,
            mode="lines",
            name="Predicted",
            line=dict(color=color_pred, dash="dash", width=1.8)
        ),
        row=1, col=1
    )

    # Interval width
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=interval_width,
            mode="lines",
            name="Interval Width",
            line=dict(color=color_width, width=2.0)
        ),
        row=2, col=1
    )

    # Absolute error
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=abs_error,
            mode="lines",
            name="Absolute Error",
            line=dict(color=color_abs, width=2.0)
        ),
        row=3, col=1
    )

    title_text = title_override if title_override is not None else "Conformal Metrics Overview"
    fig.update_layout(
        title=title_text,
        title_x=0.5,
        height=900,
        template="plotly_white",
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

# -------------------------------
# Separate conformal metric figures (improved)
# -------------------------------
def plot_conformal_metrics_separate(results, title_prefix=None):
    test_dates = results['test']['timestamp'].values
    actual = np.array(results['y_test'])
    pred = np.array(results['predictions'])
    lower = np.array(results['lower_bounds'])
    upper = np.array(results['upper_bounds'])

    in_interval = (actual >= lower) & (actual <= upper)
    interval_width = upper - lower
    abs_error = np.abs(actual - pred)

    prefix = title_prefix or ""

    color_actual_in = "#2ca02c"
    color_actual_out = "#d62728"
    color_pred = "#1f77b4"
    color_width = "#ff7f0e"
    color_abs = "#9467bd"

    # Coverage
    fig_coverage = go.Figure()
    fig_coverage.add_trace(
        go.Scatter(
            x=test_dates[in_interval],
            y=actual[in_interval],
            mode="markers+lines",
            line=dict(color=color_actual_in, width=2),
            marker=dict(color=color_actual_in, size=7),
            name="Actual (inside interval)"
        )
    )
    fig_coverage.add_trace(
        go.Scatter(
            x=test_dates[~in_interval],
            y=actual[~in_interval],
            mode="markers",
            marker=dict(color=color_actual_out, size=8, symbol="x"),
            name="Actual (outside interval)"
        )
    )
    fig_coverage.add_trace(
        go.Scatter(
            x=test_dates,
            y=pred,
            mode="lines",
            name="Predicted",
            line=dict(color=color_pred, dash="dash", width=2)
        )
    )
    fig_coverage.update_layout(
        title=f"{prefix}Coverage Indicator (Target {confidence_level:.0f}%)",
        xaxis_title="Time",
        yaxis_title="Value",
        height=500,
        width=900,
        margin=dict(l=70, r=40, t=80, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white"
    )
    fig_coverage.update_xaxes(tickangle=-25)

    # Interval width
    fig_width = go.Figure()
    fig_width.add_trace(
        go.Scatter(
            x=test_dates,
            y=interval_width,
            mode="lines+markers",
            line=dict(color=color_width, width=2),
            marker=dict(size=6),
            name="Interval Width"
        )
    )
    fig_width.update_layout(
        title=f"{prefix}Interval Width Over Time",
        xaxis_title="Time",
        yaxis_title="Width",
        height=500,
        width=900,
        margin=dict(l=70, r=40, t=80, b=70),
        template="plotly_white"
    )
    fig_width.update_xaxes(tickangle=-25)

    # Absolute error
    fig_abs = go.Figure()
    fig_abs.add_trace(
        go.Scatter(
            x=test_dates,
            y=abs_error,
            mode="lines+markers",
            line=dict(color=color_abs, width=2),
            marker=dict(size=6),
            name="Absolute Error"
        )
    )
    fig_abs.update_layout(
        title=f"{prefix}Absolute Error Over Time",
        xaxis_title="Time",
        yaxis_title="Absolute Error",
        height=500,
        width=900,
        margin=dict(l=70, r=40, t=80, b=70),
        template="plotly_white"
    )
    fig_abs.update_xaxes(tickangle=-25)

    return fig_coverage, fig_width, fig_abs

# -------------------------------
# Run multiple models and collect leaderboard (cached)
# -------------------------------
@st.cache_data(show_spinner=False)
def run_models_compare_cached(df_serialized, models_list, test_steps, config):
    df = pd.read_csv(io.BytesIO(df_serialized), parse_dates=['timestamp'])
    results_by_model = {}
    leaderboard_rows = []

    for m in models_list:
        start = time.time()
        try:
            globals().update(config.get('globals', {}))
            res = run_forecast(df, m, test_steps)
            end = time.time()
            runtime = end - start

            in_interval = ((res['y_test'] >= res['lower_bounds']) & (res['y_test'] <= res['upper_bounds']))
            empirical_coverage = float(np.mean(in_interval) * 100)
            avg_interval_width = float(np.mean(res['upper_bounds'] - res['lower_bounds']))

            leaderboard_rows.append({
                'model': m,
                'rmse': float(res['rmse']),
                'mape': float(res['mape']),
                'empirical_coverage': empirical_coverage,
                'avg_interval_width': avg_interval_width,
                'runtime_sec': runtime
            })
            results_by_model[m] = res
        except Exception as e:
            leaderboard_rows.append({
                'model': m,
                'rmse': np.nan,
                'mape': np.nan,
                'empirical_coverage': np.nan,
                'avg_interval_width': np.nan,
                'runtime_sec': np.nan,
                'error': str(e)
            })
            results_by_model[m] = {'error': str(e)}
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df = leaderboard_df.sort_values(by=['rmse'], na_position='last').reset_index(drop=True)
    return leaderboard_df, results_by_model

# -------------------------------
# Main App Logic
# -------------------------------
if data_source == "Sample Dataset (Delhi Temperature)":
    df = load_sample_data()
elif data_source == "Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    df = load_uploaded_data(uploaded_file) if uploaded_file else None
else:
    df = None

if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- Single-run forecast ---
    col_left, col_right = st.columns([1, 1])
    button_pressed = col_left.button("🚀 Run Forecast (single model)")

    if button_pressed:
        with st.spinner("Running forecast..."):
            try:
                results = run_forecast(df, model_type, test_steps)
            except Exception as e:
                st.error(f"❌ Error during forecasting: {e}")
                results = None

        if results:
            st.success("✅ Forecasting completed successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{results['rmse']:.4f}")
            with col2:
                st.metric("MAPE", f"{results['mape']:.2f}%")

            st.subheader("📈 Forecast Results (Test Set Only)")
            single_forecast_fig = plot_results(results)
            st.plotly_chart(single_forecast_fig, use_container_width=True)

            try:
                img_bytes = single_forecast_fig.to_image(format="png", scale=2)
                st.download_button(
                    label="💾 Download Forecast Plot (PNG)",
                    data=img_bytes,
                    file_name=f"{model_type}_{conformal_method}_forecast.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Image export not available: {e}")

            in_interval = ((results['y_test'] >= results['lower_bounds']) &
                           (results['y_test'] <= results['upper_bounds']))
            empirical_coverage = np.mean(in_interval) * 100
            avg_interval_width = np.mean(results['upper_bounds'] - results['lower_bounds'])

            st.subheader("📌 Conformal Prediction Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Target Coverage", f"{confidence_level:.0f}%")
            with c2:
                st.metric("Empirical Coverage", f"{empirical_coverage:.1f}%")
            with c3:
                st.metric("Avg Interval Width", f"{avg_interval_width:.2f}")

            st.subheader("📉 Conformal Metrics Graphs")
            single_conformal_fig = plot_conformal_metrics(results)
            st.plotly_chart(single_conformal_fig, use_container_width=True)

            st.markdown("#### 📥 Download Conformal Metrics (Separate)")
            cov_fig, width_fig, abs_fig = plot_conformal_metrics_separate(
                results,
                title_prefix=f"{model_type} ({conformal_method}) — "
            )

            # Coverage
            st.plotly_chart(cov_fig, use_container_width=True)
            try:
                cov_bytes = cov_fig.to_image(format="png", scale=2)
                st.download_button(
                    label="💾 Download Coverage Plot (PNG)",
                    data=cov_bytes,
                    file_name=f"{model_type}_{conformal_method}_coverage.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Coverage image export not available: {e}")

            # Interval width
            st.plotly_chart(width_fig, use_container_width=True)
            try:
                width_bytes = width_fig.to_image(format="png", scale=2)
                st.download_button(
                    label="💾 Download Interval Width Plot (PNG)",
                    data=width_bytes,
                    file_name=f"{model_type}_{conformal_method}_interval_width.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Interval width image export not available: {e}")

            # Absolute error
            st.plotly_chart(abs_fig, use_container_width=True)
            try:
                abs_bytes = abs_fig.to_image(format="png", scale=2)
                st.download_button(
                    label="💾 Download Absolute Error Plot (PNG)",
                    data=abs_bytes,
                    file_name=f"{model_type}_{conformal_method}_absolute_error.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Absolute error image export not available: {e}")

    # --- Model Compare UI ---
    st.markdown("---")
    st.header("🏁 Model Compare")

    if len(models_to_compare) == 0:
        st.info("Select at least one model in the sidebar under 'Model Compare' to enable leaderboard.")
    else:
        st.write(f"Selected models: **{', '.join(models_to_compare)}**")

        config_globals = {
            'arima_p': arima_p if 'arima_p' in globals() else 1,
            'arima_d': arima_d if 'arima_d' in globals() else 1,
            'arima_q': arima_q if 'arima_q' in globals() else 1,
            'sarima_p': sarima_p if 'sarima_p' in globals() else 1,
            'sarima_d': sarima_d if 'sarima_d' in globals() else 1,
            'sarima_q': sarima_q if 'sarima_q' in globals() else 1,
            'sarima_P': sarima_P if 'sarima_P' in globals() else 0,
            'sarima_D': sarima_D if 'sarima_D' in globals() else 0,
            'sarima_Q': sarima_Q if 'sarima_Q' in globals() else 0,
            'sarima_s': sarima_s if 'sarima_s' in globals() else 12,
            'yearly': yearly if 'yearly' in globals() else True,
            'weekly': weekly if 'weekly' in globals() else True,
            'daily': daily if 'daily' in globals() else False,
            'seasonality_mode': seasonality_mode if 'seasonality_mode' in globals() else 'additive',
            'conformal_method': conformal_method,
            'alpha': alpha,
            'cv_folds': cv_folds if 'cv_folds' in globals() else 5,
            'adaptive_threshold': adaptive_threshold if 'adaptive_threshold' in globals() else 0.05,
            'adaptive_method': adaptive_method if 'adaptive_method' in globals() else 'decay',
            'window_size': window_size if 'window_size' in globals() else 50,
            'sharpness_weight': sharpness_weight if 'sharpness_weight' in globals() else 0.5
        }

        df_serialized = df.to_csv(index=False).encode()

        if run_compare:
            with st.spinner("Running selected models (this may take a while)..."):
                leaderboard_df, results_by_model = run_models_compare_cached(
                    df_serialized, models_to_compare, test_steps,
                    {'globals': config_globals}
                )
            st.success("✅ Model compare finished.")

            st.subheader("🏆 Leaderboard (sorted by RMSE)")
            st.dataframe(leaderboard_df, use_container_width=True)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=leaderboard_df['model'].astype(str),
                                     y=leaderboard_df['rmse'],
                                     name='RMSE'))
            fig_bar.update_layout(
                title="RMSE by Model",
                xaxis_title="Model",
                yaxis_title="RMSE",
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🔎 Inspect model results")
            model_choice = st.selectbox("Choose model to inspect", leaderboard_df['model'].astype(str).tolist())

            chosen_res = results_by_model.get(model_choice, None)
            if chosen_res is None:
                st.error("No results found for that model.")
            elif 'error' in chosen_res:
                st.error(f"Model error: {chosen_res['error']}")
            else:
                st.write("**Summary metrics**")
                in_interval = ((chosen_res['y_test'] >= chosen_res['lower_bounds']) &
                               (chosen_res['y_test'] <= chosen_res['upper_bounds']))
                empirical_coverage = np.mean(in_interval) * 100
                avg_interval_width = np.mean(chosen_res['upper_bounds'] - chosen_res['lower_bounds'])
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("RMSE", f"{chosen_res['rmse']:.4f}")
                with m2:
                    st.metric("MAPE", f"{chosen_res['mape']:.2f}%")
                with m3:
                    st.metric("Empirical Coverage", f"{empirical_coverage:.1f}%")

                st.markdown("**Forecast plot**")
                inspect_fig = plot_results(chosen_res, title_override=f"{model_choice} with {conformal_method}")
                st.plotly_chart(inspect_fig, use_container_width=True)

                try:
                    inspect_img = inspect_fig.to_image(format="png", scale=2)
                    st.download_button(
                        label=f"💾 Download {model_choice} Forecast Plot (PNG)",
                        data=inspect_img,
                        file_name=f"{model_choice}_{conformal_method}_forecast.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Image export not available for {model_choice}: {e}")

                st.markdown("**Conformal metrics (combined)**")
                inspect_conformal_fig = plot_conformal_metrics(
                    chosen_res,
                    title_override=f"{model_choice} — Conformal Metrics"
                )
                st.plotly_chart(inspect_conformal_fig, use_container_width=True)

                st.markdown("**Conformal metrics (separate for report)**")
                cov_fig_i, width_fig_i, abs_fig_i = plot_conformal_metrics_separate(
                    chosen_res,
                    title_prefix=f"{model_choice} ({conformal_method}) — "
                )

                # Coverage
                st.plotly_chart(cov_fig_i, use_container_width=True)
                try:
                    cov_i_bytes = cov_fig_i.to_image(format="png", scale=2)
                    st.download_button(
                        label=f"💾 Download {model_choice} Coverage Plot (PNG)",
                        data=cov_i_bytes,
                        file_name=f"{model_choice}_{conformal_method}_coverage.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Coverage image export not available for {model_choice}: {e}")

                # Interval width
                st.plotly_chart(width_fig_i, use_container_width=True)
                try:
                    width_i_bytes = width_fig_i.to_image(format="png", scale=2)
                    st.download_button(
                        label=f"💾 Download {model_choice} Interval Width Plot (PNG)",
                        data=width_i_bytes,
                        file_name=f"{model_choice}_{conformal_method}_interval_width.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Interval width image export not available for {model_choice}: {e}")

                # Absolute error
                st.plotly_chart(abs_fig_i, use_container_width=True)
                try:
                    abs_i_bytes = abs_fig_i.to_image(format="png", scale=2)
                    st.download_button(
                        label=f"💾 Download {model_choice} Absolute Error Plot (PNG)",
                        data=abs_i_bytes,
                        file_name=f"{model_choice}_{conformal_method}_absolute_error.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Absolute error image export not available for {model_choice}: {e}")

                preds_df = chosen_res['test'].copy().reset_index(drop=True)
                preds_df['pred'] = chosen_res['predictions']
                preds_df['lower'] = chosen_res['lower_bounds']
                preds_df['upper'] = chosen_res['upper_bounds']
                csv_buf = preds_df.to_csv(index=False).encode()
                st.download_button(
                    label="Download predictions CSV",
                    data=csv_buf,
                    file_name=f"{model_choice}_predictions.csv",
                    mime="text/csv"
                )

        else:
            st.info("Click **Run Compare** in the sidebar to execute the selected models and produce the leaderboard.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with  ConformL Package</div>", unsafe_allow_html=True)
