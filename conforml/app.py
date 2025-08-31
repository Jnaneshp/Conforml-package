import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from conforml.models.arima import ARIMAModel
from conforml.conformal.methods import CVPlusConformal, AdaptiveConformal

st.set_page_config(page_title="Conformal Prediction Time Series Demo", layout="wide")
st.title("Conformal Prediction for Time Series")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV with time series", type=["csv"])

model_choice = st.sidebar.selectbox("Model", ["ARIMA (demo)"])
conformal_choice = st.sidebar.selectbox("Conformal Method", ["CV+", "Adaptive (decay)", "Adaptive (sliding)"])
alpha = st.sidebar.slider("Alpha (1 - coverage)", 0.01, 0.3, 0.1, 0.01)
window_size = st.sidebar.number_input("Sliding Window Size (if sliding)", min_value=10, max_value=500, value=50)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    # Try to infer columns
    time_col = st.selectbox("Timestamp column", df.columns, index=0)
    value_col = st.selectbox("Value column", df.columns, index=1 if len(df.columns)>1 else 0)
    df = df[[time_col, value_col]].dropna()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    df = df.reset_index(drop=True)
    # Split train/test
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    X_train, y_train = np.arange(len(train)).reshape(-1,1), train[value_col].values
    X_test, y_test = np.arange(len(train), len(train)+len(test)).reshape(-1,1), test[value_col].values
    # Model
    if model_choice == "ARIMA (demo)":
        model = ARIMAModel()
    # Fit model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # Conformal
    if conformal_choice == "CV+":
        conformal = CVPlusConformal(model, alpha=alpha)
        conformal.fit(X_train, y_train)
    elif conformal_choice == "Adaptive (decay)":
        conformal = AdaptiveConformal(model, alpha=alpha, method="decay")
        conformal.fit(X_train, y_train)
    else:
        conformal = AdaptiveConformal(model, alpha=alpha, method="sliding", window_size=window_size)
        conformal.fit(X_train, y_train)
    preds, lower, upper = conformal.predict(X_test)
    # Visualization
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df[time_col], df[value_col], label="Observed", color="black", alpha=0.5)
    ax.plot(test[time_col], preds, label="Prediction", color="blue")
    ax.fill_between(test[time_col], lower, upper, color="red", alpha=0.3, label="Prediction Interval")
    # Highlight anomalies
    anomalies = (y_test < lower) | (y_test > upper)
    ax.scatter(test[time_col][anomalies], y_test[anomalies], color="orange", label="Anomaly", zorder=5)
    ax.set_title("Prediction Intervals and Anomalies")
    ax.legend()
    st.pyplot(fig)
    # Metrics
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    st.write(f"**Empirical Coverage:** {coverage*100:.2f}%")
    st.write(f"**Interval Width (mean):** {np.mean(upper-lower):.3f}")
    st.write(f"**Anomalies:** {anomalies.sum()} out of {len(y_test)} test points")
else:
    st.info("Upload a CSV file to get started.")