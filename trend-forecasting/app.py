# trend_forecasting_app/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import StringIO
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Page config
st.set_page_config(page_title="AI Trend Forecasting", layout="wide")
st.title("📈 AI-Powered Trend Forecasting Dashboard")

# Upload file
uploaded_file = st.file_uploader("📤 Upload your sales CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗂️ Raw Data Preview")
    st.write(df.head())

    # Optional filtering if columns exist
    if 'Category' in df.columns:
        category = st.selectbox("📂 Select Category", df['Category'].unique())
        df = df[df['Category'] == category]

    if 'Region' in df.columns:
        region = st.selectbox("🌍 Select Region", df['Region'].unique())
        df = df[df['Region'] == region]

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Sales']].rename(columns={"Date": "ds", "Sales": "y"})

        st.subheader("📊 Sales Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Historical Sales")
        st.pyplot(fig)

        st.subheader("🧠 Forecast with Prophet")
        periods = st.slider("📆 Months to Forecast", 1, 12, 3)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30 * periods)
        forecast = model.predict(future)

        # Accuracy Metrics
        forecast_filtered = forecast.set_index('ds').loc[df['ds']]
        actual = df.set_index('ds')['y']
        predicted = forecast_filtered['yhat']

        mae = mean_absolute_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted, squared=False)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        st.subheader("📈 Forecast Accuracy Metrics")
        st.markdown(f"""
        - **MAE (Mean Absolute Error):** {mae:.2f}  
        - **RMSE (Root Mean Squared Error):** {rmse:.2f}  
        - **MAPE (Mean Absolute Percentage Error):** {mape:.2f}%
        """)

        st.subheader("📉 Forecast Visualization")
        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        st.subheader("🔍 Forecast Components")
        fig3 = model.plot_components(forecast)
        st.pyplot(fig3)

        st.subheader("📥 Download Forecast Data")
        st.download_button(
            label="Download CSV",
            data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

        st.subheader("🔢 Forecast Data Preview")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    except Exception as e:
        st.error(f"⚠️ Error processing data: {e}")
else:
    st.info("ℹ️ Please upload a CSV file with at least 'Date' and 'Sales' columns.")
