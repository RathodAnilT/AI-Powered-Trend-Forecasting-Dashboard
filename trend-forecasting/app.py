# trend_forecasting_app/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import StringIO

st.set_page_config(page_title="AI Trend Forecasting", layout="wide")
st.title("📈 AI-Powered Trend Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Expecting columns: ['Date', 'Sales']
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Sales']].rename(columns={"Date": "ds", "Sales": "y"})

        st.subheader("📊 Sales Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['ds'], df['y'], color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

        st.subheader("🧠 Forecast with Prophet")
        periods = st.slider("Months to Forecast", 1, 12, 3)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30 * periods)
        forecast = model.predict(future)

        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        st.subheader("📉 Forecast Visualization")
        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        st.subheader("🔍 Forecast Components")
        fig3 = model.plot_components(forecast)
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"⚠️ Error processing data: {e}")
else:
    st.info("Please upload a CSV file with 'Date' and 'Sales' columns.")
