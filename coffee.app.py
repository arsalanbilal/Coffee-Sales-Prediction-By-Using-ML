import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# ===============================
# Load Trained Model
# ===============================
loaded_model = joblib.load('tree_best_model.pkl')


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="☕ Coffee Sales Prediction",
    page_icon="☕",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===============================
# App Title
# ===============================
st.title("📊 Coffee Sales Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Coffee Sales Predictor** 🚀  
    Enter coffee details and date information to estimate **predicted sales revenue**.  
    """
)

# ===============================
# Sidebar Inputs
# ==============================#
with st.sidebar:
    st.header('📌 Input Features')
    coffee_name = st.selectbox("Coffee Name", ['Americano', 'Americano with Milk', 'Cappuccino', 'Cocoa', 'Cortado', 'Espresso', 'Hot Chocolate', 'Latte'])
    cash_type = st.selectbox("Payment", ['card', 'cash'])
    Day = st.slider("Day", min_value = 1, max_value = 31)
    DayofWeek = st.selectbox("DayofWeek", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Month = st.slider("Month", min_value = 1, max_value = 12)
    Year = st.selectbox("Year", [2024])
    




# ===============================
# Prediction Section
# ===============================
st.markdown("---")
st.subheader("🔮 Prediction Result")

# Predict button
if st.button("🚀 Predict"):
    data = pd.DataFrame([[coffee_Name, cash_type, Day, DayofWeek, Month, Year]],
                        columns=['coffee_name', 'cash_type', 'Day', 'DayofWeek', 'Month', 'Year'])

    predictions = best_model.predict(data)

    # Simulate a loading animation
    with st.spinner('Calculating prediction...'):
        time.sleep(1)

    # Stylish display
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Value</h2>
            <h1 style="color:#FF4500;font-size:60px;">{predictions[0]:,.2f}</h1>
            <p style="color:gray;">Estimated Uber demand based on provided features</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Optional: Metric display
    st.metric(label="📈 Predicted Uber Demand", value=f"{predictions[0]:,.2f}")

