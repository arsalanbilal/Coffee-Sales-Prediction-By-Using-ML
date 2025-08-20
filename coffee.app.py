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
    coffee_name = st.selectbox("Coffee Name", ['Americano', 'Americano with Milk', 'Cappuccino', 'Cocoa', 'Cortado', 'Espresso', 'Hot Chocolate', 'Latte']
    cash_type = st.selectbox("Payment", ['card', 'cash']
    Day = st.slider("Day", min_value = 1, max_value = 31)
    DayofWeek = st.selectbox("DayofWeek", [{ 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}]
    Month = st.slider("Month", min_value = 1, max_value = 12)
    Year = st.selectbox("Year", [2024])
    
i



# ===============================
# Prediction Section
# ===============================
st.markdown("---")
st.subheader("🔮 Prediction Result")

if st.button("Predict Sales"):
    try:
        prediction = loaded_model.predict(input_data)
        st.success(f"💰 **Predicted Sales: {prediction[0]:,.2f} currency units**")
        
        # Fancy Result Box
        st.markdown(
            f"""
            <div style="background-color:#f0f8ff;
                        padding:20px;
                        border-radius:15px;
                        text-align:center;
                        box-shadow:2px 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color:#2E86C1;">☕ {coffee_name}</h2>
                <h3 style="color:#117A65;">📅 {day}-{month}-{year} (Day {day_of_week})</h3>
                <h1 style="color:#B03A2E;">💰 {prediction[0]:,.2f}</h1>
                <p><b>Estimated Sales Revenue</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Model: Decision Tree Regressor")

