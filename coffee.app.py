import streamlit as st
import pickle
import numpy as np

# Load the trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

st.set_page_config(page_title="Coffee Sales Prediction ☕", layout="centered")

st.title("☕ Coffee Sales Prediction Dashboard")
st.markdown("""
Welcome to the **Coffee Sales Predictor**.  
Provide details about the coffee and date to estimate the **predicted sales revenue (money)**.
""")

# Sidebar
st.sidebar.header("📌 Input Features")

# Inputs (match your dataset features)
coffee_name = st.sidebar.selectbox("Coffee Name", 
                                   ["Latte", "Espresso", "Cappuccino", "Americano", "Mocha"])  # Update with unique values
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
year = st.sidebar.number_input("Year", min_value=2020, max_value=2025, value=2023)
day_of_week = st.sidebar.number_input("Day of Week (1=Mon ... 7=Sun)", min_value=1, max_value=7, value=3)

# Encoding coffee_name (simple ordinal for demo – update if OneHot used)
coffee_mapping = {"Latte": 0, "Espresso": 1, "Cappuccino": 2, "Americano": 3, "Mocha": 4}
coffee_encoded = coffee_mapping[coffee_name]

# Create input array
input_data = np.array([[coffee_name_encoded, day, month, year, day_of_week]])

# Predict
if st.button("🔮 Predict Sales"):
    prediction = loaded_model.predict(np.array(input_data).reshape(1, -1))
    st.success(f"💰 Predicted Sales: **{prediction[0]:.2f}** currency units")

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Model: Decision Tree Regressor")
