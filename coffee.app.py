import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# ===============================
# Load Trained Model
# ===============================
with open("coffee_sales_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)




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
# ===============================
st.sidebar.header("📌 Input Features")

coffee_options = ["Latte", "Espresso", "Cappuccino", "Americano", "Mocha"]
coffee_mapping = {name: idx for idx, name in enumerate(coffee_options)}

coffee_name = st.sidebar.selectbox("Coffee Name", coffee_options)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
year = st.sidebar.number_input("Year", min_value=2020, max_value=2025, value=2023)
day_of_week = st.sidebar.number_input("Day of Week (1=Mon ... 7=Sun)", min_value=1, max_value=7, value=3)

# Encode coffee
coffee_encoded = coffee_mapping[coffee_name]

# Prepare input
input_data = np.array([[coffee_encoded, day, month, year, day_of_week]])

# ===============================
# Prediction Section
# ===============================
st.markdown("---")
st.subheader("🔮 Prediction Result")

if st.button("Predict Sales"):
    try:
        prediction = pipeline.predict(user_inputs_df)
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

