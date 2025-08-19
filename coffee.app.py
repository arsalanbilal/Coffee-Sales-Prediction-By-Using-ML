import streamlit as st
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
    coffee_name_Americano = st.slider("Americano", 0, 1)
    coffee_name_Americano_with_Milk = st.slider("Americano_with Milk", 0, 1)
    coffee_name_Cappuccino = st.slider("Cappuccino", 0, 1)
    coffee_name_Cocoa = st.slider("Cocoa", 0, 1)
    coffee_name_Cortado = st.slider("Cortado", 0, 1)
    coffee_name_Espresso = st.slider("Espresso", 0, 1)
    coffee_name_Hot_Chocolate = st.slider("Hot Chocolate", 0, 1)
    coffee_name_Latte = st.slider("Latte", 0, 1)
    cash_type_card = st.slider("Card", 0, 1)
    cash_type_cash = st.slider("Cash", 0, 1)
    Day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
    Month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
    Year = st.sidebar.number_input("Year", min_value=2020, max_value=2025, value=2023)
    DayofWeek = st.sidebar.number_input("Day of Week", 1, 31)



input_data = pd.DataFrame([['Americano', 'Americano_with Milk', 'Cappuccino', 'Cocoa', 'Cortado', 'Espresso', 'Hot Chocolate', 'Latte', 'Card', 'Cash', 'Day', 'Month', 'Year', 'DayofWeek']],
                        columns=['coffee_name_Americano', 'coffee_name_Americano_with_Milk', 'coffee_name_Cappuccino', 'coffee_name_Cocoa', 'coffee_name_Cortado', 'coffee_name_Espresso', 
                                'coffee_name_Hot_Chocolate', 'coffee_name_Latte', 'cash_type_card', 'cash_type_cash', 'Day', 'Month', 'Year', 'DayofWeek'])



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

