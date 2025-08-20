
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

# ===============================
# Helpers
# ===============================
COFFEE_CATEGORIES = [
    "Americano", "Americano with Milk", "Cappuccino", "Cocoa",
    "Cortado", "Espresso", "Hot Chocolate", "Latte"
]
CASH_CATEGORIES = ["card", "cash"]

DAYOFWEEK_NAME_TO_NUM = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

def one_row_dataframe(coffee_name: str, cash_type: str, day: int, dow_name: str, month: int, year: int) -> pd.DataFrame:
    """Build a single-row dataframe with OHE columns expected by the model.
    Falls back to a reasonable default column order if the model does not expose feature names.
    """
    # Base numeric features
    data = {
        "Day": [int(day)],
        "Month": [int(month)],
        "Year": [int(year)],
        "DayofWeek": [int(DAYOFWEEK_NAME_TO_NUM.get(dow_name, 0))],
    }

    # One-hot features (full space so we can align with model.feature_names_in_ if available)
    for cat in COFFEE_CATEGORIES:
        data[f"coffee_name_{cat}"] = [1 if coffee_name == cat else 0]
    for cat in CASH_CATEGORIES:
        data[f"cash_type_{cat}"] = [1 if cash_type == cat else 0]

    return pd.DataFrame(data)


@st.cache_resource(show_spinner=False)
def load_model():
    # Try a few common filenames
    candidate_names = [
        "tree_best_model.pkl",
        "tree_best_model (1).pkl",
        "model.pkl",
        "best_model.pkl",
    ]
    for name in candidate_names:
        p = Path(name)
        if p.exists():
            return joblib.load(p.as_posix())
    # Also try the uploaded exact name (sometimes spaces are underscores, try both)
    for name in Path(".").glob("tree_best_model*"):
        try:
            return joblib.load(name.as_posix())
        except Exception:
            continue
    raise FileNotFoundError("Model file not found. Expected one of: " + ", ".join(candidate_names))


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="☕ Coffee Sales Prediction",
    page_icon="☕",
    layout="centered",
    initial_sidebar_state="expanded",
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
with st.sidebar:
    st.header("📌 Input Features")
    coffee_name = st.selectbox("Coffee Name", COFFEE_CATEGORIES, index=0)
    cash_type = st.selectbox("Payment Type", CASH_CATEGORIES, index=0)
    day = st.slider("Day", min_value=1, max_value=31, value=1)
    dow = st.selectbox("Day of Week", list(DAYOFWEEK_NAME_TO_NUM.keys()), index=0)
    month = st.slider("Month", min_value=1, max_value=12, value=1)
    year = st.selectbox("Year", [2023, 2024, 2025], index=1)

st.markdown("---")
st.subheader("🔮 Prediction Result")

# Load model (cached)
load_error = None
model = None
try:
    model = load_model()
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"❌ Could not load model: {load_error}")
else:
    # Predict button
    if st.button("🚀 Predict"):
        X = one_row_dataframe(coffee_name, cash_type, day, dow, month, year)

        # If the model knows its training feature names, align columns exactly
        try:
            expected_cols = list(getattr(model, "feature_names_in_", []))
        except Exception:
            expected_cols = []

        if expected_cols:
            # Add any missing expected columns with 0, and order columns
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[expected_cols]
        else:
            # Fallback – use the current X with all engineered columns
            pass

        with st.spinner("Calculating prediction..."):
            time.sleep(0.6)
            try:
                y_pred = model.predict(X)
                pred = float(np.ravel(y_pred)[0])
                st.success("✅ Prediction Completed!")
                st.markdown(
                    f"""
                    <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
                        <h2 style="color:#1E90FF;">Predicted Sales</h2>
                        <h1 style="color:#FF4500;font-size:48px;">{pred:,.2f}</h1>
                        <p style="color:gray;">Estimated coffee sales (currency units)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric(label="📈 Predicted Sales", value=f"{pred:,.2f}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.caption(
                    "Tip: Ensure the model was trained with features "
                    "Day, Month, Year, DayofWeek and one-hot columns for coffee_name_* and cash_type_*."
                )
