"""
app.py
------
Streamlit UI for FMCG order profit prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.pipeline.predict_pipeline import PredictInput, PredictPipeline

st.set_page_config(page_title="FMCG Profit Predictor", layout="wide")
st.title("FMCG Order Profit Predictor")
st.markdown(
    "Fill in the order details below and click **Predict** to get an estimated profit "
    "for that order. All inputs map to the raw fields used during training."
)

# ── Sidebar: Order Context ─────────────────────────────────────────────────────
st.sidebar.header("Order Context")

year    = st.sidebar.selectbox("Year",    [2023, 2024, 2025], index=2)
quarter = st.sidebar.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
month   = st.sidebar.slider("Month", 1, 12, 6)
month_name = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}[month]

# ── Main area: two columns ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Geography & Sales")
    region       = st.selectbox("Region",       ["Europe", "Asia", "North America", "South America", "Africa"])
    country      = st.text_input("Country",      value="Germany")
    city         = st.text_input("City",         value="Berlin")
    sales_person = st.text_input("Sales Person", value="SP-001")

    customer_type = st.selectbox("Customer Type",  ["B2B", "B2C"])
    sales_channel = st.selectbox("Sales Channel",  ["Modern Trade", "Online", "Distributor", "Wholesale"])

with col2:
    st.subheader("Product & Promotion")
    product_category = st.selectbox("Product Category", ["Beverages", "Snacks", "Personal Care", "Household", "Dairy"])
    brand            = st.text_input("Brand",            value="AquaGlow")
    product_name     = st.text_input("Product Name",     value="AquaGlow 500ml")
    promotion_type   = st.selectbox("Promotion Type", [
        "No Promo", "Seasonal Campaign", "Bundle Offer",
        "Flash Discount", "Loyalty Reward", "Clearance Sale", "New Launch"
    ])

st.subheader("Commercial Details")
c1, c2, c3, c4 = st.columns(4)
units_sold          = c1.number_input("Units Sold",           min_value=1,   max_value=2000, value=150)
unit_price_usd      = c2.number_input("Unit Price (USD)",     min_value=0.1, max_value=500.0, value=12.50, step=0.5)
discount_pct        = c3.number_input("Discount %",           min_value=0.0, max_value=30.0,  value=10.0,  step=0.5)
marketing_spend_usd = c4.number_input("Marketing Spend (USD)", min_value=0.0, max_value=2000.0, value=75.0, step=5.0)

# ── Prediction ─────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Predict Profit", type="primary"):
    try:
        pipeline = PredictPipeline()
        input_data = PredictInput(
            year=year,
            quarter=quarter,
            month=month,
            month_name=month_name,
            region=region,
            country=country,
            city=city,
            sales_person=sales_person,
            customer_type=customer_type,
            sales_channel=sales_channel,
            promotion_type=promotion_type,
            product_category=product_category,
            brand=brand,
            product_name=product_name,
            units_sold=units_sold,
            unit_price_usd=unit_price_usd,
            discount_pct=discount_pct,
            marketing_spend_usd=marketing_spend_usd,
        )
        prediction = pipeline.predict(input_data)
        colour = "green" if prediction >= 0 else "red"
        st.markdown(
            f"<h2 style='color:{colour};'>Predicted Profit: ${prediction:,.2f}</h2>",
            unsafe_allow_html=True,
        )
        if prediction < 0:
            st.warning("This order is predicted to be unprofitable. Consider reducing discount or marketing spend.")
        elif prediction < 50:
            st.info("Margin is thin. Review discount and channel mix.")
        else:
            st.success("Order looks profitable.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure you have run the training pipeline first (make train).")
