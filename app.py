import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

st.title("Employee Attrition Prediction")
st.write("Fill employee details to predict attrition")

# ---------------- LOAD MODEL ----------------
with open("xgboost.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- USER INPUTS ----------------
age = st.slider("Age", min_value=18, max_value=65, value=30)

length_of_service = st.number_input(
    "Length of Service (Years)",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)

city_name = st.text_input("City Name")
department_name = st.text_input("Department Name")
job_title = st.text_input("Job Title")

store_name = st.number_input("Store Number", min_value=0, max_value=1000, value=1, step=1)

gender_short = st.selectbox("Gender (Short)", ["M", "F"])
gender_full = st.selectbox("Gender (Full)", ["Male", "Female"])

STATUS_YEAR = st.selectbox("Status Year", list(range(2000, 2026)))

BUSINESS_UNIT = st.selectbox("Business Unit", ["Stores", "Head Office"])  # <-- NEW

# ---------------- PREDICTION ----------------
if st.button("Predict Attrition", key="predict_btn"):

    # Validation: check mandatory text fields
    if not city_name or not department_name or not job_title:
        st.warning("⚠️ Please enter all required values before prediction.")
    else:
        input_df = pd.DataFrame([{
            "age": age,
            "length_of_service": length_of_service,
            "city_name": city_name,
            "department_name": department_name,
            "job_title": job_title,
            "store_name": store_name,
            "gender_short": gender_short,
            "gender_full": gender_full,
            "STATUS_YEAR": STATUS_YEAR,
            "BUSINESS_UNIT": BUSINESS_UNIT
        }])

        try:
            prediction = model.predict(input_df)[0]   # 0 or 1
            probability = model.predict_proba(input_df)[0]

            st.subheader("Prediction Result")
            if prediction == 1:   # assuming 1 = Terminated
                st.error("Employee likely to leave ❌")
                st.write(f"Confidence: {probability[1]*100:.2f}%")
            else:                 # 0 = Active/Stay
                st.success("Employee likely to stay ✅")
                st.write(f"Confidence: {probability[0]*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
        