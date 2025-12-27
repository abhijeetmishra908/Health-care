import streamlit as st
import pandas as pd
import joblib

# --- Load saved artifacts ---
model_xgb = joblib.load("xgbmodel.pkl")
columns = joblib.load("columns.pkl")

st.title("🩺 CKD Prediction App (XGBoost)")

st.write("Enter patient details below to predict CKD status:")

# --- Input fields ---
age = st.number_input("Age", min_value=1, max_value=120, value=45)
bun = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=18.0)
creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.2)

protein_in_urine = st.selectbox("Protein in Urine", ["Normal", "Trace", "1+", "2+", "3+"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
medication = st.selectbox("Medication", ["ACE Inhibitor", "ARB", "Diuretic", "None"])

# --- Convert inputs into dataframe ---
input_dict = {
    "Age": age,
    "BUN": bun,
    "Creatinine": creatinine,
    "Protein_in_Urine": protein_in_urine,
    "Diabetes": diabetes,
    "Hypertension": hypertension,
    "Medication": medication,
}
input_df = pd.DataFrame([input_dict])

# --- One-hot encode categorical features ---
cat_cols = ["Protein_in_Urine", "Diabetes", "Hypertension", "Medication"]
input_encoded = pd.get_dummies(input_df[cat_cols], drop_first=False)

# --- Combine numeric + encoded categorical ---
num_cols = ["Age", "BUN", "Creatinine"]
input_final = pd.concat([input_df[num_cols], input_encoded], axis=1)

# --- Align with training columns ---
input_final = input_final.reindex(columns=columns, fill_value=0)

# --- Prediction ---
if st.button("Predict CKD Status"):
    prediction = model_xgb.predict(input_final)[0]
    prob = model_xgb.predict_proba(input_final)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of CKD detected (probability: {prob:.2f})")
    else:
        st.success(f"✅ No CKD detected (probability: {prob:.2f})")
