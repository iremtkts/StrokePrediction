# streamlit_app.py
import os
import json
import requests
import pandas as pd
import streamlit as st



st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺", layout="centered")
st.title(" Stroke Risk Predictor")
st.caption("Streamlit interface with FastAPI + Uvicorn backend")

DEFAULT_API_URL = "http://localhost:8000"
API_URL = st.sidebar.text_input("🔗 API URL", os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"



with st.expander("Backend Health", expanded=False):
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        st.success("Backend OK")
        st.json(r.json())
    except Exception as e:
        st.error(f"Backend' Error: {e}")
st.markdown("---")


gender_map = {"Male": 0, "Female": 1, "Other": -1}
work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": -1, "Never_worked": -2}
smoking_status_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": -1}
yes_no_map = {"Yes": 1, "No": 0}

FEATURES = [
    "gender","age","hypertension","heart_disease","work_type",
    "avg_glucose_level","bmi","smoking_status","ever_married"
]


st.subheader("Patient Information")
with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", list(gender_map.keys()), index=1)
        age = st.number_input("Age", min_value=0, max_value=120, value=60, step=1)
        hypertension = st.selectbox("Hypertension", list(yes_no_map.keys()), index=0)
        heart_disease = st.selectbox("Heart Disease", list(yes_no_map.keys()), index=1)
        ever_married = st.selectbox("Ever Married", list(yes_no_map.keys()), index=0)
    with c2:
        work_type = st.selectbox("Work Type", list(work_type_map.keys()), index=0)
        avg_glucose_level = st.number_input("Avg Glucose Level", min_value=0.0, max_value=500.0, value=110.0, step=0.1)
        bmi = st.number_input("BMI", min_value=5.0, max_value=100.0, value=27.5, step=0.1)
        smoking_status = st.selectbox("Smoking Status", list(smoking_status_map.keys()), index=0)

   
    threshold = st.slider("Threshold for class decision (show only)", 0.0, 1.0, 0.5, 0.01)
    submitted = st.form_submit_button("Prediction")


if submitted:
    payload = {
        "gender": gender_map[gender],
        "age": float(age),
        "hypertension": yes_no_map[hypertension],
        "heart_disease": yes_no_map[heart_disease],
        "work_type": work_type_map[work_type],
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "smoking_status": smoking_status_map[smoking_status],
        "ever_married": yes_no_map[ever_married]
    }

    st.info(f"POST {PREDICT_URL}")
    with st.spinner("The model predicts..."):
        try:
            res = requests.post(PREDICT_URL, json=payload, timeout=15)
            res.raise_for_status()
            out = res.json()

            
            proba = float(out.get("stroke_proba", 0.0))
            pred_backend = int(out.get("stroke_pred", 0))
            pred_ui = int(proba >= threshold)

            st.success("Prediction is ready")

        
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Stroke Probability (model)", f"{proba*100:.2f}%")
            with c2:
                st.metric("Predicted Class (backend, 0.5 threshold)", "1 (There is risk)" if pred_backend == 1 else "0 (No Risk")

            st.write("Class decision based on threshold slider (UI): **{}**".format("1 (There is risk)" if pred_ui == 1 else "0 (No Risk)"))
            st.progress(min(max(proba, 0.0), 1.0))

          
            tabs = st.tabs(["Sent Data", " Backend Response", " cURL"])
            with tabs[0]:
                st.dataframe(pd.DataFrame([payload], columns=FEATURES))
            with tabs[1]:
                st.json(out)
            with tabs[2]:
                curl = (
                    f"curl -X POST '{PREDICT_URL}' "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{json.dumps(payload)}'"
                )
                st.code(curl, language="bash")

        except requests.HTTPError as e:
            st.error(f"HTTP Error: {e}\n{res.text if 'res' in locals() else ''}")
        except requests.exceptions.ConnectionError:
            st.error("Connection error: Is the backend URL correct and working?")
        except Exception as e:
            st.error(f"Error during request: {e}")

st.markdown("---")
