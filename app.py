import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="CardioExplain - AI Risk Predictor", layout="wide", page_icon="❤️")

# Load model artifacts
@st.cache_resource
def load_artifacts():
    with open("heart_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
X_train = artifacts["X_train"]
X_test = artifacts["X_test"]

# Cache only the calculations (fast and memory-safe)
@st.cache_resource
def get_explainers():
    explainer_shap = shap.TreeExplainer(model)
    explainer_lime = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )
    return explainer_shap, explainer_lime

explainer_shap, explainer_lime = get_explainers()

# Cache only the calculations (fast and memory-safe)
@st.cache_resource
def get_global_shap_values():
    return explainer_shap(X_test)

# Title and description
st.title("❤️ CardioExplain: Explainable AI for Cardiovascular Risk Prediction")
st.write("A clinical decision support tool using optimized XGBoost, SHAP, and LIME for transparent predictions.")

# Layout: Split into sidebar for inputs, main area for tabs
st.sidebar.header("🩺 Patient Biomarkers")

# --- SIDEBAR: ALL SLIDERS AND RADIO BUTTONS (NO DROPDOWNS) ---

age = st.sidebar.slider("Age", 20, 100, 50)
sex_label = st.sidebar.radio("Sex", ["Female", "Male"], horizontal=True)
sex = 1 if sex_label == "Male" else 0

cp_label = st.sidebar.radio(
    "Chest Pain Type (cp)", 
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    horizontal=False
)
cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
cp = cp_map[cp_label]

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs_label = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], horizontal=True)
fbs = 1 if fbs_label == "True" else 0

restecg_label = st.sidebar.radio(
    "Resting Electrocardiographic Results",
    ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
)
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
restecg = restecg_map[restecg_label]

thalach = st.sidebar.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)

exang_label = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True)
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.sidebar.slider("ST depression induced by exercise", 0.0, 6.2, 1.0, step=0.1)

slope_label = st.sidebar.radio(
    "Slope of peak exercise ST segment",
    ["Upsloping", "Flat", "Downsloping"],
    horizontal=True
)
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
slope = slope_map[slope_label]

ca = st.sidebar.slider("Number of major vessels (ca) colored by fluoroscopy", 0, 3, 0)

thal_label = st.sidebar.radio(
    "Thalassemia (thal)",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)
thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
thal = thal_map[thal_label]

# Compile inputs into a single row DataFrame matching training features
patient_data = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Patient Diagnostics", "🔍 Individual Explanations (SHAP & LIME)", "📈 Population Analysis (Global SHAP)"])

with tab1:
    st.subheader("Cardiovascular Risk Score")
    
    # Run prediction
    risk_proba = model.predict_proba(patient_data)[0, 1]
    risk_percent = risk_proba * 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display Risk Score
        st.metric(label="Risk Percentage", value=f"{risk_percent:.1f}%")
        
        if risk_proba < 0.3:
            st.success("🟢 Low Risk: Standard clinical monitoring recommended.")
        elif risk_proba < 0.7:
            st.warning("🟡 Moderate Risk: Lifestyle intervention and further testing advised.")
        else:
            st.error("🔴 High Risk: Urgent cardiology consultation recommended.")
            
    with col2:
        # Visual risk bar
        st.write("### Clinical Alert Level")
        st.progress(float(risk_proba))
        
        # Display biomarker summary
        st.write("### Active Biomarker Summary")
        st.dataframe(patient_data, hide_index=True)

with tab2:
    st.subheader("Explainable AI (XAI) Diagnostics Report")
    st.write("Click the button below to generate LIME and SHAP local explanations for this specific patient.")
    
    if st.button("Generate Explainability Reports", type="primary"):
        col_xai1, col_xai2 = st.columns(2)
        
        with col_xai1:
            st.write("### 1. SHAP Local Explanation")
            with st.spinner("Generating SHAP values..."):
                shap_values = explainer_shap(patient_data)
                
                # Plot local SHAP bar plot
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.bar(shap_values[0], max_display=7, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Free memory
                st.caption("Positive SHAP values (red/right) push the risk HIGHER. Negative values (blue/left) push it LOWER.")

        with col_xai2:
            st.write("### 2. LIME Local Explanation")
            with st.spinner("Generating LIME explanation..."):
                exp = explainer_lime.explain_instance(
                    data_row=patient_data.iloc[0],
                    predict_fn=model.predict_proba,
                    num_features=5
                )
                # Render LIME explanation as HTML inside an iframe
                components.html(exp.as_html(), height=380, scrolling=True)
                st.caption("LIME approximates the model locally to show how much each biomarker contributed to this specific decision.")
    else:
        st.info("ℹ️ Click 'Generate Explainability Reports' above to view the XAI charts.")

with tab3:
    st.subheader("Global Feature Importance (Dataset Level)")
    st.write("This tab shows which features are most important across the entire population, useful for clinical research.")
    
    with st.spinner("Loading global population summary..."):
        # Get cached values
        shap_values_test = get_global_shap_values()
        
        # Draw the plot freshly
        fig_global, ax_global = plt.subplots(figsize=(8, 5))
        shap.plots.beeswarm(shap_values_test, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_global)
        plt.close(fig_global) # Free memory
        st.caption("SHAP Beeswarm Plot: The position on the X-axis shows the impact on risk, and the color represents feature values (Red = High, Blue = Low).")