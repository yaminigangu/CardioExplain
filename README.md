# ❤️ CardioExplain: Explainable AI (XAI) for Cardiovascular Risk Prediction

📄 **[Read the Full Research Paper (PDF)](./CardioExplain_Research_Paper.pdf)**  
📄 **[View LaTeX Source Code](./cardioexplain_ieee.tex)**

---

**CardioExplain** is an advanced clinical decision-support tool that predicts cardiovascular disease risk with high accuracy and provides transparent, mathematically justified explanations using **SHAP** and **LIME**.

---

## 📝 Abstract
While machine learning models achieve high accuracy in predicting cardiovascular risks, their adoption in clinical practice is limited because they act as "black boxes." Doctors cannot ethically or legally act on a prediction without understanding the underlying medical reasoning. 

CardioExplain bridges the gap between high accuracy and interpretability. We train an optimized **XGBoost Classifier** on the UCI Cleveland Heart Disease dataset, achieving a state-of-the-art predictive accuracy of **90.16%** and a **0.96 ROC-AUC score**. To make the model transparent, we implement a dual-explainability framework: **SHAP** (Shapley Additive exPlanations) for population-level (global) feature importances and local risk contributions, and **LIME** (Local Interpretable Model-agnostic Explanations) for patient-specific (local) risk justifications. This transparency builds clinical trust and enables personalized medical interventions.

---

## ⚠️ Problem Statement
1. **The Black-Box Dilemma:** High-performing models (like XGBoost, Neural Networks) do not explain *why* a patient is classified as high-risk.
2. **Clinical Accountability:** Healthcare regulations require clinical decisions to be explainable to prevent diagnostic bias.
3. **Personalized Care:** Doctors need to know which specific patient biomarkers (e.g., high cholesterol vs. max heart rate) are driving the risk to prescribe targeted treatments.

---

## 🎯 Project Objectives
- **High-Accuracy Classification:** Train an ensemble model to detect heart disease with an accuracy exceeding 88%.
- **Local Interpretability:** Generate real-time patient-specific visual reports showing the exact positive and negative medical drivers of their risk.
- **Global Interpretability:** Map the feature importance across the entire dataset to reveal general population trends for clinical research.
- **Interactive Clinical Dashboard:** Build a completely dropdown-free, responsive UI using sliders and radio buttons for seamless usage.

---

## 📊 Model Performance Metrics
The optimized XGBoost model achieved outstanding results on the test set:
- **Accuracy:** 90.16%
- **Precision:** 86.67%
- **Recall:** 92.86%
- **F1-Score:** 89.66%
- **ROC-AUC:** 0.9600

---

## ⚙️ Methodology & XAI Framework
1. **Pre-processing:** Handle missing data and binarize the heart disease target variable.
2. **Model Training:** Fit an optimized `XGBClassifier` with tuned hyper-parameters (learning rate, max depth, subsample).
3. **LIME (Local Interpretability):** Fits an interpretable local linear model around the patient's individual data point to show how features offset the prediction baseline.
4. **SHAP (Global & Local Interpretability):** 
   - *Local:* Computes Shapley values to show how much each biomarker pushed the patient's risk score away from the average.
   - *Global:* Uses a Beeswarm plot on the test set to display the impact direction of all features (e.g., high chest pain increases risk, while high max heart rate decreases it).

---

## 🛠️ Tech Stack & Dependencies
- **Machine Learning:** XGBoost, Scikit-Learn
- **Explainable AI (XAI):** SHAP, LIME
- **Frontend Dashboard:** Streamlit
- **Data Science & Plots:** Pandas, NumPy, Matplotlib

---

## 📦 Installation & Setup

1. Activate your conda environment:
   ```bash
   conda activate cardioexplain_env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Fetch the UCI Cleveland dataset:
   ```bash
   python download_data.py
   ```
4. Train and save the XGBoost model:
   ```bash
   python train.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

---

## 🔮 Future Scope
- **Time-Series Integration:** Extend the model to ingest continuous ECG data streams for real-time risk tracking.
- **Counterfactual Explanations:** Add actionable advice for patients (e.g., *"If you reduce your serum cholesterol by 30 mg/dl, your risk will drop from High to Low"*).