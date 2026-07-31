
---

###  README for CardioExplain 
Create a new file named **`README.md`** inside your `CardioExplain` folder, paste this content inside, and save:

```markdown
# ❤️ CardioExplain: Explainable AI for Cardiovascular Risk Prediction

CardioExplain is a clinical decision-support tool that predicts cardiovascular disease risk with high accuracy and provides transparent, interpretable explanations using **Explainable AI (XAI)** methods (**SHAP** and **LIME**). 

## 🚀 Features
- **High Performance ML:** Built on an optimized **XGBoost Classifier** achieving **90.16% Accuracy** and **0.96 ROC-AUC**.
- **Dual-Explainability Framework:**
  - **Local Interpretability (SHAP & LIME):** Interactive patient-specific visual reports showing exactly how individual biomarkers (e.g., cholesterol, age) contributed to their risk score.
  - **Global Interpretability (SHAP Beeswarm):** Visualizes the overall impact of clinical features across the entire population.
- **Interactive UI:** Smooth, responsive dashboard designed entirely using sliders and radio buttons for a glitch-free user experience.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **ML Algorithm:** XGBoost Classifier
- **Explainable AI:** SHAP (Shapley Additive exPlanations) & LIME (Local Interpretable Model-agnostic Explanations)
- **Data & Evaluation:** Scikit-Learn, Pandas, NumPy, Matplotlib

## 📦 How to Run Locally

1. Activate your conda environment:
   ```bash
   conda activate cardioexplain_env