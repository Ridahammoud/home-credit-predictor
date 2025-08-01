import streamlit as st
import pandas as pd
import joblib
import numpy as np
from predict_model import HomeCreditPredictor

# ====== Chargement du modèle ======
@st.cache_resource
def load_model():
    return joblib.load("model/final_lgb_model.pkl")

predictor = HomeCreditPredictor("model/final_lgb_model.pkl")

st.title("🔮 Prédiction Home Credit - LightGBM")

# --- Section 1 : Batch CSV ---
st.header("📂 Prédire à partir d'un fichier CSV")
uploaded_file = st.file_uploader("Chargez un fichier CSV (app_test)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    results = predictor.predict_batch(df)
    st.dataframe(results.head(20))
    st.download_button("⬇️ Télécharger les prédictions", results.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")

# --- Section 2 : Prédiction instantanée ---
st.header("⚡ Prédire un seul client (entrée manuelle)")

st.write("👉 Renseignez les paramètres manuellement pour tester une prédiction")

# Exemple avec quelques features importantes
feature_inputs = {}
example_features = list(model.feature_names_)[:10]  # prendre les 10 premières features

for feat in example_features:
    feature_inputs[feat] = st.number_input(f"{feat}", value=0.0)

if st.button("Prédire"):
    X_manual = pd.DataFrame([feature_inputs])
    proba = model.predict_proba(X_manual)[:, 1][0]
    pred = model.predict(X_manual)[0]

    st.success(f"🎯 **Prédiction (TARGET)** : {int(pred)}")
    st.info(f"📊 **Probabilité estimée** : {proba:.4f}")
