import streamlit as st
import pandas as pd
from predict_model import HomeCreditPredictor
import shap
import plotly.express as px
import numpy as np

# Initialisation du prédicteur

predictor = HomeCreditPredictor(
    model_path="models/final_lgb_model_f.pkl",
    imputer_path="models/final_imputer_f.pkl",
    features_path="feature_importances.csv"
)

st.title("🏦 Dashboard de credit scoring - Projet 8")

tab1, tab2 = st.tabs(["📂 Prédictions sur fichier CSV", "🔮 Prédiction manuelle"])

# Page 1 : Prédiction batch (CSV)

with tab1:
    st.header("📂 Charger un fichier CSV (ex: app_test.csv)")
    uploaded_file = st.file_uploader("Déposez ici votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Aperçu des données chargées :")
            st.dataframe(df.head())

            if "SK_ID_CURR" in df.columns:
                ids = df["SK_ID_CURR"]
            else:
                ids = pd.Series(range(len(df)), name="SK_ID_CURR")

            # Prédictions via ton predictor
            results = predictor.predict_batch(df)
            results.insert(0, "SK_ID_CURR", ids)

            seuil = st.slider("⚖️ Choisissez le seuil de probabilité de défaut %",
                              min_value=0, max_value=100, value=50, step=1) / 100.0

            results_filtered = results[results["PROBA_DEFAULT"] >= seuil]

            search_id = st.text_input("🔎 Rechercher un client par SK_ID_CURR :", "")
            results_filtered_c = results_filtered.copy()

            if search_id:
                try:
                    search_id = int(search_id)
                    results_filtered_c = results_filtered_c[results_filtered_c["SK_ID_CURR"] == search_id]
                except:
                    st.warning("Veuillez entrer un identifiant numérique valide.")

            st.success("✅ Prédictions effectuées avec succès !")
            st.dataframe(results_filtered_c.head(20))

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les prédictions",
                data=csv,
                file_name="predictions_home_credit.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")

    st.header("📂 Les Features Importances ")
    fi = pd.read_csv('feature_importances.csv')
    st.dataframe(fi.head(10))

    top_features = fi.head(10)
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 des variables les plus importantes (globales)",
        labels={"importance": "Importance", "feature": "Variable"}
    )
    st.plotly_chart(fig, use_container_width=True)

# Page 2 : Prédiction manuelle

with tab2:
    st.header("🔮 Saisir manuellement les données client")

    input_data = {}
    input_data["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", min_value=0.0, max_value=99999.0, step=1.0)
    input_data["AMT_CREDIT"] = st.number_input("AMT_CREDIT", min_value=0.0, max_value=1000000.0, step=100.0)
    input_data["AMT_ANNUITY"] = st.number_input("AMT_ANNUITY", min_value=0.0, max_value=1000000.0, step=100.0)
    input_data["DAYS_ID_PUBLISH"] = st.number_input("DAYS_ID_PUBLISH", min_value=0.0,max_value=1000000.0, step=100.0)
    input_data["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", min_value=0.0, max_value=1000000.0, step=100.0)
    input_data["AMT_GOODS_PRICE"] = st.number_input("AMT_GOODS_PRICE", min_value=0.0, max_value=1000000.0, step=100.0)
    input_data["DAYS_REGISTRATION"] = st.number_input("DAYS_REGISTRATION", min_value=0.0, max_value=1000000.0, step=100.0)
    input_data["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", min_value=0.0, max_value = 1000000.0, step=100.0)
    input_data["CNT_CHILDREN"] = st.number_input("CNT_CHILDREN", min_value=0.0, max_value = 20.0, step = 1.0)

    threshold = st.slider("⚖️ Choisir le seuil de refus", 0.0, 1.0, 0.5, 0.01)

    if st.button("🔎 Prédire le risque"):
        try:
            result = predictor.predict_single(input_data, threshold=threshold)
            proba_default = result["PROBA_DEFAULT"]
            pred = result["TARGET_PRED"]

            st.info(f"📊 Probabilité de défaut : **{proba_default:.2%}** (seuil = {threshold:.0%})")

            if proba_default >= threshold:
                st.error("❌ Prêt refusé (risque trop élevé)")
            else:
                st.success("✅ Prêt accepté")

            st.metric(
                label="Score de risque en %",
                value=f"{proba_default*100:.1f}%",
                delta=f"{(proba_default-threshold)*100:.1f} vs seuil"
            )
            st.success(f"✅ Prédiction effectuée : **{'Défaut' if pred==1 else 'Pas de Défaut'}**")

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")

    
    # Construire un DataFrame avec les données du client
    client_data = pd.DataFrame([input_data]) # un seul client
    # Explication d’un seul client
    explainer = shap.TreeExplainer(predictor.model)   # mon modèl lightgbm
    
    # Appliquer le preprocessing (OHE + alignement + imputation)
    X_client = predictor.preprocess(client_data)

    shap_values = explainer.shap_values(X_client)

    st.subheader("🌟 Explication locale de la prédiction")
    # shap_values est un np.array 2D
    shap_values_client = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": predictor.feature_names,
        "shap_value": shap_values_client
    })

    # Trier par importance absolue
    shap_df["abs_value"] = np.abs(shap_df["shap_value"])
    shap_df = shap_df.sort_values("abs_value", ascending=False).head(5)

    fig = px.bar(
        shap_df,
        x="shap_value",
        y="feature",
        orientation="h",
        title="Top 5 des variables qui influencent la prédiction du client",
        labels={"shap_value": "Impact sur le risque", "feature": "Variable"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # shap_values alignés avec les features
    X_client = predictor.preprocess(client_data)
    explainer = shap.TreeExplainer(predictor.model)
    shap_values = explainer.shap_values(X_client)
    # on prend les contributions de la classe 1 (défaut)
    shap_client = shap_values[1][0, :]  
    top_influencers = pd.Series(shap_client, index=client_data.columns) \
                    .sort_values(key=abs, ascending=False) \
                    .head(3)
    explication = "Les facteurs qui influencent le plus cette prédiction sont : "
    for var, val in top_influencers.items():
        sens = "augmentent" if val > 0 else "diminuent"
        explication += f"**{var}** ({sens} le risque), "
    st.markdown(explication.rstrip(", "))


