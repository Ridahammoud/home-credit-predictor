import streamlit as st
import pandas as pd
from predict_model import HomeCreditPredictor

# Initialisation du modèle avec features importantes
predictor = HomeCreditPredictor(
    model_path="models/final_lgb_model_2.pkl",
    features_path="feature_importances.csv"
)

# Titre de l'application
st.title("🏦 Prédiction du risque de défaut - Home Credit")

# Onglets
tab1, tab2 = st.tabs(["📂 Prédictions sur fichier CSV", "🔮 Prédiction manuelle"])

# ============================
# 1️⃣ Prédiction en batch (CSV)
# ============================
with tab1:
    st.header("📂 Charger un fichier CSV (ex: application_test.csv)")
    uploaded_file = st.file_uploader("Déposez ici votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Lecture du fichier CSV brut
            df = pd.read_csv(uploaded_file)

            st.write("✅ Aperçu des données chargées :")
            st.dataframe(df.head())

            # Sauvegarder SK_ID_CURR s'il est présent
            if "SK_ID_CURR" in df.columns:
                ids = df["SK_ID_CURR"]
            else:
                ids = pd.Series(range(len(df)), name="SK_ID_CURR")

            # Prédiction
            results = predictor.predict_batch(df)
            results.insert(0, "SK_ID_CURR", ids)

            # Barre de recherche par ID
            search_id = st.text_input("🔎 Rechercher un client par SK_ID_CURR :", "")
            results_filtered = results.copy()

            if search_id:
                try:
                    search_id = int(search_id)
                    results_filtered = results_filtered[results_filtered["SK_ID_CURR"] == search_id]
                except ValueError:
                    st.warning("Veuillez entrer un identifiant numérique valide.")

            st.success("✅ Prédictions effectuées avec succès !")
            st.dataframe(results_filtered.head(10))

            # Téléchargement des résultats
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les prédictions",
                data=csv,
                file_name="predictions_home_credit.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")

# ============================
# 2️⃣ Prédiction manuelle
# ============================
with tab2:
    st.header("🔮 Saisir manuellement les données client")

    st.info("ℹ️ Renseignez uniquement quelques champs pour tester la prédiction (les colonnes manquantes seront mises à zéro).")

    # Exemple minimal : saisie utilisateur
    input_data = {}
    input_data["EXT_SOURCE_1"] = st.number_input("EXT_SOURCE_1", min_value=0.0, max_value=1.0, step=0.01)
    input_data["EXT_SOURCE_2"] = st.number_input("EXT_SOURCE_2", min_value=0.0, max_value=1.0, step=0.01)
    input_data["EXT_SOURCE_3"] = st.number_input("EXT_SOURCE_3", min_value=0.0, max_value=1.0, step=0.01)
    input_data["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", min_value=0.0, step=100.0)
    input_data["AMT_CREDIT"] = st.number_input("AMT_CREDIT", min_value=0.0, step=100.0)
    input_data["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH (négatif)", value=-10000, step=1)

    if st.button("🔎 Prédire le risque"):
        try:
            result = predictor.predict_single(input_data)
            st.success(f"✅ Prédiction effectuée : **{'Défaut' if result['TARGET_PRED']==1 else 'Pas de Défaut'}**")
            st.info(f"📊 Probabilité de défaut : **{result['PROBA_DEFAULT']:.2%}**")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
