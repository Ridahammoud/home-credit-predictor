import streamlit as st
import pandas as pd
from predict_model import HomeCreditPredictor

# Initialisation du modèle avec les features importantes
predictor = HomeCreditPredictor(
    model_path="model/final_lgb_model_2.pkl",
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
    st.header("📂 Charger un fichier CSV (ex: app_test.csv)")
    uploaded_file = st.file_uploader("Déposez ici votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Lecture du CSV
            df = pd.read_csv(uploaded_file)

            st.write("✅ Aperçu des données chargées :")
            st.dataframe(df.head())

            # Récupérer SK_ID_CURR si présent
            if "SK_ID_CURR" in df.columns:
                ids = df["SK_ID_CURR"]
            else:
                ids = pd.Series(range(len(df)), name="SK_ID_CURR")

            # Prédictions
            results = predictor.predict_batch(df)
            results.insert(0, "SK_ID_CURR", ids)

            # Seuil choisi par l'utilisateur
            seuil = st.slider("⚖️ Choisissez le seuil de probabilité de défaut %",
                              min_value=0, max_value=100, value=50, step=1) / 100.0

            # Filtrage des résultats supérieurs au seuil 
            results_filtered = results[results["PROBA_DEFAULT"] >= seuil ]

            # Barre de recherche
            search_id = st.text_input("🔎 Rechercher un client par SK_ID_CURR :", "")
            results_filtered_c = results_filtered.copy()

            if search_id:
                try:
                    search_id = int(search_id)
                    results_filtered_c = results_filtered_c[results_filtered_c["SK_ID_CURR"] == search_id]
                except:
                    st.warning("Veuillez entrer un identifiant numérique valide.")

            st.success("✅ Prédictions effectuées avec succès !")
            st.dataframe(results_filtered.head(20))

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

    st.header("📂 Les Features Importances ")
    fi = pd.read_csv('feature_importances.csv')
    st.dataframe(fi.head(10))

# ============================
# 2️⃣ Prédiction manuelle
# ============================
with tab2:
    st.header("🔮 Saisir manuellement les données client")
    st.info("ℹ️ Renseignez uniquement quelques champs pour tester la prédiction (les colonnes manquantes seront mises à zéro).")

    input_data = {}
    input_data["EXT_SOURCE_1"] = st.number_input("EXT_SOURCE_1", min_value=0.0, max_value=100000.0, step=0.01)
    input_data["EXT_SOURCE_2"] = st.number_input("EXT_SOURCE_2", min_value=0.0, max_value=100000.0, step=0.01)
    input_data["EXT_SOURCE_3"] = st.number_input("EXT_SOURCE_3", min_value=0.0, max_value=100000.0, step=0.01)
    input_data["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", min_value=0.0, step=100.0)
    input_data["AMT_CREDIT"] = st.number_input("AMT_CREDIT", min_value=0.0, step=100.0)
    input_data["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH (négatif)", value=-10000, step=1)

    # Seuil choisi par l'utilisateur
    threshold = st.slider("⚖️ Choisir le seuil de refus", 0.0, 1.0, 0.5, 0.01)

    if st.button("🔎 Prédire le risque"):
        try:
            result = predictor.predict_single(input_data)
            proba_default = result['PROBA_DEFAULT']

            st.info(f"📊 Probabilité de défaut : **{proba_default:.2%}** (seuil = {threshold:.0%})")

            # Affichage avec comparaison au seuil
            if proba_default < threshold:
                st.error("❌ Prêt refusé (risque trop élevé)")
                st.progress(min(1.0, proba_default))  # barre proportionnelle
            else:
                st.success("✅ Prêt accepté")
                st.progress(min(1.0, proba_default))  # même barre mais sous le seuil

            # Résultat binaire
            decision = "❌ Prêt Refusé" if proba_default >= threshold else "✅ Prêt Accepté"

            # Barre de progression
            st.progress(min(int(proba_default*100),100))

            #indicateur visuel avec jauge
            st.metric(
                label="Score de risque en %",
                value=f"{proba_default*100:.1f}%",
                delta=f"{(proba_default-threshold)*100:.1f} vs seuil"
            )
            st.success(f"✅ Prédiction effectuée : **{'Défaut' if result['TARGET_PRED']==1 else 'Pas de Défaut'}**")
            st.info(f"📊 Probabilité de défaut : **{result['PROBA_DEFAULT']:.2%}**")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
