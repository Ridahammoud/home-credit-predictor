import pandas as pd
import joblib
import os

class HomeCreditPredictor:
    def __init__(self, model_path="models/final_lgb_model_2.pkl", features_path="feature_importances.csv"):
        # Vérification des fichiers
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Modèle introuvable : {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"⚠️ Fichier de features introuvable : {features_path}")

        # Charger le modèle
        self.model = joblib.load(model_path)

        # Charger les features importantes et exclure SK_ID_CURR
        feature_importances = pd.read_csv(features_path)
        self.feature_names = [f for f in feature_importances["feature"].tolist() if f != "SK_ID_CURR"]

    def preprocess(self, df):
        """Prépare les données comme à l'entraînement (One-Hot Encoding + alignement sur features importantes)."""
        # Retirer SK_ID_CURR si présent
        if 'SK_ID_CURR' in df.columns:
            df = df.drop(columns=['SK_ID_CURR'])

        # Encodage One-Hot
        df_encoded = pd.get_dummies(df)

        # Réalignement exact avec les colonnes d'entraînement
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Convertir en float
        return df_aligned.astype(float)

    def predict_batch(self, df):
        """Prédiction pour plusieurs clients."""
        features = self.preprocess(df)
        probas = self.model.predict_proba(features)[:, 1]
        predictions = (probas >= 0.5).astype(int)
        return pd.DataFrame({
            "TARGET_PRED": predictions,
            "PROBA_DEFAULT": probas
        })

    def predict_single(self, input_data):
        """Prédiction pour un seul client."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        return self.predict_batch(df).iloc[0]
