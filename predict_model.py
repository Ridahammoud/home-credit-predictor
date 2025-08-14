import pandas as pd
import joblib
import os

class HomeCreditPredictor:
    def __init__(self, model_path="models/final_lgb_model_2.pkl", features_path="feature_importances.csv"):
        # Vérifie l'existence des fichiers
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Modèle introuvable : {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"⚠️ Fichier features introuvable : {features_path}")
        
        # Charger le modèle
        self.model = joblib.load(model_path)
        
        # Charger la liste des colonnes importantes
        self.feature_names = pd.read_csv(features_path)["feature"].tolist()
        # Retirer SK_ID_CURR si présent
        self.feature_names = [f for f in self.feature_names if f != "SK_ID_CURR"]

    def preprocess(self, df):
        """Prépare les données comme à l'entraînement (One-Hot Encoding + alignement)."""
        # Supprimer l'ID si présent
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df)

        # Réaligner sur les colonnes d'entraînement
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Conversion en float
        df_aligned = df_aligned.astype(float)

        return df_aligned

    def predict_batch(self, df, return_proba=True):
        """Prédiction en batch (plusieurs lignes)."""
        features = self.preprocess(df)
        probas = self.model.predict_proba(features)[:, 1]
        
        if return_proba:
            return pd.DataFrame({
                "TARGET_PRED": (probas >= 0.5).astype(int),
                "PROBA_DEFAULT": probas
            })
        else:
            return (probas >= 0.5).astype(int)

    def predict_single(self, input_data):
        """Prédiction sur un seul client."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        return self.predict_batch(df).iloc[0]
