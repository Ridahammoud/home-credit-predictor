import pandas as pd
import joblib
import os

class HomeCreditPredictor:
    def __init__(self, model_path="model/final_lgb_model.pkl", features_path="feature_importances.csv"):
        # Vérifie l'existence des fichiers
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Modèle introuvable : {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"⚠️ Fichier features introuvable : {features_path}")
        
        # Charger le modèle
        self.model = joblib.load(model_path)
        # Charger la liste des colonnes importantes
        feature_importances = pd.read_csv(features_path)
        self.feature_names = [f for f in feature_importances["feature"].tolist() if f != "SK_ID_CURR"]

    def preprocess(self, df):
        """Prépare les données comme à l'entraînement (One-Hot Encoding + alignement sur features importantes)."""
        if 'SK_ID_CURR' in df.columns:
            df = df.drop(columns=['SK_ID_CURR'])

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df)

        # Réalignement exact avec les colonnes d'entraînement (important_features)
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Convertir en float
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
