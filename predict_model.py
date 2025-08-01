import pandas as pd
import joblib
import os

class HomeCreditPredictor:
    def __init__(self, model_path="models/final_lgb_model.pkl"):
        # Vérifie l'existence du fichier modèle
        if not os.path.exists(model_path):
            raise FileNotFoundError("⚠️ Modèle introuvable. Assurez-vous que 'final_lgb_model.pkl' est dans le dossier /models")
        
        # Chargement du modèle
        self.model = joblib.load(model_path)
        
        # Récupération des colonnes d'entraînement directement depuis le modèle
        if hasattr(self.model, "feature_name_"):
            self.feature_names = list(self.model.feature_name_)
        else:
            self.feature_names = list(self.model.booster_.feature_name())

    def preprocess(self, df):
        """Prépare les données comme à l'entraînement (One-Hot Encoding + alignement)."""
        # Supprimer l'ID si présent
        if 'SK_ID_CURR' in df.columns:
            df = df.drop(columns=['SK_ID_CURR'])

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df)

        # Aligner avec les colonnes utilisées lors de l'entraînement
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Convertir en float (LightGBM n'accepte pas object ou string)
        df_aligned = df_aligned.astype(float)

        return df_aligned

    def predict_batch(self, df):
        """Prédiction en batch (plusieurs lignes)."""
        features = self.preprocess(df)
        probas = self.model.predict_proba(features)[:, 1]
        predictions = (probas >= 0.5).astype(int)
        return pd.DataFrame({
            "TARGET_PRED": predictions,
            "PROBA_DEFAULT": probas
        })

    def predict_single(self, input_data):
        """Prédiction sur un seul client (dictionnaire ou DataFrame)."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        return self.predict_batch(df).iloc[0]
