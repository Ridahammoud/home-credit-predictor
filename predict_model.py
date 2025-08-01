import pandas as pd
import joblib

class HomeCreditPredictor:
    def __init__(self, model_path: str = "model/final_lgb_model.pkl"):
        """
        Classe utilitaire pour charger un modèle LightGBM entraîné et effectuer des prédictions.
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Charge le modèle sauvegardé."""
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise FileNotFoundError(f"Erreur lors du chargement du modèle : {e}")

    def predict_batch(self, df: pd.DataFrame):
        """
        Prédit la variable TARGET et la probabilité pour un DataFrame complet.

        :param df: DataFrame contenant les features. Si 'SK_ID_CURR' est présent, il sera conservé.
        :return: DataFrame avec SK_ID_CURR, TARGET_PRED et PROBA
        """
        ids = df["SK_ID_CURR"] if "SK_ID_CURR" in df.columns else pd.Series(range(len(df)))
        features = df.drop(columns=["SK_ID_CURR"], errors="ignore")

        probas = self.model.predict_proba(features)[:, 1]
        preds = self.model.predict(features)

        results = pd.DataFrame({
            "SK_ID_CURR": ids,
            "TARGET_PRED": preds,
            "PROBA": probas
        })
        return results

    def predict_single(self, features_dict: dict):
        """
        Prédit pour un seul individu à partir d'un dictionnaire de features.

        :param features_dict: dictionnaire {feature: valeur}
        :return: tuple (prediction, probabilité)
        """
        df_input = pd.DataFrame([features_dict])
        proba = self.model.predict_proba(df_input)[:, 1][0]
        pred = self.model.predict(df_input)[0]
        return int(pred), float(proba)
