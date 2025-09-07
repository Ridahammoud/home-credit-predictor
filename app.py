#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Flask pour HomeCreditPredictor
Expose:
 - POST /predict_single  (JSON body)
 - POST /predict_batch   (multipart/form-data file upload OR JSON array)
"""

import os
import io
import logging
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

from predict_model import HomeCreditPredictor


# Configuration & Logging

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("home-credit-api")

# Taille max upload (optionnel)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB


# Init Flask

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)  # Autorise les appels cross-origin (utile pour Streamlit hébergé séparément)

# Charger le modèle une seule fois 

# Si tes chemins sont relatifs au repo, laisse tel quel. Sinon utilise des variables d'env.
MODEL_PATH = os.environ.get("MODEL_PATH", "models/final_lgb_model_f.pkl")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "feature_importances.csv")

logger.info("Initialisation du HomeCreditPredictor...")
try:
    predictor = HomeCreditPredictor(model_path=MODEL_PATH, features_path=FEATURES_PATH)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.exception("Erreur au chargement du modèle : %s", e)
    # On ne crash pas l'app directement pour permettre le debug; mais les endpoints renverront une erreur.
    predictor = None


def ensure_predictor():
    if predictor is None:
        raise RuntimeError("Le modèle n'a pas été chargé. Vérifie MODEL_PATH et dependencies.")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé simple"""
    ok = predictor is not None
    return jsonify({"status": "ok" if ok else "error", "model_loaded": ok})

@app.route("/predict_single", methods=["POST"])
def predict_single():
    """
    Attendu: JSON body contenant les features (ex: {"EXT_SOURCE_1": 0.5, "AMT_CREDIT": 100000, ...})
    Optional: "threshold" (float entre 0 et 1) si tu veux obtenir la décision selon un seuil.
    Retour: JSON avec proba, TARGET_PRED, decision (selon threshold si fourni).
    """
    try:
        ensure_predictor()

        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        payload: Dict[str, Any] = request.get_json()

        # Extraire threshold si présent
        threshold = payload.pop("threshold", None)

        # Prévenir entrée vide
        if not payload:
            return jsonify({"error": "Body JSON vide : envoyez les features du client."}), 400

        # Appel à la méthode existante
        result = predictor.predict_single(payload)

        # Vérification format attendu
        if not isinstance(result, dict) or "PROBA_DEFAULT" not in result:
            # tolérance: ton code Streamlit utilisait result['PROBA_DEFAULT']
            logger.warning("predict_single a retourné un format inattendu : %s", type(result))
        
        proba = result.get("PROBA_DEFAULT", None) or result.get("PROBA_DEFAULT".upper(), None)
        # fallback to 'PROBA_DEFAULT' key if used in uppercase or different naming
        proba = result.get("PROBA_DEFAULT", proba)

        response = {
            "result_raw": result,
        }

        if proba is not None:
            response["PROBA_DEFAULT"] = float(proba)
            # target pred if present
            if "TARGET_PRED" in result:
                response["TARGET_PRED"] = int(result["TARGET_PRED"])
            # décision selon threshold si fourni
            if threshold is not None:
                try:
                    thr = float(threshold)
                    response["decision"] = "REFUSE" if proba >= thr else "ACCEPTE"
                    response["threshold"] = thr
                except Exception:
                    response["threshold_error"] = "threshold doit être convertible en float"
        else:
            logger.warning("Aucune PROBA_DEFAULT trouvée dans le résultat du predicteur.")

        return jsonify(response), 200

    except RuntimeError as re:
        logger.exception("Erreur runtime: %s", re)
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.exception("Erreur lors de /predict_single: %s", e)
        return jsonify({"error": f"Erreur interne: {e}"}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Accepts:
      - multipart/form-data with a file field named "file" (CSV),
      OR
      - application/json with "data": list-of-rows (list of dicts)
    Optional query/form param:
      - seuil or threshold (0-1 float) to filter returned results by PROBA_DEFAULT
      - download = "csv" to receive a CSV file back as attachment
    Returns:
      JSON with keys:
        - columns: list
        - predictions: list of rows (dict)
    Or if download=csv, returns a CSV file.
    """
    try:
        ensure_predictor()

        # Priorité: fichier uploadé
        df = None
        if "file" in request.files:
            uploaded = request.files["file"]
            if uploaded.filename == "":
                return jsonify({"error": "Aucun fichier fourni"}), 400
            # pandas peut lire le flux
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                logger.exception("Impossible de lire le CSV uploadé : %s", e)
                return jsonify({"error": f"Impossible de lire le CSV : {e}"}), 400

        elif request.is_json:
            payload = request.get_json()
            data = payload.get("data", None)
            if data is None:
                return jsonify({"error": "Pour JSON, envoyez {'data': [ ... ]}"}), 400
            try:
                df = pd.DataFrame(data)
            except Exception as e:
                logger.exception("Impossible de convertir JSON->DataFrame: %s", e)
                return jsonify({"error": f"Impossible de convertir en tableau : {e}"}), 400
        else:
            return jsonify({"error": "Aucun fichier uploadé et pas de JSON envoyé. Utilisez multipart/form-data (file) ou JSON."}), 400

        # garder les SK_ID_CURR si présent
        if "SK_ID_CURR" in df.columns:
            ids = df["SK_ID_CURR"]
        else:
            ids = pd.Series(range(len(df)), name="SK_ID_CURR")

        # On suppose qu'il retourne un DataFrame
        try:
            results = predictor.predict_batch(df)
        except Exception as e:
            logger.exception("predict_batch a échoué : %s", e)
            return jsonify({"error": f"Erreur lors de la prédiction batch : {e}"}), 500

        # On insère l'ID en première colonne si nécessaire
        if isinstance(results, pd.DataFrame):
            results.insert(0, "SK_ID_CURR", ids)
        else:
            # si la méthode retourne une liste de dicts
            try:
                results = pd.DataFrame(results)
                results.insert(0, "SK_ID_CURR", ids)
            except Exception:
                # si pas convertible, on renvoie ce qu'on a
                return jsonify({"error": "Le format retourné par predictor.predict_batch n'est pas un DataFrame convertible."}), 500

        # Filtrage par seuil si fourni (query param or form)
        seuil = request.args.get("seuil") or request.form.get("seuil") or request.json and request.json.get("seuil")
        if seuil is not None:
            try:
                s = float(seuil)
                results_filtered = results[results["PROBA_DEFAULT"] >= s]
            except Exception:
                logger.warning("Seuil fourni non convertible en float : %s", seuil)
                results_filtered = results.copy()
        else:
            results_filtered = results.copy()

        # Si l'utilisateur veut le CSV en retour
        download = request.args.get("download", request.form.get("download", None))
        if download == "csv":
            csv_bytes = df_to_csv_bytes(results_filtered)
            return send_file(
                io.BytesIO(csv_bytes),
                mimetype="text/csv",
                as_attachment=True,
                download_name="predictions_home_credit.csv"
            )

        # Sinon retourne JSON
        return jsonify({
            "n_input_rows": len(df),
            "n_predictions": len(results_filtered),
            "columns": list(results_filtered.columns),
            "predictions": results_filtered.to_dict(orient="records")
        }), 200

    except RuntimeError as re:
        logger.exception("Erreur runtime: %s", re)
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.exception("Erreur lors de /predict_batch: %s", e)
        return jsonify({"error": f"Erreur interne: {e}"}), 500



# Lancement 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug à False pour la prod ; utilise gunicorn en prod
    app.run(host="0.0.0.0", port=port, debug=False)
