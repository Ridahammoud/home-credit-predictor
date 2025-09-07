#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import pandas as pd
from werkzeug.utils import secure_filename

# Config
API_URL = os.environ.get("API_URL", "https://home-credit-predictor-g51k.onrender.com")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict_single_form", methods=["POST"])
def predict_single_form():
    try:
        # Récupérer les données du formulaire
        data = {k: float(v) for k, v in request.form.items()}
        response = requests.post(f"{API_URL}/predict_single", json=data, timeout=15)
        if response.status_code != 200:
            flash(f"Erreur API: {response.text}", "danger")
            return redirect(url_for("index"))
        result = response.json()
        return render_template("result.html", result=result)
    except Exception as e:
        logger.exception("Erreur prédiction single")
        flash(f"Erreur serveur: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/predict_batch_upload", methods=["POST"])
def predict_batch_upload():
    try:
        if "file" not in request.files:
            flash("Aucun fichier sélectionné", "warning")
            return redirect(url_for("index"))
        file = request.files["file"]
        if file.filename == "":
            flash("Nom de fichier vide", "warning")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Format non supporté. CSV uniquement.", "warning")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        # Envoyer le fichier à l'API
        with open(filepath, "rb") as f:
            response = requests.post(f"{API_URL}/predict_batch", files={"file": f}, timeout=600)
        if response.status_code != 200:
            flash(f"Erreur API: {response.text}", "danger")
            return redirect(url_for("index"))

        result = response.json()
        return render_template("batch_result.html", result=result)

    except Exception as e:
        logger.exception("Erreur prédiction batch")
        flash(f"Erreur serveur: {e}", "danger")
        return redirect(url_for("index"))

# Lancement
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
