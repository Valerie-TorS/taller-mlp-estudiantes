"""
Servicio web del MLP: entrena, expone el endpoint /predict y sirve un formulario.
Ejecutar local:  python app.py     (http://localhost:5000)
Producción:      gunicorn app:app
"""
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from train import train, MODEL_PATH, FEATURES

app = Flask(__name__)

# Si no existe un modelo entrenado, el servicio lo entrena al arrancar.
if not os.path.exists(MODEL_PATH):
    train(make_plots=False)
bundle = joblib.load(MODEL_PATH)


def predecir(datos: dict):
    """Recibe un dict con las 6 variables; los vacíos se imputan con la mediana."""
    fila = []
    for f in FEATURES:
        v = datos.get(f)
        fila.append(np.nan if v in (None, "") else float(v))
    X = np.array(fila, dtype=float).reshape(1, -1)
    X = bundle["imputer"].transform(X)
    X = bundle["scaler"].transform(X)
    proba = bundle["mlp"].predict_proba(X)[0]
    clase = int(np.argmax(proba))
    return {"aprobado": clase,
            "resultado": "Aprobado" if clase == 1 else "Reprobado",
            "probabilidad_aprobar": round(float(proba[1]), 4)}


@app.get("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    datos = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predecir(datos))
    except (ValueError, TypeError) as e:
        return jsonify(error=f"Datos inválidos: {e}"), 400


@app.post("/train")
def retrain():
    """Reentrena el modelo con el CSV y lo recarga en memoria."""
    global bundle
    m = train(make_plots=False)
    bundle = joblib.load(MODEL_PATH)
    return jsonify(mensaje="Modelo reentrenado", accuracy_test=round(m["accuracy"], 4))


@app.get("/metrics")
def metrics():
    with open("metricas.json") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
