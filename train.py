"""
Entrenamiento del MLP para predecir si un estudiante aprueba (1) o reprueba (0).
Pasos: carga -> imputación (mediana) -> split -> estandarización -> MLP -> evaluación.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score)

DATA_PATH = "dataset_estudiantes_balanceado.csv"
MODEL_PATH = "modelo_mlp.joblib"

# Calificacion_Final se EXCLUYE: Aprobado se calculó a partir de ella (fuga de datos).
FEATURES = ["Tareas", "Examenes", "Asistencia",
            "Estudio_Horas", "Sueno_Horas", "Extracurriculares"]
TARGET = "Aprobado"


def train(data_path=DATA_PATH, model_path=MODEL_PATH, make_plots=True):
    # 1. Carga
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    # 2. Split estratificado (el test queda independiente y también 50/50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Imputación con mediana (se ajusta solo con train)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 4. Estandarización: media 0, varianza 1 (se ajusta solo con train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. MLP: 2 capas ocultas (16 y 8 neuronas), ReLU, Adam, early stopping
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        solver="adam", alpha=1e-3, max_iter=1000,
                        early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=30, random_state=42)
    mlp.fit(X_train, y_train)

    # 6. Evaluación sobre el set de prueba independiente
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Reprobado", "Aprobado"],
                                   output_dict=True)
    print("Accuracy (test):", round(acc, 4))
    print(classification_report(y_test, y_pred,
                                target_names=["Reprobado", "Aprobado"]))
    print("Matriz de confusión:\n", cm)
    print("Iteraciones:", mlp.n_iter_)

    if make_plots:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=["Reprobado", "Aprobado"]
                               ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusión - Set de prueba")
        fig.tight_layout()
        fig.savefig("matriz_confusion.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mlp.loss_curve_, label="Pérdida (train)")
        ax.plot(1 - np.array(mlp.validation_scores_), label="Error (validación)")
        ax.set_xlabel("Época"); ax.set_ylabel("Valor"); ax.legend()
        ax.set_title("Curva de aprendizaje")
        fig.tight_layout()
        fig.savefig("curva_aprendizaje.png", dpi=150)
        plt.close(fig)

    # 7. Guardar todo lo necesario para predecir
    joblib.dump({"imputer": imputer, "scaler": scaler, "mlp": mlp,
                 "features": FEATURES}, model_path)
    metrics = {"accuracy": acc, "confusion_matrix": cm.tolist(),
               "report": report, "n_train": len(y_train), "n_test": len(y_test)}
    with open("metricas.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    train()
