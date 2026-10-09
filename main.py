"""Entrena y evalúa los modelos de Perceptrón.

Uso:
    python main.py --datos datos/pacientes.csv --objetivo diagnostico
"""
import argparse
import sys

from src.datos import cargar_datos, limpiar, estandarizar, dividir
from src.perceptron import Perceptron
from src.metricas import accuracy
from src.excepciones import DatosInvalidosError, ModeloNoEntrenadoError

FEATURES_BASE = ["radio", "textura", "perimetro", "area"]


def main():
    parser = argparse.ArgumentParser(description="Clasificador con Perceptrón")
    parser.add_argument("--datos", default="datos/pacientes.csv")
    parser.add_argument("--objetivo", default="diagnostico")
    args = parser.parse_args()

    df = cargar_datos(args.datos, args.objetivo)

    # ---------------- Modelo 1: básico ----------------
    datos = limpiar(df, FEATURES_BASE)
    X = estandarizar(datos[FEATURES_BASE].to_numpy(dtype=float))
    y = datos[args.objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    modelo1 = Perceptron(tasa_aprendizaje=0.01, epocas=30)
    modelo1.entrenar(X_tr, y_tr)
    y_pred = modelo1.predecir(X_te)
    print("Modelo 1 (tasa 0.01)")
    print("  accuracy:", round(accuracy(y_te, y_pred), 3))
    print("  errores por época:", modelo1.errores_por_epoca[:10], "...")

    # ---------------- Modelo 2: tasa grande ----------------
    datos = limpiar(df, FEATURES_BASE)
    X = estandarizar(datos[FEATURES_BASE].to_numpy(dtype=float))
    y = datos[args.objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    modelo2 = Perceptron(tasa_aprendizaje=0.5, epocas=30)
    modelo2.entrenar(X_tr, y_tr)
    y_pred = modelo2.predecir(X_te)
    print("Modelo 2 (tasa 0.5)")
    print("  accuracy:", round(accuracy(y_te, y_pred), 3))
    print("  errores por época:", modelo2.errores_por_epoca[:10], "...")

    # ---------------- Modelo 3: otras features ----------------
    # Tarea 5: permitir elegir las features con --features


if __name__ == "__main__":
    try:
        main()
    except (DatosInvalidosError, ModeloNoEntrenadoError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)