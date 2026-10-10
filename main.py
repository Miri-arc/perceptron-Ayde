"""Entrena y evalúa los modelos de Perceptrón.

Uso:
    python main.py --datos datos/pacientes.csv --objetivo diagnostico
"""
import argparse
import sys

from src.datos import cargar_datos, limpiar, estandarizar, dividir
from src.perceptron import Perceptron
from src.metricas import accuracy, error_clasificacion, matriz_confusion
from src.excepciones import DatosInvalidosError, ModeloNoEntrenadoError

FEATURES_BASE = ["radio", "textura", "perimetro", "area"]


def main():
    parser = argparse.ArgumentParser(description="Clasificador con Perceptrón")
    parser.add_argument("--datos", default="datos/pacientes.csv")
    parser.add_argument("--objetivo", default="diagnostico")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["concavidad", "puntos_concavos", "area", "textura"],
        help="Lista de características para el Modelo 3",
    )
    args = parser.parse_args()

    df = cargar_datos(args.datos, args.objetivo)

    # ---------------- Modelo 1: básico ----------------
    datos1 = limpiar(df, FEATURES_BASE)
    X1 = estandarizar(datos1[FEATURES_BASE].to_numpy(dtype=float))
    y1 = datos1[args.objetivo].to_numpy()
    X_tr1, X_te1, y_tr1, y_te1 = dividir(X1, y1)
    modelo1 = Perceptron(tasa_aprendizaje=0.01, epocas=30)
    modelo1.entrenar(X_tr1, y_tr1)
    y_pred1 = modelo1.predecir(X_te1)
    print("Modelo 1 (tasa 0.01)")
    print("  accuracy:", round(accuracy(y_te1, y_pred1), 3))
    print("  error:", round(error_clasificacion(y_te1, y_pred1), 3))
    print("  matriz de confusión:\n", matriz_confusion(y_te1, y_pred1))
    print("  errores por época:", modelo1.errores_por_epoca[:10], "...")

    # ---------------- Modelo 2: tasa grande con decaimiento ----------------
    datos2 = limpiar(df, FEATURES_BASE)
    X2 = estandarizar(datos2[FEATURES_BASE].to_numpy(dtype=float))
    y2 = datos2[args.objetivo].to_numpy()
    X_tr2, X_te2, y_tr2, y_te2 = dividir(X2, y2)
    modelo2 = Perceptron(tasa_aprendizaje=0.5, epocas=30, decaimiento=0.1)
    modelo2.entrenar(X_tr2, y_tr2)
    y_pred2 = modelo2.predecir(X_te2)
    print("\nModelo 2 (tasa 0.5, decaimiento 0.1)")
    print("  accuracy:", round(accuracy(y_te2, y_pred2), 3))
    print("  error:", round(error_clasificacion(y_te2, y_pred2), 3))
    print("  matriz de confusión:\n", matriz_confusion(y_te2, y_pred2))
    print("  errores por época:", modelo2.errores_por_epoca[:10], "...")

    # ---------------- Modelo 3: otras features ----------------
    datos3 = limpiar(df, args.features)
    X3 = estandarizar(datos3[args.features].to_numpy(dtype=float))
    y3 = datos3[args.objetivo].to_numpy()
    X_tr3, X_te3, y_tr3, y_te3 = dividir(X3, y3)
    modelo3 = Perceptron(tasa_aprendizaje=0.01, epocas=30)
    modelo3.entrenar(X_tr3, y_tr3)
    y_pred3 = modelo3.predecir(X_te3)
    print("\nModelo 3 (otras features)")
    print("  accuracy:", round(accuracy(y_te3, y_pred3), 3))
    print("  error:", round(error_clasificacion(y_te3, y_pred3), 3))
    print("  matriz de confusión:\n", matriz_confusion(y_te3, y_pred3))
    print("  errores por época:", modelo3.errores_por_epoca[:10], "...")


if __name__ == "__main__":
    try:
        main()
    except (DatosInvalidosError, ModeloNoEntrenadoError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)