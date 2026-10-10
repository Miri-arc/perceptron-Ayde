"""Carga y preparación de los datos para el Perceptrón."""
import os
import numpy as np
import pandas as pd
from src.excepciones import DatosInvalidosError


def cargar_datos(ruta, objetivo):
    """Lee el CSV y devuelve el DataFrame validando la existencia del archivo,
    la presencia de la columna objetivo y que esta sea binaria.
    """
    if not os.path.exists(ruta):
        raise DatosInvalidosError(f"No existe el archivo {ruta}")

    df = pd.read_csv(ruta)

    if objetivo not in df.columns:
        raise DatosInvalidosError(f"La columna objetivo '{objetivo}' no existe en el archivo.")

    valores_unicos = df[objetivo].nunique()
    if valores_unicos != 2:
        raise DatosInvalidosError(f"La columna '{objetivo}' no es binaria (posee {valores_unicos} valores distintos).")

    return df


def limpiar(df, features):
    """Valida que existan las columnas y devuelve una copia con los nulos de cada feature rellenados con su mediana."""
    columnas_faltantes = [f for f in features if f not in df.columns]
    if columnas_faltantes:
        raise DatosInvalidosError(f"Columnas inexistentes: {', '.join(columnas_faltantes)}")

    datos = df.copy()
    for col in features:
        datos[col] = datos[col].fillna(datos[col].median())
    return datos


def estandarizar(X):
    """Deja cada columna con media 0 y desviación 1."""
    media = X.mean(axis=0)
    desv = X.std(axis=0)
    desv = np.where(desv == 0, 1, desv)
    return (X - media) / desv


def dividir(X, y, prueba=0.2, semilla=42):
    """Mezcla las filas y separa entrenamiento y prueba."""
    idx = np.random.default_rng(semilla).permutation(len(X))
    n_prueba = int(len(X) * prueba)
    te, tr = idx[:n_prueba], idx[n_prueba:]
    return X[tr], X[te], y[tr], y[te]