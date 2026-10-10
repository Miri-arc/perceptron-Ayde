"""Métricas para evaluar un clasificador binario."""
import numpy as np


def accuracy(y, y_pred):
    """Proporción de predicciones correctas (entre 0 y 1)."""
    return float(np.mean(np.asarray(y) == np.asarray(y_pred)))


def error_clasificacion(y, y_pred):
    """Proporción de predicciones incorrectas (entre 0 y 1)."""
    return 1.0 - accuracy(y, y_pred)


def matriz_confusion(y, y_pred):
    """Devuelve la matriz 2x2 [[TN, FP], [FN, TP]] como array de NumPy."""
    y_real = np.asarray(y)
    y_p = np.asarray(y_pred)
    
    tn = np.sum((y_real == 0) & (y_p == 0))
    fp = np.sum((y_real == 0) & (y_p == 1))
    fn = np.sum((y_real == 1) & (y_p == 0))
    tp = np.sum((y_real == 1) & (y_p == 1))
    
    return np.array([[tn, fp], [fn, tp]])