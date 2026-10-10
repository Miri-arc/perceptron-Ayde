class DatosInvalidosError(Exception):
    """Excepción lanzada cuando los datos cargados no son válidos."""
    pass

class ModeloNoEntrenadoError(Exception):
    """Excepción lanzada al intentar predecir sin haber entrenado el modelo."""
    pass

    import sys
from src.excepciones import DatosInvalidosError, ModeloNoEntrenadoError

# En tu bloque principal o función main():
try:
    # ... todo tu flujo actual de main.py ...
    pass
except (DatosInvalidosError, ModeloNoEntrenadoError, ValueError) as e:
    print(f"Error: {e}")
    sys.exit(1)