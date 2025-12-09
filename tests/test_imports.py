import os
import sys

# Empêche Pandas d'utiliser PyArrow (évite l'erreur null bytes)
os.environ["PANDAS_BACKEND"] = "numpy"

# Ajoute la racine du projet au PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_imports():
    try:
        import scripts.train_transformer
    except Exception as e:
        assert False, f"Erreur d'import: {e}"

    assert True
