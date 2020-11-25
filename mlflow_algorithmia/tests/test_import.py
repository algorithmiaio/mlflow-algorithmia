def test_import():
    import mlflow_algorithmia

    assert mlflow_algorithmia.__version__ is not None
    assert mlflow_algorithmia.__version__ != "0.0.0"
    assert len(mlflow_algorithmia.__version__) > 0
