model = None
mlflow_bundle = "{{ mlflow_bundle_file }}"


def get_model():
    global model
    if model is None:
        try:
            from .mlflow_wrapper import MLflowWrapper
        except ImportError:
            from mlflow_wrapper import MLflowWrapper

        model = MLflowWrapper()
        model.load_model(mlflow_bundle)
    return model


def apply(input):
    try:
        model = get_model()
        predictions = model.predict(input)
        return predictions.tolist()
    except Exception as ex:
        raise ex
