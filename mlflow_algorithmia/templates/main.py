model = None
mlflow_bundle = "{{ mlflow_bundle }}"


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


if __name__ == "__main__":
    in1 = '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}'
    print(apply(in1))

    in2 = {
        "columns": [
            "alcohol",
            "chlorides",
            "citric acid",
            "density",
            "fixed acidity",
            "free sulfur dioxide",
            "pH",
            "residual sugar",
            "sulphates",
            "total sulfur dioxide",
            "volatile acidity",
        ],
        "data": [[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]],
    }
    print(apply(in2))
