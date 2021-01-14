import json

import mlflow
from Algorithmia.errors import AlgorithmException
from mlflow import pyfunc
from mlflow.pyfunc import scoring_server


try:
    from . import algorithmia_utils
except ImportError:
    import algorithmia_utils


class MLflowWrapper(object):
    def __init__(self, model_fpath=None):
        self.model = None

        if model_fpath:
            self.load_model(model_fpath)

    def load_model(self, model_fpath):
        if model_fpath.startswith("data://"):
            model_fpath = algorithmia_utils.get_file(model_fpath)

        self.model = pyfunc.load_model(model_fpath)

    def predict(self, input):
        if isinstance(input, dict):
            input = json.dumps(input)

        if isinstance(input, str):
            df = scoring_server.parse_json_input(input)
        else:
            raise AlgorithmException("Input should be str or json")

        return self.model.predict(df)
