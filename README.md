# mlflow-algorithmia

[![PyPI](https://badge.fury.io/py/mlflow-algorithmia.svg)](https://pypi.org/project/mlflow-algorithmia/)
[![Testing](https://github.com/algorithmia/mlflow-algorithmia/workflows/test/badge.svg)](https://github.com/algorithmia/mlflow-algorithmia/actions)
[![License](http://img.shields.io/:license-Apache%202-blue.svg)](https://github.com/algorithmia/mlflow-algorithmia/blob/master/LICENSE.txt)

Deploy MLflow models to Algorithmia

## Install

```
pip install git+git://github.com/algorithmia/mlflow-algorithmia.git

pip install mlflow-algorithmia
```

## Usage

This is based on the [mlflow tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
we reproduce some steps here but for more details look at the official [mlflow docs](https://www.mlflow.org/docs).

```
python examples/sklearn_elasticnet_wine/train.py
```

This will create an `mlruns` directory that contains the trained model, you can
view the UI running `mlflow ui` and start the mlflow server running:

```
$ mlflow models serve -m mlruns/0/<run-id>/artifacts/model -p 1234

# Make a test query

$ curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations

[5.120775719594933]
```

Now let's deploy the same endpoint in Algorithmia. You will need:
1. An Algorithmia API key with `Read + Write` data access
2. The path to the model (under `mlruns`) you want to deploy, for example: `mlruns/0/<run-id>/artifacts/model`

```
# Set your Algorithmia API key
export ALGORITHMIA_USERNAME=<username>
export ALGORITHMIA_API_KEY=<api-key>

# Create a deployment
mlflow deployments create -t algorithmia --name mlflow_sklearn_demo -m <path-to-model-dir>
```

Query the model in Algorithmia:
- You need the `ALGORITHMIA_USERNAME` and `ALGORITHMIA_API_KEY` variables from before and the `<version>` you want to query
- Note that if there are no published versions need to use a build hash as `<version>`

```
curl -X POST -d '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' -H 'Content-Type: application/json' -H 'Authorization: Simple '${ALGORITHMIA_API_KEY} https://api.algorithmia.com/v1/algo/${ALGORITHMIA_USERNAME}/mlflow_sklearn_demo/<version>
```

You can also use `mlflow deployments predict` to query the model, on this case it will always query the latest published version of the model.

```
echo '{"alcohol":{"0":12.8},"chlorides":{"0":0.029},"citric acid":{"0":0.48},"density":{"0":0.98},"fixed acidity":{"0":6.2},"free sulfur dioxide":{"0":29},"pH":{"0":3.33},"residual sugar":{"0":1.2},"sulphates":{"0":0.39},"total sulfur dioxide":{"0":75},"volatile acidity":{"0":0.66}}' > predict_input.json

mlflow deployments predict -t algorithmia --name mlflow_sklearn_demo -I predict_input.json
```

To update deployment for example after training a new model

```
mlflow deployments update -t algorithmia --name mlflow_sklearn_demo -m <path-to-new-model-dir>
```

To delete the deployment

```
mlflow deployments delete -t algorithmia --name mlflow_sklearn_demo
```
