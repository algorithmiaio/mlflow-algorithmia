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
2. The path to the model you want to deploy (under `mlruns`)

```
# Set your Algorithmia API key
export ALGORITHMIA_USERNAME=<username>
export ALGORITHMIA_API_KEY=<api-key>

# Create a deployment
mlflow deployments create -t algorithmia --name mlflow_sklearn_demo -m <path-to-model>

# Test query

curl -X POST -d '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' -H 'Content-Type: text/plain' -H 'Authorization: Simple <api-key>' https://api.test.algorithmia.com/v1/algo/danielfrg/mlflow_sklearn_demo5?timeout=300
```

To update deployment for example after training a new model

```
mlflow deployments update -t algorithmia --name mlflow_sklearn_demo -m <path-to-new-model>
```
