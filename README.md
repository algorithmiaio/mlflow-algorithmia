# mlflow-algorithmia

[![PyPI](https://badge.fury.io/py/mlflow-algorithmia.svg)](https://pypi.org/project/mlflow-algorithmia/)
[![Testing](https://github.com/algorithmia/mlflow-algorithmia/workflows/test/badge.svg)](https://github.com/algorithmia/mlflow-algorithmia/actions)
[![License](http://img.shields.io/:license-Apache%202-blue.svg)](https://github.com/algorithmia/mlflow-algorithmia/blob/master/LICENSE.txt)

Deploy MLflow models to Algorithmia

## Install

```
pip install git+git://github.com/algorithmia/mlflor-algorithmia.git
```

## Example usage from sklearn

```
# Create model
python examples/sklearn_elasticnet_wine/train.py

# This will create a mlruns dir
# <path-to-models>: mlruns/0/b3c1d94f189945779174c0d8304660d3/artifacts/model

# Create a deployment
mlflow deployments create --name <name> -t algorithmia -m <path-to-model>

# Update deployment (after running the model again)
mlflow deployments update --name <name> -t algorithmia -m <path-to-model>
```
