# mlflow-algorithmia

[![PyPI](https://badge.fury.io/py/mlflow-algorithmia.svg)](https://pypi.org/project/mlflow-algorithmia/)
[![License](http://img.shields.io/:license-Apache%202-blue.svg)](https://github.com/algorithmiaio/mlflow-algorithmia/blob/master/LICENSE.txt)

Deploy MLflow models to Algorithmia

## Install

```
pip install mlflow-algorithmia
```

## Usage

This is based on the [mlflow tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
we reproduce some steps here but for more details please see the official [mlflow docs](https://www.mlflow.org/docs).

Create a model by running this script:

```
mlflow run examples/sklearn_elasticnet_wine/
```

This will create an `mlruns` directory that contains the trained model,
you can run the mlflow UI running `mlflow ui` or serve the model locally using the
mlflow server running:

```
mlflow models serve -m mlruns/0/<run-id>/artifacts/model -p 1234
```

And make a test query

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations

[5.120775719594933]
```

Now let's deploy the same model in Algorithmia, you will need:

1. An Algorithmia API key with `Read + Write` data access
2. The path to the model (under `mlruns`) you want to deploy, for example: `mlruns/0/<run-id>/artifacts/model`

Set your Algorithmia API key and username as environment variables:

```
export ALGORITHMIA_USERNAME=<username>
export ALGORITHMIA_API_KEY=<api-key>
```

Create the deployment:

```
mlflow deployments create -t algorithmia --name mlflow_sklearn_demo -m mlruns/0/<run-id>/artifacts/model
```

```
INFO: Creating Mlflow bundle
INFO: Uploading Mlflow bundle
INFO: MLflow bundle uploaded to: ...
INFO: Cloning algorithm source to: ./algorithmia_tmp/
INFO: Updating algorithm source and building model
INFO: Algorithm repo updated: Update - MLflow run_id: 6df340cd6d294fe59d1b4652fb25969a
INFO: New model version ready: c6b883b325ee0bb63d91dd0cadfe0baf6bd84fb3
```

Save the new model version from the output to query the model.

### Query the model in Algorithmia

You need the new model version from above and the `ALGORITHMIA_USERNAME` and `ALGORITHMIA_API_KEY` variables you used before.

Replace `<version>` with the model version from the previous command output.

```
curl -X POST -d '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' -H 'Content-Type: application/json' -H 'Authorization: Simple '${ALGORITHMIA_API_KEY} https://api.algorithmia.com/v1/algo/$ALGORITHMIA_USERNAME/mlflow_sklearn_demo/<version>
```

You can also use `mlflow deployments predict` command to query the model, on this case it will always query the **latest public published** version of the model, to query a specific version use the method described above.

First create a `predict_input.json` file:

```
echo '{"alcohol":{"0":12.8},"chlorides":{"0":0.029},"citric acid":{"0":0.48},"density":{"0":0.98},"fixed acidity":{"0":6.2},"free sulfur dioxide":{"0":29},"pH":{"0":3.33},"residual sugar":{"0":1.2},"sulphates":{"0":0.39},"total sulfur dioxide":{"0":75},"volatile acidity":{"0":0.66}}' > predict_input.json
```

Now query the latest public published version of the model:

```
mlflow deployments predict -t algorithmia --name mlflow_sklearn_demo -I predict_input.json
```

To update deployment, for example after training a new model:

```
mlflow deployments update -t algorithmia --name mlflow_sklearn_demo -m <path-to-new-model-dir>
```

To delete the deployment:

```
mlflow deployments delete -t algorithmia --name mlflow_sklearn_demo
```

## Algorithm settings

To control the different algorithm specific deployment options such as the
algorithmia environment using environment variables. For example:

```
export ALGO_PACKAGE_SET=python38
mlflow deployments create -t algorithmia --name mlflow_sklearn_demo -m mlruns/0/<run-id>/artifacts/model
```

Will create an algorithm with the Package Set Python 3.8 instead of the default of 3.7.

A complete list of variables and its defaults:

| Variable  | Default | Description |
| --- | --- | --- |
| `ALGO_LANGUAGE` | `python3` | Package set |
| `ALGO_ENV_ID` | `` | Defaults to the Python 3.8 environment ID found on the cluster |
| `ALGO_SRC_VISIBILITY` | `closed` | Algorithm source visibility `closed` or `open` |
| `ALGO_LICENSE` | `apl` | Algorithm license |
| `ALGO_NETWORK_ACCESS` | `full` | Network Access |
| `ALGO_PIPELINE` | `True` | Algorithm pipeline enabled or not |
| `ALGO_PACKAGE_SET` |  | Optional legacy environment package set name |
