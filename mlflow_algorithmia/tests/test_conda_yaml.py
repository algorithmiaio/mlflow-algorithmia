# Test for reading and parsing conda.yaml

from mlflow_algorithmia.conda_env import Environment


def test_conda_yaml_1():
    env = Environment.from_yamlstr(
        """
dependencies:
- python=3.6.12
- scikit-learn=0.23.2
- pip
- pip:
  - mlflow
  - cloudpickle==1.6.0
name: mlflow-env
"""
    )
    assert env.name == "mlflow-env"
    assert env.channels == []
    assert env.list_deps() == ["scikit-learn==0.23.2", "mlflow", "cloudpickle==1.6.0"]


def test_conda_yaml_2():
    env = Environment.from_yamlstr(
        """
channels:
- defaults
- conda-forge
dependencies:
- python=3.6.12
- scikit-learn=0.23.2
- tensorflow>=2.0
- pytorch>1.0,<2.0
- boto
- pip
- pip:
  - mlflow
  - dask>1.0
  - cloudpickle==1.6.0
name: mlflow-complex-env
"""
    )
    assert env.name == "mlflow-complex-env"
    assert env.channels == ["defaults", "conda-forge"]
    assert env.list_deps() == [
        "scikit-learn==0.23.2",
        "tensorflow>==2.0",
        "pytorch<2.0,>1.0",
        "boto",
        "mlflow",
        "dask>1.0",
        "cloudpickle==1.6.0",
    ]
