import pytest
from mlflow.exceptions import MlflowException

from mlflow_algorithmia.deployment import Settings


@pytest.fixture
def mock_env_set(monkeypatch):
    monkeypatch.setenv("ALGORITHMIA_API_KEY", "algo-key")
    monkeypatch.setenv("ALGORITHMIA_USERNAME", "test_user")


@pytest.fixture
def mock_env_missing(monkeypatch):
    monkeypatch.delenv("ALGORITHMIA_API_KEY", raising=False)
    monkeypatch.delenv("ALGORITHMIA_USERNAME", raising=False)


def test_missing_api_key(mock_env_missing):
    with pytest.raises(MlflowException):
        s = Settings()


def test_missing_username(mock_env_set, monkeypatch):
    monkeypatch.delenv("ALGORITHMIA_USERNAME", raising=False)
    with pytest.raises(MlflowException):
        s = Settings()


def test_git_endpoint(mock_env_set):
    s = Settings()
    assert s["git_endpoint"] == "git.algorithmia.com"


def test_git_endpoint_test(mock_env_set, monkeypatch):
    monkeypatch.setenv("ALGORITHMIA_API", "https://api.test.algorithmia.com")
    s = Settings()
    assert s["git_endpoint"] == "git.test.algorithmia.com"


def test_git_endpoint_other(mock_env_set, monkeypatch):
    monkeypatch.setenv("ALGORITHMIA_API", "https://api.algorithmia.mycorp.net")
    s = Settings()
    assert s["git_endpoint"] == "git.algorithmia.mycorp.net"
