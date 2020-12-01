import os
import sys
import yaml
import urllib
import logging
import tarfile
from urllib.parse import urlparse

import Algorithmia
from git import Git, Repo, remote
from jinja2 import Environment, FileSystemLoader
from mlflow.exceptions import MlflowException
from mlflow.deployments import BaseDeploymentClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s — %(name)s.%(funcName)s:%(lineno)d — %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

DEPLOYMENT_NAME = "algorithmia"
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TMP_DIR = "./algorithmia_tmp/"


class AlgorithmiaDeploymentClient(BaseDeploymentClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = None
        self.mlmodel = None
        self.settings = Settings()

        if self.settings["api_key"] is None:
            raise MlflowException(
                "Environment variable: ALGORITHMIA_API_KEY is not set."
            )

        if self.settings["username"] is None:
            raise MlflowException(
                "Environment variable: ALGORITHMIA_USERNAME is not set."
            )

        self.client = Algorithmia.client(self.settings["api_key"])

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        """
        Create the project on Algorithmia
        """
        logger.info("Creating Algorithm %s", name)
        self.settings.update(config)

        username = self.settings["username"]
        algo_namespace = f"{username}/{name}"
        print(algo_namespace)

        if config and config.get("raiseError") == "True":
            raise RuntimeError("Error requested")

        details = {
            "label": name,
            "summary": self.settings["summary"],
            "tagline": self.settings["tagline"],
        }
        settings = {
            # "environment": "cpu",
            # "language": "python3-1",
            "package_set": "python37",
            "source_visibility": "closed",
            "license": "apl",
            "network_access": "full",
            "pipeline_enabled": True,
        }

        self.client.algo(algo_namespace).create(details=details, settings=settings)

        self.update_deployment(
            name=name, model_uri=model_uri, flavor=flavor, config=config
        )
        return {"name": DEPLOYMENT_NAME, "flavor": flavor}

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        """
        1. Compress and upload MLflow model
        2. Update the code to use the new model
        """
        self.read_mlmodel(model_uri)

        os.makedirs(TMP_DIR, exist_ok=True)
        tar_file = self.compress_mlflow(model_uri)
        algo_tar_file = self.upload_model(name, tar_file)

        repo_path = self.clone_algorithm_repo(name)
        self.update_source(name, repo_path, mlflow_bundle=algo_tar_file)
        self.update_repo()
        return {"flavor": flavor}

    def delete_deployment(self, name):
        return None

    def list_deployments(self):
        if os.environ.get("raiseError") == "True":
            raise RuntimeError("Error requested")
        return [DEPLOYMENT_NAME]

    def get_deployment(self, name):
        return {"key1": "val1", "key2": "val2"}

    def predict(self, name, df):
        return 1

    def clone_algorithm_repo(self, name):
        """
        Clone a repository from Algorithmia if it doesnt exist
        If its already cloned then pull
        """
        os.makedirs(TMP_DIR, exist_ok=True)
        repo_path = os.path.join(TMP_DIR, name)

        if not os.path.exists(repo_path):
            # Encoding the API key, so we can use it in the git URL

            username = self.settings["username"]
            encoded_api_key = urllib.parse.quote_plus(self.settings["api_key"])
            git_endpoint = self.settings["git_endpoint"]
            username = self.settings["username"]
            algo_name = name
            algo_repo = f"https://{username}:{encoded_api_key}@{git_endpoint}/git/{username}/{algo_name}.git"
            self.repo = Repo.clone_from(algo_repo, repo_path, progress=Progress())
        else:
            self.repo = Repo(repo_path)
            origin = self.repo.remote(name="origin")
            origin.pull()

        return repo_path

    def update_repo(self):
        print("Updating Algorithmia repo and building model")
        commit_msg = f"Update - MLflow run_id: {self.run_id}"
        self.repo.git.add(".")
        self.repo.index.commit(commit_msg)
        origin = self.repo.remote(name="origin")
        origin.push()
        print(f"Updated Algorithmia repo: {commit_msg}")

    def read_mlmodel(self, model_uri):
        """
        Read MLmodel file and return it as an object
        """
        mlmodel_path = os.path.join(model_uri, "MLmodel")
        with open(mlmodel_path, "r") as file:
            mlmodel = yaml.load(file, Loader=yaml.FullLoader)

        self.mlmodel = mlmodel
        self.run_id = self.mlmodel["run_id"]

    def compress_mlflow(self, model_uri):
        """
        Creates a .tar.gz file from the MLflow model
        """
        tar_fname = f"model-{self.run_id}.tar.gz"
        tar_fpath = os.path.join(TMP_DIR, tar_fname)
        with tarfile.open(tar_fpath, "w:gz") as tar:
            source_dir = model_uri
            tar.add(source_dir, arcname=tar_fname[: -len(".tar.gz")])

        return tar_fpath

    def upload_model(self, name, tar_fpath):
        """
        Upload compress MLflow model to algorithmia
        """
        username = self.settings["username"]
        algo_data_dir = f"data://{username}/{name}"

        if not self.client.dir(algo_data_dir).exists():
            self.client.dir(algo_data_dir).create()

        tar_fname = os.path.basename(tar_fpath)
        algo_file = os.path.join(algo_data_dir, tar_fname)
        self.client.file(algo_file).putFile(tar_fpath)
        print(f"Uploaded MLflow bundle to: {algo_file}")
        return algo_file

    def update_source(self, name, repo_path, **kwargs):
        _ = os.path.join(repo_path, ".gitignore")
        self.render_file("gitignore_repo", _, **kwargs)

        _ = os.path.join(repo_path, "requirements.txt")
        self.render_file("requirements.txt", _, **kwargs)

        _ = os.path.join(repo_path, "src", f"{name}.py")
        self.render_file("main.py", _, **kwargs)

        _ = os.path.join(repo_path, "src", "algo_utils.py")
        self.render_file("algo_utils.py", _, **kwargs)

        _ = os.path.join(repo_path, "src", "mlflow_wrapper.py")
        self.render_file("mlflow_wrapper.py", _, **kwargs)

        _ = os.path.join(repo_path, "models")
        os.makedirs(_, exist_ok=True)

        _ = os.path.join(repo_path, "models", ".gitignore")
        self.render_file("gitignore_all", _, **kwargs)

    def render_file(self, inp, out, **kwargs):
        file_loader = FileSystemLoader(os.path.join(THIS_DIR, "templates"))
        env = Environment(loader=file_loader)
        template = env.get_template(inp)
        output = template.render(**kwargs)

        with open(out, "w") as f:
            f.write(output)


class Settings(dict):
    def __init__(self):
        super().__init__()
        self["api_endpoint"] = os.environ.get(
            "ALGORITHMIA_API", "https://api.algorithmia.com"
        )
        self["api_key"] = os.environ.get("ALGORITHMIA_API_KEY", None)
        self["username"] = os.environ.get("ALGORITHMIA_USERNAME", None)
        self["tagline"] = os.environ.get("ALGORITHM_TAGLINE", "Mlflow deployment")
        self["summary"] = os.environ.get("ALGORITHN_SUMMARY", "MLflow deployment")

        url = urlparse(self["api_endpoint"]).netloc
        git_endpoint = url[4:] if url.startswith("www.") else url
        git_endpoint = git_endpoint.replace("api.", "git.")
        self["git_endpoint"] = git_endpoint


class Progress(remote.RemoteProgress):
    def line_dropped(self, line):
        print(line)

    def update(self, *args):
        print(self._cur_line)


def run_local(name, model_uri, flavor=None, config=None):
    logger.info(
        "Deployed locally at the key {} using the model from {}. ".format(
            name, model_uri
        )
        + "It's flavor is {} and config is {}".format(flavor, config)
    )


def target_help():
    return "Deploy MLflow models to Algorithmia"
