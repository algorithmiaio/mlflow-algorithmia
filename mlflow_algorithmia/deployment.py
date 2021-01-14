import json
import logging
import os
import shutil
import sys
import tarfile
import urllib
from urllib.parse import urlparse

import Algorithmia
import requests
import ruamel.yaml as yaml
from Algorithmia.errors import raiseAlgoApiError
from algorithmia_api_client.rest import ApiException
from git import Git, Repo, remote
from jinja2 import Environment, FileSystemLoader
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException

from mlflow_algorithmia.conda_env import Environment as CondaEnvironment


logger = logging.getLogger(__name__)
CURDIR = os.path.dirname(os.path.realpath(__file__))


class AlgorithmiaDeploymentClient(BaseDeploymentClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo = None
        self.algo = None
        self.run_id = None
        self.mlmodel = None
        self.settings = Settings()
        self.client = Algorithmia.client(self.settings["api_key"])

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        """
        Creates a deployment of the MLflow model in Algorithmia
        1. Creates and uploads a model bundle
        2. Creates the source code
        """
        self.settings.update(config)

        if config and config.get("raiseError") == "True":
            raise RuntimeError("Error requested")

        self.create_algorithm(name)
        self.update_deployment(
            name=name, model_uri=model_uri, flavor=flavor, config=config
        )
        return {"name": name, "flavor": "Algorithmia"}

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        """
        Updates a deployment in Algorithmia
        1. Creates and uploads a new model bundle
        2. Updates the source code
        """
        self.read_model_metadata(model_uri)

        os.makedirs(self.settings["tmp_dir"], exist_ok=True)
        tar_file = self.create_bundle(model_uri)
        algo_tar_file = self.upload_bundle(name, tar_file)

        repo_path = self.repo_clone_or_pull(name)
        config = {
            "mlflow_bundle_file": algo_tar_file,
            "dependencies": self.get_requirements(model_uri),
        }
        self.update_source(name, repo_path, **config)

        self.repo_commit_and_push()

        version = self.get_builds(name)[0]["commit_sha"]
        logger.info("New model version ready: %s", version)
        return {"name": name, "flavor": "Algorithmia"}

    def list_deployments(self):
        return "To see Algorithmia deployments go to the Algorithmia homepage"

    def delete_deployment(self, name):
        """
        Deletes a deployment in algorithmia and removes local
        temp directory
        """
        if os.path.exists(self.settings["tmp_dir"]):
            shutil.rmtree(self.settings["tmp_dir"])
        self.delete_algorithm(name)

    def get_deployment(self, name):
        username = self.settings["username"]
        algo_namespace = f"{username}/{name}"
        algo = self.client.algo(algo_namespace)

        return {
            "name": algo.algoname,
            "username": algo.username,
            "url": algo.url,
        }

    def predict(self, deployment_name, df):
        username = self.settings["username"]
        algo_namespace = f"{username}/{deployment_name}"
        algo = self.client.algo(algo_namespace)

        query = df.to_json(orient="split")
        return algo.pipe(query).result

    # Util functions

    def create_algorithm(self, name):
        """
        Create an algorithm in Algorithmia
        """
        logger.info("Creating Algorithm %s", name)

        details = {
            "label": name,
            "summary": self.settings["summary"],
            "tagline": self.settings["tagline"],
        }
        settings = {
            "package_set": "python37",
            "source_visibility": "closed",
            "license": "apl",
            "network_access": "full",
            "pipeline_enabled": True,
        }

        self.algo_(name).create(details=details, settings=settings)
        logger.info("Algorithm %s created", name)

    def algo_(self, name):
        """
        Internal representation for the algorithmia algorithm
        """
        username = self.settings["username"]
        algo_namespace = f"{username}/{name}"
        return self.client.algo(algo_namespace)

    def get_builds(self, name):
        api = self.settings["api_endpoint"]
        username = self.settings["username"]
        url = f"{api}/v1/algorithms/{username}/{name}/builds"

        api_key = self.settings["api_key"]
        headers = {"Authorization": f"Simple {api_key}"}
        r = requests.get(url, headers=headers)
        return r.json()["results"]

    def delete_algorithm(self, name):
        """Deletes an algorithm in Algorithmia"""
        logger.info("Deleting %s deployment in Algorithmia", name)
        try:
            api_response = self.client.manageApi.delete_algorithm(
                self.settings["username"], name
            )
            return api_response
        except ApiException as ex:
            error_message = json.loads(ex.body)
            raiseAlgoApiError(error_message)
        logger.info("Algorithm %s deleted", name)

    def read_model_metadata(self, model_uri):
        """
        Read MLmodel file
        """
        mlmodel_path = os.path.join(model_uri, "MLmodel")
        with open(mlmodel_path, "r") as file:
            mlmodel = yaml.load(file, Loader=yaml.Loader)

        self.mlmodel = mlmodel
        self.run_id = self.mlmodel["run_id"]

    def create_bundle(self, model_uri):
        """
        Creates a .tar.gz bundle from the MLflow model
        """
        logger.info("Creating Mlflow bundle")
        tar_fname = f"model-{self.run_id}.tar.gz"
        tar_fpath = os.path.join(self.settings["tmp_dir"], tar_fname)
        with tarfile.open(tar_fpath, "w:gz") as tar:
            source_dir = model_uri
            tar.add(source_dir, arcname=tar_fname[: -len(".tar.gz")])

        return tar_fpath

    def upload_bundle(self, name, tar_fpath):
        """
        Upload the MLflow bundle to algorithmia
        """
        logger.info("Uploading Mlflow bundle")
        username = self.settings["username"]
        algo_data_dir = f"data://{username}/{name}"

        if not self.client.dir(algo_data_dir).exists():
            self.client.dir(algo_data_dir).create()

        tar_fname = os.path.basename(tar_fpath)
        algo_file = os.path.join(algo_data_dir, tar_fname)
        self.client.file(algo_file).putFile(tar_fpath)
        logger.info("MLflow bundle uploaded to: %s", algo_file)
        return algo_file

    def repo_clone_or_pull(self, name):
        """
        Clones a repository from Algorithmia to the temp directory
        If repo it's already cloned then pulls any changes
        """
        target_dir = self.settings["tmp_dir"]
        logger.info("Cloning algorithm source to: %s", target_dir)
        os.makedirs(target_dir, exist_ok=True)
        repo_path = os.path.join(target_dir, name)

        if not os.path.exists(repo_path):
            username = self.settings["username"]
            encoded_api_key = self.settings["encoded_api_key"]
            git_endpoint = self.settings["git_endpoint"]
            username = self.settings["username"]
            algo_name = name
            algo_repo = f"https://{username}:{encoded_api_key}@{git_endpoint}/git/{username}/{algo_name}.git"
            self.repo = Repo.clone_from(algo_repo, repo_path)
            # self.repo = Repo.clone_from(algo_repo, repo_path, progress=Progress())
        else:
            self.repo = Repo(repo_path)
            origin = self.repo.remote(name="origin")
            origin.pull()

        return repo_path

    def repo_commit_and_push(self):
        logger.info("Updating algorithm source and building model")
        commit_msg = f"Update - MLflow run_id: {self.run_id}"
        self.repo.git.add(".")
        self.repo.index.commit(commit_msg)
        origin = self.repo.remote(name="origin")
        origin.push()
        logger.info("Algorithm repo updated: %s", commit_msg)

    def update_source(self, name, repo_path, **kwargs):
        _ = os.path.join(repo_path, ".gitignore")
        self.render_file("gitignore_repo", _, **kwargs)

        _ = os.path.join(repo_path, "requirements.txt")
        self.render_file("requirements.txt", _, **kwargs)

        _ = os.path.join(repo_path, "src", f"{name}.py")
        self.render_file("entrypoint.py", _, **kwargs)

        _ = os.path.join(repo_path, "src", "algorithmia_utils.py")
        self.render_file("algorithmia_utils.py", _, **kwargs)

        _ = os.path.join(repo_path, "src", "mlflow_wrapper.py")
        self.render_file("mlflow_wrapper.py", _, **kwargs)

        _ = os.path.join(repo_path, "models")
        os.makedirs(_, exist_ok=True)

        _ = os.path.join(repo_path, "models", ".gitignore")
        self.render_file("gitignore_all", _, **kwargs)

    def render_file(self, inp, out, **kwargs):
        file_loader = FileSystemLoader(os.path.join(CURDIR, "templates"))
        env = Environment(loader=file_loader)
        template = env.get_template(inp)
        output = template.render(**kwargs)

        with open(out, "w") as file:
            file.write(output)

    def get_requirements(self, model_uri):
        """
        Return a list of requirements based on the MLflow conda.yaml
        """
        conda_yaml_path = os.path.join(model_uri, "conda.yaml")
        environemnt = CondaEnvironment.from_file(conda_yaml_path)
        return environemnt.list_deps()


class Settings(dict):
    def __init__(self):
        super().__init__()
        self["api_key"] = os.environ.get("ALGORITHMIA_API_KEY", None)
        self["username"] = os.environ.get("ALGORITHMIA_USERNAME", None)

        if self["api_key"] is None:
            raise MlflowException(
                "Environment variable ALGORITHMIA_API_KEY is not set."
            )

        if self["username"] is None:
            raise MlflowException(
                "Environment variable ALGORITHMIA_USERNAME is not set."
            )

        default_api = "https://api.algorithmia.com"
        self["api_endpoint"] = os.environ.get("ALGORITHMIA_API", default_api)
        # Encoding the API key, so we can use it in the git URL
        self["encoded_api_key"] = urllib.parse.quote_plus(self["api_key"])
        self["tagline"] = os.environ.get("ALGORITHM_TAGLINE", "Mlflow deployment")
        self["summary"] = os.environ.get("ALGORITHM_SUMMARY", "Mlflow deployment")

        # Make the git_endpoint from the Algorithmia API
        url = urlparse(self["api_endpoint"]).netloc
        url = url[4:] if url.startswith("www.") else url
        url = url[4:] if url.startswith("api.") else url
        default_git_endpoint = f"git.{url}"
        self["git_endpoint"] = os.environ.get(
            "ALGORITHMIA_GIT_ENDPOINT", default_git_endpoint
        )

        default_tmp_dir = "./algorithmia_tmp/"
        self["tmp_dir"] = os.environ.get("MLFLOW_ALGO_TMP_DIR", default_tmp_dir)


class Progress(remote.RemoteProgress):
    def line_dropped(self, line):
        print(line)

    def update(self, op_code, cur_count, max_count=None, message=""):
        print(self._cur_line)


def run_local(name, model_uri, flavor=None, config=None):
    logger.info("Use `mlflow models serve` to run this model locally")


def target_help():
    return "Deploy MLflow models to Algorithmia"
