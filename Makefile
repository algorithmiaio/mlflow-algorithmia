SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

PWD := $(shell pwd)
TEST_FILTER ?= ""


first: help

clean:  ## Clean build files
	@rm -rf build dist site htmlcov .pytest_cache .eggs
	@rm -f .coverage coverage.xml mlflow_algorithmia/_generated_version.py
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .ipynb_checkpoints -exec rm -rf {} +


cleanall: clean   ## Clean everything
	@rm -rf *.egg-info


help:  ## Show this help menu
	@grep -E '^[0-9a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?##"; OFS="\t\t"}; {printf "\033[36m%-30s\033[0m %s\n", $$1, ($$2==""?"":$$2)}'


# ------------------------------------------------------------------------------
# Package build, test and docs

env:
	mamba env create


develop:  ## Install package for development
	python -m pip install --no-build-isolation -e .


build: package  ## Build everything


package:  ## Build Python package (sdist)
	python setup.py sdist


check:  ## Check linting
	@flake8
	@isort --check-only --diff --recursive --project mlflow_algorithmia --section-default THIRDPARTY .
	@black --check .


fmt:  ## Format source
	@isort --recursive --project mlflow_algorithmia --section-default THIRDPARTY .
	@black .


upload-pypi:  ## Upload package to PyPI
	twine upload dist/*.tar.gz


upload-test:  ## Upload package to test PyPI
	twine upload --repository test dist/*.tar.gz


test:  ## Run tests
	pytest -k $(TEST_FILTER)


test-report:  ## Generate coverage reports
	@coverage xml
	@coverage html


docs:  ## Build mkdocs
	mkdocs build --config-file $(CURDIR)/mkdocs.yml


serve-docs:  ## Serve docs
	mkdocs serve

# ------------------------------------------------------------------------------
# Project specific
