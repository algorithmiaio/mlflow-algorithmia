import os

from setuptools import find_packages, setup


setup_dir = os.path.abspath(os.path.dirname(__file__))


def read_file(filename):
    filepath = os.path.join(setup_dir, filename)
    with open(filepath) as file:
        return file.read()


setup(
    name="mlflow-algorithmia",
    use_scm_version=True,
    packages=find_packages(),
    # package_dir={"": "src"},
    zip_safe=False,
    include_package_data=True,
    # package_data={"mlflow_algorithmia": ["includes/*"]},
    # data_files=data_files,
    # cmdclass={"install": InstallCmd},
    entry_points={
        "mlflow.deployments": "algorithmia=mlflow_algorithmia.deployment",
    },
    options={"bdist_wheel": {"universal": "1"}},
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    install_requires=read_file("requirements-package.txt").splitlines(),
    extras_require={
        "test": ["pytest"],
        "dev": read_file("requirements.txt").splitlines(),
    },
    description="",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    maintainer="Algorithmia",
    maintainer_email="drodriguez@algorithmia.io",
    url="https://github.com/algorithmia/mlflow-algorithmia",
    keywords=[],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
