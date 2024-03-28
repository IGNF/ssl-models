from pathlib import Path

from setuptools import find_packages, setup  # type: ignore #

with open(str(Path(__file__).parent / "README.md"), "r") as fh:
    long_description = fh.read()
with open(str(Path(__file__).parent / "VERSION"), "r") as fh:
    version = fh.read().strip()
setup(
    name="ssl-models",
    version=version,
    description="SSL pretraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IGNF/ssl-models",
    project_urls={"Bug Tracker": "https://github.com/IGNF/ssl-models/issues"},
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=["odeon"],
    include_package_data=True,
    package_data={"*": ["*.yml"]},
)
