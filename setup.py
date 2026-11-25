from setuptools import setup, find_packages

setup(
    name="citation_index",
    version="1.0.0-beta.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic",
        "requests",
        "lxml",
    ],
    python_requires=">=3.11",
)
