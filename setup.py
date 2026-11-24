from setuptools import setup, find_packages

setup(
    name="citation_index",
    version="0.5.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic",
        "requests",
        "lxml",
    ],
    python_requires=">=3.11",
)
