from setuptools import setup, find_packages

setup(
    name="citation_index",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic",
    ],
    python_requires=">=3.8",
) 