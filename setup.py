# setup.py
from setuptools import setup, find_packages

setup(
    name="student_success",  # Replace with your project name
    version="0.1",
    packages=find_packages(where="Source Code"),
    package_dir={"": "Source Code"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost"
    ],
    python_requires=">=3.8",
)