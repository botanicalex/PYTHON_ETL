from setuptools import setup, find_packages

setup(
    name="credit-risk-trainer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "google-cloud-bigquery>=3.11.0",
        "google-cloud-storage>=2.10.0",
        "db-dtypes>=1.1.0",
        "pyarrow>=12.0.0",
        "joblib>=1.2.0",
    ],
)
