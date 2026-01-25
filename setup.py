from setuptools import setup, find_packages

setup(
    name="cardiovascular_risk",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "mlflow>=2.5.0",
        "pyyaml>=6.0",
        "pytest>=7.4.0",
        "joblib>=1.2.0"
    ],
    extras_require={
        "dev": ["black", "flake8", "mypy", "pre-commit"],
        "viz": ["matplotlib", "seaborn", "plotly"],
    },
)
