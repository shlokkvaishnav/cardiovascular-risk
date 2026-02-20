from setuptools import setup, find_packages
from pathlib import Path

def load_requirements() -> list:
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return []
    return [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="cardiovascular_risk",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=load_requirements(),
    extras_require={
        "dev": ["black", "flake8", "mypy", "pre-commit"],
        "viz": ["matplotlib", "seaborn", "plotly"],
    },
)
