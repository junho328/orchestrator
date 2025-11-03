"""Setup script for orchestrator package."""
from setuptools import setup, find_packages

setup(
    name="orchestrator",
    version="0.1.0",
    description="Orchestrator training package",
    packages=["orchestrator", "orchestrator.data_utils", "orchestrator.trainers", "orchestrator.utils"],
    package_dir={
        "orchestrator": ".",
        "orchestrator.data_utils": "data_utils",
        "orchestrator.trainers": "trainers",
        "orchestrator.utils": "utils",
    },
    python_requires=">=3.8",
)

