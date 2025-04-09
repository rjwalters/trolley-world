# setup.py
from setuptools import setup, find_packages

setup(
    name="trolleyworld",
    version="0.1.0",
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # List packages under src/
    install_requires=["matplotlib", "numpy", "stumpy"],
    extras_require={
        "dev": ["pytest", "black", "flake8", "setuptools", "pylance", "mypy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
)
