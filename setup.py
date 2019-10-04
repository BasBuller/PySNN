from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

requirements = [
    "torch>=1.2.0",
    "torchvision",
    "numpy",
    "pandas",
    "sklearn"
]

setup(
    name="pysnn", 
    version="0.1",
    description="Framework for engineering and simulating spiking neural networks, built on top of PyTorch.",
    long_description=readme,
    author="Bas Buller",
    author_email="bas.buller@gmail.com",
    packages=find_packages(exclude=("tests",)),
    # packages=find_packages(),
    install_requires=requirements
)