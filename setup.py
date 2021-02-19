#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mlift",
    description="Utilities for inference in generative models using Mici",
    author="Matt Graham",
    url="https://github.com/thiery-lab/manifold_lifting.git",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "mici>=0.1.10",
        "numpy>=1.18",
        "scipy>=1.5",
        "jax>=0.2.9",
        "jaxlib>=0.1.60",
    ],
)
