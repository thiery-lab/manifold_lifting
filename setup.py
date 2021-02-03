#!/usr/bin/env python3

from setuptools import setup

setup(
    name="mlift",
    description="Utilities for inference in generative models using Mici",
    author="Matt Graham",
    url="https://github.com/thiery-lab/manifold_lifting.git",
    packages=["mlift"],
    python_requires=">=3.6",
    install_requires=[
        "mici>=0.1.10",
        "numpy>=1.17",
        "scipy>=1.1",
        "jax>=0.2.8",
        "jaxlib>=0.1.59",
    ],
)
