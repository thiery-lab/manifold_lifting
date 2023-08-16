#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mlift",
    version="0.1.1",
    description="Utilities for inference in generative models using Mici",
    author="Matt Graham",
    url="https://github.com/thiery-lab/manifold_lifting.git",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mici>=0.2.1",
        "numpy>=1.22",
        "scipy>=1.5",
        "jax>=0.4",
        "jaxlib>=0.4",
        "matplotlib>=3.1",
        "sympy>=1.7",
        "symnum>=0.2.0",
        "arviz>=0.16",
        "multiprocess>=0.70"
    ],
)
