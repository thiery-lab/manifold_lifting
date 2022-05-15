#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mlift",
    version="0.1.0",
    description="Utilities for inference in generative models using Mici",
    author="Matt Graham",
    url="https://github.com/thiery-lab/manifold_lifting.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "mici>=0.1.10",
        "numpy>=1.20",
        "scipy>=1.5",
        "jax>=0.2.11",
        "jaxlib>=0.1.65",
        "matplotlib>=3.1",
        "sympy>=1.7",
        "symnum>=0.1.2",
        "arviz>=0.11",
        "multiprocess>=0.70"
    ],
)
