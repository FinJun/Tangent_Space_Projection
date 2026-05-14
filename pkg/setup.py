#!/usr/bin/env python
# coding: utf-8

import setuptools

setuptools.setup(
    include_package_data=True,
    name="pyepo",
    packages=setuptools.find_packages(),
    description="PyTorch-based End-to-End Predict-then-Optimize Tool",
    version="0.2.4",
    url="https://github.com/khalil-research/PyEPO",
    install_requires=[
        "numpy",
        "scipy",
        "pathos",
        "tqdm",
        "Pyomo>=6.1.2",
        "gurobipy>=9.1.2",
        "scikit_learn",
        "torch>=1.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
