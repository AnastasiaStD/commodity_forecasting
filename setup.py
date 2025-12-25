#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commodity Forecasting - Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="commodity-forecasting",
    version="1.0.0",
    author="Commodity Forecasting Team",
    author_email="anstolyarowa@yandex.ru",
    description="Библиотека для прогнозирования цен на сырьевые товары",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnastasiaStD/commodity_forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "commodity-forecast=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)
