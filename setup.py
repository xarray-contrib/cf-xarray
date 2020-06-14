#!/usr/bin/env python
from setuptools import setup

setup(
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="A lightweight convenience wrapper for using CF attributes on xarray objects. ",
    url="https://cf-xarray.readthedocs.io",
)
