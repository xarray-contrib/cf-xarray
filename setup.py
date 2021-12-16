#!/usr/bin/env python
from setuptools import setup

setup(
    # The package metadata is specified in setup.cfg but GitHub's downstream dependency graph
    # does not work unless we put the name this here too.
    name="cf_xarray",
    use_scm_version={
        "write_to": "cf_xarray/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
