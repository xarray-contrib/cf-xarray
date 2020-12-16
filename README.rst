.. image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue&style=for-the-badge
    :target: https://binder.pangeo.io/v2/gh/xarray-contrib/cf-xarray/main?urlpath=lab
    :alt: Binder

.. image:: https://img.shields.io/readthedocs/cf-xarray/latest.svg?style=for-the-badge
    :target: https://cf-xarray.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/workflow/status/xarray-contrib/cf-xarray/CI?logo=github&style=for-the-badge
    :target: https://github.com/xarray-contrib/cf-xarray/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/github/workflow/status/xarray-contrib/cf-xarray/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/xarray-contrib/cf-xarray/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/xarray-contrib/cf-xarray.svg?style=for-the-badge
    :target: https://codecov.io/gh/xarray-contrib/cf-xarray
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/v/cf-xarray.svg?style=for-the-badge
    :target: https://pypi.org/project/cf-xarray
    :alt: Python Package Index

.. image:: https://img.shields.io/conda/vn/conda-forge/cf_xarray.svg?style=for-the-badge
    :target: https://anaconda.org/conda-forge/cf_xarray
    :alt: Conda Version


cf-xarray
=========

A lightweight convenience wrapper for using CF attributes on xarray objects. 

For example you can use ``.cf.mean("latitude")`` instead of ``.mean("lat")`` if appropriate attributes are set! This allows you to write code that does not require knowledge of specific dimension or coordinate names particular to a dataset.

See more in the introductory notebook `here <https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html>`_.
