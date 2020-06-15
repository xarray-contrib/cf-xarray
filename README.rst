.. image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue&style=for-the-badge
    :target: https://binder.pangeo.io/v2/gh/xarray-contrib/cf-xarray/master?urlpath=lab
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

.. If you want the following badges to be visible, please remove this line, and unindent the lines below

	.. image:: https://img.shields.io/conda/vn/conda-forge/cf-xarray.svg?style=for-the-badge
        :target: https://anaconda.org/conda-forge/cf-xarray
        :alt: Conda Version


cf-xarray
=========

A lightweight convenience wrapper for using CF attributes on xarray objects. Right now all of this works:

.. code-block:: python

    import cf_xarray
    import xarray as xr

    ds = xr.tutorial.load_dataset("air_temperature").isel(time=slice(4))

    ds.air.cf.var("X")

    ds.air.cf.resample(T="M").var()

    ds.air.cf.groupby("T").var("Y")

    (
    	ds.air
    	.cf.isel(T=slice(4))
    	.cf.plot.contourf(x="Y", y="X", col="T", col_wrap=4)
    )

    ds.air.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")

    ds.air.attrs["cell_measures"] = "area: cell_area"
    ds.coords["cell_area"] = (
	np.cos(ds.air.cf["latitude"] * np.pi / 180)
        * xr.ones_like(ds.air.cf["longitude"])
        * 105e3
        * 110e3
    )
    ds.air.cf.weighted("area").sum("latitude")
