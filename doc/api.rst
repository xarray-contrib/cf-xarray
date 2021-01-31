.. currentmodule:: xarray

API
===

DataArray
---------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.cf.axes
    DataArray.cf.cell_measures
    DataArray.cf.coordinates
    DataArray.cf.standard_names
    DataArray.cf.plot

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.cf.__getitem__
    DataArray.cf.describe
    DataArray.cf.guess_coord_axis
    DataArray.cf.keys
    DataArray.cf.rename_like

Dataset
-------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.cf.axes
    Dataset.cf.cell_measures
    Dataset.cf.coordinates
    Dataset.cf.standard_names

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.cf.__getitem__
    Dataset.cf.add_bounds
    Dataset.cf.bounds_to_vertices
    Dataset.cf.decode_vertical_coords
    Dataset.cf.describe
    Dataset.cf.get_bounds
    Dataset.cf.guess_coord_axis
    Dataset.cf.keys
    Dataset.cf.rename_like

.. currentmodule:: cf_xarray

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    bounds_to_vertices
    vertices_to_bounds
