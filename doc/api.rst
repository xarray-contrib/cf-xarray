API
===

.. currentmodule:: cf_xarray

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    bounds_to_vertices
    vertices_to_bounds


.. currentmodule:: xarray

DataArray
---------

.. _daattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.cf.axes
    DataArray.cf.cell_measures
    DataArray.cf.coordinates
    DataArray.cf.standard_names
    DataArray.cf.plot


.. _dameth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.cf.__getitem__
    DataArray.cf.__repr__
    DataArray.cf.guess_coord_axis
    DataArray.cf.keys
    DataArray.cf.rename_like

Dataset
-------

.. _dsattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.cf.axes
    Dataset.cf.cell_measures
    Dataset.cf.coordinates
    Dataset.cf.standard_names

.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.cf.__getitem__
    Dataset.cf.__repr__
    Dataset.cf.add_bounds
    Dataset.cf.bounds_to_vertices
    Dataset.cf.decode_vertical_coords
    Dataset.cf.get_bounds
    Dataset.cf.get_bounds_dim_name
    Dataset.cf.guess_coord_axis
    Dataset.cf.keys
    Dataset.cf.rename_like
