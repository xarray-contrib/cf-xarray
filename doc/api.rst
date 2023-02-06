API
===

.. currentmodule:: cf_xarray

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    bounds_to_vertices
    vertices_to_bounds
    shapely_to_cf
    cf_to_shapely
    set_options
    encode_multi_index_as_compress
    decode_compress_to_multi_index

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
    DataArray.cf.cf_roles
    DataArray.cf.coordinates
    DataArray.cf.formula_terms
    DataArray.cf.grid_mapping_name
    DataArray.cf.is_flag_variable
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
    DataArray.cf.add_canonical_attributes
    DataArray.cf.differentiate
    DataArray.cf.guess_coord_axis
    DataArray.cf.keys
    DataArray.cf.rename_like

Flag Variables
++++++++++++++

cf_xarray supports rich comparisons for `CF flag variables`_. Flag masks are not yet supported.

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.cf.__lt__
    DataArray.cf.__le__
    DataArray.cf.__eq__
    DataArray.cf.__ne__
    DataArray.cf.__ge__
    DataArray.cf.__gt__
    DataArray.cf.isin


Dataset
-------

.. _dsattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.cf.axes
    Dataset.cf.bounds
    Dataset.cf.cell_measures
    Dataset.cf.cf_roles
    Dataset.cf.coordinates
    Dataset.cf.formula_terms
    Dataset.cf.grid_mapping_names
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
    Dataset.cf.add_canonical_attributes
    Dataset.cf.bounds_to_vertices
    Dataset.cf.decode_vertical_coords
    Dataset.cf.differentiate
    Dataset.cf.get_bounds
    Dataset.cf.get_bounds_dim_name
    Dataset.cf.guess_coord_axis
    Dataset.cf.keys
    Dataset.cf.rename_like


.. _`CF flag variables`: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
