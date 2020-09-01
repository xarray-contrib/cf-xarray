.. currentmodule:: cf_xarray

.. ipython:: python
    :suppress:

    import cf_xarray.accessor


.. _contribut:

Contributing
------------

This section will be expanded later. For now it lists docstrings for a number of internal variables, classes and functions.

Variables
~~~~~~~~~

.. autodata:: cf_xarray.accessor._AXIS_NAMES
.. autodata:: cf_xarray.accessor._CELL_MEASURES
.. autodata:: cf_xarray.accessor._COORD_NAMES
.. autodata:: cf_xarray.accessor._WRAPPED_CLASSES


Attribute parsing
+++++++++++++++++

This dictionary contains criteria for identifying axis and coords using CF attributes. It was copied from MetPy

.. autosummary::
   :toctree: generated/

   ~accessor.coordinate_criteria

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_class.rst


    ~accessor.CFAccessor
    ~accessor._CFWrappedClass
    ~accessor._CFWrappedPlotMethods

Functions
~~~~~~~~~


Primarily for developer reference. Some of these could become public API if necessary.

.. autosummary::
   :toctree: generated/

    ~accessor._getattr
    ~accessor._get_axis_coord
    ~accessor._get_axis_coord_single
    ~accessor._get_measure
    ~accessor._get_measure_variable
    accessor._CFWrappedPlotMethods._plot_decorator
