.. cf_xarray documentation master file, created by
   sphinx-quickstart on Mon Jun  1 06:30:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/full-logo.png
    :align: center
    :width: 80%
   
|
|

Welcome to cf_xarray's documentation!
=====================================


``cf_xarray`` is a lightweight accessor that allows you to interpret CF attributes present
on xarray objects.


Installing
----------

``cf_xarray`` can be installed using ``pip``

    >>> pip install cf_xarray


or using ``conda``

    >>> conda install -c conda-forge cf_xarray


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: In-depth Examples

   examples/introduction


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   quickstart
   howtouse
   faq
   coord_axes
   selecting
   flags
   units
   parametricz
   bounds
   geometry
   plotting
   custom-criteria
   provenance
   API Reference <api>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For contributors

   Contributing Guide <contributing>
   Development Roadmap <roadmap>
   Whats New <whats-new>
   GitHub repository <https://github.com/xarray-contrib/cf-xarray>
