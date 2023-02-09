.. currentmodule:: xarray

What's New
----------

v0.8.0 (Feb 8, 2023)
====================

- Support interpreting `SGRID Conventions <https://sgrid.github.io/sgrid/>`_ to identify
  X, Y, Z axes. By `Deepak Cherian`_.
- Add a `rich <https://rich.readthedocs.io>`_ repr. (:pr:`409`).
  Use ``rich.print(ds.cf)`` or ``%load_ext rich`` in a Jupyter session to
  view a much richer representation of the ``.cf`` accessor. By `Deepak Cherian`_.
- Support interpreting the ``grid_mapping`` attribute: (:pr:`391`).
  See :py:meth:`Dataset.cf.grid_mapping_names` and ``Dataset.cf["grid_mapping"]``,
  ``DataArray.cf["grid_mapping"]``. By `Lars Buntemeyer`_ and `Deepak Cherian`_.


v0.7.9 (Jan 31, 2023)
=====================

- Fix packaging of v0.7.8. That release was yanked off PyPI.

v0.7.8 (Jan 31, 2023)
=====================

- Optionally use the `regex` package to continue supporting global flags in regular expressions that are not at start of pattern (:pr:`408`). By `Kristen Thyng`_
- Added link to docs for a new example "COSIMA ocean-sea ice model demo" (:pr:`397`). By `Aidan Heerdegen`_

v0.7.7 (Jan 14, 2023)
=====================

- Fix to ``geometry.points_to_cf`` to support shapely 2.0. (:pr:`386`).
  By `Pascal Bourgault`_

v0.7.6 (Dec 07, 2022)
=====================

- Fix to ``cf.add_bounds`` to support all types of curved grids (:pr:`376`).
  By `Pascal Bourgault`_
- Allow custom criteria to match the variable name of DataArray objects (:pr:`379`). By `Mathias Hauser`_
- Support new Xarray indexes API when creating MultiIndex for "compression by gathering" datasets.
  (:pr:`381`). By `Deepak Cherian`_

v0.7.5 (Nov 15, 2022)
=====================

- ``cf.add_bounds`` can estimate 2D bounds using an approximate linear interpolation (:pr:`370`).
  By `Pascal Bourgault`_.
- Improve detection of bounds order by rellaxing check a bit (:pr:`361`).
  By `Lars Buntemeyer`_.
- Performance improvements. (:pr:`358`). By `Luke Davis`_
- Fix coordinate/axis detection (:pr:`359`). By `Martin Schupfner`_

v0.7.4 (July 14, 2022)
======================

- Major performance improvement for ``__getitem_`` (:pr:`349`).
  By `Deepak Cherian`_.
- Raise warning instead of error with malformed ``cell_measures`` attributes. Warnings
  now print names of variables with malformed attributes (:pr:`350`).
  By `Deepak Cherian`_.

v0.7.3 (June 30, 2022)
======================
- :py:meth:`Dataset.cf.guess_coord_axis` now skips known axes/coordinates and only returns a single guess per variable.
  Additional attributes such as `units` must be added to known axes/coordinates using :py:meth:`Dataset.cf.add_canonical_attributes`.
  By `Mattia Almansi`_.
- Increased support for ``cf_role`` variables. Added :py:attr:`Dataset.cf.cf_roles` By `Deepak Cherian`_.

v0.7.2 (April 5, 2022)
======================
- added encoder and decoder for writing pandas MultiIndex-es to file using "compression by gathering".
  See :doc:`coding` for more. By `Deepak Cherian`_.
- added another type of vertical coordinate to decode: ``ocean_sigma_coordinate``. By `Kristen Thyng`_.

v0.7.0 (January 24, 2022)
=========================
- Many improvements to autoguessing for plotting. By `Deepak Cherian`_
- Fix detection of datetime-like variables. By `Romain Caneill`_
- Integrate more unit aliases from ``xclim`` (Thanks!). By `Deepak Cherian`_
- Significantly expanded and improved documention. By `Deepak Cherian`_.
- integrate ``xclim``'s CF-compliant unit formatter. :pr:`278`. By `Justus Magin`_.
- Fix dropping of bad variable names. :pr:`291`. By `Tom Vo`_.

v0.6.3 (December 16, 2021)
==========================
- Packaging improvements. By `Filipe Fernandes`_.
- cf_xarray now works with Datasets created using ``_to_temp_dataset``. By `Pascal Bourgault`_.

v0.6.2 (December 7, 2021)
=========================
- Various bug fixes.
- New ``cf_xarray.geometry`` submodule with functions :py:func:`shapely_to_cf`` and :py:func:`cf_to_shapely` to convert between CF-compliant geometry datasets and DataArrays storing shapely geometries (new optional dependency). By `Pascal Bourgault`_.

v0.6.1 (August 16, 2021)
========================
- Support detecting pint-backed Variables with units-based criteria. By `Deepak Cherian`_.
- Support reshaping nD bounds arrays to (n-1)D vertex arrays. By `Deepak Cherian`_.
- Support rich comparisons  with ``DataArray.cf`` and :py:meth:`DataArray.cf.isin` for `flag variables`_.
  By `Deepak Cherian`_ and `Julius Busecke`_

v0.6.0 (June 29, 2021)
======================
- Support indexing by ``cf_role`` attribute. By `Deepak Cherian`_.
- Implemented :py:meth:`Dataset.cf.add_canonical_attributes` and :py:meth:`DataArray.cf.add_canonical_attributes`
  to add CF canonical attributes. By `Mattia Almansi`_.
- Begin adding support for units with a unit registry for pint arrays. :pr:`197`.
  By `Jon Thielen`_ and `Justus Magin`_.
- :py:meth:`Dataset.cf.rename_like` also updates the ``bounds`` and ``cell_measures`` attributes. By `Mattia Almansi`_.
- Support of custom vocabularies/criteria: user can input criteria for identifying variables by their name and attributes to be able to refer to them by custom names like `ds.cf["ssh"]`. :pr:`234`. By `Kristen Thyng`_ and `Deepak Cherian`_.

v0.5.2 (May 11, 2021)
=====================

- Add some explicit support for CMIP6 output. By `Deepak Cherian`_.
- Replace the ``dims`` argument of :py:meth:`Dataset.cf.add_bounds` with ``keys``, allowing to use CF keys. By `Mattia Almansi`_.
- Added :py:attr:`DataArray.cf.formula_terms` and :py:attr:`Dataset.cf.formula_terms`.
  By `Deepak Cherian`_.
- Added :py:attr:`Dataset.cf.bounds` to return a dictionary mapping valid keys to the variable names of their bounds. By `Mattia Almansi`_.
- :py:meth:`DataArray.cf.differentiate` and :py:meth:`Dataset.cf.differentiate` can optionally correct
  sign of the derivative by interpreting the ``"positive"`` attribute. By `Deepak Cherian`_.

v0.5.1 (Feb 24, 2021)
=====================

Minor bugfix release, thanks to `Pascal Bourgault`_.

v0.5.0 (Feb 24, 2021)
=====================

- Replace ``cf.describe()`` with :py:meth:`Dataset.cf.__repr__`. By `Mattia Almansi`_.
- Automatically set ``x`` or ``y`` for :py:attr:`DataArray.cf.plot`. By `Deepak Cherian`_.
- Added scripts to document coordinate and axes criteria with tables. By `Mattia Almansi`_.
- Support for ``.drop_vars()``, ``.drop_sel()``, ``.drop_dims()``, ``.set_coords()``, ``.reset_coords()``. By `Mattia Almansi`_.
- Support for using ``standard_name`` in more functions. (:pr:`128`) By `Deepak Cherian`_
- Allow :py:meth:`DataArray.cf.__getitem__` with standard names. By `Deepak Cherian`_
- Rewrite the ``values`` of :py:attr:`Dataset.coords` and :py:attr:`Dataset.data_vars` with objects returned
  by :py:meth:`Dataset.cf.__getitem__`. This allows extraction of DataArrays when there are clashes
  between DataArray names and "special" CF names like ``T``.
  (:issue:`129`, :pr:`130`). By `Deepak Cherian`_
- Retrieve bounds dimension name with :py:meth:`Dataset.cf.get_bounds_dim_name`. By `Pascal Bourgault`_.
- Fix iteration and arithmetic with ``GroupBy`` objects. By `Deepak Cherian`_.

v0.4.0 (Jan 22, 2021)
=====================
- Support for arbitrary cell measures indexing. By `Mattia Almansi`_.
- Avoid using ``grid_latitude`` and ``grid_longitude`` for detecting latitude and longitude variables.
  By `Pascal Bourgault`_.

v0.3.1 (Nov 25, 2020)
=====================
- Support :py:attr:`Dataset.cf.cell_measures`. By `Deepak Cherian`_.
- Added :py:attr:`Dataset.cf.axes` to return a dictionary mapping available Axis standard names to variable names of an xarray object, :py:attr:`Dataset.cf.coordinates` for Coordinates, :py:attr:`Dataset.cf.cell_measures` for Cell Measures, and :py:attr:`Dataset.cf.standard_names` for all variables. `Kristen Thyng`_ and `Mattia Almansi`_.
- Changed :py:meth:`Dataset.cf.get_valid_keys` to :py:meth:`Dataset.cf.keys`. `Kristen Thyng`_.
- Added :py:meth:`Dataset.cf.decode_vertical_coords` for decoding of parameterized vertical coordinate variables.
  (:issue:`34`, :pr:`103`). `Deepak Cherian`_.
- Added top-level :py:func:`~cf_xarray.bounds_to_vertices` and :py:func:`~cf_xarray.vertices_to_bounds` as well as :py:meth:`Dataset.cf.bounds_to_vertices`
  to convert from coordinate bounds in a CF format (shape (nx, 2)) to a vertices format (shape (nx+1)).
  (:pr:`108`). `Pascal Bourgault`_.

v0.3.0 (Sep 27, 2020)
=====================
This release brings changes necessary to make ``cf_xarray`` more useful with the ROMS
model in particular. Thanks to Kristen Thyng for opening many issues.

- ``vertical`` and ``Z`` are not synonyms any more. In particular, the attribute
  ``positive: up`` now will only match ``vertical`` and not ``Z``. `Deepak Cherian`_.
- Fixed tests that would only pass if ran in a specific order. `Julia Kent`_.

v0.2.1 (Aug 06, 2020)
=====================
- Support for the ``bounds`` attribute. (:pr:`68`, :issue:`32`). `Deepak Cherian`_.
- Add :py:meth:`Dataset.cf.guess_coord_axis` to automagically guess axis and coord names, and add
  appropriate attributes. (:pr:`67`, :issue:`46`). `Deepak Cherian`_.

v0.2.0 (Jul 28, 2020)
=====================

- ``cf_xarray`` is now available on conda-forge. Thanks to `Anderson Banihirwe`_ and `Filipe Fernandes`_
- Remap datetime accessor syntax for groupby. E.g. ``.cf.groupby("T.month")`` → ``.cf.groupby("ocean_time.month")``.
  (:pr:`64`, :issue:`6`). `Julia Kent`_.
- Added :py:meth:`Dataset.cf.rename_like` to rename matching variables. Only coordinate variables
  i.e. those that match the criteria for ``("latitude", "longitude", "vertical", "time")``
  are renamed for now. (:pr:`55`) `Deepak Cherian`_.
- Added :py:meth:`Dataset.cf.add_bounds` to add guessed bounds for 1D coordinates. (:pr:`53`) `Deepak Cherian`_.

v0.1.5
======

- Begin documenting things for contributors in :ref:`contribut`.
- Parse ``ancillary_variables`` attribute. These variables are converted to coordinate variables.
- Support :py:meth:`Dataset.reset_index`
- Wrap ``.sizes`` and ``.chunks``. (:pr:`42`) `Deepak Cherian`_.

     >>> ds.cf.sizes
     {'X': 53, 'Y': 25, 'T': 2920, 'longitude': 53, 'latitude': 25, 'time': 2920}


v0.1.4
======

- Support indexing by ``standard_name``
- Set default ``xincrease`` and ``yincrease`` by interpreting the ``positive`` attribute.

v0.1.3
======

- Support expanding key to multiple dimension names.

.. _`Mattia Almansi`: https://github.com/malmans2
.. _`Justus Magin`: https://github.com/keewis
.. _`Jon Thielen`: https://github.com/jthielen
.. _`Anderson Banihirwe`: https://github.com/andersy005
.. _`Pascal Bourgault`: https://github.com/aulemahal
.. _`Deepak Cherian`: https://github.com/dcherian
.. _`Filipe Fernandes`: https://github.com/ocefpaf
.. _`Julia Kent`: https://github.com/jukent
.. _`Kristen Thyng`: https://github.com/kthyng
.. _`Julius Busecke`: https://github.com/jbusecke
.. _`Tom Vo`: https://github.com/tomvothecoder
.. _`Romain Caneill`: https://github.com/rcaneill
.. _`Lars Buntemeyer`: https://github.com/larsbuntemeyer
.. _`Luke Davis`: https://github.com/lukelbd
.. _`Martin Schupfner`: https://github.com/sol1105
.. _`Mathias Hauser`: https://github.com/mathause
.. _`Aidan Heerdegen`: https://github.com/aidanheerdegen
.. _`flag variables`: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
