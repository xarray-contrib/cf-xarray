.. currentmodule:: xarray

What's New
----------

v0.5.1 (Feb 24, 2021)
=====================

Minor bugfix release, thanks to `Pasical Bourgault`_.

v0.5.0 (Feb 24, 2021)
=====================

- Replace ``cf.describe()`` with :py:meth:`Dataset.cf.__repr__`. By `Mattia Almansi`_.
- Automatically set ``x`` or ``y`` for :py:attr:`DataArray.cf.plot`. By `Deepak Cherian`_.
- Added scripts to document :ref:`criteria` with tables. By `Mattia Almansi`_.
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
.. _`Anderson Banihirwe`: https://github.com/andersy005
.. _`Pascal Bourgault`: https://github.com/aulemahal
.. _`Deepak Cherian`: https://github.com/dcherian
.. _`Filipe Fernandes`: https://github.com/ocefpaf
.. _`Julia Kent`: https://github.com/jukent
.. _`Kristen Thyng`: https://github.com/kthyng
