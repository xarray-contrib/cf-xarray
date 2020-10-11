What's New
----------


v0.4.0 (unreleased)
===================

- Added ``.cf.decode_vertical_coords`` for decoding of paramterized vertical coordinate variables.
  (:issue:`34`, :pr:`103`). `Deepak Cherian`_.

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
- Add ``.cf.guess_coord_axis`` to automagically guess axis and coord names, and add
  appropriate attributes. (:pr:`67`, :issue:`46`). `Deepak Cherian`_.

v0.2.0 (Jul 28, 2020)
=====================

- ``cf_xarray`` is now available on conda-forge. Thanks to `Anderson Banihirwe`_ and `Filipe Fernandes`_
- Remap datetime accessor syntax for groupby. E.g. ``.cf.groupby("T.month")`` → ``.cf.groupby("ocean_time.month")``.
  (:pr:`64`, :issue:`6`). `Julia Kent`_.
- Added ``.cf.rename_like`` to rename matching variables. Only coordinate variables
  i.e. those that match the criteria for ``("latitude", "longitude", "vertical", "time")``
  are renamed for now. (:pr:`55`) `Deepak Cherian`_.
- Added ``.cf.add_bounds`` to add guessed bounds for 1D coordinates. (:pr:`53`) `Deepak Cherian`_.

v0.1.5
======
- Wrap ``.sizes`` and ``.chunks``. (:pr:`42`) `Deepak Cherian`_.

     >>> ds.cf.sizes
     {'X': 53, 'Y': 25, 'T': 2920, 'longitude': 53, 'latitude': 25, 'time': 2920}

- Begin documenting things for contributors in :ref:`contribut`.
- Parse ``ancillary_variables`` attribute. These variables are converted to coordinate variables.
- Support ``reset_index``

v0.1.4
======

- Support indexing by ``standard_name``
- Set default ``xincrease`` and ``yincrease`` by interpreting the ``positive`` attribute.

v0.1.3
======

- Support expanding key to multiple dimension names.

.. _`Anderson Banihirwe`: https://github.com/andersy005
.. _`Deepak Cherian`: https://github.com/dcherian
.. _`Filipe Fernandes`: https://github.com/ocefpaf
.. _`Julia Kent`: https://github.com/jukent
