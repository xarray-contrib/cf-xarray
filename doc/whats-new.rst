What's New
----------

v0.1.6 (unreleased)
===================

- Added ``.cf.rename_like`` to rename matching variables. Only coordinate variables
  i.e. those that match the criteria for ``("latitude", "longitude", "vertical", "time")``
  are renamed for now. (:pr:`55`) `Deepak Cherian`_.
- Added ``.cf.add_bounds`` to add guessed bounds for 1D coordinates. (:pr:`53`) `Deepak Cherian`_.

v0.1.5
======
- Wrap ``.sizes`` and ``.chunks``. (:pr:`42`) `Deepak Cherian`_.

     >>> ds.cf.sizes
     {'X': 53, 'Y': 25, 'T': 2920, 'longitude': 53, 'latitude': 25, 'time': 2920}

- Begin documenting things for contributors in :ref:`contributing`.
- Parse ``ancillary_variables`` attribute. These variables are converted to coordinate variables.
- Support ``reset_index``

v0.1.4
======

- Support indexing by ``standard_name``
- Set default ``xincrease`` and ``yincrease`` by interpreting the ``positive`` attribute.

v0.1.3
======

- Support expanding key to multiple dimension names.

.. _`Deepak Cherian`: https://github.com/dcherian
