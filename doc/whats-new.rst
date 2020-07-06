What's New
----------

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
