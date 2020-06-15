.. currentmodule:: cf_xarray

Roadmap
-------

Goals
=====

1. Enable easy use of additional CF attributes that are not decoded by xarray.

2. Provide a consolidated set of public helper functions that other packages can use to avoid
   duplicated efforts in parsing CF attributes.

Scope
=====


1. This package will not provide a full implementation of the CF data model using xarray objects.
   This use case should be served by Iris.

2. Unit support is left to ``pint-xarray`` and future xarray support for ``pint`` until it is clear
   that there is a need for some functionality.

3. Projections and CRS stuff is left to ``rioxarray`` and other geo-xarray packages. Some helper
   functions could be folded in to ``cf-xarray`` to encourage consolidation in that sub-ecosystem.

Design principles
=================

1. Be uncomplicated.

   Avoid anything that requires saving state in accessor objects (for now).

2. Be friendly.

   Users should be allowed to mix CF names and variables names from the parent xarray object e.g.
   ``ds.cf.plot(x="X", y="model_depth")``. This allows use with "imperfectly tagged" datasets.

3. Be loud when wrapping to avoid confusion.

   For e.g. the ``repr`` for ``cf.groupby("X")`` should make it clear that this is a
   CF-wrapped ``Resample`` instance i.e. ``cf.groupby("X").mean("T")`` is allowed. Docstrings
   should also clearly indicate wrapping by ``cf-xarray``; for e.g. ``ds.cf.isel``.

4. Allow easy debugging and help build understanding.

   Since ``cf_xarray`` depends on ``attrs`` being present and since ``attrs`` are easily lost in xarray
   operations, we should allow easy diagnosis of what ``cf_xarray`` can decode for a particular
   object.
