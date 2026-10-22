.. _releasing:

Releasing cf-xarray
===================

This checklist is for maintainers with upstream release permissions.
The existing workflows, rather than a local upload, build and publish releases.
The version comes from Git tags through ``setuptools_scm``; do not manually
edit a source version string.

Before tagging
--------------

* Confirm the intended changes are merged into ``main`` and the required CI
  jobs pass. Review the changes since the previous release and prepare release
  notes using the `GitHub releases page
  <https://github.com/xarray-contrib/cf-xarray/releases>`_.
* Check the documentation build and any compatibility changes against the
  supported Python and dependency environments in
  ``.github/workflows/ci.yaml``.
* For a package-build rehearsal, a maintainer can add the ``test-build`` label
  to a pull request or manually dispatch ``testpypi-release.yaml``. Despite its
  name, that workflow currently builds and checks the distributions and installs
  the wheel, but its TestPyPI upload step is commented out. Inspect its
  ``releases`` artifact and the version printed by the installed package.
  An ordinary push to ``main`` does not pass this job's execution condition.

Local build rehearsal
---------------------

Use a full Git checkout with tags available. Run these commands in a disposable
checkout and a fresh virtual environment, not in an environment with an editable
cf-xarray installation:

.. code-block:: bash

   python -m pip install build twine
   python -m build
   python -m twine check --strict dist/*
   python -m pip install dist/cf_xarray-*.whl

Then change to a directory outside the checkout and check the installed wheel:

.. code-block:: bash

   python -m cf_xarray.scripts.print_versions
   python -c "import cf_xarray; print(cf_xarray.__version__)"

Use a clean ``dist`` directory so the install selects exactly one wheel.
A development checkout can have a development version; the final tagged build
must have the intended release version, not ``0.0.0``.
This rehearsal does not upload anything. The CI workflows clean their disposable
checkouts before building; do not copy ``git clean -xdf`` into a working checkout
that contains files you want to keep.

Tag and publish
---------------

* After the checks pass, choose the release version and create and push its
  ``v``-prefixed Git tag at the reviewed commit using the project's normal
  maintainer process.
* A pushed ``v*`` tag triggers ``pypi.yaml`` in the upstream repository. Its
  build job creates and checks the sdist and wheel; the push-event job installs
  the wheel, prints versions and uploads to TestPyPI. Confirm these jobs succeed
  before publishing the GitHub Release. A TestPyPI upload is a real publication,
  not a dry run.
* Publish the GitHub Release for the same tag, with the reviewed release notes.
  The ``release: published`` event triggers ``pypi.yaml`` again and publishes to
  PyPI using the protected ``pypi`` environment and trusted publishing.
  Complete any required environment approval and confirm the upload succeeded.
  Do not recreate or move a published tag, or attempt to reuse a published
  package version to recover from a failed upload.

After publishing
----------------

* In a fresh environment, install the released version from PyPI and run
  ``python -m cf_xarray.scripts.print_versions`` outside the source checkout.
  Check the installed version matches the GitHub Release and both distribution
  files are present on PyPI.
* Check that the documentation for the release is available and that the release
  notes describe compatibility changes and credit contributors.
* If a release is broken, preserve the failed workflow logs and artifacts,
  coordinate the correction with the maintainers, and use a new version rather
  than silently replacing the release.
