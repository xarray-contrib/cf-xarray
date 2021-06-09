import importlib
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

from cf_xarray import cf_table
from cf_xarray.scripts import cf_table_to_dicts, make_doc


def test_make_doc():

    # Create _buil/csv in a temporary directory
    owd = os.getcwd()
    with TemporaryDirectory() as tmpdirname:
        names = [
            "axes_criteria",
            "coords_criteria",
            "all_criteria",
            "all_regex",
        ]
        tables_to_check = [f"_build/csv/{name}.csv" for name in names]
        try:
            os.chdir(os.path.dirname(tmpdirname))
            make_doc.main()
            assert all(os.path.exists(path) for path in tables_to_check)
        finally:
            # Always return to original working directory
            os.chdir(owd)


def test_cf_table_to_dicts():

    actual = cf_table

    with NamedTemporaryFile("w", suffix=".py") as tmpfile:
        tmpfile.write(cf_table_to_dicts.main())
        spec = importlib.util.spec_from_file_location("tmpmodule", tmpfile.name)
        expected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expected)

        assert actual.CF_TABLE_INFO == expected.CF_TABLE_INFO
        assert actual.CF_TABLE_STD_NAMES == expected.CF_TABLE_STD_NAMES
        assert actual.CF_TABLE_ALIASES == expected.CF_TABLE_ALIASES
