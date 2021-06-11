import os
from tempfile import TemporaryDirectory

from cf_xarray.scripts import make_doc


def test_make_doc():

    names = [
        "axes_criteria",
        "coords_criteria",
        "all_criteria",
        "all_regex",
    ]
    tables_to_check = [f"_build/csv/{name}.csv" for name in names]

    # Create _build/csv in a temporary directory
    owd = os.getcwd()
    with TemporaryDirectory() as tmpdirname:
        try:
            os.chdir(os.path.dirname(tmpdirname))
            make_doc.main()
            assert all(os.path.exists(path) for path in tables_to_check)
        finally:
            # Always return to original working directory
            os.chdir(owd)
