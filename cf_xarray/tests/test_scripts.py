import os

from cf_xarray.scripts import make_doc


def remove_if_exist(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def test_make_doc():

    # Create/remove files from tests/,
    # always return to original working directory
    owd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        names = ["axes", "coordinates", "coordinate_criteria"]
        paths_to_check = [f"_build/csv/{name}.csv" for name in names]
        remove_if_exist(paths_to_check)

        make_doc.main()
        assert all(os.path.exists(path) for path in paths_to_check)
    finally:
        os.chdir(owd)
