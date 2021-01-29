import os

from cf_xarray.scripts import make_doc


def test_make_doc():

    path = "_build/csv/coordinate_criteria.csv"
    try:
        make_doc.main()
        assert os.path.exists(path)
    finally:
        os.remove(path)
