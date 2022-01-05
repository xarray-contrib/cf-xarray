#!/usr/bin/env python

import os

from pandas import DataFrame

from cf_xarray.accessor import _AXIS_NAMES, _COORD_NAMES
from cf_xarray.criteria import coordinate_criteria, regex


def main():
    """
    Make all additional files needed to build the documentations.
    """

    make_criteria_csv()
    make_regex_csv()


def make_criteria_csv():
    """
    Make criteria tables:
        _build/csv/{all,axes,coords}_criteria.csv
    """

    csv_dir = "_build/csv"
    os.makedirs(csv_dir, exist_ok=True)

    # Criteria tables
    df = DataFrame.from_dict(coordinate_criteria)
    df = df.dropna(axis=1, how="all")
    df = df.applymap(lambda x: ", ".join(sorted(x)) if isinstance(x, tuple) else x)
    df = df.sort_index(axis=0).sort_index(axis=1)

    # All criteria
    df.transpose().to_csv(os.path.join(csv_dir, "all_criteria.csv"))

    # Axes and coordinates
    for keys, name in zip([_AXIS_NAMES, _COORD_NAMES], ["axes", "coords"]):
        subdf = df[sorted(keys)].dropna(axis=1, how="all")
        subdf = subdf.dropna(axis=1, how="all").transpose()
        subdf.transpose().to_csv(os.path.join(csv_dir, f"{name}_criteria.csv"))


def make_regex_csv():
    """
    Make regex tables:
        _build/csv/all_regex.csv
    """

    csv_dir = "_build/csv"
    os.makedirs(csv_dir, exist_ok=True)
    df = DataFrame(regex, index=[0])
    df = df.applymap(lambda x: f"``{str(x)[11:-1]}``")
    df = df.sort_index(axis=1).transpose()
    df.to_csv(os.path.join(csv_dir, "all_regex.csv"), header=False)


if __name__ == "__main__":
    main()
