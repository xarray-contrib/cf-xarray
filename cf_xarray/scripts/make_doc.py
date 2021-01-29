#!/usr/bin/env python

import os

from pandas import DataFrame

from cf_xarray.accessor import coordinate_criteria


def main():

    # coordinate_criteria table
    path = "_build/csv/coordinate_criteria.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = DataFrame.from_dict(coordinate_criteria)
    df = df.fillna("")
    df = df.applymap(lambda x: ", ".join(sorted(x)))
    df = df.sort_index(0).sort_index(1)
    df.to_csv(path)


if __name__ == "__main__":
    main()
