#!/usr/bin/env python

import os

from pandas import DataFrame

from cf_xarray.accessor import _AXIS_NAMES, _COORD_NAMES, coordinate_criteria


def main():

    # axes, coordinates, and coordinate criteria tables
    axes = {
        key: {k: v for k, v in values.items() if k in _AXIS_NAMES}
        for key, values in coordinate_criteria.items()
    }
    coords = {
        key: {k: v for k, v in values.items() if k in _COORD_NAMES}
        for key, values in coordinate_criteria.items()
    }

    for mapper, name in zip(
        [coordinate_criteria, axes, coords],
        ["coordinate_criteria", "axes", "coordinates"],
    ):

        path = f"_build/csv/{name}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = DataFrame.from_dict(mapper)
        df = df.fillna("")
        df = df.applymap(lambda x: ", ".join(sorted(x)))
        df = df.sort_index(0).sort_index(1)
        df.to_csv(path)


if __name__ == "__main__":
    main()
