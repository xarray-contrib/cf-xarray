#!/usr/bin/env python

import argparse

from cf_xarray.utils import parse_cf_table

#:  Link to CF standard name table
CF_TABLE_URL = (
    "https://raw.githubusercontent.com/cf-convention/"
    "cf-convention.github.io/master/Data/cf-standard-names/current/src/"
    "cf-standard-name-table.xml"
)


def main():
    """
    Returns
    -------
    str
        Python syntax that generates dictionaries used by cf-xarray
        and inferred from the official CF standard name table.
    """

    info, table, aliases = parse_cf_table(CF_TABLE_URL)
    info, table, aliases = (
        {k: tosort[k] for k in sorted(tosort)} for tosort in (info, table, aliases)
    )
    string = [
        f"CF_TABLE_INFO = {info!r}",
        f"CF_TABLE_STD_NAMES = {table!r}",
        f"CF_TABLE_ALIASES = {aliases!r}",
    ]

    return "\n".join(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print to terminal the Python syntax that generates dictionaries"
            " used by cf-xarray and inferred from the official CF standard name table."
        ),
        epilog=(
            "To generate the latest cf_table.py:"
            " python cf_table_to_dicts.py > cf_table.py"
        ),
    )
    parser.parse_args()
    comment = [
        '"""',
        "CF_TABLE_INFO, CF_TABLE_STD_NAMES, and CF_TABLE_ALIASES.",
        f"Created by {parser.prog}.",
        '"""',
    ]
    print("\n".join(comment))
    print(main())
