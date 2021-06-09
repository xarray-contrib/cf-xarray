#!/usr/bin/env python

import argparse
from urllib.request import urlopen

#:  Link to CF standard name table
CF_TABLE_URL = (
    "https://raw.githubusercontent.com/cf-convention/"
    "cf-convention.github.io/master/Data/cf-standard-names/current/src/"
    "cf-standard-name-table.xml"
)


def main():

    return f"CF_STANDARD_NAME_TABLE = {urlopen(CF_TABLE_URL).read()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Print to terminal cf_table.py"),
        epilog=("To generate cf_table.py:" " python cf_table_to_str.py > cf_table.py"),
    )
    parser.parse_args()
    comment = [
        '"""',
        f"Created by {parser.prog}",
        f"Raw file url: {CF_TABLE_URL}",
        '"""',
    ]
    print("\n".join(comment))
    print(main())
