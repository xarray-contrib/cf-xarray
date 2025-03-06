from dataclasses import dataclass

import numpy as np
import pandas as pd
from xarray.groupers import EncodedGroups, UniqueGrouper


@dataclass
class FlagGrouper(UniqueGrouper):
    """
    Grouper object that allows convenient categorical grouping by a CF flag variable.

    Labels in the grouped output will be restricted to those listed in ``flag_meanings``.
    """

    def factorize(self, group) -> EncodedGroups:
        if "flag_values" not in group.attrs or "flag_meanings" not in group.attrs:
            raise ValueError(
                "FlagGrouper can only be used with flag variables that have"
                "`flag_values` and `flag_meanings` specified in attrs."
            )

        values = np.array(group.attrs["flag_values"])
        full_index = pd.Index(group.attrs["flag_meanings"].split(" "))

        self.labels = values

        # TODO: we could optimize here, since `group` is already factorized,
        # but there are subtleties. For example, the attrs must be up to date,
        # any value that is not in flag_values will cause an error, etc.
        ret = super().factorize(group)

        ret.codes.attrs.pop("flag_values")
        ret.codes.attrs.pop("flag_meanings")

        return EncodedGroups(
            codes=ret.codes,
            full_index=full_index,
            group_indices=ret.group_indices,
        )
