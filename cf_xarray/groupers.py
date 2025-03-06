from dataclasses import dataclass

import numpy as np
import pandas as pd
from xarray.groupers import EncodedGroups, UniqueGrouper


@dataclass
class FlagGrouper(UniqueGrouper):
    def factorize(self, group) -> EncodedGroups:
        if "flag_values" not in group.attrs or "flag_meanings" not in group.attrs:
            raise ValueError(
                "FlagGrouper can only be used with flag variables that have"
                "`flag_values` and `flag_meanings` specified in attrs."
            )

        values = np.array(group.attrs["flag_values"])
        full_index = pd.Index(group.attrs["flag_meanings"].split(" "))

        self.labels = values
        ret = super().factorize(group)

        codes_da = ret.codes
        codes_da.attrs.pop("flag_values")
        codes_da.attrs.pop("flag_meanings")

        ret.codes = codes_da
        ret.full_index = full_index

        return ret
