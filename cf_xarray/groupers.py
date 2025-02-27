import numpy as np
import pandas as pd
from xarray.groupers import EncodedGroups, Grouper


class FlagGrouper(Grouper):
    def factorize(self, group) -> EncodedGroups:
        assert "flag_values" in group.attrs
        assert "flag_meanings" in group.attrs

        values = np.array(group.attrs["flag_values"])
        full_index = pd.Index(group.attrs["flag_meanings"].split(" "))

        if group.dtype.kind in "iu" and (np.diff(values) == 1).all():
            # optimize
            codes = group.data - values[0].astype(int)
        else:
            codes, _ = pd.factorize(group.data.ravel())

        codes_da = group.copy(data=codes.reshape(group.shape))
        codes_da.attrs.pop("flag_values")
        codes_da.attrs.pop("flag_meanings")

        return EncodedGroups(codes=codes_da, full_index=full_index)

    def reset(self):
        pass
