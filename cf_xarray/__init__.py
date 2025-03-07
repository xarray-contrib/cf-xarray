import xarray
from packaging.version import Version

from . import geometry as geometry
from . import sgrid  # noqa
from .accessor import CFAccessor  # noqa
from .coding import (  # noqa
    decode_compress_to_multi_index,
    encode_multi_index_as_compress,
)
from .geometry import cf_to_shapely, shapely_to_cf  # noqa
from .helpers import bounds_to_vertices, vertices_to_bounds  # noqa
from .options import set_options  # noqa
from .utils import _get_version

if Version(xarray.__version__) >= Version("2024.07.0"):
    from . import groupers as groupers

__version__ = _get_version()
