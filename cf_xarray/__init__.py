from . import tracking  # noqa
from .accessor import CFAccessor  # noqa
from .coding import (  # noqa
    decode_compress_to_multi_index,
    encode_multi_index_as_compress,
)
from .geometry import cf_to_shapely, shapely_to_cf  # noqa
from .helpers import bounds_to_vertices, vertices_to_bounds  # noqa
from .options import set_options  # noqa
from .tracking import track_cf_attributes  # noqa
from .utils import _get_version

__version__ = _get_version()
