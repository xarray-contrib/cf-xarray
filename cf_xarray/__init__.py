from pkg_resources import DistributionNotFound, get_distribution

from .accessor import CFAccessor  # noqa
from .helpers import bounds_to_vertices, vertices_to_bounds  # noqa

try:
    __version__ = get_distribution("cf_xarray").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
