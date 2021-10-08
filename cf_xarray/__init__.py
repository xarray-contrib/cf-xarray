from .accessor import CFAccessor  # noqa
from .geometry import cf_to_shapely, shapely_to_cf  # noqa
from .helpers import bounds_to_vertices, vertices_to_bounds  # noqa
from .options import set_options  # noqa
from .utils import _get_version

__version__ = _get_version()
