from . import tracking  # noqa
from .accessor import CFAccessor  # noqa
from .helpers import bounds_to_vertices, vertices_to_bounds  # noqa
from .options import set_options  # noqa
from .tracking import track_cf_attributes  # noqa
from .utils import _get_version

__version__ = _get_version()
