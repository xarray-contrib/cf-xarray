# Connectivity attributes on a UGRID ``mesh_topology`` variable.
UGRID_CONNECTIVITY_ATTRS = [
    "face_node_connectivity",
    "edge_node_connectivity",
    "face_edge_connectivity",
    "face_face_connectivity",
    "edge_face_connectivity",
    "boundary_node_connectivity",
]

# Coordinate-variable attributes on a UGRID ``mesh_topology`` variable.
UGRID_COORD_ATTRS = [
    "node_coordinates",
    "edge_coordinates",
    "face_coordinates",
]


def get_mesh_variables(ds, mesh_var_name):
    """Return variables referenced by a UGRID ``mesh_topology`` variable.

    Reads the connectivity attributes (``face_node_connectivity``,
    ``edge_node_connectivity``, ...) and the coordinate attributes
    (``node_coordinates``, ``edge_coordinates``, ``face_coordinates``) from the
    mesh topology variable's attrs.

    Returns a ``(connectivity, coordinates)`` tuple of variable-name lists,
    each filtered to names actually present in ``ds``.
    """
    if mesh_var_name not in ds.variables:
        return [], []
    mesh_attrs = ds[mesh_var_name].attrs
    connectivity: list = []
    for attr_name in UGRID_CONNECTIVITY_ATTRS:
        if conn_str := mesh_attrs.get(attr_name):
            connectivity.extend(n for n in conn_str.split() if n in ds.variables)
    coordinates: list = []
    for attr_name in UGRID_COORD_ATTRS:
        if coord_str := mesh_attrs.get(attr_name):
            coordinates.extend(n for n in coord_str.split() if n in ds.variables)
    return connectivity, coordinates
