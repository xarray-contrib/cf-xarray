import xarray as xr
import cf_xarray as cfxr


def print_section(title):
    print(f"\n{title}")
    print("=" * len(title))


def print_bounds_and_vertices(bounds, vertices):
    print("Bounds:")
    print(bounds)
    print("Vertices:")
    print(vertices)
    print("-" * 40)


# 0a. Strictly monotonic bounds (increasing)
print_section("0a. Strictly monotonic bounds (increasing)")
bounds = xr.DataArray(
    [[10.0, 10.5], [10.5, 11.0], [11.0, 11.5], [11.5, 12.0], [12.0, 12.5]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 0b. Strictly monotonic bounds (decreasing)
print_section("0b. Strictly monotonic bounds (decreasing)")
bounds = xr.DataArray(
    [[12.5, 12.0], [12.0, 11.5], [11.5, 11.0], [11.0, 10.5], [10.5, 10.0]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 1. Descending coordinates
print_section("1. Descending coordinates")
bounds = xr.DataArray(
    [[50.5, 50.0], [51.0, 50.5], [51.5, 51.0], [52.0, 51.5], [52.5, 52.0]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 2. Descending coordinates with duplicated values
print_section("2. Descending coordinates with duplicated values")
bounds = xr.DataArray(
    [[50.5, 50.0], [51.0, 50.5], [51.0, 50.5], [52.0, 51.5], [52.5, 52.0]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 3. Ascending coordinates
print_section("3. Ascending coordinates")
bounds = xr.DataArray(
    [[50.0, 50.5], [50.5, 51.0], [51.0, 51.5], [51.5, 52.0], [52.0, 52.5]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 4. Ascending coordinates with duplicated values
print_section("4. Ascending coordinates with duplicated values")
bounds = xr.DataArray(
    [[50.0, 50.5], [50.5, 51.0], [50.5, 51.0], [51.0, 51.5], [51.5, 52.0]],
    dims=("lat", "bounds"),
)
vertices = cfxr.bounds_to_vertices(bounds, "bounds")
print_bounds_and_vertices(bounds, vertices)

# 5. 3D array (extra non-core dim)
print_section("5. 3D array (extra non-core dim)")
bounds_3d = xr.DataArray(
    [
        [
            [50.0, 50.5],
            [50.5, 51.0],
            [51.0, 51.5],
            [51.5, 52.0],
            [52.0, 52.5],
        ],
        [
            [60.0, 60.5],
            [60.5, 61.0],
            [61.0, 61.5],
            [61.5, 62.0],
            [62.0, 62.5],
        ],
    ],
    dims=("extra", "lat", "bounds"),
)
vertices_3d = cfxr.bounds_to_vertices(bounds_3d, "bounds", core_dims=["lat"])
print_bounds_and_vertices(bounds_3d, vertices_3d)

# 6. 4D array (time, extra, lat, bounds)
print_section("6. 4D array (time, extra, lat, bounds)")
bounds_4d = xr.DataArray(
    [
        [
            [
                [50.0, 50.5],
                [50.5, 51.0],
                [51.0, 51.5],
                [51.5, 52.0],
                [52.0, 52.5],
            ],
            [
                [60.0, 60.5],
                [60.5, 61.0],
                [61.0, 61.5],
                [61.5, 62.0],
                [62.0, 62.5],
            ],
        ],
        [
            [
                [70.0, 70.5],
                [70.5, 71.0],
                [71.0, 71.5],
                [71.5, 72.0],
                [72.0, 72.5],
            ],
            [
                [80.0, 80.5],
                [80.5, 81.0],
                [81.0, 81.5],
                [81.5, 82.0],
                [82.0, 82.5],
            ],
        ],
    ],
    dims=("time", "extra", "lat", "bounds"),
)
vertices_4d = cfxr.bounds_to_vertices(bounds_4d, "bounds", core_dims=["lat"])
print_bounds_and_vertices(bounds_4d, vertices_4d)
