import numpy as np
import inspect

_REGISTRY = {}

def register(name=None):
    def wrapper(func):
        func_name = name or func.__name__

        arg_spec = inspect.getfullargspec(func)

        func._requirements = set(arg_spec.args)

        if func_name not in _REGISTRY:
            _REGISTRY[func_name] = func
    return wrapper

def get_parametric_func(stdname):
    try:
        return _REGISTRY[stdname]
    except KeyError:
        raise NotImplementedError(
            f"Coordinate function for {stdname!r} not implemented yet. Contributions welcome!"
        )

@register()
def atmosphere_ln_pressure_coordinate(p0, lev):
    return p0 * np.exp(-lev)

@register()
def atmosphere_sigma_coordinate(sigma, ps, ptop):
    return ptop + sigma * (ps - ptop)

@register()
def atmosphere_hybrid_sigma_pressure_coordinate(b, ps, p0, a=None, ap=None):
    if a is None:
        value = ap + b * ps
    else:
        value = a * p + b * ps

    return value

@register()
def atmosphere_hybrid_height_coordinate(a, b, orog):
    return a + b * orog

@register()
def atmosphere_sleve_coordinate(a, b1, b2, ztop, zsurf1, zsurf2):
    return a + ztop + b1 * zsurf1 + b2 * zsurf2

@register()
def ocean_sigma_coordinate(sigma, eta, depth):
    return eta + sigma * (depth + eta)

@register()
def ocean_s_coordinate(s, eta, depth, a, b, depth_c):
    c = (1 - b) * np.sinh(a * s) / np.sinh(a) + b * (np.tanh(a * (s + 0.5)) / 2 * np.tanh(0.5 * a) - 0.5)

    return eta * (1 + s) + depth_c * s + (depth - depth_c) * c

@register()
def ocean_s_coordinate_g1(s, C, eta, depth, depth_c):
    s = depth_c * s + (depth - depth_c) * C

    return s + eta * (1 + s / depth)

@register()
def ocean_s_coordinate_g2(s, C, eta, depth, depth_c):
    s = (depth_c * s + depth * C) / (depth_c + depth)

    return eta + (eta + depth) * s

@register()
def ocean_sigma_z_coordinate(sigma, eta, depth, depth_c, nsigma, zlev):
    n, j, i = eta.shape

    k = sigma.shape[0]

    z = np.zeros((n, k, j, i))

    sigma_defined = ~np.isnan(sigma)

    zlev_defined = ~np.isnan(zlev)

    depth_min = np.minimum(depth_c, depth[np.newaxis, :, :])

    z[:, sigma_defined, :, :] = eta[:, np.newaxis, :, :] + sigma[sigma_defined, np.newaxis, np.newaxis] * (depth_min + eta[:, np.newaxis, :, :])

    z[:, zlev_defined, :, :] = zlev[zlev_defined]

    return z

@register()
def ocean_double_sigma_coordinate(sigma, depth, z1, z2, a, href, k_c):
    k = sigma.shape[0]

    j, i = depth.shape

    z = np.zeros((k, j, i))

    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(2 * a / (z1 - z2) * (depth - href))

    above_kc = sigma.k > k_c

    z[above_kc, :, :] = f + (sigma[above_kc] - 1) * (depth[np.newaxis, :, :] - f)

    z[~above_kc, :, :] = sigma[~above_kc] * f

    return z
