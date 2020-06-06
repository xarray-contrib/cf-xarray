# cf-xarray


A lightweight convenience wrapper for using CF attributes on xarray objects. Right now all of this works:

```
import cf_xarray

ds = xr.tutorial.load_dataset("air_temperature").isel(time=slice(4))

ds.air.cf.var("X")

ds.air.cf.resample(T="M").var()

ds.air.cf.groupby("T").var("Y")

(
	ds.air
	.cf.isel(T=slice(4))
	.cf.plot.contourf(x="Y", y="X", col="T", col_wrap=4)
)

ds.air.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")

ds.air.attrs["cell_measures"] = "area: cell_area"
ds.coords["cell_area"] = (
    xr.DataArray(np.cos(ds.cf["latitude"] * np.pi / 180))
    * xr.ones_like(ds.cf["longitude"])
    * 105e3
    * 110e3
)
ds.air.cf.weighted("area").sum("latitude")

```
