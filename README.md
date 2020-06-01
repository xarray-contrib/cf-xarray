# cf-xarray


A lightweight convenience wrapper for using CF attributes on xarray objects. Right now all of this works:

```
import cf_xarray

ds = xr.tutorial.load_dataset("air_temperature").isel(time=slice(4))

ds.air.cf.var("X")

ds.air.cf.resample(T="M").var("X")

(
	ds.air
	.cf.isel(T=slice(4))
	.cf.plot.contourf(x="Y", y="X", col="T", col_wrap=4)
)

ds.air.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")
```
