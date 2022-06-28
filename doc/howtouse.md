---
hide-toc: true
---

```{eval-rst}
.. currentmodule:: xarray
```

# How to use cf_xarray

There are four ways one can use cf_xarray.

## Use CF standard names

Use "CF names" like `standard_name`, coordinates like `"latitude"`, axes like `"X"` instead of actual variable names. For e.g.

## Extract actual variable names

Use `cf_xarray` to extract the appropriate variable name through the properties:

## Rename to a custom vocabulary

Use {py:meth}`Dataset.rename`, or {py:meth}`Dataset.cf.rename_like` to rename variables to your preferences

## Define custom criteria for a custom vocabulary

Define custom criteria to avoid explicity renaming but still work with your datasets using a custom vocabulary.
