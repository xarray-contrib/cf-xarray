# This module provides functions for adding CF attribtues
# and tracking history, provenance using xarray's keep_attrs
# functionality

import copy
import functools
from datetime import datetime
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, RDFS
import xarray as xr

PROV_KEY = "__prov__"

CELL_METHODS = {
    "sum": "sum",
    "max": "maximum",
    "min": "minimum",
    "median": "median",
    "mean": "mean",
    "std": "standard_deviation",
    "var": "variance",
}

def call_signature(func, **kwargs):
    callargstr = []

    for (k, v) in kwargs.items():
        if isinstance(v, (xr.DataArray)):
            callargstr.append(f"{k}=<array>")
        elif isinstance(v, (float, int, str)):
            callargstr.append(f"{k}={v!r}")  # repr so strings have ' '
        else:
            # don't take chance of having unprintable values
            callargstr.append(f"{k}={type(v)}")

    return f"{func.__name__}({callargstr})"


def add_cell_methods(attrs, context):
    """Add appropriate cell_methods attribute."""
    assert len(attrs) == 1
    cell_methods = attrs[0].get("cell_methods", "")
    return {"cell_methods": f"context.dim: {CELL_METHODS[context.func]} {cell_methods}".strip()}


def add_history(attrs, context):
    """Adds a history attribute following the NetCDF User Guide convention."""

    # https://www.unidata.ucar.edu/software/netcdf/documentation/4.7.4-pre/attribute_conventions.html
    # A global attribute for an audit trail. This is a character array with a line
    # for each invocation of a program that has modified the dataset. Well-behaved
    # generic netCDF applications should append a line containing:
    #     date, time of day, user name, program name and command arguments.

    # nco uses the ctime format
    now = datetime.now().ctime()
    history = attrs[0].get("history", [])

    new_history = (
        f"{now}:"
        f" {context.func}(args)\n"
        # TODO: should we record software versions?
    )
    return {"history": history + [new_history]}


def init_graph():
    """Create empty graph and bind namespaces."""
    g = Graph()
    # The metaclip ontology
    DS = Namespace("http://www.metaclip.org/datasource/datasource.owl#")
    g.namespace_manager.bind("ds", DS)

    # The namespace describing xarray objects (not sure if this is standard)
    XR = Namespace("xarray:")
    g.namespace_manager.bind("xr", XR)
    return g


def add_provenance(attrs, context):
    """Add provenance information related to the operational context."""
    # Fetch the DataArray graph and instantiate namespaces.
    g = attrs[0].get(PROV_KEY, init_graph())
    ns = dict(g.namespaces())
    XR = Namespace(ns["xr"])
    DS = Namespace(ns["ds"])

    # Creating vertex for the function itself
    cmd = XR[f"call:{context.func}"]
    # For now it's just a generic command, but there could be an Ontology defined for xarray functions giving more
    # information on what they're doing.
    # Also, this is limited because we don't know the arguments to the function nor the dimension it operates on.
    g.add((cmd, RDF.type, DS.Command))

    # Linking that function to the DataArray
    # Unclear how we know exactly which DataArray this command operates on.
    # Cheating a bit here...
    ref = attrs[0]["__prov_da_id__"]
    g.add((ref, DS.hadCommandCall, cmd))
    return {PROV_KEY: g}


def _tracker(
    attrs,
    context,
    strict: bool = False,
    cell_methods: bool = True,
    history: bool = True,
    prov: bool = True
):

    # can only handle single variable attrs for now
    assert len(attrs) == 1
    attrs_out = copy.deepcopy(attrs[0])

    if cell_methods and context.func in CELL_METHODS:
        attrs_out.update(add_cell_methods(attrs, context))
    if history:
        attrs_out.update(add_history(attrs, context))
    if prov:
        attrs_out.update(add_provenance(attrs, context))

    return attrs_out


def track_cf_attributes(
    *, strict: bool = False, cell_methods: bool = True, history: bool = True, prov: bool = True
):
    """Top-level user-facing function.

    Parameters
    ----------
    strict: bool
        Controls if an error is raised when an appropriate attribute cannot
        be added because of lack of information.
    cell_methods: bool
        Add cell_methods attribute when possible
    history: bool
        Adds a history attribute like NCO and follows the NUG convention.
    prov: bool
        Add provenance information to an RDF graph.
    """

    # TODO: check xarray version here.
    return functools.partial(
        _tracker, strict=strict, cell_methods=cell_methods, history=history, prov=prov
    )


def track_provenance_with_rdflib(ds, varname):
    """Create provenance document."""
    prov = ds.attrs.get("has_provenance")
    if prov is not None:
        raise NotImplementedError

    g = init_graph()
    ns = dict(g.namespaces())
    XR = Namespace(ns["xr"])
    DS = Namespace(ns["ds"])

    # Each vertex has an identifier in the graph
    e = XR[f"ds:{id(ds)}"]  # Creates a URIRef

    # Here we add an RDF triplet (subject, predicate, object)
    # What the next line does is tell the graph entity `e` has type `ds:Dataset`
    g.add((e, RDF.type, DS.Dataset))

    if "project_id" in ds.attrs:
        label = ds.attrs["project_id"]
        ref = XR[f"project:{label}"]
        g.add((ref, RDF.type, DS.Project))
        g.add((e, DS.hadProject, ref))
        g.add((ref, RDFS.label, Literal(label)))

    if "institute_id" in ds.attrs:
        label = ds.attrs["institute_id"]
        ref = XR[f"institute:{label.replace(' ', '_')}"]
        g.add((ref, RDF.type, DS.ModellingCenter))
        g.add((e, DS.hadModellingCenter, ref))
        g.add((ref, RDFS.label, Literal(label)))

    # Add vertex for the variable
    key = varname
    da = ds[key]
    # Copy or deepcopy does not make an independent object. The copies still link to the original graph.
    # This will look weird if we want to assign a provenance graph to each variable (they'll all be identical)
    da.attrs[PROV_KEY] = vg = copy.copy(g)
    v = XR[f"da:{key}:{id(da)}"]
    da.attrs["__prov_da_id__"] = v
    # Create DatasetSubset
    vg.add((v, RDF.type, DS.DatasetSubset))
    vg.add((e, DS.hadDatasetSubset, v))
    # Create Variable
    vg.add((v, DS.hasVariable, XR[key]))
    vg.add((XR[key], RDF.type, DS.Variable))
    vg.add((XR[key], RDFS.label, Literal(key)))
    if "units" in da.attrs:
        vg.add((XR[key], DS.withUnits, Literal(da.attrs["units"])))
    # TODO: add info about temporal and spatial extent

    return ds


def track_provenance_with_prov(ds, varname):
    """Not working for now."""
    import prov
    from prov.model import ProvDocument
    from prov.identifier import Namespace
    from uuid import uuid4
    # Create an xarray namespace for what happens here
    XARRAY = Namespace("xarray", uri="urn:xarray:")

    def get_record(label, klass, ns={}):
        """Search namespaces to find a class instance with the given label.

        Use the output to create a new provenance entity or activity.
        """
        # TODO: Search into ns
        # Default when label is not found
        identifier = XARRAY[f"{klass}.{uuid4()}"]
        attributes = {prov.model.PROV_LABEL: label,
                      prov.model.PROV_TYPE: klass}
        # PROV class, identifier, None, attributes
        return dict(identifier=identifier, other_attributes=attributes)

    # Create the provenance document
    doc = ProvDocument()

    # Identify namespaces, here we're using the METACLIP ontologies
    ns = {"ds": "http://www.metaclip.org/datasource/datasource.owl#",
          "ipcc": "http://www.metaclip.org/ipcc_terms/ipcc_terms.owl#",
          "veri": "http://www.metaclip.org/verification/verification.owl#",
          "cal": "http://www.metaclip.org/calibration/calibration.owl#",
          "go": "http://www.metaclip.org/graphical_output/graphical_output.owl#"}
    for key, uri in ns.items():
        doc.add_namespace(key, uri)

    # Create a `Dataset` entity with an identifier that uniquely identifies this object
    # I suppose this could be a __hash__
    ds_id = id(ds)

    # ds:Dataset is a subclass of entity
    e = doc.entity(XARRAY[f"dataset_{ds_id}"],
                   {prov.model.PROV_TYPE: "ds:Dataset"})

    # Add attributes
    # Some attributes might have a corresponding node in the ontology. In that case, we want to link it here.
    # Otherwise, we create an new `instance` of the attribute class.
    # TODO: these attributes are CMIP5 specific. CMIP6 has slight differences in how attributes are named. Ideally,
    # users could create mappings from dataset attributes to ontology classes. More ideally, an inference engine
    # could do this mapping automatically.
    if "project_id" in ds.attrs:
        label = ds.attrs["project_id"]
        # Project is a subclass of prov:activity
        a = doc.activity(**get_record(label, "ds:Project", ns))
        # ds:hadProject is a sub property of prov:wasGeneratedBy
        e.wasGeneratedBy(a, attributes={prov.model.PROV_TYPE: "ds:hadProject"})

    if "institute_id" in ds.attrs:
        label = ds.attrs["institute_id"]
        # ModellingCenter is a subclass of prov.Organization
        a = doc.agent(**get_record(label, "ds:ModellingCenter", ns))
        e.wasAttributedTo(a)

    # ...

    for key in ds.data_vars:
        # ds:DatasetSubset is a subclass of ds: Step, which is a subclass of prov:Derivation
        # A variable is a prov:Entity
        # ds:hasVariable is a property
        da = ds[key]
        da.attrs[PROV_KEY] = vdoc = copy.copy(doc)
        identifier = id(da)
        se = vdoc.entity(XARRAY[f"subset_{identifier}"],
                         {prov.model.PROV_TYPE: "ds:DatasetSubset"})
        se.wasDerivedFrom(e, {prov.model.PROV_TYPE: "ds:hadDatasetSubset"})
        try:
            v = vdoc.entity(XARRAY[f"dataarray_{identifier}"],
                            {prov.model.PROV_TYPE: "ds:Variable",
                             prov.model.PROV_LABEL: key,
                             "ds:withUnits": da.attrs["units"],
                             })
            # ... ?
        except KeyError:
            pass

        # Don't know how to make an edge with ds:hasVariable


def test_prov_tracking():
    ds = xr.open_dataset("/home/david/data/cmip5/pr_Amon_GFDL-CM3_historical_r1i1p1_186001-186412.nc")

    # Create RDF graph in attribute '__prov__'
    track_provenance_with_rdflib(ds, "pr")

    # Run operation
    with xr.set_options(keep_attrs=track_cf_attributes(prov=True)):
        ds.pr.mean(dim="time")

    print(ds.pr.__prov__.serialize())
