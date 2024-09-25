from enum import Enum
from keyword import kwlist
from os import PathLike
from typing import (
    Any,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import pytest
from modflow_devtools.misc import run_cmd
from numpy.typing import NDArray

from autotest.conftest import get_project_root_path

PROJ_ROOT = get_project_root_path()
DFNS_PATH = PROJ_ROOT / "flopy" / "mf6" / "data" / "dfn"
SCALAR_TYPES = {
    "keyword": bool,
    "integer": int,
    "double precision": float,
    "string": str,
}
NP_SCALAR_TYPES = {
    "keyword": np.bool_,
    "integer": np.int_,
    "double precision": np.float64,
    "string": np.str_,
}

Array = NDArray
Scalar = Union[bool, int, float, str]
Definition = Dict[str, Dict[str, Scalar]]


def fullname(t: type) -> str:
    """Convert a type to a name suitable for templating."""
    origin = get_origin(t)
    args = get_args(t)
    if origin is Literal:
        return (
            f"{Literal.__name__}[{', '.join(['"' + a + '"' for a in args])}]"
        )
    elif origin is Union:
        if len(args) == 2 and args[1] is type(None):
            return f"{Optional.__name__}[{fullname(args[0])}]"
        return f"{Union.__name__}[{', '.join([fullname(a) for a in args])}]"
    elif origin is tuple:
        return f"{Tuple.__name__}[{', '.join([fullname(a) for a in args])}]"
    elif origin is list:
        return f"{List.__name__}[{', '.join([fullname(a) for a in args])}]"
    elif origin is np.ndarray:
        return f"NDArray[np.{fullname(args[1].__args__[0])}]"
    elif origin is np.dtype:
        return str(t)
    elif isinstance(t, ForwardRef):
        return t.__forward_arg__
    elif t is Ellipsis:
        return "..."
    elif isinstance(t, type):
        return t.__qualname__
    else:
        return str(t)


def load_dfn(f) -> Tuple[Definition, List[str]]:
    """
    Load an input definition file. Returns a tuple containing
    a dictionary variables and a list of metadata attributes.
    """
    meta = list()
    vars = dict()
    var = dict()

    for line in f:
        # remove whitespace/etc from the line
        line = line.strip()

        # record flopy metadata attributes but
        # skip all other comment lines
        if line.startswith("#"):
            _, sep, tail = line.partition("flopy")
            if sep == "flopy":
                meta.append(tail.strip())
            continue

        # if we hit a newline and the parameter dict
        # is nonempty, we've reached the end of its
        # block of attributes
        if not any(line):
            if any(var):
                vars[var["name"]] = var
                var = dict()
            continue

        # split the attribute's key and value and
        # store it in the parameter dictionary
        key, _, value = line.partition(" ")
        var[key] = value

    # add the final parameter
    if any(var):
        vars[var["name"]] = var

    return vars, meta


def test_load_dfn_gwf_ic():
    dfn_path = DFNS_PATH / "gwf-ic.dfn"
    with open(dfn_path, "r") as f:
        vars, meta = load_dfn(f)

    assert len(vars) == 2
    assert set(vars.keys()) == {"export_array_ascii", "strt"}
    assert not any(meta)


def test_load_dfn_prt_prp():
    dfn_path = DFNS_PATH / "prt-prp.dfn"
    with open(dfn_path, "r") as f:
        vars, meta = load_dfn(f)

    assert len(vars) == 40
    assert len(meta) == 1


def get_template_context(
    component: str,
    subcomponent: str,
    definition: Definition,
    metadata: List[str],
) -> dict:
    """
    Convert an input definition to a template rendering context.

    TODO: pull out a class for the input definition, and expose
    this as an instance method?
    """

    def _map_var(var: dict, wrap: bool = False) -> dict:
        """
        Transform a variable from its original representation in
        an input definition to a form suitable for type hints and
        and docstrings.

        This involves expanding nested type hierarchies, converting
        input types to equivalent Python primitives and composites,
        and various other shaping.

        Notes
        -----
        If a `default_value` is not provided, keywords are `False`
        by default. Everything else is `None` by default.

        If `wrap` is true, scalars will be wrapped as records with
        keywords represented as string literals. This is useful for
        unions, to distinguish between choices having the same type.
        """
        var_ = {
            **var,
            # some flags the template uses for formatting.
            # these are ofc derivable in Python but Jinja
            # doesn't allow arbitrary expressions, and it
            # doesn't seem to have `subclass`-ish filters.
            # (we convert the variable type to string too
            # before returning, for the same reason.)
            "is_array": False,
            "is_list": False,
            "is_record": False,
            "is_union": False,
            "is_variadic": False,
            "is_choice": False,
        }
        name_ = var["name"]
        type_ = var["type"]
        shape = var.get("shape", None)
        shape = None if shape == "" else shape

        # utilities for generating records
        # as named tuples.

        def _get_record_fields(name: str) -> dict:
            """
            Call `_map_var` recursively on each field
            of the record variable with the given name.

            Notes
            -----
            This function is provided because records
            need extra processing; we remove keywords
            and 'filein'/'fileout', which are details
            of the mf6io format, not of python/flopy.
            """
            record = definition[name]
            names = record["type"].split()[1:]
            fields = {
                n: {**_map_var(field), "optional": field.get("optional", True)}
                for n, field in definition.items()
                if n in names
            }
            field_names = list(fields.keys())

            # if the record represents a file...
            if "file" in name:
                # remove filein/fileout
                for term in ["filein", "fileout"]:
                    if term in field_names:
                        fields.pop(term)

                # remove leading keyword
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

                # should just have one remaining field, the file path
                n, path = fields.popitem()
                if any(fields):
                    raise ValueError(
                        f"File record has too many fields: {fields}"
                    )
                path["type"] = PathLike
                fields[n] = path

            # if tagged, remove the leading keyword
            elif record.get("tagged", False):
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

            return fields

        # list input can have records or unions as rows.
        # lists which have a consistent record type are
        # regular, inconsistent record types irregular.
        if type_.startswith("recarray"):
            # make sure columns are defined
            names = type_.split()[1:]
            n_names = len(names)
            if n_names < 1:
                raise ValueError(f"Missing recarray definition: {type_}")

            # regular tabular/columnar data (1 record type) can be
            # defined with a nested record (i.e. explicit) or with
            # fields directly inside the recarray (implicit). list
            # data for unions/keystrings necessarily comes nested.

            def _is_explicit_record():
                return len(names) == 1 and definition[names[0]][
                    "type"
                ].startswith("record")

            def _is_implicit_record():
                types = [
                    fullname(v["type"])
                    for n, v in definition.items()
                    if n in names
                ]
                scalar_types = list(SCALAR_TYPES.keys())
                return all(t in scalar_types for t in types)

            if _is_explicit_record():
                name = names[0]
                record_type = _map_var(definition[name])
                var_["type"] = List[record_type["type"]]
                var_["children"] = {name: record_type}
                var_["is_list"] = True
            elif _is_implicit_record():
                # record implicitly defined, make it on the fly
                name = name_
                fields = _get_record_fields(name)
                record_type = Tuple[
                    tuple([f["type"] for f in fields.values()])
                ]
                record = {
                    "name": name,
                    "type": record_type,
                    "children": fields,
                    "is_array": False,
                    "is_record": True,
                    "is_union": False,
                    "is_list": False,
                    "is_variadic": False,
                    "is_choice": False,
                }
                var_["type"] = List[record_type]
                var_["children"] = {name: record}
                var_["is_list"] = True
            else:
                # irregular recarray, rows can be any of several types
                children = {n: _map_var(definition[n]) for n in names}
                var_["type"] = List[
                    Union[tuple([c["type"] for c in children.values()])]
                ]
                var_["children"] = children
                var_["is_list"] = True

        # now the basic composite types...
        # union (product) type, children are choices of records
        elif type_.startswith("keystring"):
            names = type_.split()[1:]
            children = {n: _map_var(definition[n], wrap=True) for n in names}
            var_["type"] = Union[tuple([c["type"] for c in children.values()])]
            var_["children"] = children
            var_["is_union"] = True

        # record (sum) type, children are fields
        elif type_.startswith("record"):
            name = name_
            fields = _get_record_fields(name)
            if len(fields) > 1:
                record_type = Tuple[
                    tuple([c["type"] for c in fields.values()])
                ]
            elif len(fields) == 1:
                t = list(fields.values())[0]["type"]
                # make sure we don't double-wrap tuples
                record_type = t if get_origin(t) is tuple else Tuple[(t,)]
            # TODO: if record has 1 field, accept value directly?
            var_["type"] = record_type
            var_["children"] = fields
            var_["is_record"] = True

        # are we wrapping a choice in a union?
        # if so, use a literal for the leading
        # keyword like tuple (Literal[...], T)
        elif wrap:
            name = name_
            field = _map_var(var)
            fields = {name: field}
            field_type = (
                Literal[name] if field["type"] is bool else field["type"]
            )
            record_type = (
                Tuple[Literal[name]]
                if field["type"] is bool
                else Tuple[Literal[name], field["type"]]
            )
            fields[name] = {**field, "type": field_type}
            var_["type"] = record_type
            var_["children"] = fields
            var_["is_record"] = True
            var_["is_choice"] = True

        # at this point, if it has a shape, it's an array.
        # but if it's in a record use a variadic tuple.
        elif shape is not None:
            if var.get("in_record", False):
                if type_ not in SCALAR_TYPES.keys():
                    raise TypeError(f"Unsupported repeating type: {type_}")
                var_["type"] = Tuple[SCALAR_TYPES[type_], ...]
                var_["is_variadic"] = True
            elif type_ == "string":
                var_["type"] = Tuple[SCALAR_TYPES[type_], ...]
                var_["is_variadic"] = True
            else:
                if type_ not in NP_SCALAR_TYPES.keys():
                    raise TypeError(f"Unsupported array type: {type_}")
                var_["type"] = NDArray[NP_SCALAR_TYPES[type_]]
                var_["is_array"] = True

        # finally a bog standard scalar
        else:
            var_["type"] = SCALAR_TYPES[type_]

        # make optional if needed
        if var_.get("optional", True):
            var_["type"] = (
                Optional[var_["type"]]
                if (
                    var_["type"] is not bool
                    and var_.get("optional", True)
                    and not var_.get("in_record", False)
                    and not wrap
                )
                else var_["type"]
            )

            # keywords default to False, everything else to None
            var_["default"] = var.pop(
                "default", False if var_["type"] is bool else None
            )

        # remove backslashes from description
        var_["description"] = var_.get("description", "").replace("\\", "")

        # if name is a reserved keyword,
        # add trailing underscore to it
        var_["name"] = (
            f"{var_['name']}_" if var_["name"] in kwlist else var_["name"]
        )

        return var_

    def _qualify(var: dict) -> dict:
        """
        Recursively convert the variable's type to a fully qualified string.
        """

        var["type"] = fullname(var["type"])
        children = var.get("children", dict())
        if any(children):
            var["children"] = {n: _qualify(c) for n, c in children.items()}
        return var

    def _variables(vars: dict) -> dict:
        return {
            name: _qualify(_map_var(var))
            for name, var in vars.items()
            # filter components of composites
            # since we've inflated the parent
            # types in the hierarchy already
            if not var.get("in_record", False)
        }

    def _dfn(vars: dict, meta: list) -> list:
        """
        Currently, generated classes have a `.dfn` property that
        reproduces the corresponding DFN sans a few attributes.
        """

        def _var_dfn(var: dict) -> List[str]:
            exclude = ["longname", "description"]
            return [
                " ".join([k, v]) for k, v in var.items() if k not in exclude
            ]

        return [["header"] + [attr for attr in meta]] + [
            _var_dfn(var) for var in vars.values()
        ]

    return {
        "component": component,
        "subcomponent": subcomponent,
        "variables": _variables(definition),
        "dfn": _dfn(definition, metadata),
    }


@pytest.mark.parametrize(
    "dfn, n_flat, n_nested", [("gwf-ic", 2, 2), ("prt-prp", 40, 18)]
)
def test_get_template_context(dfn, n_flat, n_nested):
    component, subcomponent = dfn.split("-")

    with open(DFNS_PATH / f"{dfn}.dfn") as f:
        variables, metadata = load_dfn(f)

    context = get_template_context(
        component, subcomponent, variables, metadata
    )
    assert context["component"] == component
    assert context["subcomponent"] == subcomponent
    assert len(context["variables"]) == n_nested
    assert len(context["dfn"]) == n_flat + 1  # +1 for metadata


from jinja2 import Environment, PackageLoader


class ComponentType(Enum):
    Package = "package"
    Simulation = "simulation"


TEMPLATE_ENV = Environment(loader=PackageLoader("flopy", "mf6/templates/"))


def get_component_type(component, subcomponent) -> ComponentType:
    if component == "sim" and subcomponent == "nam":
        return ComponentType.Simulation
    else:
        return ComponentType.Package


@pytest.mark.parametrize(
    "dfn",
    [
        # packages
        "gwf-ic",
        "prt-prp",
        "gwe-ctp",
        "gwe-cnd",
        "gwf-dis",
        "prt-mip",
        "prt-oc",
        # models
        "gwf-nam",
        "gwt-nam",
        "gwe-nam",
        "prt-nam",
        # solutions
        "sln-ims",
        "sln-ems",
        # simulation
        "sim-nam",
    ],
)
def test_render_template(dfn, function_tmpdir):
    component, subcomponent = dfn.split("-")
    comp_name = f"{component}{subcomponent}"
    comp_type = get_component_type(component, subcomponent).value
    template = TEMPLATE_ENV.get_template(f"{comp_type}.jinja")

    with open(DFNS_PATH / f"{dfn}.dfn", "r") as f:
        variables, metadata = load_dfn(f)

    context = get_template_context(
        component, subcomponent, variables, metadata
    )
    source = template.render(**context)
    source_path = function_tmpdir / f"{comp_name}.py"
    with open(source_path, "w") as f:
        f.write(source)
        run_cmd("ruff", "format", source_path, verbose=True)
