import os
from os import PathLike
from typing import Iterable, Optional, Union, get_args, get_origin

from numpy.typing import ArrayLike
from pandas import DataFrame

from flopy.mf6.utils.codegen.spec import Var, VarKind


def _add_exg_vars(ctx):
    """
    Add initializer parameters for an exchange input context.
    Exchanges need different parameters than a typical package.
    """
    d = dict(ctx)
    a = d["name"].r[:3]
    b = d["name"].r[:3]
    default = f"{a.upper()}6-{b.upper()}6"
    vars_ = d["variables"].copy()
    vars_ = {
        "parent": Var(
            name="parent",
            _type="MFSimulation",
            description=(
                "Simulation that this package is a part of. "
                "Package is automatically added to simulation "
                "when it is initialized."
            ),
        ),
        "loading_package": Var(
            name="loading_package",
            _type=bool,
            description=(
                "Do not set this parameter. It is intended for "
                "debugging and internal processing purposes only."
            ),
            default=False,
        ),
        "exgtype": Var(
            name="exgtype",
            _type=str,
            default=default,
            description="The exchange type.",
        ),
        "exgmnamea": Var(
            name="exgmnamea",
            _type=str,
            description="The name of the first model in the exchange.",
        ),
        "exgmnameb": Var(
            name="exgmnameb",
            _type=str,
            description="The name of the second model in the exchange.",
        ),
        **vars_,
        "filename": Var(
            name="filename",
            _type=Union[str, PathLike],
            description="File name for this package.",
        ),
        "pname": Var(
            name="pname",
            _type=str,
            description="Package name for this package.",
        ),
    }

    if d["references"]:
        for key, ref in d["references"].items():
            if key not in vars_:
                continue
            vars_[ref["val"]] = Var(
                name=ref["val"],
                description=ref.get("description", None),
                reference=ref,
            )

    d["variables"] = vars_
    return d


def _add_pkg_vars(ctx):
    """Add variables for a package context."""
    d = dict(ctx)
    parent_name = "parent"
    vars_ = d["variables"].copy()
    vars_ = {
        parent_name: Var(
            name=parent_name,
            _type=d["parent"],
            description="Parent that this package is part of.",
        ),
        "loading_package": Var(
            name="loading_package",
            _type=bool,
            description=(
                "Do not set this variable. It is intended for debugging "
                "and internal processing purposes only."
            ),
            default=False,
        ),
        **vars_,
        "filename": Var(
            name="filename",
            _type=str,
            description="File name for this package.",
        ),
        "pname": Var(
            name="pname",
            _type=str,
            description="Package name for this package.",
        ),
    }

    if d["name"].l == "utl":
        vars_["parent_file"] = Var(
            name="parent_file",
            _type=Union[str, PathLike],
            description=(
                "Parent package file that references this package. Only needed "
                "for utility packages (mfutl*). For example, mfutllaktab package "
                "must have a mfgwflak package parent_file."
            ),
        )

    if d["references"] and d["name"] != (None, "nam"):
        for key, ref in d["references"].items():
            if key not in vars_:
                continue
            vars_[ref["val"]] = Var(
                name=ref["val"],
                description=ref.get("description", None),
                reference=ref,
            )

    d["variables"] = vars_
    return d


def _add_mdl_vars(ctx):
    """Add variables for a model context."""
    d = dict(ctx)
    vars_ = d["variables"].copy()
    vars_ = {
        "simulation": Var(
            name="simulation",
            _type="MFSimulation",
            description=(
                "Simulation that this model is part of. "
                "Model is automatically added to the simulation "
                "when it is initialized."
            ),
        ),
        "modelname": Var(
            name="modelname",
            _type=str,
            description="The name of the model.",
            default="model",
        ),
        "model_nam_file": Var(
            name="model_nam_file",
            _type=Optional[Union[str, PathLike]],
            description=(
                "The relative path to the model name file from model working folder."
            ),
        ),
        "version": Var(
            name="version",
            _type=str,
            description="The version of modflow",
            default="mf6",
        ),
        "exe_name": Var(
            name="exe_name",
            _type=str,
            description="The executable name.",
            default="mf6",
        ),
        "model_rel_path": Var(
            name="model_ws",
            _type=Union[str, PathLike],
            description="The model working folder path.",
            default=os.curdir,
        ),
        **vars_,
    }

    if d["references"]:
        for key, ref in d["references"].items():
            if key not in vars_:
                continue
            vars_[ref["val"]] = Var(
                name=ref["val"],
                description=ref.get("description", None),
                reference=ref,
            )

    d["variables"] = vars_
    return d


def _add_sim_vars(ctx):
    """Add variables for a simulation context."""
    d = dict(ctx)
    vars_ = d["variables"].copy()
    skip_init = [
        "tdis6",
        "models",
        "exchanges",
        "mxiter",
        "solutiongroup",
    ]
    for k in skip_init:
        var = vars_.get(k, None)
        if var:
            var["init_param"] = False
            vars_[k] = var
    vars_ = {
        "sim_name": Var(
            name="sim_name",
            _type=str,
            default="sim",
            description="Name of the simulation.",
        ),
        "version": Var(
            name="version",
            _type=str,
            default="mf6",
        ),
        "exe_name": Var(
            name="exe_name",
            _type=Union[str, PathLike],
            default="mf6",
        ),
        "sim_ws": Var(
            name="sim_ws",
            _type=Union[str, PathLike],
            default=os.curdir,
        ),
        "verbosity_level": Var(
            name="verbosity_level",
            _type=int,
            default=1,
        ),
        "write_headers": Var(
            name="write_headers",
            _type=bool,
            default=True,
        ),
        "use_pandas": Var(
            name="use_pandas",
            _type=bool,
            default=True,
        ),
        "lazy_io": Var(
            name="lazy_io",
            _type=bool,
            default=False,
        ),
        **vars_,
    }

    if d["references"]:
        for key, ref in d["references"].items():
            if key not in vars_:
                continue
            vars_[ref["val"]] = Var(
                name=ref["val"],
                description=ref.get("description", None),
                reference=ref,
            )

    d["variables"] = vars_
    return d


def _add_ctx_vars(o):
    d = dict(o)
    if d["name"].base == "MFSimulationBase":
        return _add_sim_vars(d)
    elif d["name"].base == "MFModel":
        return _add_mdl_vars(d)
    elif d["name"].base == "MFPackage":
        if d["name"].l == "exg":
            return _add_exg_vars(d)
        else:
            return _add_pkg_vars(d)
    return d


def _is_ctx(o) -> bool:
    d = dict(o)
    return "name" in d and "base" in d


def _is_var(o) -> bool:
    d = dict(o)
    return "name" in d and "_type" in d


def _init_param(o) -> bool:
    """Whether the var is an `__init__` method parameter."""
    d = dict(o)
    if d["name"] in [
        "packages",
        "tdis6",
        "models",
        "exchanges",
        "mxiter",
        "solutiongroup",
    ]:
        return False
    if d.get("ref", None):
        return False
    return True


def _init_assign(o) -> bool:
    """
    Whether to assign arguments to self in the
    `__init__` method. if this is false, assume
    the template has conditionals for any more
    involved initialization needs.
    """
    d = dict(o)
    return d["name"] in ["exgtype", "exgnamea", "exgnameb"]


def _init_build(o) -> bool:
    """
    Whether to call `build_mfdata()` on the variable.
    in the `__init__` method.
    """
    d = dict(o)
    ref = d.get("ref", None)
    if ref:
        return False
    if d["name"] in [
        "parent",
        "loading_package",
        "exgtype",
        "exgnamea",
        "exgnameb",
        "filename",
        "pname",
        "parent_file" "simulation",
        "modelname",
        "model_nam_file",
        "version",
        "exe_name",
        "model_rel_path",
        "sim_name",
        "sim_ws",
        "verbosity_level",
        "write_headers",
        "use_pandas",
        "lazy_io",
    ]:
        return False
    return True


def _init_super(o) -> bool:
    """
    Whether to pass the variable to `super().__init__()`
    by name in the `__init__` method."""
    d = dict(o)
    return d["name"] in [
        "parent",
        "loading_package",
        "filename",
        "pname",
        "simulation",
        "modelname",
        "model_nam_file",
        "version",
        "exe_name",
        "model_rel_path",
        "sim_name",
        "sim_ws",
        "verbosity_level",
        "write_headers",
        "use_pandas",
        "lazy_io",
    ]


def _class_attr(o) -> bool:
    """Whether to add a class attribute for the variable."""
    d = dict(o)
    if d.get("ref", None):
        return True
    kind = VarKind.from_type(d["_type"])
    if kind != VarKind.Scalar:
        return True
    return False


def _kind(o) -> VarKind:
    # the variable's general shape. because Jinja
    # doesn't allow arbitrary expressions, and it
    # doesn't seem to have a subclass test filter,
    # we need this for template conditional exprs.
    d = dict(o)
    return VarKind.from_type(d["_type"])


def _loose_type(o) -> type:
    """
    Derive a "loose" (lenient) typing attribute
    from the variable's type, which can be more
    accepting than the variable's specification.
    Used for init method params, while the spec
    itself (in e.g. the class docstring) can be
    the more descriptive (i.e. unmodified) type.
    """
    d = dict(o)
    if d["kind"] == VarKind.Array:
        # arrays can be described as NDArray with a
        # type parameter, or ndarray with type and
        # shape parameters, while init params can
        # be specified more loosely as ArrayLike.
        return ArrayLike
    if d["kind"] == VarKind.List:
        # lists can be iterables regardless whether
        # regular. if regular then accept dataframe
        _iterable = Iterable[get_args(d["_type"])[0]]
        children = list(d["children"].values())
        if (
            any(children)
            and VarKind.from_type(children[0]["_type"]) == VarKind.Union
        ):
            return _iterable
        return Union[_iterable, DataFrame]
    # TODO transient lists:
    # map of lists by stress period, or...
    # iterable appled to all stress periods
    return d["_type"]


SHIM = {
    "keep_none": ["default", "block"],
    "quote_str": ["default"],
    "type_name": ["_type"],
    "transform": [
        # context-specific parameters
        # for the `__init__()` method.
        # do it as a `transform` (not
        # `add_entry`) so we are able
        # to control the param order.
        (_is_ctx, _add_ctx_vars)
    ],
    "add_entry": [
        (
            _is_var,
            [
                ("kind", _kind),
                ("loose_type", _loose_type),
                ("init_param", _init_param),
                ("init_assign", _init_assign),
                ("init_build", _init_build),
                ("init_super", _init_super),
                ("class_attr", _class_attr),
            ],
        ),
    ],
}
"""
Arguments for `renderable` as applied to `Context`
to support the current `flopy.mf6` input framework.
"""
