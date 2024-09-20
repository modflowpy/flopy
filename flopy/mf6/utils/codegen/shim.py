"""
The purpose of this module is to keep special handling
necessary to support the current `flopy.mf6` generated
classes separate from more general templating and code
generation infrastructure.
"""

import os
from keyword import kwlist
from typing import List, Optional

from flopy.mf6.utils.codegen.dfn import Metadata
from flopy.mf6.utils.codegen.var import VarKind


def _is_ctx(o) -> bool:
    """Whether the object is an input context."""
    d = dict(o)
    return "name" in d and "base" in d


def _is_var(o) -> bool:
    """Whether the object is a input context variable."""
    d = dict(o)
    return "name" in d and "_type" in d


def _is_init_param(o) -> bool:
    """Whether the object is an `__init__` method parameter."""
    d = dict(o)
    return not d.get("ref", None)


def _is_container_init_param(o) -> bool:
    """
    Whether the object is a parameter of the corresponding
    package container class. This is only relevant for some
    subpackage contexts.
    """
    return True


def _add_exg_params(ctx: dict) -> dict:
    """
    Add initializer parameters for an exchange input context.
    Exchanges need different parameters than a typical package.
    """
    vars_ = ctx["variables"].copy()
    vars_ = {
        "loading_package": {
            "name": "loading_package",
            "_type": "bool",
            "description": (
                "Do not set this parameter. It is intended for "
                "debugging and internal processing purposes only."
            ),
            "default": False,
            "init_param": True,
        },
        "exgtype": {
            "name": "exgtype",
            "_type": "str",
            "default": f"{ctx['name'].r[:3].upper()}6-{ctx['name'].r[:3].upper()}6",
            "description": "The exchange type.",
            "init_param": True,
        },
        "exgmnamea": {
            "name": "exgmnamea",
            "_type": "str",
            "description": "The name of the first model in the exchange.",
            "default": None,
            "init_param": True,
        },
        "exgmnameb": {
            "name": "exgmnameb",
            "_type": "str",
            "description": "The name of the second model in the exchange.",
            "default": None,
            "init_param": True,
        },
        **vars_,
        "filename": {
            "name": "filename",
            "_type": "pathlike",
            "description": "File name for this package.",
            "default": None,
            "init_param": True,
        },
        "pname": {
            "name": "pname",
            "_type": "str",
            "description": "Package name for this package.",
            "default": None,
            "init_param": True,
        },
    }

    if ctx["references"]:
        for key, ref in ctx["references"].items():
            if key not in vars_:
                continue
            vars_[ref["val"]] = {
                "name": ref["val"],
                "description": ref.get("description", None),
                "reference": ref,
                "init_param": True,
                "default": None,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["variables"] = vars_
    return ctx


def _add_pkg_params(ctx: dict) -> dict:
    """Add variables for a package context."""
    vars_ = ctx["variables"].copy()

    if ctx["name"].r == "nam":
        init_skip = ["export_netcdf", "nc_filerecord"]
    elif ctx["name"] == ("utl", "ts"):
        init_skip = ["method", "interpolation_method_single", "sfac"]
    else:
        init_skip = []
    for k in init_skip:
        var = vars_.get(k, None)
        if var:
            var["init_param"] = False
            var["init_skip"] = True
            vars_[k] = var

    vars_ = {
        "loading_package": {
            "name": "loading_package",
            "_type": "bool",
            "description": (
                "Do not set this variable. It is intended for debugging "
                "and internal processing purposes only."
            ),
            "default": False,
            "init_param": True,
        },
        **vars_,
        "filename": {
            "name": "filename",
            "_type": "str",
            "description": "File name for this package.",
            "default": None,
            "init_param": True,
        },
        "pname": {
            "name": "pname",
            "_type": "str",
            "description": "Package name for this package.",
            "default": None,
            "init_param": True,
        },
    }

    if ctx["name"].l == "utl":
        vars_["parent_file"] = {
            "name": "parent_file",
            "_type": "pathlike",
            "description": (
                "Parent package file that references this package. Only needed "
                "for utility packages (mfutl*). For example, mfutllaktab package "
                "must have a mfgwflak package parent_file."
            ),
        }

    if ctx["references"]:
        for key, ref in ctx["references"].items():
            if key not in vars_:
                continue
            vars_[key] = {
                "name": ref["val"],
                "description": ref.get("description", None),
                "reference": ref,
                "init_param": ctx["name"].r != "nam",
                "default": None,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["variables"] = vars_
    return ctx


def _add_mdl_params(ctx: dict) -> dict:
    """Add variables for a model context."""
    vars_ = ctx["variables"].copy()
    init_skip = ["packages", "export_netcdf", "nc_filerecord"]
    for k in init_skip:
        var = vars_.get(k, None)
        if var:
            var["init_param"] = False
            var["init_skip"] = True
            vars_[k] = var
    vars_ = {
        "modelname": {
            "name": "modelname",
            "_type": "str",
            "description": "The name of the model.",
            "default": "model",
            "init_param": True,
        },
        "model_nam_file": {
            "name": "model_nam_file",
            "_type": "pathlike",
            "default": None,
            "description": (
                "The relative path to the model name file from model working folder."
            ),
            "init_param": True,
        },
        "version": {
            "name": "version",
            "_type": "str",
            "description": "The version of modflow",
            "default": "mf6",
            "init_param": True,
        },
        "exe_name": {
            "name": "exe_name",
            "_type": "str",
            "description": "The executable name.",
            "default": "mf6",
            "init_param": True,
        },
        "model_rel_path": {
            "name": "model_rel_path",
            "_type": "pathlike",
            "description": "The model working folder path.",
            "default": os.curdir,
            "init_param": True,
        },
        **vars_,
    }

    if ctx["references"]:
        for key, ref in ctx["references"].items():
            if key not in vars_:
                continue
            vars_[key] = {
                "name": ref["val"],
                "description": ref.get("description", None),
                "reference": ref,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["variables"] = vars_
    return ctx


def _add_sim_params(ctx: dict) -> dict:
    """Add variables for a simulation context."""
    vars_ = ctx["variables"].copy()
    init_skip = [
        "tdis6",
        "models",
        "exchanges",
        "mxiter",
        "solutiongroup",
    ]
    for k in init_skip:
        var = vars_.get(k, None)
        if var:
            var["init_param"] = False
            var["init_skip"] = True
            vars_[k] = var
    vars_ = {
        "sim_name": {
            "name": "sim_name",
            "_type": "str",
            "default": "sim",
            "description": "Name of the simulation.",
            "init_param": True,
        },
        "version": {
            "name": "version",
            "_type": "str",
            "default": "mf6",
            "init_param": True,
        },
        "exe_name": {
            "name": "exe_name",
            "_type": "pathlike",
            "default": "mf6",
            "init_param": True,
        },
        "sim_ws": {
            "name": "sim_ws",
            "_type": "pathlike",
            "default": ".",
            "init_param": True,
        },
        "verbosity_level": {
            "name": "verbosity_level",
            "_type": "int",
            "default": 1,
            "init_param": True,
        },
        "write_headers": {
            "name": "write_headers",
            "_type": "bool",
            "default": True,
            "init_param": True,
        },
        "use_pandas": {
            "name": "use_pandas",
            "_type": "bool",
            "default": True,
            "init_param": True,
        },
        "lazy_io": {
            "name": "lazy_io",
            "_type": "bool",
            "default": False,
            "init_param": True,
        },
        **vars_,
    }

    if ctx["references"] and ctx["name"] != (None, "nam"):
        for key, ref in ctx["references"].items():
            if key not in vars_:
                continue
            vars_[key] = {
                "name": ref["param"],
                "description": ref.get("description", None),
                "reference": ref,
                "init_param": True,
                "default": None,
            }

    ctx["variables"] = vars_
    return ctx


def _add_parent_param(ctx: dict) -> dict:
    vars_ = ctx["variables"]
    parent = ctx["parent"]
    if ctx.get("reference"):
        parent = f"parent_{parent}"
    ctx["variables"] = {
        parent: {
            "name": parent,
            "_type": str(ctx["parent"]),
            "description": f"Parent {parent} that this package is part of.",
            "init_param": True,
        },
        **vars_,
    }
    return ctx


def _add_init_params(o):
    """Add context-specific `__init__()` method parameters."""
    ctx = dict(o)
    if ctx["name"].base == "MFSimulationBase":
        ctx = _add_sim_params(ctx)
    elif ctx["name"].base == "MFModel":
        ctx = _add_mdl_params(ctx)
        ctx = _add_parent_param(ctx)
    elif ctx["name"].base == "MFPackage":
        if ctx["name"].l == "exg":
            ctx = _add_exg_params(ctx)
        else:
            ctx = _add_pkg_params(ctx)
        ctx = _add_parent_param(ctx)
    return ctx


def _transform_context(o):
    # add vars depending on the
    # specific type of context.
    # do this as a transform so
    # we can control the order
    # they appear in `__init__`
    # or other method signatures.
    return _add_init_params(o)


def _var_attrs(ctx: dict) -> str:
    """
    Get class attributes for the context.
    """
    ctx_name = ctx["name"]

    def _attr(var: dict) -> Optional[str]:
        var_name = var["name"]
        var_kind = var.get("kind", None)
        var_block = var.get("block", None)

        if var_kind is None or var_kind == VarKind.Scalar.value:
            return None

        if var_name in ["cvoptions", "output"]:
            return None

        if (
            ctx_name.l is not None and ctx_name.r == "nam"
        ) and var_name != "packages":
            return None

        if var_kind in [
            VarKind.List.value,
            VarKind.Record.value,
            VarKind.Union.value,
        ]:
            if not var_block:
                raise ValueError("Need block")
            args = [f"'{ctx_name.r}'", f"'{var_block}'", f"'{var_name}'"]
            if ctx_name.l is not None and ctx_name.l not in [
                "sim",
                "sln",
                "utl",
                "exg",
            ]:
                args.insert(0, f"'{ctx_name.l}6'")
            return f"{var_name} = ListTemplateGenerator(({', '.join(args)}))"

        if var_kind == VarKind.Array.value:
            if not var_block:
                raise ValueError("Need block")
            args = [f"'{ctx_name.r}'", f"'{var_block}'", f"'{var_name}'"]
            if ctx_name.l is not None and ctx_name.l not in [
                "sim",
                "sln",
                "utl",
                "exg",
            ]:
                args.insert(0, f"'{ctx_name.l}6'")
            return f"{var_name} = ArrayTemplateGenerator(({', '.join(args)}))"

        return None

    attrs = [_attr(var) for var in ctx["variables"].values()]
    return "\n    ".join([a for a in attrs if a])


def _init_body(ctx: dict) -> str:
    """
    Get the `__init__` method body for the context.
    """

    def _super_call() -> Optional[str]:
        """
        Whether to pass the variable to `super().__init__()`
        by name in the `__init__` method.
        """

        if ctx["base"] == "MFPackage":
            parent = ctx["parent"]
            if ctx["reference"]:
                parent = f"parent_{parent}"
            pkgtyp = ctx["name"].r
            args = [
                parent,
                f"'{pkgtyp}'",
                "filename",
                "pname",
                "loading_package",
                "**kwargs",
            ]
        elif ctx["base"] == "MFModel":
            parent = ctx["parent"]
            mdltyp = ctx["name"].l
            args = [
                parent,
                f"'{mdltyp}6'",
                "modelname=modelname",
                "model_nam_file=model_nam_file",
                "version=version",
                "exe_name=exe_name",
                "model_rel_path=model_rel_path",
                "**kwargs",
            ]
        elif ctx["base"] == "MFSimulationBase":
            args = [
                "sim_name=sim_name",
                "version=version",
                "exe_name=exe_name",
                "sim_ws=sim_ws",
                "verbosity_level=verbosity_level",
                "write_headers=write_headers",
                "lazy_io=lazy_io",
                "use_pandas=use_pandas",
            ]

        return f"super().__init__({', '.join(args)})"

    def _should_assign(var: dict) -> bool:
        """
        Whether to assign arguments to self in the
        `__init__` method. if this is false, assume
        the template has conditionals for any more
        involved initialization needs.
        """
        return var["name"] in ["exgtype", "exgmnamea", "exgmnameb"]

    def _should_build(var: dict) -> bool:
        """
        Whether to call `build_mfdata()` on the variable.
        in the `__init__` method.
        """
        if var.get("reference", None):
            return False
        name = var["name"]
        if name in [
            "simulation",
            "model",
            "package",
            "parent_model",
            "parent_package",
            "loading_package",
            "parent_model_or_package",
            "exgtype",
            "exgmnamea",
            "exgmnameb",
            "filename",
            "pname",
            "parent_file",
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
            "export_netcdf",
            "nc_filerecord",
            "method",
            "interpolation_method_single",
            "sfac",
            "output",
        ]:
            return False
        return True

    def _body() -> Optional[str]:
        if ctx["base"] in ["MFSimulationBase", "MFModel"]:
            statements = []
            references = {}
            for var in ctx["variables"].values():
                if not var.get("kind", None) or var.get("init_skip", False):
                    continue
                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"
                ref = var.get("reference", None)
                statements.append(f"self.name_file.{name}.set_data({name})")
                statements.append(f"self.{name} = self.name_file.{name}")
                if ref and ref["key"] not in references:
                    references[ref["key"]] = ref
                    statements.append(
                        f"self._{ref['param']} = self._create_package('{ref['abbr']}', {ref['param']})"
                    )
        else:
            statements = []
            references = {}
            for var in ctx["variables"].values():
                name = var["name"]
                ref = var.get("reference", None)
                if name in kwlist:
                    name = f"{name}_"

                if _should_assign(var):
                    statements.append(f"self.{name} = {name}")
                    if name == "exgmnameb":
                        statements.append(
                            "simulation.register_exchange_file(self)"
                        )
                elif _should_build(var):
                    lname = name[:-1] if name.endswith("_") else name
                    statements.append(
                        f"self.{'_' if ref else ''}{name} = self.build_mfdata('{lname}', {name if var.get('init_param', True) else 'None'})"
                    )

                if (
                    ref
                    and ref["key"] not in references
                    and ctx["name"].r != "nam"
                ):
                    references[ref["key"]] = ref
                    statements.append(
                        f"self._{ref['key']} = self.build_mfdata('{ref['key']}', None)"
                    )
                    statements.append(
                        f"self._{ref['abbr']}_package = self.build_child_package('{ref['abbr']}', {ref['val']}, '{ref['param']}', self._{ref['key']})"
                    )

        return (
            None
            if not any(statements)
            else "\n".join(["        " + s for s in statements])
        )

    sections = [_super_call(), _body()]
    sections = [s for s in sections if s]
    return "\n".join(sections)


def _dfn(o) -> List[Metadata]:
    """
    Get a list of the class' original definition attributes
    as a partial, internal reproduction of the DFN contents.

    Notes
    -----
    Currently, generated classes have a `.dfn` property that
    reproduces the corresponding DFN sans a few attributes.
    This represents the DFN in raw form, before adapting to
    Python, consolidating nested types, etc.
    """

    ctx = dict(o)
    dfn = ctx["definition"]

    def _fmt_var(var: dict) -> List[str]:
        exclude = ["longname", "description"]

        def _fmt_name(k, v):
            return v.replace("-", "_") if k == "name" else v

        return [
            " ".join([k, str(_fmt_name(k, v))]).strip()
            for k, v in var.items()
            if k not in exclude
        ]

    meta = dfn.metadata or list()
    _dfn = []
    for name, var in dfn:
        var_ = ctx["variables"].get(name, None)
        if var_ and "construct_package" in var_:
            var["construct_package"] = var_["construct_package"]
            var["construct_data"] = var_["construct_data"]
            var["parameter_name"] = var_["parameter_name"]
        _dfn.append((name, var))
    return [["header"] + [m for m in meta]] + [_fmt_var(v) for k, v in _dfn]


def _qual_base(ctx: dict):
    base = ctx["base"]
    if base == "MFSimulationBase":
        module = "mfsimbase"
    elif base == "MFModel":
        module = "mfmodel"
    else:
        module = "mfpackage"
    return f"{module}.{base}"


SHIM = {
    "keep_none": ["default", "block", "metadata"],
    "quote_str": ["default"],
    "set_pairs": [
        (
            _is_ctx,
            [
                ("dfn", _dfn),
                ("qual_base", _qual_base),
                ("var_attrs", _var_attrs),
                ("init_body", _init_body),
            ],
        ),
        (
            _is_var,
            [
                ("init_param", _is_init_param),
                ("container_init_param", _is_container_init_param),
            ],
        ),
    ],
    "transform": [(_is_ctx, _transform_context)],
}
"""
Arguments for `renderable` as applied to `Context`
to support the current `flopy.mf6` input framework.
"""
