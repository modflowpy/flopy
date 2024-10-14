"""
The purpose of this module is to keep special handling
necessary to support the current `flopy.mf6` generated
classes separate from more general templating and code
generation infrastructure. It has no dependency on the
rest of the `flopy.mf6.utils.codegen` module.
"""

import os
from keyword import kwlist
from typing import List, Optional


def _is_context(o) -> bool:
    """Whether the object is an input context."""
    d = dict(o)
    return "name" in d and "base" in d


def _is_var(o) -> bool:
    """Whether the object is an input context variable."""
    d = dict(o)
    return "name" in d and "kind" in d


def _is_init_param(o) -> bool:
    """Whether the object is an `__init__` method parameter."""
    d = dict(o)
    if d.get("ref", None):
        return False
    if d["name"] in ["output"]:
        return False
    return True


def _is_container_init_param(o) -> bool:
    """
    Whether the object is a parameter of the corresponding
    package container class. This is only relevant for some
    subpackage contexts.
    """
    d = dict(o)
    if d["name"] in ["output"]:
        return False
    return True


def _set_exg_vars(ctx: dict) -> dict:
    """
    Modify variables for an exchange context.
    """
    vars_ = ctx["vars"].copy()
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

    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = vars_.get(key, None)
            if not key_var:
                continue
            vars_[key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
                "init_param": True,
                "default": None,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["vars"] = vars_
    return ctx


def _set_pkg_vars(ctx: dict) -> dict:
    """Modify variables for a package context."""
    vars_ = ctx["vars"].copy()

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

    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = vars_.get(key, None)
            if not key_var:
                continue
            vars_[key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
                "init_param": ctx["name"].r != "nam",
                "default": None,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["vars"] = vars_
    return ctx


def _set_mdl_vars(ctx: dict) -> dict:
    """Modify variables for a model context."""
    vars_ = ctx["vars"].copy()
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

    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = vars_.get(key, None)
            if not key_var:
                continue
            vars_[key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
                "construct_package": ref["abbr"],
                "construct_data": ref["val"],
                "parameter_name": ref["param"],
            }

    ctx["vars"] = vars_
    return ctx


def _set_sim_vars(ctx: dict) -> dict:
    """Modify variables for a simulation context."""
    vars_ = ctx["vars"].copy()
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

    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs) and ctx["name"] != (None, "nam"):
        for key, ref in refs.items():
            key_var = vars_.get(key, None)
            if not key_var:
                continue
            vars_[key] = {
                **key_var,
                "name": ref["param"],
                "description": ref.get("description", None),
                "ref": ref,
                "init_param": True,
                "init_skip": True,
                "default": None,
            }

    ctx["vars"] = vars_
    return ctx


def _set_parent(ctx: dict) -> dict:
    vars_ = ctx["vars"]
    parent = ctx["parent"]
    ctx["vars"] = {
        parent: {
            "name": parent,
            "_type": str(ctx["parent"]),
            "description": f"Parent {parent} that this package is part of.",
            "init_param": True,
        },
        **vars_,
    }
    return ctx


def _map_context(o):
    """
    Transform an input context as needed depending on its type.

    Notes
    -----
    This includes adding extra variables for the `__init__` method;
    This is done as a transform instead of with `set_pairs` so we
    can control the order they appear in the method signature.
    """

    ctx = dict(o)
    if ctx["name"].base == "MFSimulationBase":
        ctx = _set_sim_vars(ctx)
    elif ctx["name"].base == "MFModel":
        ctx = _set_mdl_vars(ctx)
        ctx = _set_parent(ctx)
    elif ctx["name"].base == "MFPackage":
        ctx = (
            _set_exg_vars(ctx)
            if ctx["name"].l == "exg"
            else _set_pkg_vars(ctx)
        )
        ctx = _set_parent(ctx)
    return ctx


def _class_attrs(ctx: dict) -> str:
    """
    Get class attributes for the context.
    """
    ctx_name = ctx["name"]

    def _attr(var: dict) -> Optional[str]:
        var_name = var["name"]
        var_kind = var.get("kind", None)
        var_block = var.get("block", None)
        var_ref = var.get("meta", dict()).get("ref", None)

        if var_kind is None or var_kind == "scalar":
            return None

        if var_name in ["cvoptions", "output"]:
            return None

        if (
            ctx_name.l is not None and ctx_name.r == "nam"
        ) and var_name != "packages":
            return None

        if ctx_name.r == "dis" and var_name == "packagedata":
            return None

        if var_kind in ["list", "record", "union"]:
            if not var_block:
                raise ValueError("Need block")

            if var_ref:
                # if the variable is a subpackage reference, use the original key
                # (which has been replaced already with the referenced variable)
                args = [
                    f"'{ctx_name.r}'",
                    f"'{var_block}'",
                    f"'{var_ref['key']}'",
                ]
                if ctx_name.l is not None and ctx_name.l not in [
                    "sim",
                    "sln",
                    "utl",
                    "exg",
                ]:
                    args.insert(0, f"'{ctx_name.l}6'")
                return f"{var_ref['key']} = ListTemplateGenerator(({', '.join(args)}))"

            args = [f"'{ctx_name.r}'", f"'{var_block}'", f"'{var_name}'"]
            if ctx_name.l is not None and ctx_name.l not in [
                "sim",
                "sln",
                "utl",
                "exg",
            ]:
                args.insert(0, f"'{ctx_name.l}6'")
            return f"{var_name} = ListTemplateGenerator(({', '.join(args)}))"

        elif var_kind == "array":
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

    attrs = [_attr(v) for v in ctx["vars"].values()]
    return "\n    ".join([a for a in attrs if a])


def _init_body(ctx: dict) -> str:
    """
    Get the `__init__` method body for the context.
    """

    def _super() -> Optional[str]:
        """
        Whether to pass the variable to `super().__init__()`
        by name in the `__init__` method.
        """

        if ctx["base"] == "MFPackage":
            args = [
                ctx["parent"]
                if ctx.get("meta", dict()).get("ref", None)
                else ctx['parent'],
                f"'{ctx['name'].r}'",
                "filename",
                "pname",
                "loading_package",
                "**kwargs",
            ]
        elif ctx["base"] == "MFModel":
            args = [
                ctx["parent"],
                f"'{ctx['name'].l}6'",
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

    def _assign(var: dict) -> bool:
        """
        Whether to assign arguments to self in the
        `__init__` method. if this is false, assume
        the template has conditionals for any more
        involved initialization needs.
        """
        return var["name"] in ["exgtype", "exgmnamea", "exgmnameb"]

    def _build(var: dict) -> bool:
        """
        Whether to call `build_mfdata()` on the variable.
        in the `__init__` method.
        """
        if var.get("meta", dict()).get("ref", None) and ctx["name"] != (
            None,
            "nam",
        ):
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
            for var in ctx["vars"].values():
                ref = var.get("meta", dict()).get("ref", None)
                if not var.get("kind", None):
                    continue

                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                if not var.get("init_skip", False):
                    statements.append(
                        f"self.name_file.{name}.set_data({name})"
                    )
                    statements.append(f"self.{name} = self.name_file.{name}")
                if ref and ref["key"] not in references:
                    references[ref["key"]] = ref
                    statements.append(
                        f"self.{ref['param']} = self._create_package('{ref['abbr']}', {ref['param']})"
                    )
        else:
            statements = []
            references = {}
            for var in ctx["vars"].values():
                name = var["name"]
                ref = var.get("meta", dict()).get("ref", None)
                if name in kwlist:
                    name = f"{name}_"

                if _assign(var):
                    statements.append(f"self.{name} = {name}")
                    if name == "exgmnameb":
                        statements.append(
                            "simulation.register_exchange_file(self)"
                        )
                elif _build(var):
                    if ref and ctx["name"] == (None, "nam"):
                        statements.append(
                            f"self.{'_' if ref else ''}{ref['key']} = self.build_mfdata('{ref['key']}', None)"
                        )
                    else:
                        # hack...
                        _name = name[:-1] if name.endswith("_") else name
                        if _name == "steady_state":
                            _name = "steady-state"
                        statements.append(
                            f"self.{'_' if ref else ''}{name} = self.build_mfdata('{_name}', {name if var.get('init_param', True) else 'None'})"
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

    sections = [_super(), _body()]
    sections = [s for s in sections if s]
    return "\n".join(sections)


def _dfn(o) -> List[List[str]]:
    """
    Get a list of the class' original definition attributes
    as a partial, internal reproduction of the DFN contents.

    Notes
    -----
    Currently, generated classes have a `.dfn` property that
    reproduces the corresponding DFN sans a few attributes.
    Once `mfstructure.py` etc is reworked to introspect the
    context classes instead of this property, it can go.
    """

    ctx = dict(o)
    dfn, meta = ctx["meta"]["dfn"]

    def _meta():
        exclude = ["subpackage", "parent_name_type"]
        return [v for v in meta if not any(p in v for p in exclude)]

    def _dfn():
        def _var(var: dict) -> List[str]:
            exclude = ["longname", "description"]
            name = var["name"]
            var_ = ctx["vars"].get(name, None)
            keys = ["construct_package", "construct_data", "parameter_name"]
            if var_ and keys[0] in var_:
                for k in keys:
                    var[k] = var_[k]
            return [
                " ".join([k, v]).strip()
                for k, v in var.items()
                if k not in exclude
            ]

        return [_var(var) for var in dfn]

    return [["header"] + _meta()] + _dfn()


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
            _is_context,
            [
                ("dfn", _dfn),
                ("qual_base", _qual_base),
                ("class_attrs", _class_attrs),
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
    "transform": [(_is_context, _map_context)],
}
"""
Arguments for `renderable` as applied to `Context`
to support the current `flopy.mf6` input framework.
"""
