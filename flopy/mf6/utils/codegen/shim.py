"""
The purpose of this module is to keep special handling
necessary to support the current `flopy.mf6` generated
classes separate from more general templating and code
generation infrastructure.
"""

from keyword import kwlist
from pprint import pformat
from typing import List, Optional


def _cls_attrs(ctx: dict) -> List[str]:
    ctx_name = ctx["name"]

    def _attr(var: dict) -> Optional[str]:
        var_name = var["name"]
        var_kind = var.get("kind", None)
        var_block = var.get("block", None)
        var_ref = var.get("meta", dict()).get("ref", None)

        if (
            var_kind is None
            or var_kind == "scalar"
            or var_name in ["cvoptions", "output"]
            or (ctx_name.r == "dis" and var_name == "packagedata")
            or (
                var_name != "packages"
                and (ctx_name.l is not None and ctx_name.r == "nam")
            )
        ):
            return None

        if var_kind in ["list", "record", "union", "array"]:
            if not var_block:
                raise ValueError("Need block")

            if var_kind != "array":
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

            def _args():
                args = [f"'{ctx_name.r}'", f"'{var_block}'", f"'{var_name}'"]
                if ctx_name.l is not None and ctx_name.l not in [
                    "sim",
                    "sln",
                    "utl",
                    "exg",
                ]:
                    args.insert(0, f"'{ctx_name.l}6'")
                return args

            kind = var_kind if var_kind == "array" else "list"
            return f"{var_name} = {kind.title()}TemplateGenerator(({', '.join(_args())}))"

        return None

    def _dfn() -> List[List[str]]:
        dfn, meta = ctx["meta"]["dfn"]

        def _meta():
            exclude = ["subpackage", "parent_name_type"]
            return [v for v in meta if not any(p in v for p in exclude)]

        def _dfn():
            def _var(var: dict) -> List[str]:
                exclude = ["longname", "description"]
                name = var["name"]
                var_ = ctx["vars"].get(name, None)
                keys = [
                    "construct_package",
                    "construct_data",
                    "parameter_name",
                ]
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

    attrs = list(filter(None, [_attr(v) for v in ctx["vars"].values()]))

    if ctx["base"] == "MFModel":
        attrs.append(f"model_type = {ctx_name.l}")
    elif ctx["base"] == "MFPackage":
        attrs.extend(
            [
                f"package_abbr = '{ctx_name.r}'"
                if ctx_name.l == "exg"
                else f"package_abbr = '{'' if ctx_name.l in ['sln', 'sim', 'exg', None] else ctx_name.l}{ctx_name.r}'",
                f"_package_type = '{ctx_name.r}'",
                f"dfn_file_name = '{ctx_name.l}-{ctx_name.r}.dfn'"
                if ctx_name.l == "exg"
                else f"dfn_file_name = '{ctx_name.l or 'sim'}-{ctx_name.r}.dfn'",
                f"dfn = {pformat(_dfn(), indent=10)}",
            ]
        )

    return attrs


def _init_body(ctx: dict) -> List[str]:
    def _statements() -> Optional[List[str]]:
        base = ctx["base"]
        if base == "MFSimulationBase":

            def _should_set(var: dict) -> bool:
                return var["name"] not in [
                    "tdis6",
                    "models",
                    "exchanges",
                    "mxiter",
                    "solutiongroup",
                    "hpc_data",
                ]

            stmts = []
            refs = {}
            for var in ctx["vars"].values():
                ref = var.get("meta", dict()).get("ref", None)
                if not var.get("kind", None):
                    continue

                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                if _should_set(var):
                    stmts.append(f"self.name_file.{name}.set_data({name})")
                    stmts.append(f"self.{name} = self.name_file.{name}")
                if ref and ref["key"] not in refs:
                    refs[ref["key"]] = ref
                    stmts.append(
                        f"self.{ref['param']} = self._create_package('{ref['abbr']}', {ref['param']})"
                    )
        elif base == "MFModel":

            def _should_set(var: dict) -> bool:
                return var["name"] not in [
                    "export_netcdf",
                    "nc_filerecord",
                    "packages",
                ]

            stmts = []
            refs = {}
            for var in ctx["vars"].values():
                ref = var.get("meta", dict()).get("ref", None)
                if not var.get("kind", None):
                    continue

                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                if _should_set(var):
                    stmts.append(f"self.name_file.{name}.set_data({name})")
                    stmts.append(f"self.{name} = self.name_file.{name}")
                if ref and ref["key"] not in refs:
                    refs[ref["key"]] = ref
                    stmts.append(
                        f"self.{ref['param']} = self._create_package('{ref['abbr']}', {ref['param']})"
                    )
        elif base == "MFPackage":

            def _should_build(var: dict) -> bool:
                if var.get("meta", dict()).get("ref", None) and ctx[
                    "name"
                ] != (
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
                    "parent_model_or_package",
                    "parent_file",
                    "modelname",
                    "model_nam_file",
                    "export_netcdf",
                    "nc_filerecord",
                    "method",
                    "interpolation_method_single",
                    "sfac",
                    "output",
                ]:
                    return False
                return True

            stmts = []
            refs = {}
            for var in ctx["vars"].values():
                name = var["name"]
                ref = var.get("meta", dict()).get("ref", None)
                if name in kwlist:
                    name = f"{name}_"

                if _should_build(var):
                    if ref and ctx["name"] == (None, "nam"):
                        stmts.append(
                            f"self.{'_' if ref else ''}{ref['key']} = self.build_mfdata('{ref['key']}', None)"
                        )
                    else:
                        # hack...
                        _name = name[:-1] if name.endswith("_") else name
                        if _name == "steady_state":
                            _name = "steady-state"
                        stmts.append(
                            f"self.{'_' if ref else ''}{name} = self.build_mfdata('{_name}', {name if var.get('init_param', True) else 'None'})"
                        )

                if ref and ref["key"] not in refs and ctx["name"].r != "nam":
                    refs[ref["key"]] = ref
                    stmts.append(
                        f"self._{ref['key']} = self.build_mfdata('{ref['key']}', None)"
                    )
                    stmts.append(
                        f"self._{ref['abbr']}_package = self.build_child_package('{ref['abbr']}', {ref['val']}, '{ref['param']}', self._{ref['key']})"
                    )

        return stmts

    return list(filter(None, _statements()))


def _init_skip(ctx: dict) -> List[str]:
    name = ctx["name"]
    base = name.base
    if base == "MFSimulationBase":
        skip = [
            "tdis6",
            "models",
            "exchanges",
            "mxiter",
            "solutiongroup",
        ]
        refs = ctx.get("meta", dict()).get("refs", dict())
        return skip
    elif base == "MFModel":
        skip = ["packages", "export_netcdf", "nc_filerecord"]
        refs = ctx.get("meta", dict()).get("refs", dict())
        if any(refs) and ctx["name"] != (None, "nam"):
            for key in refs.keys():
                if ctx["vars"].get(key, None):
                    skip.append(key)
        return skip
    elif base == "MFPackage":
        if name.r == "nam":
            return ["export_netcdf", "nc_filerecord"]
        elif name == ("utl", "ts"):
            return ["method", "interpolation_method_single", "sfac"]
        else:
            return []


def _is_context(o) -> bool:
    d = dict(o)
    return "name" in d and "base" in d


def _parent(ctx: dict) -> str:
    ref = ctx["meta"].get("ref", None)
    if ref:
        return ref["parent"]
    name = ctx["name"]
    ref = ctx["meta"].get("ref", None)
    if name == ("sim", "nam"):
        return None
    elif name.l is None or name.r is None or name.l in ["sim", "exg", "sln"]:
        return "simulation"
    elif ref:
        if name.l == "utl" and name.r == "hpc":
            return "simulation"
        return "package"
    return "model"


def _replace_refs_exg(ctx: dict) -> dict:
    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = ctx["vars"].get(key, None)
            if not key_var:
                continue
            ctx["vars"][key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
                "default": None,
            }
    return ctx


def _replace_refs_pkg(ctx: dict) -> dict:
    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = ctx["vars"].get(key, None)
            if not key_var:
                continue
            ctx["vars"][key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
                "default": None,
            }
    return ctx


def _replace_refs_mdl(ctx: dict) -> dict:
    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = ctx["vars"].get(key, None)
            if not key_var:
                continue
            ctx["vars"][key] = {
                **key_var,
                "name": ref["val"],
                "description": ref.get("description", None),
                "ref": ref,
            }
    return ctx


def _replace_refs_sim(ctx: dict) -> dict:
    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs) and ctx["name"] != (None, "nam"):
        for key, ref in refs.items():
            key_var = ctx["vars"].get(key, None)
            if not key_var:
                continue
            ctx["vars"][key] = {
                **key_var,
                "name": ref["param"],
                "description": ref.get("description", None),
                "ref": ref,
                "default": None,
            }
    return ctx


def _transform_context(o):
    ctx = dict(o)
    ctx_name = ctx["name"]
    ctx_base = ctx_name.base
    if ctx_base == "MFSimulationBase":
        return _replace_refs_sim(ctx)
    elif ctx_base == "MFModel":
        return _replace_refs_mdl(ctx)
    elif ctx_base == "MFPackage":
        if ctx_name.l == "exg":
            return _replace_refs_exg(ctx)
        else:
            return _replace_refs_pkg(ctx)


SHIM = {
    "keep_none": ["default", "block", "metadata"],
    "quote_str": ["default"],
    "set_pairs": [
        (
            _is_context,
            [
                ("cls_attrs", _cls_attrs),
                ("init_skip", _init_skip),
                ("init_body", _init_body),
                ("parent", _parent),
            ],
        ),
    ],
    "transform": [(_is_context, _transform_context)],
}
