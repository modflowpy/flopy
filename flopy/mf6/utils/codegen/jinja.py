from enum import Enum
from keyword import kwlist
from pprint import pformat
from typing import Any, List, Optional

from jinja2 import pass_context


def try_get_enum_value(v: Any) -> Any:
    """
    Get the enum's value if the object is an instance
    of an enumeration, otherwise return it unaltered.
    """
    return v.value if isinstance(v, Enum) else v


class Filters:
    @pass_context
    def cls_attrs(ctx, ctx_name) -> List[str]:
        def _attr(var: dict) -> Optional[str]:
            var_name = var["name"]
            var_kind = try_get_enum_value(var.get("kind", None))
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
                    args = [
                        f"'{ctx_name.r}'",
                        f"'{var_block}'",
                        f"'{var_name}'",
                    ]
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

        if ctx_name.base == "MFModel":
            attrs.append(f"model_type = {ctx_name.l}")
        elif ctx_name.base == "MFPackage":
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

    @pass_context
    def init_body(ctx, ctx_name) -> List[str]:
        def _statements() -> Optional[List[str]]:
            base = ctx_name.base
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

                    if (
                        ref
                        and ref["key"] not in refs
                        and ctx["name"].r != "nam"
                    ):
                        refs[ref["key"]] = ref
                        stmts.append(
                            f"self._{ref['key']} = self.build_mfdata('{ref['key']}', None)"
                        )
                        stmts.append(
                            f"self._{ref['abbr']}_package = self.build_child_package('{ref['abbr']}', {ref['val']}, '{ref['param']}', self._{ref['key']})"
                        )

            return stmts

        return list(filter(None, _statements()))

    @pass_context
    def parent(ctx, ctx_name):
        ref = ctx["meta"].get("ref", None)
        if ref:
            return ref["parent"]
        if ctx_name == ("sim", "nam"):
            return None
        elif (
            ctx_name.l is None
            or ctx_name.r is None
            or ctx_name.l in ["sim", "exg", "sln"]
        ):
            return "simulation"
        elif ref:
            if ctx_name.l == "utl" and ctx_name.r == "hpc":
                return "simulation"
            return "package"
        return "model"

    def prefix(ctx_name):
        return "MF" if ctx_name.base == "MFSimulationBase" else "Modflow"

    @pass_context
    def skip(ctx, ctx_name):
        base = ctx_name.base
        if base == "MFSimulationBase":
            return [
                "tdis6",
                "models",
                "exchanges",
                "mxiter",
                "solutiongroup",
            ]
        elif base == "MFModel":
            skip = ["packages", "export_netcdf", "nc_filerecord"]
            refs = ctx.get("meta", dict()).get("refs", dict())
            if any(refs) and ctx["name"] != (None, "nam"):
                for k in refs.keys():
                    if ctx["vars"].get(k, None):
                        skip.append(k)
            return skip
        else:
            if ctx_name.r == "nam":
                return ["export_netcdf", "nc_filerecord"]
            elif ctx_name == ("utl", "ts"):
                return ["method", "interpolation_method_single", "sfac"]
            return []
