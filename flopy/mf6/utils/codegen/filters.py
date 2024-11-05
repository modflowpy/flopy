from enum import Enum
from keyword import kwlist
from pprint import pformat
from typing import Any, List, Optional

from jinja2 import pass_context

from flopy.mf6.utils.codegen.dfn import _SCALARS


def try_get_enum_value(v: Any) -> Any:
    """
    Get the enum's value if the object is an instance
    of an enumeration, otherwise return it unaltered.
    """
    return v.value if isinstance(v, Enum) else v


class Filters:
    class Cls:
        def base(ctx_name) -> str:
            """Base class from which the input context should inherit."""
            _, r = ctx_name
            if ctx_name == ("sim", "nam"):
                return "MFSimulationBase"
            if r is None:
                return "MFModel"
            return "MFPackage"

        def title(ctx_name) -> str:
            """
            The input context's unique title. This is not
            identical to `f"{l}{r}` in some cases, but it
            remains unique. The title is substituted into
            the file name and class name.
            """
            l, r = ctx_name
            if (l, r) == ("sim", "nam"):
                return "simulation"
            if l is None:
                return r
            if r is None:
                return l
            if l == "sim":
                return r
            if l in ["sln", "exg"]:
                return r
            return l + r

        def description(ctx_name) -> str:
            """A description of the input context."""
            l, r = ctx_name
            base = Filters.Cls.base(ctx_name)
            title = Filters.Cls.title(ctx_name).title()
            if base == "MFPackage":
                return f"Modflow{title} defines a {r.upper()} package."
            elif base == "MFModel":
                return f"Modflow{title} defines a {l.upper()} model."
            elif base == "MFSimulationBase":
                return (
                    "MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation."
                    " A MFSimulation object must be created before creating any of the MODFLOW"
                    " 6 model objects."
                )

        def prefix(ctx_name) -> str:
            base = Filters.Cls.base(ctx_name)
            return "MF" if base == "MFSimulationBase" else "Modflow"

        @pass_context
        def parent(ctx, ctx_name) -> str:
            subpkg = ctx["meta"].get("subpkg", None)
            if subpkg:
                return subpkg["parent"]
            if ctx_name == ("sim", "nam"):
                return None
            elif (
                ctx_name.l is None
                or ctx_name.r is None
                or ctx_name.l in ["sim", "exg", "sln"]
            ):
                return "simulation"
            elif subpkg:
                if ctx_name.l == "utl" and ctx_name.r == "hpc":
                    return "simulation"
                return "package"
            return "model"

        @pass_context
        def skip(ctx, ctx_name) -> List[str]:
            base = Filters.Cls.base(ctx_name)
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
                refs = ctx.get("meta", dict()).get("fkeys", dict())
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

    class Var:
        def maybe_file(var: dict) -> dict:
            name = var["name"]
            tagged = var.get("tagged", False)
            fields = var.get("children", None)

            if not fields:
                return var

            # if tagged, remove the leading keyword
            elif tagged:
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

            # if the record represents a file...
            elif "file" in name:
                # remove filein/fileout
                field_names = list(fields.keys())
                for term in ["filein", "fileout"]:
                    if term in field_names:
                        fields.pop(term)

                # remove leading keyword
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

            var["children"] = fields
            return var

        def type(var: dict) -> str:
            _type = var["type"]
            shape = var.get("shape", None)
            children = var.get("children", None)
            if children:
                if _type == "list":
                    children = ", ".join(
                        [v["name"] for v in children.values()]
                    )
                    return f"[{children}]"
                elif _type == "record":
                    children = ", ".join(
                        [v["name"] for v in children.values()]
                    )
                    return f"({children})"
                elif _type == "union":
                    return " | ".join([v["name"] for v in children.values()])
            if shape:
                return f"[{_type}]"
            return var["type"]

    class Vars:
        @pass_context
        def attrs(ctx, vars) -> List[str]:
            ctx_name = ctx["name"]
            base = Filters.Cls.base(ctx_name)

            def _attr(var: dict) -> Optional[str]:
                var_name = var["name"]
                var_type = var["type"]
                var_shape = var.get("shape", None)
                var_block = var.get("block", None)
                var_ref = var.get("fkey", None)

                if (
                    (var_type in _SCALARS and not var_shape)
                    or var_name in ["cvoptions", "output"]
                    or (ctx_name.r == "dis" and var_name == "packagedata")
                    or (
                        var_name != "packages"
                        and (ctx_name.l is not None and ctx_name.r == "nam")
                    )
                ):
                    return None

                is_array = (
                    var_type in ["integer", "double precision"] and var_shape
                )
                is_composite = var_type in ["list", "record", "union"]
                if is_array or is_composite:
                    if not var_block:
                        raise ValueError("Need block")

                    if not is_array:
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

                    kind = "array" if is_array else "list"
                    return f"{var_name} = {kind.title()}TemplateGenerator(({', '.join(_args())}))"

                return None

            def _dfn() -> List[List[str]]:
                dfn, meta = ctx["meta"]["dfn"]

                def _meta():
                    exclude = ["subpackage", "parent_name_type"]
                    return [
                        v for v in meta if not any(p in v for p in exclude)
                    ]

                def _dfn():
                    def _var(var: dict) -> List[str]:
                        exclude = ["longname", "description"]
                        name = var["name"]
                        var_ = vars.get(name, None)
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

            attrs = list(filter(None, [_attr(v) for v in vars.values()]))

            if base == "MFModel":
                attrs.append(f"model_type = {ctx_name.l}")
            elif base == "MFPackage":
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
        def init(ctx, vars) -> List[str]:
            ctx_name = ctx["name"]
            base = Filters.Cls.base(ctx_name)

            def _statements() -> Optional[List[str]]:
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
                    for var in vars.values():
                        name = var["name"]
                        if name in kwlist:
                            name = f"{name}_"

                        if _should_set(var):
                            stmts.append(
                                f"self.name_file.{name}.set_data({name})"
                            )
                            stmts.append(
                                f"self.{name} = self.name_file.{name}"
                            )

                        fkey = var.get("fkey", None)
                        if fkey and fkey["key"] not in refs:
                            refs[fkey["key"]] = fkey
                            stmts.append(
                                f"self.{fkey['param']} = self._create_package('{fkey['abbr']}', {fkey['param']})"
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
                    for var in vars.values():
                        name = var["name"]
                        if name in kwlist:
                            name = f"{name}_"

                        if _should_set(var):
                            stmts.append(
                                f"self.name_file.{name}.set_data({name})"
                            )
                            stmts.append(
                                f"self.{name} = self.name_file.{name}"
                            )

                        fkey = var.get("fkey", None)
                        if fkey and fkey["key"] not in refs:
                            refs[fkey["key"]] = fkey
                            stmts.append(
                                f"self.{fkey['param']} = self._create_package('{fkey['abbr']}', {fkey['param']})"
                            )
                elif base == "MFPackage":

                    def _should_build(var: dict) -> bool:
                        if var.get("fkey", None) and ctx_name != (
                            None,
                            "nam",
                        ):
                            return False
                        return var["name"] not in [
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
                        ]

                    stmts = []
                    refs = {}
                    for var in vars.values():
                        name = var["name"]
                        if name in kwlist:
                            name = f"{name}_"

                        fkey = var.get("fkey", None)
                        if _should_build(var):
                            if fkey and ctx["name"] == (None, "nam"):
                                stmts.append(
                                    f"self.{'_' if fkey else ''}{fkey['key']} = self.build_mfdata('{fkey['key']}', None)"
                                )
                            else:
                                _name = (
                                    name[:-1] if name.endswith("_") else name
                                )
                                name = name.replace("-", "_")
                                stmts.append(
                                    f"self.{'_' if fkey else ''}{name} = self.build_mfdata('{_name}', {name})"
                                )

                        if (
                            fkey
                            and fkey["key"] not in refs
                            and ctx["name"].r != "nam"
                        ):
                            refs[fkey["key"]] = fkey
                            stmts.append(
                                f"self._{fkey['key']} = self.build_mfdata('{fkey['key']}', None)"
                            )
                            stmts.append(
                                f"self._{fkey['abbr']}_package = self.build_child_package('{fkey['abbr']}', {fkey['val']}, '{fkey['param']}', self._{fkey['key']})"
                            )

                return stmts

            return list(filter(None, _statements()))

    def safe_str(v: str) -> str:
        return (f"{v}_" if v in kwlist else v).replace("-", "_")

    def escape_trailing_underscore(v: str) -> str:
        return f"{v[:-1]}\\\\_" if v.endswith("_") else v

    def value(v: Any) -> str:
        v = try_get_enum_value(v)
        if isinstance(v, str) and v[0] not in ["'", '"']:
            v = f"'{v}'"
        return v
