from enum import Enum
from keyword import kwlist
from typing import Any, List, Optional

from jinja2 import pass_context


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
            if ctx_name == ("sim", "nam"):
                return "MFSimulationBase"
            if ctx_name[1] is None:
                return "MFModel"
            return "MFPackage"

        def title(ctx_name) -> str:
            """
            The input context's unique title. This is not
            identical to `f"{l}{r}` in some cases, but it
            remains unique. The title is substituted into
            the file name and class name.
            """
            if ctx_name == ("sim", "nam"):
                return "simulation"
            l, r = ctx_name
            if l is None:
                return r
            if r is None:
                return l
            if l == "sim":
                return r
            if l in ["sln", "exg"]:
                return r
            return l + r

        def package_abbr(ctx_name) -> str:
            if ctx_name[0] in ["sim", "sln", "exg", None]:
                return ctx_name[1]
            return "".join(ctx_name)

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
            """The input context class name prefix, e.g. 'MF' or 'Modflow'."""
            base = Filters.Cls.base(ctx_name)
            return "MF" if base == "MFSimulationBase" else "Modflow"

        @pass_context
        def parent(ctx, ctx_name) -> str:
            """The input context's parent context type, if it can have a parent."""
            subpkg = ctx.get("subpackage", None)
            if subpkg:
                return subpkg["parent"]
            if ctx_name == ("sim", "nam"):
                return None
            elif (
                ctx_name[0] is None
                or ctx_name[1] is None
                or ctx_name[0] in ["sim", "exg", "sln"]
            ):
                return "simulation"
            return "model"

        @pass_context
        def skip_init(ctx, ctx_name) -> List[str]:
            """Variables to skip in input context's `__init__` method."""
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
                return ["packages", "export_netcdf", "nc_filerecord"]
            else:
                if ctx_name[1] == "nam":
                    return ["export_netcdf", "nc_filerecord"]
                elif ctx_name == ("utl", "ts"):
                    return ["method", "interpolation_method_single", "sfac"]
                return []

    class Var:
        def untag(var: dict) -> dict:
            """
            If the variable is a tagged record, remove the leading
            tag field. If the variable is a tagged file path input
            record, remove both leading tag and 'filein'/'fileout'
            keyword following it.
            """
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
            """
            Get a readable representation of the variable's type.
            TODO: eventually replace this with a proper `type` in
            the variable spec when we add type hints
            """
            _type = var["type"]
            shape = var.get("shape", None)
            children = var.get("children", None)
            if children:
                if _type == "list":
                    if len(children) == 1:
                        first = list(children.values())[0]
                        if first["type"] in ["record", "union"]:
                            return f"[{Filters.Var.type(first)}]"
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
        def attrs(ctx, variables) -> List[str]:
            """
            Map the context's input variables to corresponding class attributes,
            where applicable. TODO: this should get much simpler if we can drop
            all the `ListTemplateGenerator`/`ArrayTemplateGenerator` attributes.
            Ultimately I (WPB) think we can aim for context classes consisting
            of just a class attr for each variable, with anything complicated
            happening in a decorator or base class.
            """
            from modflow_devtools.dfn import _MF6_SCALARS

            name = ctx["name"]
            base = Filters.Cls.base(name)

            def _attr(var: dict) -> Optional[str]:
                var_name = var["name"]
                var_type = var["type"]
                var_shape = var.get("shape", None)
                var_block = var.get("block", None)
                var_subpkg = var.get("subpackage", None)

                if (
                    (var_type in _MF6_SCALARS and not var_shape)
                    or var_name in ["cvoptions", "output"]
                    or (name[1] == "dis" and var_name == "packagedata")
                    or (
                        var_name != "packages"
                        and (name[0] is not None and name[1] == "nam")
                    )
                ):
                    return None

                is_array = (
                    var_type in ["string", "integer", "double precision"]
                    and var_shape
                )
                is_composite = var_type in ["list", "record", "union"]
                if is_array or is_composite:
                    if not var_block:
                        raise ValueError("Need block")

                    if not is_array:
                        if var_subpkg:
                            # if the variable is a subpackage reference, use the original key
                            # (which has been replaced already with the referenced variable)
                            args = [
                                f"'{name[1]}'",
                                f"'{var_block}'",
                                f"'{var_subpkg['key']}'",
                            ]
                            if name[0] is not None and name[0] not in [
                                "sim",
                                "sln",
                                "utl",
                                "exg",
                            ]:
                                args.insert(0, f"'{name[0]}6'")
                            return f"{var_subpkg['key']} = ListTemplateGenerator(({', '.join(args)}))"

                    def _args():
                        args = [
                            f"'{name[1]}'",
                            f"'{var_block}'",
                            f"'{var_name}'",
                        ]
                        if name[0] is not None and name[0] not in [
                            "sim",
                            "sln",
                            "utl",
                            "exg",
                        ]:
                            args.insert(0, f"'{name[0]}6'")
                        return args

                    kind = "array" if is_array else "list"
                    return f"{var_name} = {kind.title()}TemplateGenerator(({', '.join(_args())}))"

                return None

            attrs = list(filter(None, [_attr(v) for v in variables.values()]))

            if base == "MFPackage":
                attrs.extend(
                    [
                        f"package_abbr = '{Filters.Cls.package_abbr(name)}'",
                        f"_package_type = '{name[1]}'",
                        f"dfn_file_name = '{name[0]}-{name[1]}.dfn'"
                        if name[0] == "exg"
                        else f"dfn_file_name = '{name[0] or 'sim'}-{name[1]}.dfn'",
                    ]
                )

            return attrs

        @pass_context
        def init(ctx, vars) -> List[str]:
            """
            Map the context's input variables to statements in the class'
            `__init__` method body, if applicable. TODO: consider how we
            can dispatch as necessary based on a variable's type instead
            of explicitly choosing among:

            - self.var = var
            - self.var = self.build_mfdata(...)
            - self.subppkg_var = self._create_package(...)
            - ...

            """
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

                        subpkg = var.get("subpackage", None)
                        if subpkg and subpkg["key"] not in refs:
                            refs[subpkg["key"]] = subpkg
                            args = f"'{subpkg['abbr']}', {subpkg['param']}"
                            stmts.append(
                                f"self.{subpkg['param']} = self._create_package({args})"
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

                        subpkg = var.get("subpackage", None)
                        if subpkg and subpkg["key"] not in refs:
                            refs[subpkg["key"]] = subpkg
                            args = f"'{subpkg['abbr']}', {subpkg['param']}"
                            stmts.append(
                                f"self.{subpkg['param']} = self._create_package({args})"
                            )
                elif base == "MFPackage":

                    def _should_build(var: dict) -> bool:
                        subpkg = var.get("subpackage", None)
                        if subpkg and ctx_name != (None, "nam"):
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

                        subpkg = var.get("subpackage", None)
                        if _should_build(var):
                            if subpkg and ctx["name"] == (None, "nam"):
                                stmts.append(
                                    f"self.{'_' if subpkg else ''}{subpkg['key']} "
                                    f"= self.build_mfdata('{subpkg['key']}', None)"
                                )
                            else:
                                _name = (
                                    name[:-1] if name.endswith("_") else name
                                )
                                name = name.replace("-", "_")
                                stmts.append(
                                    f"self.{'_' if subpkg else ''}{name} "
                                    f"= self.build_mfdata('{_name}', {name})"
                                )

                        if (
                            subpkg
                            and subpkg["key"] not in refs
                            and ctx["name"][1] != "nam"
                        ):
                            refs[subpkg["key"]] = subpkg
                            stmts.append(
                                f"self._{subpkg['key']} "
                                f"= self.build_mfdata('{subpkg['key']}', None)"
                            )
                            args = (
                                f"'{subpkg['abbr']}', {subpkg['val']}, "
                                f"'{subpkg['param']}', self._{subpkg['key']}"
                            )
                            stmts.append(
                                f"self._{subpkg['abbr']}_package "
                                f"= self.build_child_package({args})"
                            )

                return stmts

            return list(filter(None, _statements()))

    def safe_name(name: str) -> str:
        """
        Make sure a string is safe to use as a variable name in Python code.
        If the string is a reserved keyword, add a trailing underscore to it.
        Also replace any hyphens with underscores.
        """
        return (f"{name}_" if name in kwlist else name).replace("-", "_")

    def escape_trailing_underscore(v: str) -> str:
        """If the string has a trailing underscore, escape it."""
        return f"{v[:-1]}\\\\_" if v.endswith("_") else v

    def value(v: Any) -> str:
        """
        Format a value to appear in the RHS of an assignment or argument-
        passing expression: if it's an enum, get its value; if it's `str`,
        quote it.
        """
        v = try_get_enum_value(v)
        if isinstance(v, str) and v[0] not in ["'", '"']:
            v = f"'{v}'"
        return v
