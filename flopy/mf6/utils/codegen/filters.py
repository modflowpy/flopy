import sys
from enum import Enum
from keyword import kwlist
from pathlib import Path
from pprint import pformat
from typing import Any, List, Optional

from boltons.iterutils import default_enter, remap


def _try_get_enum_value(v: Any) -> Any:
    """
    Get the enum's value if the object is an instance
    of an enumeration, otherwise return it unaltered.
    """
    return v.value if isinstance(v, Enum) else v


def _get_vars(d: dict, developmode: bool = True) -> dict[str, dict]:
    vars_ = dict()

    def visit(p, k, v):
        if (
            isinstance(v, dict)
            and "type" in v
            # support 'prerelease' for now but it's been deprecated for 'developmode'
            and (developmode or not v.get("developmode", v.get("prerelease", False)))
        ):
            vars_[k] = v
        return True

    def enter(p, k, v):
        if isinstance(v, dict) and "type" in v:
            return (v, False)
        return default_enter(p, k, v)

    dd = d.copy()
    del dd["legacy_dfn"]
    del dd["legacy_meta"]
    remap(dd, enter=enter, visit=visit)
    return vars_


def base(component_name: tuple[str, str]) -> str:
    """Base class from which the input context should inherit."""
    if component_name == ("sim", "nam"):
        return "MFSimulationBase"
    if component_name[1] is None:
        return "MFModel"
    return "MFPackage"


def title(component_name: tuple[str, str]) -> str:
    """
    The input context's unique title. This is not
    identical to `f"{l}{r}` in some cases, but it
    remains unique. The title is substituted into
    the file name and class name.
    """
    if component_name == ("sim", "nam"):
        return "simulation"
    l, r = component_name
    if l is None:
        return r
    if r is None:
        return l
    if l == "sim":
        return r
    if l in ["sln", "exg"]:
        return r
    return l + r


def package_abbr(component_name: tuple[str, str]) -> str:
    if component_name[0] in ["sim", "sln", "exg", None]:
        return component_name[1]
    return "".join(component_name)


def description(component_name: tuple[str, str]) -> str:
    """A description of the input context."""
    l, r = component_name
    component_base = base(component_name)
    component_title = title(component_name).title()
    if component_base == "MFPackage":
        return f"Modflow{component_title} defines a {r.upper()} package."
    elif component_base == "MFModel":
        return f"Modflow{component_title} defines a {l.upper()} model."
    elif component_base == "MFSimulationBase":
        return (
            "MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation."
            " A MFSimulation object must be created before creating any of the MODFLOW"
            " 6 model objects."
        )


def prefix(component_name: tuple[str, str]) -> str:
    """The input context class name prefix, e.g. 'MF' or 'Modflow'."""
    component_base = base(component_name)
    return "MF" if component_base == "MFSimulationBase" else "Modflow"


def dfn_file_name(component_name: tuple[str, str]) -> str:
    if component_name[0] == "exg":
        return f"{'-'.join(component_name)}.dfn"
    if tuple(component_name) in [
        (None, "mvr"),
        (None, "gnc"),
    ]:
        return f"gwf-{component_name[1]}.dfn"
    if tuple(component_name) in [(None, "mvt")]:
        return f"gwt-{component_name[1]}.dfn"
    return f"{component_name[0] or 'sim'}-{component_name[1]}.dfn"


def parent(dfn: dict, component_name: tuple[str, str]) -> str:
    # TODO should be no longer needed when parents are explicit in dfns
    """The input context's parent context type, if it can have a parent."""
    subpkg = dfn.get("ref", None)
    if subpkg:
        return subpkg["parent"]
    if component_name == ("sim", "nam"):
        return None
    elif component_name[1] is None or component_name[0] in [None, "sim", "exg", "sln"]:
        return "simulation"
    return "model"


def skip_init(component_name: tuple[str, str]) -> List[str]:
    """Variables to skip in input context's `__init__` method."""
    component_base = base(component_name)
    if component_base == "MFSimulationBase":
        return [
            "tdis6",
            "models",
            "exchanges",
            "mxiter",
            "solutiongroup",
        ]
    elif component_base == "MFModel":
        return ["packages"]
    else:
        # if component_name[1] == "nam":
        #     return ["export_netcdf", "nc_filerecord"]
        if component_name == ("utl", "ts"):
            return ["method", "interpolation_method_single", "sfac"]
        return []


def untag(var: dict) -> dict:
    """
    If the variable is a tagged record, remove the leading
    tag field. If the variable is a tagged file path input
    record, remove both leading tag and 'filein'/'fileout'
    keyword following it.
    """
    name = var["name"]
    tagged = var.get("tagged", False)
    fields = var.get("fields", None)

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

    var["fields"] = fields
    return var


def type(var: dict) -> str:
    """
    Get a readable representation of the variable's type.
    TODO: eventually replace this with a proper `type` in
    the variable spec when we add type hints
    """
    _type = var["type"]
    shape = var.get("shape", None)
    children_vars = children(var)
    if children_vars:
        if _type == "recarray":
            if len(children_vars) == 1:
                first = list(children_vars.values())[0]
                if first["type"] in ["record", "keystring"]:
                    return f"[{type(first)}]"
            children_vars = ", ".join([v["name"] for v in children_vars.values()])
            return f"[{children_vars}]"
        elif _type == "record":
            children_vars = ", ".join([v["name"] for v in children_vars.values()])
            return f"({children_vars})"
        elif _type == "keystring":
            return " | ".join([v["name"] for v in children_vars.values()])
    elif shape:
        return f"[{_type}]"
    return var["type"]


def children(var: dict) -> Optional[dict]:
    _type = var["type"]
    item = var.get("item", None)
    fields = var.get("fields", None)
    choices = var.get("choices", None)
    if item:
        assert _type == "recarray"
        return {item["name"]: item}
    if fields:
        assert _type == "record"
        return fields
    if choices:
        assert _type == "keystring"
        return choices
    return None


def default_value(var: dict) -> Any:
    _default = var.get("default", None)
    if _default is not None:
        return _default
    return None


def variables(dfn: dict, developmode: bool = True) -> List[str]:
    return _get_vars(dfn, developmode=developmode)


def attrs(
    dfn: dict, component_name: tuple[str, str], developmode: bool = True
) -> List[str]:
    """
    Map the context's input variables to corresponding class attributes,
    where applicable. TODO: this should get much simpler if we can drop
    all the `ListTemplateGenerator`/`ArrayTemplateGenerator` attributes.
    """
    try:
        from modflow_devtools.dfn import SCALAR_TYPES
    except ImportError:
        from modflow_devtools.dfn import _SCALAR_TYPES as SCALAR_TYPES  # noqa: PLC2701

    component_base = base(component_name)
    component_vars = variables(dfn, developmode=developmode)

    def _attr(var: dict) -> Optional[str]:
        var_name = var["name"]
        var_type = var["type"]
        var_block = var["block"]
        var_shape = var.get("shape", None)
        var_subpkg = var.get("ref", None)

        if (
            (var_type in SCALAR_TYPES and not var_shape)
            or var_name in ["cvoptions", "output"]
            # or (component_name[1] == "dis" and var_name == "packagedata")
        ):
            return None

        if var_subpkg:
            # if the variable is a subpackage reference, use the original key
            # (which has been replaced already with the referenced variable)
            args = [
                f"'{component_name[1]}'",
                f"'{var_block}'",
                f"'{var_subpkg['key']}'",
            ]
            if component_name[0] not in [
                None,
                "sim",
                "sln",
                "utl",
                "exg",
            ]:
                args.insert(0, f"'{component_name[0]}6'")
            return f"{var_subpkg['key']} = ListTemplateGenerator(({', '.join(args)}))"
        is_array = var_type in ["string", "integer", "double precision"] and var_shape
        is_composite = var_type in ["recarray", "record", "keystring"]
        if is_array or is_composite:

            def _args():
                args = [
                    f"'{component_name[1]}'",
                    f"'{var_block}'",
                    f"'{var_name}'",
                ]
                if component_name[0] is not None and component_name[0] not in [
                    "sim",
                    "sln",
                    "utl",
                    "exg",
                ]:
                    args.insert(0, f"'{component_name[0]}6'")
                return args

            kind = "array" if is_array else "list"
            return (
                f"{var_name} = {kind.title()}TemplateGenerator(({', '.join(_args())}))"
            )

        return None

    attrs = list(filter(None, [_attr(v) for v in component_vars.values()]))

    dfn_header = ["header"]
    if dfn.get("multi", None):
        dfn_header.append("multi-package")
    if dfn.get("advanced", None):
        dfn_header.append("package-type advanced-stress-package")
    if dfn.get("sln", None):
        dfn_header.append(["solution_package", "*"])

    dfn_dir = Path(__file__).parents[2] / "data" / "dfn"

    def _dfn(definition, metadata) -> List[List[str]]:
        def _meta():
            exclude = ["subpackage", "parent_name_type"]
            return [v for v in metadata if not any(p in v for p in exclude)]

        def __dfn():
            def _var(var: dict) -> List[str]:
                exclude = ["longname", "description"]
                name = var["name"]
                subpkg = dfn.get("fkeys", dict()).get(name, None)
                if subpkg:
                    var["construct_package"] = subpkg["abbr"]
                    var["construct_data"] = subpkg["val"]
                    var["parameter_name"] = subpkg["param"]
                return [
                    " ".join([k, v]).strip() for k, v in var.items() if k not in exclude
                ]

            return [_var(var) for var in list(definition.values(multi=True))]

        return [["header"] + _meta()] + __dfn()

    def _filter_metadata(metadata):
        meta_ = list()
        for m in metadata:
            if "multi" in m:
                meta_.append(m)
            elif "solution" in m:
                s = m.split()
                meta_.append([s[0], s[2]])
            elif "package-type" in m:
                s = m.split()
                meta_.append(" ".join(s))
        return meta_

    dfn_copy = dfn.copy()
    legacy_dfn = dfn_copy.pop("legacy_dfn", {})
    legacy_meta = dfn_copy.pop("legacy_meta", [])
    legacy_dfn = _dfn(legacy_dfn, _filter_metadata(legacy_meta))
    if component_base == "MFPackage":
        attrs.extend(
            [
                f"package_abbr = '{package_abbr(component_name)}'",
                f"_package_type = '{component_name[1]}'",
                f"dfn_file_name = '{dfn_file_name(component_name)}'",
                f"dfn = {pformat(legacy_dfn, indent=10, width=sys.maxsize)}",
                f"spec = {pformat(dfn_copy, indent=10, width=sys.maxsize)}",
            ]
        )

    return attrs


def init(
    dfn: dict, component_name: tuple[str, str], developmode: bool = True
) -> List[str]:
    component_base = base(component_name)
    component_vars = variables(dfn, developmode=developmode)

    def _statements() -> Optional[List[str]]:
        if component_base == "MFSimulationBase":

            def _should_set(var: dict) -> bool:
                return var["name"] not in [
                    "tdis6",
                    "models",
                    "exchanges",
                    "mxiter",
                    "solutiongroup",
                ]

            stmts = []
            refs = {}
            for var in component_vars.values():
                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                subpkg = var.get("ref", None)

                if _should_set(var):
                    if name not in ["hpc_data"]:
                        stmts.append(f"self.name_file.{name}.set_data({name})")
                    if not subpkg:
                        stmts.append(f"self.{name} = self.name_file.{name}")

                if subpkg and subpkg["key"] not in refs:
                    refs[subpkg["key"]] = subpkg
                    args = f"'{subpkg['abbr']}', {subpkg['param']}"
                    stmts.append(
                        f"self.{subpkg['param']} = self._create_package({args})"
                    )
        elif component_base == "MFModel":

            def _should_set(var: dict) -> bool:
                return var["name"] not in [
                    "packages",
                ]

            stmts = []
            refs = {}
            for var in component_vars.values():
                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                if _should_set(var):
                    stmts.append(f"self.name_file.{name}.set_data({name})")
                    stmts.append(f"self.{name} = self.name_file.{name}")

                subpkg = var.get("ref", None)
                if subpkg and subpkg["key"] not in refs:
                    refs[subpkg["key"]] = subpkg
                    args = f"'{subpkg['abbr']}', {subpkg['param']}"
                    stmts.append(
                        f"self.{subpkg['param']} = self._create_package({args})"
                    )
        elif component_base == "MFPackage":

            def _should_build(var: dict) -> bool:
                subpkg = var.get("ref", None)
                if subpkg and component_name != (None, "nam"):
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
                    "method",
                    "interpolation_method_single",
                    "sfac",
                    "output",
                ]

            stmts = []
            refs = {}
            for var in component_vars.values():
                name = var["name"]
                if name in kwlist:
                    name = f"{name}_"

                subpkg = var.get("ref", None)
                if _should_build(var):
                    if subpkg and component_name == (None, "nam"):
                        stmts.append(
                            f"self.{'_' if subpkg else ''}{subpkg['key']} "
                            f"= self.build_mfdata('{subpkg['key']}', None)"
                        )
                    else:
                        _name = name[:-1] if name.endswith("_") else name
                        name = name.replace("-", "_")
                        stmts.append(
                            f"self.{'_' if subpkg else ''}{name} "
                            f"= self.build_mfdata('{_name}', {name})"
                        )

                if subpkg and subpkg["key"] not in refs and component_name[1] != "nam":
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


def safe_name(v: str) -> str:
    """
    Make sure a string is safe to use as a variable name in Python code.
    If the string is a reserved keyword, add a trailing underscore to it.
    Also replace any hyphens with underscores.
    """
    return (f"{v}_" if v in kwlist else v).replace("-", "_")


def math(v: str) -> str:
    """Massage latex equations"""
    v = v.replace("$<$", "<")
    v = v.replace("$>$", ">")
    if "$" in v:
        descsplit = v.split("$")
        mylist = [
            i.replace("\\", "") + ":math:`" + j.replace("\\", "\\\\") + "`"
            for i, j in zip(descsplit[::2], descsplit[1::2])
        ]
        mylist.append(descsplit[-1].replace("\\", ""))
        v = "".join(mylist)
    else:
        v = v.replace("\\", "")
    return v


def clean(v: str) -> str:
    """Clean description"""
    replace_pairs = [
        ("``", '"'),  # double quotes
        ("''", '"'),
        ("`", "'"),  # single quotes
        ("~", " "),  # non-breaking space
        (r"\mf", "MODFLOW 6"),
        (r"\citep{konikow2009}", "(Konikow et al., 2009)"),
        (r"\citep{hill1990preconditioned}", "(Hill, 1990)"),
        (r"\ref{table:ftype}", "in mf6io.pdf"),
        (r"\ref{table:gwf-obstypetable}", "in mf6io.pdf"),
    ]
    for s1, s2 in replace_pairs:
        if s1 in v:
            v = v.replace(s1, s2)
    return v


def value(v: Any) -> str:
    """
    Format a value to appear in the RHS of an assignment or argument-
    passing expression: if it's an enum, get its value; if it's `str`,
    quote it.
    """
    v = _try_get_enum_value(v)
    if isinstance(v, str) and v[0] not in ["'", '"']:
        v = f"'{v}'"
    return v
