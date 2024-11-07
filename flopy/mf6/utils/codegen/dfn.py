from ast import literal_eval
from collections.abc import Mapping
from os import PathLike
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
)
from warnings import warn

from boltons.dictutils import OMD

_SCALARS = {
    "keyword",
    "integer",
    "double precision",
    "string",
}


def _try_literal_eval(value: str) -> Any:
    """
    Try to parse a string as a literal. If this fails,
    return the value unaltered.
    """
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _try_parse_bool(value: Any) -> Any:
    """
    Try to parse a boolean from a string as represented
    in a DFN file, otherwise return the value unaltered.
    """
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "false"]:
            return value == "true"
    return value


Vars = Dict[str, "Var"]
Refs = Dict[str, "Ref"]
Dfns = Dict[str, "Dfn"]


class Var(TypedDict):
    """An input variable specification."""

    name: str
    type: str
    shape: Optional[Any] = None
    block: Optional[str] = None
    default: Optional[Any] = None
    children: Optional["Vars"] = None
    description: Optional[str] = None


class Ref(TypedDict):
    """
    This class is used to represent subpackage references:
    a foreign-key-like reference between a file input variable
    and another input definition. This allows an input context
    to refer to another input context by including a filepath
    variable as a foreign key. The former's `__init__` method
    is modified such that the variable named `val` replaces
    the `key` variable.
    """

    key: str
    val: str
    abbr: str
    param: str
    parent: str
    description: Optional[str]


class Sln(TypedDict):
    abbr: str
    pattern: str


DfnFmtVersion = Literal[1]
"""DFN format version number."""


class Dfn(TypedDict):
    """
    MODFLOW 6 input definition. An input definition
    file specifies a component of an MF6 simulation,
    e.g. a model or package.
    """

    class Name(NamedTuple):
        """
        Uniquely identifies an input definition.
        Consists of a left term and a right term.
        """

        l: str
        r: str

        @classmethod
        def parse(cls, v: str) -> "Dfn.Name":
            try:
                return cls(*v.split("-"))
            except:
                raise ValueError(f"Bad DFN name format: {v}")

        def __str__(self) -> str:
            return "-".join(self)

    name: Name
    vars: Vars

    @staticmethod
    def _load_v1_flat(
        f, common: Optional[dict] = None
    ) -> Tuple[Mapping, List[str]]:
        var = dict()
        flat = list()
        meta = list()
        common = common or dict()

        for line in f:
            # remove whitespace/etc from the line
            line = line.strip()

            # record context name and flopy metadata
            # attributes, skip all other comment lines
            if line.startswith("#"):
                _, sep, tail = line.partition("flopy")
                if sep == "flopy":
                    if (
                        "multi-package" in tail
                        or "solution_package" in tail
                        or "subpackage" in tail
                        or "parent" in tail
                    ):
                        meta.append(tail.strip())
                _, sep, tail = line.partition("package-type")
                if sep == "package-type":
                    meta.append(f"package-type {tail.strip()}")
                continue

            # if we hit a newline and the parameter dict
            # is nonempty, we've reached the end of its
            # block of attributes
            if not any(line):
                if any(var):
                    flat.append((var["name"], var))
                    var = dict()
                continue

            # split the attribute's key and value and
            # store it in the parameter dictionary
            key, _, value = line.partition(" ")
            if key == "default_value":
                key = "default"
            var[key] = value

            # make substitutions from common variable definitions,
            # remove backslashes, TODO: generate/insert citations.
            descr = var.get("description", None)
            if descr:
                descr = (
                    descr.replace("\\", "")
                    .replace("``", "'")
                    .replace("''", "'")
                )
                _, replace, tail = descr.strip().partition("REPLACE")
                if replace:
                    key, _, subs = tail.strip().partition(" ")
                    subs = literal_eval(subs)
                    cvar = common.get(key, None)
                    if cvar is None:
                        warn(
                            "Can't substitute description text, "
                            f"common variable not found: {key}"
                        )
                    else:
                        descr = cvar.get("description", "")
                        if any(subs):
                            descr = descr.replace("\\", "").replace(
                                "{#1}", subs["{#1}"]
                            )
                var["description"] = descr

        # add the final parameter
        if any(var):
            flat.append((var["name"], var))

        # the point of the OMD is to losslessly handle duplicate variable names
        return OMD(flat), meta

    @classmethod
    def _load_v1(cls, f, name, **kwargs) -> "Dfn":
        """
        Temporary load routine for the v1 DFN format.
        This can go away once we convert to v2 (TOML).
        """

        # if we have any subpackage references
        # we need to watch for foreign key vars
        # (file input vars) and register matches
        refs = kwargs.pop("refs", dict())
        fkeys = dict()

        # load dfn as flat multidict and str metadata
        flat, meta = Dfn._load_v1_flat(f, **kwargs)

        # pass the original dfn representation on
        # the dfn since it is reproduced verbatim
        # in generated classes for now. drop this
        # later when we figure out how to unravel
        # mfstructure.py etc
        def _meta():
            meta_ = list()
            for m in meta:
                if "multi" in m:
                    meta_.append(m)
                elif "solution" in m:
                    s = m.split()
                    meta_.append([s[0], s[2]])
                elif "package-type" in m:
                    s = m.split()
                    meta_.append(" ".join(s))
            return meta_

        dfn = list(flat.values(multi=True)), _meta()

        def _load_variable(var: Dict[str, Any]) -> Var:
            """
            Convert an input variable from its original representation
            in a definition file to a structured, Python-friendly form.

            This involves trimming unneeded attributes and setting
            some others.

            Notes
            -----
            If a variable does not have a `default` attribute, it will
            default to `False` if it is a keyword, otherwise to `None`.

            A filepath variable whose name functions as a foreign key
            for a separate context will be given a reference to it.
            """

            # parse booleans from strings. everything else can
            # stay a string except default values, which we'll
            # try to parse as arbitrary literals below, and at
            # some point types, once we introduce type hinting
            var = {k: _try_parse_bool(v) for k, v in var.items()}

            _name = var["name"]
            _type = var.get("type", None)
            shape = var.get("shape", None)
            shape = None if shape == "" else shape
            block = var.get("block", None)
            children = dict()
            default = var.get("default", None)
            default = (
                _try_literal_eval(default) if _type != "string" else default
            )
            description = var.get("description", "")
            ref = refs.get(_name, None)

            # if var is a foreign key, register it
            if ref:
                fkeys[_name] = ref

            def _items() -> Vars:
                """Load a list's children (items: record or union of records)."""

                names = _type.split()[1:]
                types = [
                    v["type"]
                    for v in flat.values(multi=True)
                    if v["name"] in names and v.get("in_record", False)
                ]
                n_names = len(names)
                if n_names < 1:
                    raise ValueError(f"Missing recarray definition: {_type}")

                # list input can have records or unions as rows. lists
                # that have a consistent item type can be considered
                # tabular. lists that can possess multiple item types
                # (unions) are considered irregular. regular lists can
                # be defined with a nested record (explicit) or with a
                # set of fields directly in the recarray (implicit). an
                # irregular list is always defined with a nested union.
                is_explicit = n_names == 1 and (
                    types[0].startswith("record")
                    or types[0].startswith("keystring")
                )

                if is_explicit:
                    child = next(iter(flat.getlist(names[0])))
                    return {names[0]: _load_variable(child)}
                elif all(t in _SCALARS for t in types):
                    # implicit simple record (all fields are scalars)
                    fields = _fields()
                    return {
                        _name: Var(
                            name=_name,
                            type="record",
                            block=block,
                            children=fields,
                            description=description.replace(
                                "is the list of", "is the record of"
                            ),
                        )
                    }
                else:
                    # implicit complex record (some fields are records or unions)
                    fields = {
                        v["name"]: _load_variable(v)
                        for v in flat.values(multi=True)
                        if v["name"] in names and v.get("in_record", False)
                    }
                    first = list(fields.values())[0]
                    single = len(fields) == 1
                    name_ = first["name"] if single else _name
                    child_type = (
                        "union"
                        if single and "keystring" in first["type"]
                        else "record"
                    )
                    return {
                        name_: Var(
                            name=name_,
                            type=child_type,
                            block=block,
                            children=first["children"] if single else fields,
                            description=description.replace(
                                "is the list of", f"is the {child_type} of"
                            ),
                        )
                    }

            def _choices() -> Vars:
                """Load a union's children (choices)."""
                names = _type.split()[1:]
                return {
                    v["name"]: _load_variable(v)
                    for v in flat.values(multi=True)
                    if v["name"] in names and v.get("in_record", False)
                }

            def _fields() -> Vars:
                """Load a record's children (fields)."""
                names = _type.split()[1:]
                fields = dict()
                for name in names:
                    v = flat.get(name, None)
                    if (
                        not v
                        or not v.get("in_record", False)
                        or v["type"].startswith("record")
                    ):
                        continue
                    fields[name] = v
                return fields

            if _type.startswith("recarray"):
                children = _items()
                _type = "list"

            elif _type.startswith("keystring"):
                children = _choices()
                _type = "union"

            elif _type.startswith("record"):
                children = _fields()
                _type = "record"

            # for now, we can tell a var is an array if its type
            # is scalar and it has a shape. once we have proper
            # typing, this can be read off the type itself.
            elif shape is not None and _type not in _SCALARS:
                raise TypeError(f"Unsupported array type: {_type}")

            # if var is a foreign key, return subpkg var instead
            if ref:
                return Var(
                    name=ref["param" if name == ("sim", "nam") else "val"],
                    type=_type,
                    shape=shape,
                    block=block,
                    children=None,
                    description=(
                        f"* Contains data for the {ref['abbr']} package. Data can be "
                        f"stored in a dictionary containing data for the {ref['abbr']} "
                        "package with variable names as keys and package data as "
                        f"values. Data just for the {ref['val']} variable is also "
                        f"acceptable. See {ref['abbr']} package documentation for more "
                        "information"
                    ),
                    default=None,
                    subpackage=ref,
                )

            return Var(
                name=_name,
                type=_type,
                shape=shape,
                block=block,
                children=children,
                description=description,
                default=default,
            )

        # load top-level variables. any nested
        # variables will be loaded recursively
        vars_ = {
            var["name"]: _load_variable(var)
            for var in flat.values(multi=True)
            if not var.get("in_record", False)
        }

        def _package_type() -> Optional[str]:
            line = next(
                iter(
                    m
                    for m in meta
                    if isinstance(m, str) and m.startswith("package-type")
                ),
                None,
            )
            return line.split()[-1] if line else None

        def _subpackage() -> Optional["Ref"]:
            def _parent():
                line = next(
                    iter(
                        m
                        for m in meta
                        if isinstance(m, str) and m.startswith("parent")
                    ),
                    None,
                )
                if not line:
                    return None
                split = line.split()
                return split[1]

            def _rest():
                line = next(
                    iter(
                        m
                        for m in meta
                        if isinstance(m, str) and m.startswith("subpac")
                    ),
                    None,
                )
                if not line:
                    return None
                _, key, abbr, param, val = line.split()
                matches = [v for v in vars_.values() if v["name"] == val]
                if not any(matches):
                    descr = None
                else:
                    if len(matches) > 1:
                        warn(f"Multiple matches for referenced variable {val}")
                    match = matches[0]
                    descr = match["description"]

                return {
                    "key": key,
                    "val": val,
                    "abbr": abbr,
                    "param": param,
                    "description": descr,
                }

            parent = _parent()
            rest = _rest()
            if parent and rest:
                return Ref(parent=parent, **rest)
            return None

        def _solution() -> Optional[Sln]:
            sln = next(
                iter(
                    m
                    for m in meta
                    if isinstance(m, str) and m.startswith("solution_package")
                ),
                None,
            )
            if sln:
                abbr, pattern = sln.split()[1:]
                return Sln(abbr=abbr, pattern=pattern)
            return None

        def _multi() -> bool:
            return any("multi-package" in m for m in meta)

        package_type = _package_type()
        subpackage = _subpackage()
        solution = _solution()
        multi = _multi()

        return cls(
            name=name,
            vars=vars_,
            dfn=dfn,
            foreign_keys=fkeys,
            package_type=package_type,
            subpackage=subpackage,
            solution=solution,
            multi=multi,
        )

    @classmethod
    def load(
        cls,
        f,
        name: Optional[Name] = None,
        version: DfnFmtVersion = 1,
        **kwargs,
    ) -> "Dfn":
        """
        Load an input definition from a DFN file.
        """

        if version == 1:
            return cls._load_v1(f, name, **kwargs)
        else:
            raise ValueError(
                f"Unsupported version, expected one of {version.__args__}"
            )

    @staticmethod
    def _load_all_v1(dfndir: PathLike) -> Dfns:
        # find definition files
        paths = [
            p
            for p in dfndir.glob("*.dfn")
            if p.stem not in ["common", "flopy"]
        ]

        # try to load common variables
        common_path = dfndir / "common.dfn"
        if not common_path.is_file:
            common = None
        else:
            with open(common_path, "r") as f:
                common, _ = Dfn._load_v1_flat(f)

        # load subpackage references first
        refs: Refs = {}
        for path in paths:
            with open(path) as f:
                name = Dfn.Name.parse(path.stem)
                dfn = Dfn.load(f, name=name, common=common)
                subpkg = dfn.get("subpackage", None)
                if subpkg:
                    refs[subpkg["key"]] = subpkg

        # load all the input definitions
        dfns: Dfns = {}
        for path in paths:
            with open(path) as f:
                name = Dfn.Name.parse(path.stem)
                dfn = Dfn.load(f, name=name, common=common, refs=refs)
                dfns[name] = dfn

        return dfns

    @staticmethod
    def load_all(dfndir: PathLike, version: DfnFmtVersion = 1) -> Dfns:
        """Load all input definitions from the given directory."""

        if version == 1:
            return Dfn._load_all_v1(dfndir)
        else:
            raise ValueError(
                f"Unsupported version, expected one of {version.__args__}"
            )
