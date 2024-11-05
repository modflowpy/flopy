from ast import literal_eval
from collections import UserDict
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

from flopy.utils.utl_import import import_optional_dependency

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
    children: Optional[Vars] = None
    description: Optional[str] = None


class Ref(TypedDict):
    """
    A foreign-key-like reference between a file input variable
    and another input definition. This allows an input context
    to refer to another input context, by including a filepath
    variable whose name acts as a foreign key for a different
    input context. The referring context's `__init__` method
    is modified such that the variable named `val` replaces
    the `key` variable.

    This class is used to represent subpackage references.
    """

    key: str
    val: str
    abbr: str
    param: str
    parent: str
    description: Optional[str]


class Dfn(UserDict):
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

    Version = Literal[1]

    name: Optional[Name]
    meta: Optional[Dict[str, Any]]

    def __init__(
        self,
        data: Optional[Vars] = None,
        name: Optional[Name] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.data = data or dict()
        self.name = name
        self.meta = meta

    @staticmethod
    def _load_v1_flat(
        f, common: Optional[dict] = None, **kwargs
    ) -> Tuple[OMD, List[str]]:
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
                    tail = tail.strip()
                    if "solution_package" in tail:
                        tail = tail.split()
                        tail.pop(1)
                    meta.append(tail)
                    continue
                _, sep, tail = line.partition("package-type")
                if sep == "package-type":
                    if meta is None:
                        meta = list
                    meta.append(f"{sep} {tail.strip()}")
                    continue
                _, sep, tail = line.partition("solution_package")
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
        flat, meta = Dfn._load_v1_flat(f, **kwargs)
        refs = kwargs.pop("refs", dict())
        fkeys = dict()

        def _map(spec: Dict[str, Any]) -> Var:
            """
            Convert an input variable specification from its shape
            in a classic definition file to a Python-friendly form.

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
            spec = {k: _try_parse_bool(v) for k, v in spec.items()}

            _name = spec["name"]
            _type = spec.get("type", None)
            shape = spec.get("shape", None)
            shape = None if shape == "" else shape
            block = spec.get("block", None)
            children = dict()
            default = spec.get("default", None)
            default = (
                _try_literal_eval(default) if _type != "string" else default
            )
            description = spec.get("description", "")
            fkey = refs.get(_name, None)

            # if var is a foreign key, register it
            if fkey:
                fkeys[_name] = fkey

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
                    return {names[0]: _map(child)}
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
                        v["name"]: _map(v)
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
                    v["name"]: _map(v)
                    for v in flat.values(multi=True)
                    if v["name"] in names and v.get("in_record", False)
                }

            def _fields() -> Vars:
                """Load a record's children (fields)."""
                names = _type.split()[1:]
                return {
                    v["name"]: _map(v)
                    for v in flat.values(multi=True)
                    if v["name"] in names
                    and v.get("in_record", False)
                    and not v["type"].startswith("record")
                }

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
            if fkey:
                return Var(
                    name=fkey["param" if name == ("sim", "nam") else "val"],
                    type=_type,
                    shape=shape,
                    block=block,
                    children=None,
                    description=(
                        f"* Contains data for the {fkey['abbr']} package. Data can be "
                        f"stored in a dictionary containing data for the {fkey['abbr']} "
                        "package with variable names as keys and package data as "
                        f"values. Data just for the {fkey['val']} variable is also "
                        f"acceptable. See {fkey['abbr']} package documentation for more "
                        "information"
                    ),
                    default=None,
                    fkey=fkey,
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

        vars_ = {
            var["name"]: _map(var)
            for var in flat.values(multi=True)
            if not var.get("in_record", False)
        }

        def _subpkg() -> Optional["Ref"]:
            lines = {
                "subpkg": next(
                    iter(
                        m
                        for m in meta
                        if isinstance(m, str) and m.startswith("subpac")
                    ),
                    None,
                ),
                "parent": next(
                    iter(
                        m
                        for m in meta
                        if isinstance(m, str) and m.startswith("parent")
                    ),
                    None,
                ),
            }

            def __subpkg():
                line = lines["subpkg"]
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

            def _parent():
                line = lines["parent"]
                split = line.split()
                return split[1]

            return (
                Ref(**__subpkg(), parent=_parent())
                if all(v for v in lines.values())
                else None
            )

        return cls(
            vars_,
            name,
            {
                "dfn": (
                    # pass the original DFN representation as
                    # metadata so templates can use it for now,
                    # eventually we can hopefully drop this
                    list(flat.values(multi=True)),
                    meta,
                ),
                "fkeys": fkeys,
                "subpkg": _subpkg(),
            },
        )

    @classmethod
    def load(
        cls,
        f,
        name: Optional[Name] = None,
        version: Version = 1,
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
            name = Dfn.Name(*path.stem.split("-"))
            with open(path) as f:
                dfn = Dfn.load(f, name=name, common=common)
                ref = dfn.meta.get("subpkg", None)
                if ref:
                    refs[ref["key"]] = ref

        # load all the input definitions
        dfns: Dfns = {}
        for path in paths:
            name = Dfn.Name(*path.stem.split("-"))
            with open(path) as f:
                dfn = Dfn.load(f, name=name, refs=refs, common=common)
                dfns[name] = dfn

        return dfns

    @staticmethod
    def load_all(dfndir: PathLike, version: Version = 1) -> Dfns:
        """Load all input definitions from the given directory."""

        if version == 1:
            return Dfn._load_all_v1(dfndir)
        else:
            raise ValueError(
                f"Unsupported version, expected one of {version.__args__}"
            )
