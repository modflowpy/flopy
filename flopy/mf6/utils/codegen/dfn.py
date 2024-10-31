from ast import literal_eval
from collections import UserDict
from dataclasses import dataclass
from enum import Enum
from keyword import kwlist
from os import PathLike
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

from boltons.dictutils import OMD

from flopy.mf6.utils.codegen.utils import try_literal_eval, try_parse_bool

_SCALARS = {
    "keyword",
    "integer",
    "double precision",
    "string",
}


Vars = Dict[str, "Var"]
Refs = Dict[str, "Ref"]
Dfns = Dict[str, "Dfn"]


@dataclass
class Var:
    """MODFLOW 6 input variable specification."""

    class Kind(Enum):
        """
        An input variable's kind. This is an enumeration
        of the general shapes of data MODFLOW 6 accepts.
        """

        Array = "array"
        Scalar = "scalar"
        Record = "record"
        Union = "union"
        List = "list"

    name: str
    kind: Kind
    block: Optional[str] = None
    default: Optional[Any] = None
    children: Optional[Vars] = None
    description: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class Ref:
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

    @classmethod
    def from_dfn(cls, dfn: "Dfn") -> Optional["Ref"]:
        """
        Try to load a reference from the definition.
        Returns `None` if the definition cannot be
        referenced by other contexts.
        """

        # TODO: all this won't be necessary once we
        # structure DFN format; we can then support
        # subpackage references directly instead of
        # by making assumptions about `dfn.meta`

        if not dfn.meta or "dfn" not in dfn.meta:
            return None

        _, meta = dfn.meta["dfn"]

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

        def _subpkg():
            line = lines["subpkg"]
            _, key, abbr, param, val = line.split()
            matches = [v for v in dfn.values() if v.name == val]
            if not any(matches):
                descr = None
            else:
                if len(matches) > 1:
                    warn(f"Multiple matches for referenced variable {val}")
                match = matches[0]
                descr = match.description

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
            cls(**_subpkg(), parent=_parent())
            if all(v for v in lines.values())
            else None
        )


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

    name: Optional[Name]
    meta: Optional[Dict[str, Any]]

    def __init__(
        self,
        data: Optional[Vars] = None,
        name: Optional[Name] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.data = OMD(data)
        self.name = name
        self.meta = meta

    @staticmethod
    def _load(f, common: Optional[dict] = None) -> Tuple[OMD, List[str]]:
        """
        Internal use only. Loads the DFN as an ordered multi-dictionary* and
        a list of string metadata. This is later parsed into more structured
        form. We also store the original representation for now so it can be
        used by the shim.

        *The point of the OMD is to handle duplicate variable names; the only
        case of this right now is 'auxiliary' which can appear in the options
        block and again as a keyword in a record in a package data variable.

        """
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

        return OMD(flat), meta

    @classmethod
    def load(
        cls,
        f,
        name: Optional[Name] = None,
        refs: Optional[Dfns] = None,
        **kwargs,
    ) -> "Dfn":
        """Load an input definition."""

        refs = refs or dict()
        referenced = dict()
        flat, meta = Dfn._load(f, **kwargs)

        def _map(spec: Dict[str, Any]) -> Var:
            """
            Convert a variable specification from its representation
            in an input definition file to a Pythonic form.

            Notes
            -----
            This involves expanding nested type hierarchies, mapping
            types to roughly equivalent Python primitives/composites.
            The composite inflation step will not be necessary after
            DFNs move to a structured format.

            If a variable does not have a `default` attribute, it will
            default to `False` if it is a keyword, otherwise to `None`.

            Any filepath variable whose name functions as a foreign key
            for another context will be given a pointer to the context.

            """

            # parse booleans from strings. everything else can
            # stay a string except default values, which we'll
            # try to parse as arbitrary literals below, and at
            # some point types, once we introduce type hinting
            spec = {k: try_parse_bool(v) for k, v in spec.items()}

            _name = spec["name"]
            _type = spec.get("type", None)
            block = spec.get("block", None)
            shape = spec.get("shape", None)
            shape = None if shape == "" else shape
            default = spec.get("default", None)
            description = spec.get("description", "")
            children = dict()

            # if var is a foreign key, register the reference
            ref = refs.get(_name, None)
            if ref:
                referenced[_name] = ref

            def _fields(record_name: str) -> Vars:
                """Recursively load/convert a record's fields."""
                record = next(iter(flat.getlist(record_name)), None)
                assert record
                names = _type.split()[1:]
                fields = {
                    v["name"]: _map(v)
                    for v in flat.values(multi=True)
                    if v["name"] in names
                    and not v["type"].startswith("record")
                    and v.get("in_record", False)
                }

                # if the record represents a file...
                if "file" in _name:
                    # remove filein/fileout
                    for term in ["filein", "fileout"]:
                        if term in names:
                            fields.pop(term)

                    # remove leading keyword
                    keyword = next(iter(fields), None)
                    if keyword:
                        fields.pop(keyword)

                    # set the type
                    n = list(fields.keys())[0]
                    path_field = fields[n]
                    path_field._type = Union[str, PathLike]
                    fields[n] = path_field

                # if tagged, remove the leading keyword
                elif record.get("tagged", False):
                    keyword = next(iter(fields), None)
                    if keyword:
                        fields.pop(keyword)

                return fields

            # list, child is the item type
            if _type.startswith("recarray"):
                # make sure columns are defined
                names = _type.split()[1:]
                n_names = len(names)
                if n_names < 1:
                    raise ValueError(f"Missing recarray definition: {_type}")

                # list input can have records or unions as rows.
                # lists which have a consistent record type are
                # regular, inconsistent record types irregular.

                # regular tabular/columnar data (1 record type) can be
                # defined with a nested record (i.e. explicit) or with
                # fields directly inside the recarray (implicit). list
                # data for unions/keystrings necessarily comes nested.

                is_explicit_record = n_names == 1 and flat[names[0]][
                    "type"
                ].startswith("record")

                def _is_implicit_scalar_record():
                    # if the record is defined implicitly and it has
                    # only scalar fields
                    types = [
                        v["type"]
                        for v in flat.values(multi=True)
                        if v["name"] in names and v.get("in_record", False)
                    ]
                    return all(t in _SCALARS for t in types)

                if is_explicit_record:
                    record = next(iter(flat.getlist(names[0])), None)
                    children = {names[0]: _map(record)}
                    kind = Var.Kind.List
                elif _is_implicit_scalar_record():
                    fields = _fields(_name)
                    children = {
                        _name: Var(
                            name=_name,
                            kind=Var.Kind.Record,
                            block=block,
                            children=fields,
                            description=description,
                            meta={
                                "type": f"[{', '.join([f.meta['type'] for f in fields.values()])}]"
                            },
                        )
                    }
                    kind = Var.Kind.List
                else:
                    # implicit complex record (i.e. some fields are records or unions)
                    fields = {
                        v["name"]: _map(v)
                        for v in flat.values(multi=True)
                        if v["name"] in names and v.get("in_record", False)
                    }
                    first = list(fields.values())[0]
                    single = len(fields) == 1
                    name_ = first.name if single else _name
                    children = {
                        name_: Var(
                            name=name_,
                            kind=Var.Kind.Record,
                            block=block,
                            children=first.children if single else fields,
                            description=description,
                            meta={
                                "type": f"[{', '.join([v.meta['type'] for v in fields.values()])}]"
                            },
                        )
                    }
                    kind = Var.Kind.List
                type_ = f"[{', '.join([v.name for v in children.values()])}]"

            # union (product), children are choices
            elif _type.startswith("keystring"):
                names = _type.split()[1:]
                children = {
                    v["name"]: _map(v)
                    for v in flat.values(multi=True)
                    if v["name"] in names and v.get("in_record", False)
                }
                kind = Var.Kind.Union
                type_ = f"[{', '.join([v.name for v in children.values()])}]"

            # record (sum), children are fields
            elif _type.startswith("record"):
                children = _fields(_name)
                kind = Var.Kind.Record
                type_ = f"[{', '.join([v.meta['type'] for v in children.values()])}]"

            # at this point, if it has a shape, it's an array
            elif shape is not None:
                if _type not in _SCALARS:
                    raise TypeError(f"Unsupported array type: {_type}")
                elif _type == "string":
                    kind = Var.Kind.List
                else:
                    kind = Var.Kind.Array
                type_ = f"[{_type}]"

            # finally scalars
            else:
                kind = Var.Kind.Scalar
                type_ = _type

            # create var
            return Var(
                # if name is a reserved keyword, add a trailing underscore to it.
                # convert dashes to underscores since it may become a class attr.
                name=(f"{_name}_" if _name in kwlist else _name).replace(
                    "-", "_"
                ),
                kind=kind,
                block=block,
                description=description,
                default=(
                    try_literal_eval(default) if _type != "string" else default
                ),
                children=children,
                # type is a string for now, when
                # introducing type hints make it
                # a proper type...
                meta={"ref": ref, "type": type_},
            )

        # pass the original DFN representation as
        # metadata so the shim can use it for now
        _vars = list(flat.values(multi=True))

        # convert input variable specs to
        # structured form, descending into
        # composites recursively as needed
        flat = {
            var["name"]: _map(var)
            for var in flat.values(multi=True)
            if not var.get("in_record", False)
        }

        # reset the var name. we may have altered
        # it when converting the variable e.g. to
        # avoid collision with a reserved keyword
        flat = {v.name: v for v in flat.values()}

        return cls(
            flat,
            name,
            {
                "dfn": (_vars, meta),
                "refs": referenced,
            },
        )

    @staticmethod
    def load_all(dfndir: PathLike) -> Dict[str, "Dfn"]:
        """Load all input definitions from the given directory."""
        # find definition files
        paths = [
            p for p in dfndir.glob("*.dfn") if p.stem not in ["common", "flopy"]
        ]

        # try to load common variables
        common_path = dfndir / "common.dfn"
        if not common_path.is_file:
            common = None
        else:
            with open(common_path, "r") as f:
                common, _ = Dfn._load(f)

        # load subpackage references first
        refs: Refs = {}
        for path in paths:
            name = Dfn.Name(*path.stem.split("-"))
            with open(path) as f:
                dfn = Dfn.load(f, name=name, common=common)
                ref = Ref.from_dfn(dfn)
                if ref:
                    refs[ref.key] = ref

        # load all the input definitions
        dfns: Dfns = {}
        for path in paths:
            name = Dfn.Name(*path.stem.split("-"))
            with open(path) as f:
                dfn = Dfn.load(f, name=name, refs=refs, common=common)
                dfns[name] = dfn

        return dfns
