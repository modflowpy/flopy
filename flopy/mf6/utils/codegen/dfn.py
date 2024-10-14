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

_SCALARS = {
    "keyword",
    "integer",
    "double precision",
    "string",
}


Vars = Dict[str, "Var"]
Dfns = Dict[str, "Dfn"]


def _try_parse_bool(value):
    """
    Try to parse a boolean from a string as represented
    in a DFN file, otherwise return the value unaltered.
    """
    
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "false"]:
            return value == "true"
    return value


def _try_literal_eval(value: str) -> Any:
    """
    Try to parse a string as a literal. If this fails,
    return the value unaltered.
    """
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value


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
    description: Optional[str] = None
    default: Optional[Any] = None
    children: Optional[Vars] = None
    meta: Optional[Dict[str, Any]] = None


class Dfn(UserDict):
    """
    MODFLOW 6 input definition. An input definition
    file specifies a component of an MF6 simulation,
    e.g. a model or package.
    """

    class Name(NamedTuple):
        """
        Uniquely identifies an input definition. A name
        consists of a left term and optional right term.
        """

        l: str
        r: str

    name: Optional[Name]
    meta: Optional[Dict[str, Any]]

    def __init__(
        self,
        vars: Optional[Vars] = None,
        name: Optional[Name] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.data = OMD(vars)
        self.name = name
        self.meta = meta

    @staticmethod
    def _load(f, common: Optional[dict] = None) -> Tuple[OMD, List[str]]:
        """
        Internal use only. Loads the DFN as an ordered multi-dictionary, and
        a list of string metadata. This is later parsed into more structured
        form. We also store the original representation for now so it can be
        used by the shim.
        """
        var = dict()
        vars = list()
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
                    vars.append((var["name"], var))
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
                descr = descr.replace("\\", "")
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
            vars.append((var["name"], var))

        return OMD(vars), meta

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
        vars, meta = Dfn._load(f, **kwargs)

        def _map(spec: Dict[str, Any], wrap: bool = False) -> Var:
            """
            Convert a variable specification from its representation
            in an input definition file to a Pythonic form.

            Notes
            -----
            This involves expanding nested type hierarchies, mapping
            types to roughly equivalent Python primitives/composites,
            and other shaping.

            The rules for optional variable defaults are as follows:
            If a `default_value` is not provided, keywords are `False`
            by default, everything else is `None`.

            If `wrap` is true, scalars will be wrapped as records.
            This is useful to distinguish among choices in unions.

            Any filepath variable whose name functions as a foreign key
            for another context will be given a pointer to the context.

            """

            # parse booleans from strings. everything else can
            # stay a string except default values, which we'll
            # try to parse as arbitrary literals below, and at
            # some point types, once we introduce type hinting
            spec = {k: _try_parse_bool(v) for k, v in spec.items()}

            # pull off attributes we're interested in
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
                record = next(iter(vars.getlist(record_name)), None)
                assert record
                names = _type.split()[1:]
                fields = {
                    v["name"]: _map(v)
                    for v in vars.values(multi=True)
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

                is_explicit_record = n_names == 1 and vars[names[0]][
                    "type"
                ].startswith("record")

                def _is_implicit_scalar_record():
                    # if the record is defined implicitly and it has
                    # only scalar fields
                    types = [
                        v["type"]
                        for v in vars.values(multi=True)
                        if v["name"] in names and v.get("in_record", False)
                    ]
                    return all(t in _SCALARS for t in types)

                if is_explicit_record:
                    record = next(iter(vars.getlist(names[0])), None)
                    children = {names[0]: _map(record)}
                    kind = Var.Kind.List
                elif _is_implicit_scalar_record():
                    children = {
                        _name: Var(
                            name=_name,
                            kind=Var.Kind.Record,
                            block=block,
                            children=_fields(_name),
                            description=description,
                        )
                    }
                    kind = Var.Kind.List
                else:
                    # implicit complex record (i.e. some fields are records or unions)
                    fields = {
                        v["name"]: _map(v)
                        for v in vars.values(multi=True)
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
                        )
                    }
                    kind = Var.Kind.List

            # union (product), children are choices.
            # scalar choices are wrapped as records.
            elif _type.startswith("keystring"):
                names = _type.split()[1:]
                children = {
                    v["name"]: _map(v, wrap=True)
                    for v in vars.values(multi=True)
                    if v["name"] in names and v.get("in_record", False)
                }
                kind = Var.Kind.Union

            # record (sum), children are fields
            elif _type.startswith("record"):
                children = _fields(_name)
                kind = Var.Kind.Record

            # are we wrapping a var into a record
            # as a choice in a union?
            elif wrap:
                children = {_name: _map(spec)}
                kind = Var.Kind.Record

            # at this point, if it has a shape, it's an array
            elif shape is not None:
                if _type not in _SCALARS:
                    raise TypeError(f"Unsupported array type: {_type}")
                elif _type == "string":
                    kind = Var.Kind.List
                else:
                    kind = Var.Kind.Array

            # finally scalars
            else:
                kind = Var.Kind.Scalar

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
                    _try_literal_eval(default)
                    if _type != "string"
                    else default
                ),
                children=children,
                meta={"ref": ref},
            )

        # pass the original DFN representation as
        # metadata so the shim can use it for now
        _vars = list(vars.values(multi=True))

        # convert input variable specs to
        # structured form, descending into
        # composites recursively as needed
        vars = {
            var["name"]: _map(var)
            for var in vars.values(multi=True)
            if not var.get("in_record", False)
        }

        # reset the var name. we may have altered
        # it when converting the variable e.g. to
        # avoid collision with a reserved keyword
        vars = {v.name: v for v in vars.values()}

        return cls(
            vars,
            name,
            {
                "dfn": (_vars, meta),
                "refs": referenced,
            },
        )
