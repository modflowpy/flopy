"""
createpackages.py is a utility script that reads in the file definition
metadata in the .dfn files and creates the package classes in the modflow
folder. Run this script any time changes are made to the .dfn files.

To create a new package that is part of an existing model, first create a new
dfn file for the package in the mf6/data/dfn folder.
1) Follow the file naming convention <model abbr>-<package abbr>.dfn.
2) Run this script (createpackages.py), and check in your new dfn file, and
   the package class and updated __init__.py that createpackages.py created.

A subpackage is a package referenced by another package (vs being referenced
in the name file).  The tas, ts, and obs packages are examples of subpackages.
There are a few additional steps required when creating a subpackage
definition file.  First, verify that the parent package's dfn file has a file
record for the subpackage to the option block.   For example, for the time
series package the file record definition starts with:

    block options
    name ts_filerecord
    type record ts6 filein ts6_filename

Verify that the same naming convention is followed as the example above,
specifically:

    name <subpackage-abbr>_filerecord
    record <subpackage-abbr>6 filein <subpackage-abbr>6_filename

Next, create the child package definition file in the mf6/data/dfn folder
following the naming convention above.

When your child package is ready for release follow the same procedure as
other packages along with these a few additional steps required for
subpackages.

At the top of the child dfn file add two lines describing how the parent and
child packages are related. The first line determines how the subpackage is
linked to the package:

# flopy subpackage <parent record> <abbreviation> <child data>
<data name>

* Parent record is the MF6 record name of the filerecord in parent package
  that references the child packages file name
* Abbreviation is the short abbreviation of the new subclass
* Child data is the name of the child class data that can be passed in as
  parameter to the parent class. Passing in this parameter to the parent class
  automatically creates the child class with the data provided.
* Data name is the parent class parameter name that automatically creates the
  child class with the data provided.

The example below is the first line from the ts subpackage dfn:

# flopy subpackage ts_filerecord ts timeseries timeseries

The second line determines the variable name of the subpackage's parent and
the type of parent (the parent package's object oriented parent):

# flopy parent_name_type <parent package variable name>
<parent package type>

An example below is the second line in the ts subpackage dfn:

# flopy parent_name_type parent_package MFPackage

There are three possible types (or combination of them) that can be used for
"parent package type", MFPackage, MFModel, and MFSimulation. If a package
supports multiple types of parents (for example, it can be either in the model
namefile or in a package, like the obs package), include all the types
supported, separating each type with a / (MFPackage/MFModel).

To create a new type of model choose a unique three letter model abbreviation
("gwf", "gwt", ...). Create a name file dfn with the naming convention
<model abbr>-nam.dfn. The name file must have only an options and packages
block (see gwf-nam.dfn as an example). Create a new dfn file for each of the
packages in your new model, following the naming convention described above.

When your model is ready for release make sure all the dfn files are in the
flopy/mf6/data/dfn folder, run createpackages.py, and check in your new dfn
files, the package classes, and updated init.py that createpackages.py created.

"""

import collections
import os
from ast import literal_eval
from collections import UserDict, namedtuple
from dataclasses import asdict, dataclass, replace
from enum import Enum
from keyword import kwlist
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    ForwardRef,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)
from warnings import warn

import numpy as np
from jinja2 import Environment, PackageLoader
from modflow_devtools.misc import run_cmd
from numpy.typing import ArrayLike, NDArray


def _try_get_type_name(t) -> str:
    """Convert a type to a name suitable for templating."""
    origin = get_origin(t)
    args = get_args(t)
    if origin is Literal:
        args = ['"' + a + '"' for a in args]
        return f"Literal[{', '.join(args)}]"
    elif origin is Union:
        if len(args) >= 2 and args[-1] is type(None):
            if len(args) > 2:
                return f"Optional[Union[{', '.join([_try_get_type_name(a) for a in args[:-1]])}]]"
            return f"Optional[{_try_get_type_name(args[0])}]"
        return f"Union[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is tuple:
        return f"Tuple[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is collections.abc.Iterable:
        return f"Iterable[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is list:
        return f"List[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is np.ndarray:
        return f"NDArray[np.{_try_get_type_name(args[1].__args__[0])}]"
    elif origin is np.dtype:
        return str(t)
    elif isinstance(t, ForwardRef):
        return t.__forward_arg__
    elif t is Ellipsis:
        return "..."
    elif isinstance(t, type):
        return t.__qualname__
    return t


def _try_get_enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def renderable(
    maybe_cls=None,
    *,
    wrap_str: Optional[List[str]] = None,
    keep_none: Optional[List[str]] = None,
):
    """
    An object meant to be passed into a template
    as a "rendered" dictionary, where "rendering"
    means transforming key/value pairs to a form
    more convenient for use within the template.

    The object *must* be a dataclass.

    Notes
    -----
    Jinja supports attribute- and dictionary-
    based access but no arbitrary expressions,
    and only a limited set of custom filters.
    This can make it awkward to express some
    things, so convert the dataclasses we'll
    pass to `template.render(...)` to dicts,
    with a few touchups.

    These include:
        - converting types to suitably qualified type names
        - optionally removing key/value pairs whose value is None
        - optionally quoting strings forming the RHS of an assignment or
          argument passing expression

    """

    wrap_str = wrap_str or list()
    keep_none = keep_none or list()

    def __renderable(cls):
        def _render(d: dict) -> dict:
            def _render_key(k):
                return k

            def _render_val(v):
                return _try_get_type_name(_try_get_enum_value(v))

            # drop nones except where keep requested
            _d = {
                _render_key(k): _render_val(v)
                for k, v in d.items()
                if (k in keep_none or v is not None)
            }

            # wrap string values where requested
            if wrap_str:
                for k in wrap_str:
                    v = _d.get(k, None)
                    if v is not None and isinstance(v, str):
                        _d[k] = f'"{v}"'

            return _d

        def render(self) -> dict:
            """
            Recursively render the dataclass instance.
            """
            return _render(
                asdict(self, dict_factory=lambda d: _render(dict(d)))
            )

        setattr(cls, "render", render)
        return cls

    # first arg value depends on the decorator usage:
    # class if `@renderable`, `None` if `@renderable()`.
    # referenced from https://github.com/python-attrs/attrs/blob/a59c5d7292228dfec5480388b5f6a14ecdf0626c/src/attr/_next_gen.py#L405C4-L406C65
    return __renderable if maybe_cls is None else __renderable(maybe_cls)


class ContextName(NamedTuple):
    """
    Uniquely identifies an input context by its name, which
    consists of a <= 3-letter left term and optional right
    term also of <= 3 letters.

    Notes
    -----
    A single `DefinitionName` may be associated with one or
    more `ContextName`s. For instance, a model DFN file will
    produce both a NAM package class and also a model class.

    From the `ContextName` several other things are derived,
    including:

    - the input context class' name
    - a description of the context class
    - the name of the source file to write
    - the base class the context inherits from

    """

    l: str
    r: Optional[str]

    @property
    def title(self) -> str:
        """
        The input context's unique title. This is not
        identical to `f"{l}{r}` in some cases, but it
        remains unique. The title is substituted into
        the file name and class name.
        """

        l, r = self
        if self == ("sim", "nam"):
            return "simulation"
        if l is None:
            return r
        if r is None:
            return l
        if l == "sim":
            return r
        if l in ["sln", "exg"]:
            return r
        return f"{l}{r}"

    @property
    def base(self) -> str:
        """Base class from which the input context should inherit."""
        _, r = self
        if self == ("sim", "nam"):
            return "MFSimulationBase"
        if r is None:
            return "MFModel"
        return "MFPackage"

    @property
    def target(self) -> str:
        """The source file name to generate."""
        return f"mf{self.title}.py"

    @property
    def description(self) -> str:
        """A description of the input context."""
        l, r = self
        title = self.title.title()
        if self.base == "MFPackage":
            return f"Modflow{title} defines a {r.upper()} package."
        elif self.base == "MFModel":
            return f"Modflow{title} defines a {l.upper()} model."
        elif self.base == "MFSimulationBase":
            return """
    MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation.
    A MFSimulation object must be created before creating any of the MODFLOW 6
    model objects."""


class DfnName(NamedTuple):
    """
    Uniquely identifies an input definition by its name, which
    consists of a <= 3-letter left term and an optional right
    term, also <= 3 letters.

    Notes
    -----
    A single `DefinitionName` may be associated with one or
    more `ContextName`s. For instance, a model DFN file will
    produce both a NAM package class and also a model class.
    """

    l: str
    r: str

    @property
    def contexts(self) -> List[ContextName]:
        """
        Returns a list of contexts this definition will produce.

        Notes
        -----
        Model definition files produce both a model class context and
        a model namefile package context. The same goes for simulation
        definition files. All other definition files produce a single
        context.
        """
        if self.r == "nam":
            if self.l == "sim":
                return [
                    ContextName(None, self.r),  # nam pkg
                    ContextName(*self),  # simulation
                ]
            else:
                return [
                    ContextName(*self),  # nam pkg
                    ContextName(self.l, None),  # model
                ]
        elif (self.l, self.r) in [
            ("gwf", "mvr"),
            ("gwf", "gnc"),
            ("gwt", "mvt"),
        ]:
            return [ContextName(*self), ContextName(None, self.r)]
        return [ContextName(*self)]


Metadata = List[str]


@dataclass
class Dfn(UserDict):
    """
    An MF6 input definition.
    """

    name: Optional[DfnName]
    metadata: Optional[Metadata]

    def __init__(
        self,
        variables: Dict[str, Dict[str, str]],
        name: Optional[DfnName] = None,
        metadata: Optional[Metadata] = None,
    ):
        super().__init__(variables)
        self.name = name
        self.metadata = metadata


Dfns = Dict[str, Dfn]


def load_dfn(f, name: Optional[DfnName] = None) -> Dfn:
    """
    Load an input definition from a definition file.
    """

    meta = None
    vars_ = dict()
    var = dict()

    for line in f:
        # remove whitespace/etc from the line
        line = line.strip()

        # record context name and flopy metadata
        # attributes, skip all other comment lines
        if line.startswith("#"):
            _, sep, tail = line.partition("flopy")
            if sep == "flopy":
                if meta is None:
                    meta = list()
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
                vars_[var["name"]] = var
                var = dict()
            continue

        # split the attribute's key and value and
        # store it in the parameter dictionary
        key, _, value = line.partition(" ")
        if key == "default_value":
            key = "default"
        if value in ["true", "false"]:
            value = value == "true"
        var[key] = value

    # add the final parameter
    if any(var):
        vars_[var["name"]] = var

    return Dfn(variables=vars_, name=name, metadata=meta)


@dataclass
class Subpkg:
    """
    A foreign-key-like reference between a file input variable
    and a subpackage definition. This allows an input context
    to reference a subpackage by including a variable with an
    appropriate name.

    Parameters
    ----------
    key : str
        The name of the file input variable identifying the
        referenced subpackage.
    val : str
        The name of the variable containing subpackage data
        in the referenced subpackage.
    abbr : str
        An abbreviation of the subpackage's name.
    param : str
        The subpackage parameter name. TODO: explain
    parents : List[type]
        The subpackage's supported parent types.
    """

    key: str
    val: str
    abbr: str
    param: str
    parents: List[Union[type, str]]
    description: Optional[str]

    @classmethod
    def from_dfn(cls, dfn: Dfn) -> Optional["Subpkg"]:
        if not dfn.metadata:
            return None

        lines = {
            "subpkg": next(
                iter(
                    m
                    for m in dfn.metadata
                    if isinstance(m, str) and m.startswith("subpac")
                ),
                None,
            ),
            "parent": next(
                iter(
                    m
                    for m in dfn.metadata
                    if isinstance(m, str) and m.startswith("parent")
                ),
                None,
            ),
        }

        def _subpkg():
            line = lines["subpkg"]
            _, key, abbr, param, val = line.split()
            descr = dfn.get(val, dict()).get("description", None)
            return {
                "key": key,
                "val": val,
                "abbr": abbr,
                "param": param,
                "description": descr,
            }

        def _parents():
            line = lines["parent"]
            _, _, _type = line.split()
            return _type.split("/")

        return (
            cls(**_subpkg(), parents=_parents())
            if all(v for v in lines.values())
            else None
        )


Subpkgs = Dict[str, Subpkg]


class VarKind(Enum):
    """
    An input variable's kind. This is an enumeration
    of the general shapes of data MODFLOW 6 accepts.
    """

    Array = "array"
    Scalar = "scalar"
    Record = "record"
    Union = "union"
    List = "list"

    @classmethod
    def from_type(cls, t: type) -> Optional["VarKind"]:
        origin = get_origin(t)
        args = get_args(t)
        if origin is Union:
            if len(args) >= 2 and args[-1] is type(None):
                if len(args) > 2:
                    return VarKind.Union
                return cls.from_type(args[0])
            return VarKind.Union
        if origin is np.ndarray or origin is NDArray or origin is ArrayLike:
            return VarKind.Array
        elif origin is collections.abc.Iterable or origin is list:
            return VarKind.List
        elif origin is tuple:
            return VarKind.Record
        try:
            if issubclass(t, (bool, int, float, str)):
                return VarKind.Scalar
        except:
            pass
        return None


@dataclass
class Var:
    """A variable in a MODFLOW 6 input context."""

    name: str
    _type: Union[type, str]
    block: Optional[str]
    description: Optional[str]
    default: Optional[Any]
    children: Optional[Dict[str, "Var"]]
    meta: Optional[List[str]]
    subpkg: Optional[Subpkg]
    kind: Optional[VarKind]
    is_choice: bool = False
    init_param: bool = True
    init_assign: bool = False
    init_build: bool = False
    init_super: bool = False
    class_attr: bool = False

    def __init__(
        self,
        name: str,
        _type: Optional[type] = None,
        block: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        parent: Optional["Var"] = None,
        children: Optional["Vars"] = None,
        meta: Optional[Metadata] = None,
        subpkg: Optional[Subpkg] = None,
        kind: Optional[VarKind] = None,
        is_choice: bool = False,
        init_param: bool = True,
        init_assign: bool = False,
        init_build: bool = False,
        init_super: bool = False,
        class_attr: bool = False,
    ):
        self.name = name
        self._type = _type or Any
        self.block = block
        self.description = description
        self.default = default
        self.parent = parent
        self.children = children
        self.meta = meta
        self.subpkg = subpkg
        # TODO: the rest of the attributes below are
        # needed to handle complexities in the input
        # context classes; in a future version, they
        # will ideally not be necessary.
        # ---
        # the variable's general kind.
        # this is ofc derivable on demand but Jinja
        # doesn't allow arbitrary expressions, and it
        # doesn't seem to have `subclass`-ish filters.
        self.kind = kind or VarKind.from_type(_type)
        # similarly, whether the variable is a choice
        # in a union. this is derivable from .parent,
        # but awkward to do in Jinja, so use a flag.
        self.is_choice = is_choice
        # whether the var is an init method parameter
        self.init_param = init_param
        # whether to assign arguments to self in the
        # init method body. if this is false, assume
        # the template has conditionals for any more
        # involved initialization needs.
        self.init_assign = init_assign
        # whether to call `build_mfdata()` to build
        # the parameter.
        self.init_build = init_build
        # whether to pass arg to super().__init__()
        self.init_super = init_super
        # whether the variable has a corresponding
        # class attribute
        self.class_attr = class_attr


Vars = Dict[str, Var]


@renderable(wrap_str=["default"], keep_none=["block", "default"])
@dataclass
class Context:
    """
    An input context. Each of these is specified by a definition file
    and becomes a generated class. A definition file may specify more
    than one input context (e.g. model DFNs yield a model class and a
    package class).

    Notes
    -----
    A context class minimally consists of a name, a map of variables,
    a map of records, and a list of metadata.

    A separate map of record variables is maintained because we will
    generate named tuples for record types, and complex filtering of
    e.g. nested maps of variables is awkward or impossible in Jinja.

    The context class may inherit from a base class, and may specify
    a parent context within which it can be created (the parent then
    becomes the first `__init__` method parameter).

    """

    name: ContextName
    base: Optional[type]
    parent: Optional[Union[type, str]]
    description: Optional[str]
    metadata: Metadata
    variables: Vars
    records: Vars
    subpkg: bool


_SCALAR_TYPES = {
    "keyword": bool,
    "integer": int,
    "double precision": float,
    "string": str,
}
_NP_SCALAR_TYPES = {
    "keyword": np.bool_,
    "integer": np.int_,
    "double precision": np.float64,
    "string": np.str_,
}


def make_context(
    name: ContextName,
    dfn: Dfn,
    common: Optional[Dfn] = None,
    subpkgs: Optional[Subpkgs] = None,
) -> Context:
    """
    Convert an MF6 input definition to a structured descriptor
    of an input context class to create with a Jinja template.

    Notes
    -----
    Each input definition corresponds to a generated Python
    source file. A definition may produce one or more input
    context classes.

    A map of other definitions may be provided, in which case a
    parameter in this context may act as kind of "foreign key",
    identifying another context as a subpackage which this one
    is related to.
    """

    common = common or dict()
    subpkgs = subpkgs or dict()
    _subpkg = Subpkg.from_dfn(dfn)
    records = dict()

    def _nt_name(s, trims=False):
        """
        Convert a record name to the name of a corresponding named tuple.

        Notes
        -----
        Dashes and underscores are removed, with title-casing for clauses
        separated by them, and a trailing "record" is removed if present.

        """
        s = s.title().replace("record", "").replace("-", "_").replace("_", "")
        if trims:
            s = s[:-1] if s.endswith("s") else s
        return s

    def _parent() -> Optional[str]:
        """
        Get the context's parent(s), i.e. context(s) which can
        own an instance of this context. If this context is a
        subpackage which can have multiple parent types, this
        will be a Union of possible parent types, otherwise a
        single parent type.

        Notes
        -----
        We return a string directly instead of a type to avoid
        the need to import `MFSimulation` in this file (avoids
        potential for circular imports).
        """
        l, r = dfn.name
        if (l, r) == ("sim", "nam") and name == ("sim", "nam"):
            return None
        if l in ["sim", "exg", "sln"]:
            return "MFSimulation"
        if r in ["nam"] and name.l is None:
            return "MFSimulation"
        if _subpkg:
            if len(_subpkg.parents) > 1:
                return f"Union[{', '.join([_try_get_type_name(t) for t in _subpkg.parents])}]"
            return _subpkg.parents[0]
        return "MFModel"

    parent = _parent()

    def _convert(
        var: Dict[str, str],
        wrap: bool = False,
    ) -> Var:
        """
        Transform a variable from its original representation in
        an input definition to a specification suitable for type
        hints, docstrings, an `__init__` method's signature, etc.

        This involves expanding nested type hierarchies, mapping
        types to roughly equivalent Python primitives/composites,
        and other shaping.

        Notes
        -----
        The rules for optional variable defaults are as follows:
        If a `default_value` is not provided, keywords are `False`
        by default, everything else is `None`.

        If `wrap` is true, scalars will be wrapped as records with
        keywords represented as string literals. This is useful for
        unions, to distinguish between choices having the same type.

        Any variable whose name functions as a key for a subpackage
        will be provided with a subpackage reference.
        """

        # var attributes to be converted
        _name = var["name"]
        _type = var.get("type", "unknown")
        block = var.get("block", None)
        shape = var.get("shape", None)
        shape = None if shape == "" else shape
        optional = var.get("optional", True)
        in_record = var.get("in_record", False)
        tagged = var.get("tagged", False)
        description = var.get("description", "")
        children = None
        is_record = False
        class_attr = var.get("class_attr", False)

        def _description(descr: str) -> str:
            """
            Make substitutions from common variable definitions,
            remove backslashes, generate/insert citations, etc.
            TODO: insert citations.
            """
            descr = descr.replace("\\", "")
            _, replace, tail = descr.strip().partition("REPLACE")
            if replace:
                key, _, replacements = tail.strip().partition(" ")
                replacements = literal_eval(replacements)
                common_var = common.get(key, None)
                if common_var is None:
                    raise ValueError(f"Common variable not found: {key}")
                descr = common_var.get("description", "")
                if any(replacements):
                    return descr.replace("\\", "").replace(
                        "{#1}", replacements["{#1}"]
                    )
                return descr
            return descr

        def _fields(record_name: str) -> Vars:
            """
            Recursively load/convert a record's fields.

            Notes
            -----
            This function is provided because records
            need extra processing; we remove keywords
            and 'filein'/'fileout', which are details
            of the mf6io format, not of python/flopy.
            """
            record = dfn[record_name]
            field_names = record["type"].split()[1:]
            fields: Dict[str, Var] = {
                n: _convert(field, wrap=False)
                for n, field in dfn.items()
                if n in field_names
            }
            field_names = list(fields.keys())

            # if the record represents a file...
            if "file" in record_name:
                # remove filein/fileout
                for term in ["filein", "fileout"]:
                    if term in field_names:
                        fields.pop(term)

                # remove leading keyword
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

                # set the type
                n = list(fields.keys())[0]
                path_field = fields[n]
                path_field._type = Union[str, os.PathLike]
                fields[n] = path_field

            # if tagged, remove the leading keyword
            elif record.get("tagged", False):
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

            return fields

        # go through all the possible input types
        # from top (composite) to bottom (scalar):
        #
        #   - list
        #   - union
        #   - record
        #   - array
        #   - scalar
        #
        # list input can have records or unions as rows.
        # lists which have a consistent record type are
        # regular, inconsistent record types irregular.
        if _type.startswith("recarray"):
            # flag as a class attribute (ListTemplateGenerator etc)
            class_attr = var.get("class_attr", True)

            # make sure columns are defined
            names = _type.split()[1:]
            n_names = len(names)
            if n_names < 1:
                raise ValueError(f"Missing recarray definition: {_type}")

            # regular tabular/columnar data (1 record type) can be
            # defined with a nested record (i.e. explicit) or with
            # fields directly inside the recarray (implicit). list
            # data for unions/keystrings necessarily comes nested.

            is_explicit_record = len(names) == 1 and dfn[names[0]][
                "type"
            ].startswith("record")

            def _is_implicit_scalar_record():
                # if the record is defined implicitly and it has
                # only scalar fields
                types = [
                    _try_get_type_name(v["type"])
                    for n, v in dfn.items()
                    if n in names
                ]
                scalar_types = list(_SCALAR_TYPES.keys())
                return all(t in scalar_types for t in types)

            if is_explicit_record:
                record_name = names[0]
                record_spec = dfn[record_name]
                record_type = _convert(record_spec, wrap=False)
                children = {_nt_name(record_name).lower(): record_type}
                type_ = Iterable[record_type._type]
            elif _is_implicit_scalar_record():
                record_name = _name
                record_fields = _fields(record_name)
                field_types = [f._type for f in record_fields.values()]
                record_type = Tuple[tuple(field_types)]
                record = Var(
                    name=record_name,
                    _type=record_type,
                    block=block,
                    children=record_fields,
                    description=description,
                )
                records[_nt_name(record_name, trims=True)] = replace(
                    record, name=_nt_name(record_name, trims=True)
                )
                record_type = namedtuple(
                    _nt_name(record_name, trims=True),
                    [_nt_name(k) for k in record_fields.keys()],
                )
                record = replace(
                    record,
                    _type=record_type,
                    name=_nt_name(record_name, trims=True).lower(),
                )
                children = {_nt_name(record_name, trims=True): record}
                type_ = Iterable[record_type]
            else:
                # implicit complex record (i.e. some fields are records or unions)
                record_fields = {
                    n: _convert(dfn[n], wrap=False) for n in names
                }
                first = list(record_fields.values())[0]
                single = len(record_fields) == 1
                record_name = first.name if single else _name
                _t = [f._type for f in record_fields.values()]
                record_type = (
                    first._type
                    if (single and first.kind == VarKind.Union)
                    else Tuple[tuple(_t)]
                )
                record = Var(
                    name=record_name,
                    _type=record_type,
                    block=block,
                    children=first.children if single else record_fields,
                    description=description,
                )
                records[_nt_name(record_name)] = replace(
                    record, name=_nt_name(record_name)
                )
                record_type = namedtuple(
                    _nt_name(record_name),
                    [_nt_name(k) for k in record_fields.keys()],
                )
                record = replace(
                    record,
                    _type=record_type,
                    name=_nt_name(record_name).lower(),
                )
                type_ = Iterable[record_type]

        # union (product), children are record choices
        elif _type.startswith("keystring"):
            # flag as a class attribute (ListTemplateGenerator etc)
            class_attr = var.get("class_attr", True)

            names = _type.split()[1:]
            children = {n: _convert(dfn[n], wrap=True) for n in names}
            type_ = Union[tuple([c._type for c in children.values()])]

        # record (sum) type, children are fields
        elif _type.startswith("record"):
            # flag as a class attribute (ListTemplateGenerator etc)
            class_attr = var.get("class_attr", True)

            children = _fields(_name)
            if len(children) > 1:
                record_type = Tuple[
                    tuple([f._type for f in children.values()])
                ]
            elif len(children) == 1:
                t = list(children.values())[0]._type
                # make sure we don't double-wrap tuples
                record_type = t if get_origin(t) is tuple else Tuple[(t,)]
            # TODO: if record has 1 field, accept value directly?
            type_ = record_type
            is_record = True

        # are we wrapping a var into a record
        # as a choice in a union? if so use a
        # string literal for the keyword e.g.
        # `Tuple[Literal[...], T]`
        elif wrap:
            field_name = _name
            field = _convert(var, wrap=False)
            field_type = (
                Literal[field_name] if field._type is bool else field._type
            )
            record_type = (
                Tuple[Literal[field_name]]
                if field._type is bool
                else Tuple[Literal[field_name], field._type]
            )
            children = {
                field_name: replace(field, _type=field_type, is_choice=True)
            }
            type_ = record_type
            is_record = True

        # at this point, if it has a shape, it's an array..
        # but if it's in a record make it a variadic tuple,
        # and if its item type is a string use an iterable.
        elif shape is not None:
            # flag as a class attribute (ListTemplateGenerator etc)
            class_attr = var.get("class_attr", True)
            scalars = list(_SCALAR_TYPES.keys())
            if in_record:
                if _type not in scalars:
                    raise TypeError(f"Unsupported repeating type: {_type}")
                type_ = Tuple[_SCALAR_TYPES[_type], ...]
            elif _type in scalars and _SCALAR_TYPES[_type] is str:
                type_ = Iterable[_SCALAR_TYPES[_type]]
            else:
                if _type not in _NP_SCALAR_TYPES.keys():
                    raise TypeError(f"Unsupported array type: {_type}")
                type_ = NDArray[_NP_SCALAR_TYPES[_type]]

        # finally a bog standard scalar
        else:
            # if it's a keyword, there are 2 cases where we want to convert
            # it to a string literal: 1) it tags another variable, or 2) it
            # is being wrapped into a record as a choice in a union
            tag = _type == "keyword" and (tagged or wrap)
            type_ = Literal[_name] if tag else _SCALAR_TYPES.get(_type, _type)

        # format the variable description
        description = _description(description)

        # keywords default to False, everything else to None
        default = var.get("default", False if type_ is bool else None)
        if isinstance(default, str) and type_ is not str:
            try:
                default = literal_eval(default)
            except:
                pass
        if _name in ["continue", "print_input"]:  # hack...
            default = None

        # if name is a reserved keyword, add a trailing underscore to it.
        # convert dashes to underscores since it may become a class attr.
        name_ = (f"{_name}_" if _name in kwlist else _name).replace("-", "_")

        # create var
        var_ = Var(
            name=name_,
            _type=type_,
            block=block,
            description=description,
            default=default,
            children=children,
            init_param=True,
            init_build=True,
            class_attr=class_attr,
        )

        # check if the variable references a subpackage
        subpkg = subpkgs.get(_name, None)
        if subpkg:
            var_.init_build = False
            var_.subpkg = subpkg

        # if this is a record, make a named tuple for it
        if is_record:
            records[_nt_name(name_)] = replace(var_, name=_nt_name(name_))
            if children:
                type_ = namedtuple(
                    _nt_name(name_), [_nt_name(k) for k in children.keys()]
                )
                var_._type = type_

        # make optional if needed
        if optional:
            var_._type = (
                Optional[type_]
                if (type_ is not bool and not in_record and not wrap)
                else type_
            )

        return var_

    def _variables() -> Vars:
        """
        Return all input variables for an input context class.

        Notes
        -----
        Not all variables become parameters; nested variables
        will become components of composite parameters, e.g.,
        record fields, keystring (union) choices, list items.

        Variables may be added, depending on the context type.
        """

        vars_ = dfn.copy()
        vars_ = {
            name: _convert(var, wrap=False)
            for name, var in vars_.items()
            # filter composite components
            # since we've already inflated
            # their parents in the hierarchy
            if not var.get("in_record", False)
        }

        # set the name since we may have altered
        # it when creating the variable (e.g. to
        # avoid name/reserved keyword collisions.
        vars_ = {v.name: v for v in vars_.values()}

        def _add_exg_vars(_vars: Vars) -> Vars:
            """
            Add initializer parameters for an exchange context.
            Exchanges need different parameters than a typical
            package.
            """
            a = name.r[:3]
            b = name.r[:3]
            default = f"{a.upper()}6-{b.upper()}6"
            vars_ = {
                "parent": Var(
                    name="parent",
                    _type="MFSimulation",
                    description=(
                        "Simulation that this package is a part of. "
                        "Package is automatically added to simulation "
                        "when it is initialized."
                    ),
                    init_param=True,
                    init_assign=False,
                    init_build=False,
                    init_super=True,
                ),
                "loading_package": Var(
                    name="loading_package",
                    _type=bool,
                    description=(
                        "Do not set this parameter. It is intended for "
                        "debugging and internal processing purposes only."
                    ),
                    default=False,
                    init_param=True,
                    init_assign=False,
                    init_build=False,
                    init_super=True,
                ),
                "exgtype": Var(
                    name="exgtype",
                    _type=str,
                    default=default,
                    description="The exchange type.",
                    init_param=True,
                    init_assign=True,
                    init_build=False,
                    init_super=False,
                ),
                "exgmnamea": Var(
                    name="exgmnamea",
                    _type=str,
                    description="The name of the first model in the exchange.",
                    init_param=True,
                    init_assign=True,
                    init_super=False,
                ),
                "exgmnameb": Var(
                    name="exgmnameb",
                    _type=str,
                    description="The name of the second model in the exchange.",
                    init_param=True,
                    init_assign=True,
                    init_build=False,
                    init_super=False,
                ),
                **_vars,
                "filename": Var(
                    name="filename",
                    _type=Union[str, PathLike],
                    description="File name for this package.",
                    init_param=True,
                    init_assign=False,
                    init_build=False,
                    init_super=True,
                ),
                "pname": Var(
                    name="pname",
                    _type=str,
                    description="Package name for this package.",
                    init_param=True,
                    init_assign=False,
                    init_build=False,
                    init_super=True,
                ),
            }

            # if a reference map is provided,
            # find any variables referring to
            # subpackages, and attach another
            # "value" variable for them all..
            # allows passing data directly to
            # `__init__` instead of a path to
            # load the subpackage from. maybe
            # impossible if the data variable
            # doesn't appear in the reference
            # definition, though.
            if subpkgs:
                for k, subpkg in subpkgs.items():
                    key = vars_.get(k, None)
                    if not key:
                        continue
                    vars_[subpkg.key].init_param = False
                    vars_[subpkg.key].init_build = True
                    vars_[subpkg.key].class_attr = True
                    vars_[subpkg.val] = Var(
                        name=subpkg.val,
                        description=subpkg.description,
                        subpkg=subpkg,
                        init_param=True,
                        init_assign=False,
                        init_super=False,
                        init_build=False,
                    )

            return vars_

        def _add_pkg_vars(_vars: Vars) -> Vars:
            """Add variables for a package context."""
            parent_name = "parent"
            vars_ = {
                parent_name: Var(
                    name=parent_name,
                    _type=parent,
                    description="Parent that this package is part of.",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "loading_package": Var(
                    name="loading_package",
                    _type=bool,
                    description=(
                        "Do not set this variable. It is intended for debugging "
                        "and internal processing purposes only."
                    ),
                    default=False,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                **_vars,
                "filename": Var(
                    name="filename",
                    _type=str,
                    description="File name for this package.",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "pname": Var(
                    name="pname",
                    _type=str,
                    description="Package name for this package.",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
            }

            # if context is a subpackage add
            # a `parent_file` variable which
            # is the path to the subpackage's
            # parent context
            subpkg = Subpkg.from_dfn(dfn)
            if subpkg and dfn.name.l == "utl":
                vars_["parent_file"] = Var(
                    name="parent_file",
                    _type=Union[str, PathLike],
                    description=(
                        "Parent package file that references this package. Only needed "
                        "for utility packages (mfutl*). For example, mfutllaktab package "
                        "must have a mfgwflak package parent_file."
                    ),
                    init_param=True,
                    init_assign=False,
                )

            # if a reference map is provided,
            # find any variables referring to
            # subpackages, and attach another
            # "value" variable for them all..
            # allows passing data directly to
            # `__init__` instead of a path to
            # load the subpackage from. maybe
            # impossible if the data variable
            # doesn't appear in the reference
            # definition, though.
            if subpkgs and name != (None, "nam"):
                for k, subpkg in subpkgs.items():
                    key = vars_.get(k, None)
                    if not key:
                        continue
                    vars_[subpkg.key].init_param = False
                    vars_[subpkg.key].init_build = True
                    vars_[subpkg.key].class_attr = True
                    vars_[subpkg.val] = Var(
                        name=subpkg.val,
                        description=subpkg.description,
                        subpkg=subpkg,
                        init_param=True,
                        init_assign=False,
                        init_super=False,
                        init_build=False,
                    )

            return vars_

        def _add_mdl_vars(_vars: Vars) -> Vars:
            """Add variables for a model context."""
            vars_ = _vars.copy()
            packages = _vars.get("packages", None)
            if packages:
                packages.init_param = False
                vars_["packages"] = packages

            vars_ = {
                "simulation": Var(
                    name="simulation",
                    _type="MFSimulation",
                    description=(
                        "Simulation that this model is part of. "
                        "Model is automatically added to the simulation "
                        "when it is initialized."
                    ),
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "modelname": Var(
                    name="modelname",
                    _type=str,
                    description="The name of the model.",
                    default="model",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "model_nam_file": Var(
                    name="model_nam_file",
                    _type=Optional[Union[str, PathLike]],
                    description=(
                        "The relative path to the model name file from model working folder."
                    ),
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "version": Var(
                    name="version",
                    _type=str,
                    description="The version of modflow",
                    default="mf6",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "exe_name": Var(
                    name="exe_name",
                    _type=str,
                    description="The executable name.",
                    default="mf6",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "model_rel_path": Var(
                    name="model_ws",
                    _type=Union[str, PathLike],
                    description="The model working folder path.",
                    default=os.curdir,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                **vars_,
            }

            # if a reference map is provided,
            # find any variables referring to
            # subpackages, and attach another
            # "value" variable for them all..
            # allows passing data directly to
            # `__init__` instead of a path to
            # load the subpackage from. maybe
            # impossible if the data variable
            # doesn't appear in the reference
            # definition, though.
            if subpkgs:
                for k, subpkg in subpkgs.items():
                    key = vars_.get(k, None)
                    if not key:
                        continue
                    vars_[subpkg.key].init_param = False
                    vars_[subpkg.key].class_attr = True
                    vars_[subpkg.val] = Var(
                        name=subpkg.val,
                        description=subpkg.description,
                        subpkg=subpkg,
                        init_param=True,
                        init_assign=False,
                        init_super=False,
                        init_build=False,
                    )

            return vars_

        def _add_sim_params(_vars: Vars) -> Vars:
            """Add variables for a simulation context."""
            vars_ = _vars.copy()
            skip_init = [
                "tdis6",
                "models",
                "exchanges",
                "mxiter",
                "solutiongroup",
            ]
            for k in skip_init:
                var = vars_.get(k, None)
                if var:
                    var.init_param = False
                    vars_[k] = var
            vars_ = {
                "sim_name": Var(
                    name="sim_name",
                    _type=str,
                    default="sim",
                    description="Name of the simulation.",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "version": Var(
                    name="version",
                    _type=str,
                    default="mf6",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "exe_name": Var(
                    name="exe_name",
                    _type=Union[str, PathLike],
                    default="mf6",
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "sim_ws": Var(
                    name="sim_ws",
                    _type=Union[str, PathLike],
                    default=os.curdir,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "verbosity_level": Var(
                    name="verbosity_level",
                    _type=int,
                    default=1,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "write_headers": Var(
                    name="write_headers",
                    _type=bool,
                    default=True,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "use_pandas": Var(
                    name="use_pandas",
                    _type=bool,
                    default=True,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                "lazy_io": Var(
                    name="lazy_io",
                    _type=bool,
                    default=False,
                    init_param=True,
                    init_assign=False,
                    init_super=True,
                ),
                **vars_,
            }

            # if a reference map is provided,
            # find any variables referring to
            # subpackages, and attach another
            # "value" variable for them all..
            # allows passing data directly to
            # `__init__` instead of a path to
            # load the subpackage from. maybe
            # impossible if the data variable
            # doesn't appear in the reference
            # definition, though.
            if subpkgs:
                for k, subpkg in subpkgs.items():
                    key = vars_.get(k, None)
                    if not key:
                        continue
                    vars_[subpkg.key].init_param = False
                    vars_[subpkg.key].init_build = False
                    vars_[subpkg.key].class_attr = True
                    vars_[subpkg.param] = Var(
                        name=subpkg.param,
                        description=subpkg.description,
                        subpkg=subpkg,
                        init_param=True,
                        init_assign=False,
                        init_super=False,
                        init_build=False,
                    )

            return vars_

        # add initializer method parameters
        # for this particular context type
        if name.base == "MFSimulationBase":
            vars_ = _add_sim_params(vars_)
        elif name.base == "MFModel":
            vars_ = _add_mdl_vars(vars_)
        elif name.base == "MFPackage":
            if name.l == "exg":
                vars_ = _add_exg_vars(vars_)
            else:
                vars_ = _add_pkg_vars(vars_)

        return vars_

    def _metadata() -> List[Metadata]:
        """
        Get a list of the class' original definition attributes
        as a partial, internal reproduction of the DFN contents.

        Notes
        -----
        Currently, generated classes have a `.dfn` property that
        reproduces the corresponding DFN sans a few attributes.
        This represents the DFN in raw form, before adapting to
        Python, consolidating nested types, etc.
        """

        def _fmt_var(var: Var) -> List[str]:
            exclude = ["longname", "description"]

            def _fmt_name(k, v):
                return v.replace("-", "_") if k == "name" else v

            return [
                " ".join([k, str(_fmt_name(k, v))]).strip()
                for k, v in var.items()
                if k not in exclude
            ]

        meta = dfn.metadata or list()
        return [["header"] + [m for m in meta]] + [
            _fmt_var(var) for var in dfn.values()
        ]

    return Context(
        name=name,
        base=name.base,
        parent=parent,
        description=name.description,
        metadata=_metadata(),
        variables=_variables(),
        records=records,
        subpkg=bool(_subpkg),
    )


def make_contexts(
    dfn: Dfn,
    common: Optional[Dfn] = None,
    subpkgs: Optional[Subpkgs] = None,
) -> Iterator[Context]:
    for name in dfn.name.contexts:
        yield make_context(name=name, dfn=dfn, common=common, subpkgs=subpkgs)


_TEMPLATE_ENV = Environment(
    loader=PackageLoader("flopy", "mf6/utils/templates/"),
)


def make_targets(
    dfn: Dfn,
    outdir: Path,
    common: Optional[Dfn] = None,
    subpkgs: Optional[Subpkgs] = None,
    verbose: bool = False,
):
    """
    Generate Python source file(s) from the given input definition.

    Notes
    -----

    Model definitions will produce two files / classes, one for the
    model itself and one for its corresponding control file package.

    All other definitions currently produce a single file and class.
    """

    template = _TEMPLATE_ENV.get_template("context.py.jinja")
    for context in make_contexts(dfn=dfn, common=common, subpkgs=subpkgs):
        target = outdir / context.name.target
        with open(target, "w") as f:
            source = template.render(**context.render())
            f.write(source)
            if verbose:
                print(f"Wrote {target}")


def make_all(dfndir: Path, outdir: Path, verbose: bool = False):
    """Generate Python source files from the DFN files in the given location."""

    # find definition files
    paths = [
        p for p in dfndir.glob("*.dfn") if p.stem not in ["common", "flopy"]
    ]

    # try to load common variables
    common_path = dfndir / "common.dfn"
    if not common_path.is_file:
        warn("No common input definition file...")
        common = None
    else:
        with open(common_path, "r") as f:
            common = load_dfn(f)

    # load all definitions first before we generate targets,
    # so we can identify subpackages and create references
    # between package/subpackage contexts.
    dfns = dict()
    subpkgs = dict()
    for p in paths:
        name = DfnName(*p.stem.split("-"))
        with open(p) as f:
            dfn = load_dfn(f, name=name)
            dfns[name] = dfn
            subpkg = Subpkg.from_dfn(dfn)
            if subpkg:
                # key is the name of the file record
                # that corresponds to the subpackage
                subpkgs[subpkg.key] = subpkg

    # generate target files
    for dfn in dfns.values():
        with open(p) as f:
            make_targets(
                dfn=dfn,
                outdir=outdir,
                subpkgs=subpkgs,
                common=common,
                verbose=verbose,
            )

    # write __init__.py file
    init_path = outdir / "__init__.py"
    with open(init_path, "w") as f:
        for dfn in dfns.values():
            for context in dfn.name.contexts:
                prefix = (
                    "MF" if context.base == "MFSimulationBase" else "Modflow"
                )
                f.write(
                    f"from .mf{context.title} import {prefix}{context.title.title()}\n"
                )

    # format the generated files
    run_cmd("ruff", "format", outdir, verbose=verbose)
    run_cmd("ruff", "check", "--fix", outdir, verbose=True)


_MF6_PATH = Path(__file__).parents[1]
_DFN_PATH = _MF6_PATH / "data" / "dfn"
_TGT_PATH = _MF6_PATH / "modflow"


if __name__ == "__main__":
    make_all(_DFN_PATH, _TGT_PATH)
