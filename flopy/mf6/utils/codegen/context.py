from ast import literal_eval
from collections import namedtuple
from dataclasses import dataclass, replace
from keyword import kwlist
from os import PathLike
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    get_origin,
)

import numpy as np
from numpy.typing import NDArray

from flopy.mf6.utils.codegen.dfn import Dfn, DfnName, Metadata
from flopy.mf6.utils.codegen.ref import Refs
from flopy.mf6.utils.codegen.render import renderable
from flopy.mf6.utils.codegen.shim import SHIM
from flopy.mf6.utils.codegen.spec import Ref, Var, VarKind, Vars
from flopy.mf6.utils.codegen.utils import _try_get_type_name

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


def get_context_names(dfn_name: DfnName) -> List[ContextName]:
    """
    Returns a list of contexts this definition produces.

    Notes
    -----
    An input definition may produce one or more input contexts.

    Model definition files produce both a model class context and
    a model namefile package context. The same goes for simulation
    definition files. All other definition files produce a single
    context.
    """
    if dfn_name.r == "nam":
        if dfn_name.l == "sim":
            return [
                ContextName(None, dfn_name.r),  # nam pkg
                ContextName(*dfn_name),  # simulation
            ]
        else:
            return [
                ContextName(*dfn_name),  # nam pkg
                ContextName(dfn_name.l, None),  # model
            ]
    elif (dfn_name.l, dfn_name.r) in [
        ("gwf", "mvr"),
        ("gwf", "gnc"),
        ("gwt", "mvt"),
    ]:
        return [ContextName(*dfn_name), ContextName(None, dfn_name.r)]
    return [ContextName(*dfn_name)]


@renderable(**SHIM)
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

    The context class may inherit from a base class, and may specify
    a parent context within which it can be created (the parent then
    becomes the first `__init__` method parameter).

    A separate map of record variables is maintained because we will
    generate named tuples for record types, and complex filtering of
    e.g. nested maps of variables is awkward or impossible in Jinja.
    TODO: make this a prerendering step

    """

    name: ContextName
    base: Optional[type]
    parent: Optional[Union[type, str]]
    description: Optional[str]
    metadata: List[Metadata]
    variables: Vars
    records: Vars
    references: Refs


def make_context(
    name: ContextName,
    dfn: Dfn,
    common: Optional[Dfn] = None,
    references: Optional[Refs] = None,
) -> Context:
    """
    Extract a context descriptor from an input definition:
    a structured representation of the input context that
    can be used to generate an input data interface layer.

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
    references = references or dict()
    ref = Ref.from_dfn(dfn)  # this a ref?
    refs = dict()  # referenced contexts
    records = dict()  # record variables

    def _ntname(s):
        """
        Convert a record name to the name of a corresponding named tuple.

        Notes
        -----
        Dashes and underscores are removed, with title-casing for clauses
        separated by them, and a trailing "record" is removed if present.

        """
        return (
            s.title().replace("record", "").replace("-", "_").replace("_", "")
        )

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
        if name.r is None:
            return "MFSimulation"
        if ref:
            if len(ref.parents) > 1:
                return f"Union[{', '.join([_try_get_type_name(t) for t in ref.parents])}]"
            return ref.parents[0]
        return "MFModel"

    parent = _parent()

    def _convert(var: Dict[str, Any], wrap: bool = False) -> Var:
        """
        Transform a variable from its original representation in
        an input definition to a specification suitable for type
        hints, docstrings, an `__init__` method's signature, etc.

        This involves expanding nested type hierarchies, mapping
        types to roughly equivalent Python primitives/composites,
        and other shaping.

        The rules for optional variable defaults are as follows:
        If a `default_value` is not provided, keywords are `False`
        by default, everything else is `None`.

        If `wrap` is true, scalars will be wrapped as records with
        keywords represented as string literals. This is useful for
        unions, to distinguish between choices having the same type.

        Any filepath variable whose name functions as a foreign key
        for another context will be given a pointer to the context.


        Notes
        -----
        This function does most of the work in the whole module.
        A bit of a beast, and Codacy complains it's too complex,
        but having it here allows using the outer function scope
        (including the input definition, etc) without a bunch of
        extra function parameters. And what it's doing is fairly
        straightforward: map a variable specification from a DFN
        into a corresponding Python representation.

        """

        _name = var["name"]
        _type = var.get("type", "unknown")
        block = var.get("block", None)
        shape = var.get("shape", None)
        shape = None if shape == "" else shape
        optional = var.get("optional", True)
        in_record = var.get("in_record", False)
        tagged = var.get("tagged", False)
        description = var.get("description", "")
        children = dict()
        is_record = False

        def _description(descr: str) -> str:
            """
            Make substitutions from common variable definitions,
            remove backslashes, TODO: generate/insert citations.
            """
            descr = descr.replace("\\", "")
            _, replace, tail = descr.strip().partition("REPLACE")
            if replace:
                key, _, subs = tail.strip().partition(" ")
                subs = literal_eval(subs)
                cmn_var = common.get(key, None)
                if cmn_var is None:
                    raise ValueError(f"Common variable not found: {key}")
                descr = cmn_var.get("description", "")
                if any(subs):
                    return descr.replace("\\", "").replace(
                        "{#1}", subs["{#1}"]
                    )
                return descr
            return descr

        def _fields(record_name: str) -> Vars:
            """Recursively load/convert a record's fields."""
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
                path_field._type = Union[str, PathLike]
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

        # list input, child is the item type
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
                record = _convert(record_spec, wrap=False)
                children = {_ntname(record_name).lower(): record}
                type_ = Iterable[record._type]
            elif _is_implicit_scalar_record():
                record_name = _name
                fields = _fields(record_name)
                field_types = [f._type for f in fields.values()]
                record_type = Tuple[tuple(field_types)]
                record = Var(
                    name=record_name,
                    _type=record_type,
                    block=block,
                    children=fields,
                    description=description,
                )
                records[_ntname(record_name)] = replace(
                    record, name=_ntname(record_name)
                )
                record_type = namedtuple(
                    _ntname(record_name),
                    [_ntname(k) for k in fields.keys()],
                )
                record = replace(
                    record,
                    _type=record_type,
                    name=_ntname(record_name).lower(),
                )
                children = {_ntname(record_name): record}
                type_ = Iterable[record_type]
            else:
                # implicit complex record (i.e. some fields are records or unions)
                fields = {n: _convert(dfn[n], wrap=False) for n in names}
                first = list(fields.values())[0]
                single = len(fields) == 1
                record_name = first.name if single else _name
                field_types = [f._type for f in fields.values()]
                record_type = (
                    first._type
                    if (single and get_origin(first._type) is Union)
                    else Tuple[tuple(field_types)]
                )
                record = Var(
                    name=record_name,
                    _type=record_type,
                    block=block,
                    children=first.children if single else fields,
                    description=description,
                )
                records[_ntname(record_name)] = replace(
                    record, name=_ntname(record_name)
                )
                record_type = namedtuple(
                    _ntname(record_name),
                    [_ntname(k) for k in fields.keys()],
                )
                record = replace(
                    record,
                    _type=record_type,
                    name=_ntname(record_name).lower(),
                )
                children = {_ntname(record_name): record}
                type_ = Iterable[record_type]

        # union (product), children are record choices
        elif _type.startswith("keystring"):
            names = _type.split()[1:]
            children = {n: _convert(dfn[n], wrap=True) for n in names}
            type_ = Union[tuple([c._type for c in children.values()])]

        # record (sum) type, children are fields
        elif _type.startswith("record"):
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
            children = {field_name: replace(field, _type=field_type)}
            type_ = record_type
            is_record = True

        # at this point, if it has a shape, it's an array..
        # but if it's in a record make it a variadic tuple,
        # and if its item type is a string use an iterable.
        elif shape is not None:
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
        )

        # if the var is a foreign key, register the referenced context
        ref_ = references.get(_name, None)
        if ref_:
            var_.reference = ref_
            refs[_name] = ref_

        # if the var is a record, make a named tuple for it
        if is_record:
            records[_ntname(name_)] = replace(var_, name=_ntname(name_))
            if any(children):
                type_ = namedtuple(
                    _ntname(name_), [_ntname(k) for k in children.keys()]
                )
                var_._type = type_

        # wrap the var's type with Optional if it's optional
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
        # avoid name/reserved keyword collisions)
        return {v.name: v for v in vars_.values()}

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

        def _fmt_var(var: Union[Var, List[Var]]) -> List[str]:
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
            _fmt_var(var) for var in dfn.omd.values(multi=True)
        ]

    return Context(
        name=name,
        base=name.base,
        parent=parent,
        description=name.description,
        metadata=_metadata(),
        variables=_variables(),
        records=records,
        references=refs,
    )


def make_contexts(
    dfn: Dfn,
    common: Optional[Dfn] = None,
    refs: Optional[Refs] = None,
) -> Iterator[Context]:
    """Generate one or more input contexts from the given input definition."""
    for name in get_context_names(dfn.name):
        yield make_context(name=name, dfn=dfn, common=common, references=refs)
