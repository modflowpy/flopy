from ast import literal_eval
from dataclasses import dataclass
from keyword import kwlist
from os import PathLike
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
)

from flopy.mf6.utils.codegen.dfn import Dfn, DfnName
from flopy.mf6.utils.codegen.ref import Ref, Refs
from flopy.mf6.utils.codegen.render import renderable
from flopy.mf6.utils.codegen.shim import SHIM
from flopy.mf6.utils.codegen.var import Var, VarKind, Vars

_SCALAR_TYPES = {
    "keyword",
    "integer",
    "double precision",
    "string",
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


@renderable(
    # shim for implementation details in the
    # generated context classes which aren't
    # really concerns of the core framework,
    # and may eventually go away
    **SHIM
)
@dataclass
class Context:
    """
    An input context. Each of these is specified by a definition file
    and becomes a generated class. A definition file may specify more
    than one input context (e.g. model DFNs yield a model class and a
    package class).

    Notes
    -----
    A context class minimally consists of a name, a definition, and a
    map of variables. The definition and variables are redundant (the
    latter are generated from the former) but for now, the definition
    is needed. When generated classes no longer reproduce definitions
    verbatim, it can be removed.

    The context class may inherit from a base class, and may specify
    a parent context within which it can be created (the parent then
    becomes the first `__init__` method parameter).

    The context class may reference other contexts via foreign key
    relations held by its variables, and may itself be referenced
    by other contexts if desired.

    """

    name: ContextName
    definition: Dfn
    variables: Vars
    base: Optional[type] = None
    parent: Optional[str] = None
    description: Optional[str] = None
    reference: Optional[Ref] = None
    references: Optional[Refs] = None


def make_context(
    name: ContextName,
    definition: Dfn,
    commonvars: Optional[Dfn] = None,
    references: Optional[Refs] = None,
) -> Context:
    """
    Extract from an input definition a context descriptor:
    a structured representation of an input context class.

    Each input definition yields one or more input contexts.
    The `name` parameter selects which context to make.

    A map of common variables may be provided, which can be
    referenced in the given context's variable descriptions.

    A map of other definitions may be provided, in which case a
    parameter in this context may act as kind of "foreign key",
    identifying another context as a subpackage which this one
    is related to.

    Notes
    -----
    This function does most of the work in the whole module.
    A bit of a beast, but convenient to use the outer scope
    (including the input definition, etc) in inner functions
    without sending around a lot of parameters. And it's not
    complicated; we just map a variable specification from a
    definition file to a corresponding Python representation.
    """

    _definition = dict(definition)
    _commonvars = dict(commonvars or dict())
    _references = dict(references or dict())

    # is this context referenceable?
    reference = Ref.from_dfn(definition)

    # contexts referenced by this one
    referenced = dict()

    def _parent() -> Optional[str]:
        """
        Get a string parameter name for the context's parent(s),
        i.e. context(s) which can own an instance of this one.

        If this context is a subpackage with multiple possible
        parent types "x" and "y, this will be of form "x_or_y".

        """
        l, r = definition.name
        if (l, r) == ("sim", "nam") and name == ("sim", "nam"):
            return None
        if l in ["sim", "exg", "sln"] or name.r is None:
            return "simulation"
        if reference:
            if len(reference.parents) > 1:
                return "_or_".join(reference.parents)
            return reference.parents[0]
        return "model"

    def _convert(var: Dict[str, Any], wrap: bool = False) -> Var:
        """
        Transform a variable from its original representation in
        an input definition to a Python specification appropriate
        for generating an input context class.

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

        _name = var["name"]
        _type = var.get("type", None)
        block = var.get("block", None)
        shape = var.get("shape", None)
        shape = None if shape == "" else shape
        default = var.get("default", None)
        descr = var.get("description", "")

        # if the var is a foreign key, register the referenced context
        ref = _references.get(_name, None)
        if ref:
            referenced[_name] = ref

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
                cmn_var = _commonvars.get(key, None)
                if cmn_var is None:
                    raise ValueError(f"Common variable not found: {key}")
                descr = cmn_var.get("description", "")
                if any(subs):
                    return descr.replace("\\", "").replace(
                        "{#1}", subs["{#1}"]
                    )
                return descr
            return descr

        def _default(value: str) -> Any:
            """
            Try to parse a default value as a literal.
            """
            if _type != "string":
                try:
                    return literal_eval(value)
                except (SyntaxError, ValueError):
                    return value

        def _fields(record_name: str) -> Vars:
            """Recursively load/convert a record's fields."""
            record = _definition[record_name]
            field_names = record["type"].split()[1:]
            fields: Dict[str, Var] = {
                n: _convert(field, wrap=False)
                for n, field in _definition.items()
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

        def _var() -> Var:
            """
            Create the variable.

            Notes
            -----
            Goes through all the possible input kinds
            from top (composites) to bottom (scalars):

            - list
            - union
            - record
            - array
            - scalar

            Creates and returs a variable of the proper
            kind. This may be a composite variable, in
            which case nested variables are recursively
            created as needed to produce the composite.
            """

            children = dict()

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

                is_explicit_record = len(names) == 1 and _definition[names[0]][
                    "type"
                ].startswith("record")

                def _is_implicit_scalar_record():
                    # if the record is defined implicitly and it has
                    # only scalar fields
                    types = [
                        v["type"] for n, v in _definition.items() if n in names
                    ]
                    return all(t in _SCALAR_TYPES for t in types)

                if is_explicit_record:
                    record_name = names[0]
                    record_spec = _definition[record_name]
                    record = _convert(record_spec, wrap=False)
                    children = {record_name: record}
                    kind = VarKind.List
                elif _is_implicit_scalar_record():
                    record_name = _name
                    fields = _fields(record_name)
                    record = Var(
                        name=record_name,
                        _type=_type.split()[0],
                        kind=VarKind.Record,
                        block=block,
                        children=fields,
                        description=descr,
                    )
                    children = {record_name: record}
                    kind = VarKind.List
                else:
                    # implicit complex record (i.e. some fields are records or unions)
                    fields = {
                        n: _convert(_definition[n], wrap=False) for n in names
                    }
                    first = list(fields.values())[0]
                    single = len(fields) == 1
                    record_name = first.name if single else _name
                    record = Var(
                        name=record_name,
                        _type=_type.split()[0],
                        kind=VarKind.Record,
                        block=block,
                        children=first.children if single else fields,
                        description=descr,
                    )
                    children = {record_name: record}
                    kind = VarKind.List

            # union (product), children are record choices
            elif _type.startswith("keystring"):
                names = _type.split()[1:]
                children = {
                    n: _convert(_definition[n], wrap=True) for n in names
                }
                kind = VarKind.Union

            # record (sum), children are fields
            elif _type.startswith("record"):
                children = _fields(_name)
                kind = VarKind.Record

            # are we wrapping a var into a record
            # as a choice in a union?
            elif wrap:
                field_name = _name
                field = _convert(var, wrap=False)
                children = {field_name: field}
                kind = VarKind.Record

            # at this point, if it has a shape, it's an array
            elif shape is not None:
                if _type not in _SCALAR_TYPES:
                    raise TypeError(f"Unsupported array type: {_type}")
                elif _type == "string":
                    kind = VarKind.List
                else:
                    kind = VarKind.Array

            # finally scalars
            else:
                kind = VarKind.Scalar

            # create var
            return Var(
                # if name is a reserved keyword, add a trailing underscore to it.
                # convert dashes to underscores since it may become a class attr.
                name=(f"{_name}_" if _name in kwlist else _name).replace(
                    "-", "_"
                ),
                _type=_type,
                kind=kind,
                block=block,
                description=_description(descr),
                default=_default(default),
                children=children,
                reference=ref,
            )

        return _var()

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

        vars_ = _definition.copy()
        vars_ = {
            name: _convert(var, wrap=False)
            for name, var in vars_.items()
            # skip composites, we already inflated
            # their parents in the var hierarchy
            if not var.get("in_record", False)
        }

        # reset var name since we may have altered
        # it when creating the variable e.g. to
        # avoid a reserved keyword collision
        return {v.name: v for v in vars_.values()}

    return Context(
        name=name,
        definition=definition,
        variables=_variables(),
        base=name.base,
        parent=_parent(),
        description=name.description,
        reference=reference,
        references=referenced,
    )


def make_contexts(
    definition: Dfn,
    commonvars: Optional[Dfn] = None,
    references: Optional[Refs] = None,
) -> Iterator[Context]:
    """Generate input contexts from the given input definition."""
    for name in get_context_names(definition.name):
        yield make_context(
            name=name,
            definition=definition,
            commonvars=commonvars,
            references=references,
        )
