from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
)

from flopy.mf6.utils.codegen.dfn import Dfn, Vars
from flopy.mf6.utils.codegen.ref import Ref
from flopy.mf6.utils.codegen.render import renderable
from flopy.mf6.utils.codegen.shim import SHIM


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
    A context minimally consists of a name and a map of variables.

    The context class may inherit from a base class, and may specify
    a parent context within which it can be created (the parent then
    becomes the first `__init__` method parameter).

    The context class may reference other contexts via foreign key
    relations held by its variables, and may itself be referenced
    by other contexts if desired.

    """

    class Name(NamedTuple):
        """
        Uniquely identifies an input context. A context
        consists of a left term and optional right term.

        Notes
        -----
        A single definition may be associated with one or more
        contexts. For instance, a model DFN file will produce
        both a namefile package class and a model class.

        From the context name several other things are derived:

        - the input context class' name
        - a description of the context class
        - the name of the source file to write
        - the base class the context inherits from
        - the name of the parent parameter in the context
        class' `__init__` method, if it can have a parent

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

        def parent(self, ref: Optional[Ref] = None) -> Optional[str]:
            """
            Return the name of the parent `__init__` method parameter,
            or `None` if the context cannot have parents. Contexts can
            have more than one possible parent, in which case the name
            of the parameter is of the pattern `name1_or_..._or_nameN`.
            """
            if ref:
                return ref.parent
            if self == ("sim", "nam"):
                return None
            elif (
                self.l is None
                or self.r is None
                or self.l in ["sim", "exg", "sln"]
            ):
                return "simulation"
            return "model"

        @staticmethod
        def from_dfn(dfn: Dfn) -> List["Context.Name"]:
            """
            Returns a list of context names this definition produces.

            Notes
            -----
            An input definition may produce one or more input contexts.

            Model definition files produce both a model class context and
            a model namefile package context. The same goes for simulation
            definition files. All other definition files produce a single
            context.
            """
            if dfn.name.r == "nam":
                if dfn.name.l == "sim":
                    return [
                        Context.Name(None, dfn.name.r),  # nam pkg
                        Context.Name(*dfn.name),  # simulation
                    ]
                else:
                    return [
                        Context.Name(*dfn.name),  # nam pkg
                        Context.Name(dfn.name.l, None),  # model
                    ]
            elif (dfn.name.l, dfn.name.r) in [
                ("gwf", "mvr"),
                ("gwf", "gnc"),
                ("gwt", "mvt"),
            ]:
                return [
                    Context.Name(*dfn.name),
                    Context.Name(None, dfn.name.r),
                ]
            return [Context.Name(*dfn.name)]

    name: Name
    vars: Vars
    base: Optional[type] = None
    parent: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dfn(cls, dfn: Dfn) -> Iterator["Context"]:
        """
        Extract context class descriptor(s) from an input definition.
        These are structured representations of input context classes.
        Each input definition yields one or more input contexts.
        """

        meta = dfn.meta.copy()
        ref = Ref.from_dfn(dfn)
        if ref:
            meta["ref"] = ref

        for name in Context.Name.from_dfn(dfn):
            yield Context(
                name=name,
                vars=dfn.data,
                base=name.base,
                parent=name.parent(ref),
                description=name.description,
                meta=meta,
            )
