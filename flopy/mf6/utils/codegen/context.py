from typing import (
    Iterator,
    List,
    NamedTuple,
    Optional,
    TypedDict,
)

from flopy.mf6.utils.codegen.dfn import Dfn, Vars


class Context(TypedDict):
    """
    An input context. Each of these is specified by a definition file
    and becomes a generated class. A definition file may specify more
    than one input context (e.g. model DFNs yield a model class and a
    package class).

    A context consists minimally of a name and a map of variables.
    """

    class Name(NamedTuple):
        """
        Uniquely identifies an input context. The name
        consists of a left (component) term and optional
        right (subcomponent) term.

        Notes
        -----
        A single definition may be associated with one or more
        contexts. For instance, a model DFN file will produce
        both a namefile package class and a model class. These
        share a single DFN name but have different context names.
        """

        l: str
        r: Optional[str]

        @staticmethod
        def from_dfn(dfn: Dfn) -> List["Context.Name"]:
            """
            Returns a list of context names this definition produces.
            An input definition may produce one or more input contexts.
            """
            name = dfn["name"]
            if name.r == "nam":
                if name.l == "sim":
                    return [
                        Context.Name(None, name.r),  # nam pkg
                        Context.Name(*name),  # simulation
                    ]
                else:
                    return [
                        Context.Name(*name),  # nam pkg
                        Context.Name(name.l, None),  # model
                    ]
            elif name in [
                ("gwf", "mvr"),
                ("gwf", "gnc"),
                ("gwt", "mvt"),
            ]:
                # TODO: deduplicate mfmvr.py/mfgwfmvr.py etc and remove special cases
                return [
                    Context.Name(*name),
                    Context.Name(None, name.r),
                ]
            return [Context.Name(*name)]

    name: Name
    vars: Vars

    @staticmethod
    def from_dfn(dfn: Dfn) -> Iterator["Context"]:
        """
        Extract context class descriptor(s) from an input definition.
        These are structured representations of input context classes.

        Each input definition yields one or more input contexts.
        """

        def _ctx(name, _dfn):
            _dfn = _dfn.copy()
            _dfn.pop("name", None)
            _vars = _dfn.pop("vars", dict())
            return Context(name=name, vars=_vars, **_dfn)

        for name in Context.Name.from_dfn(dfn):
            yield _ctx(name, dfn)
