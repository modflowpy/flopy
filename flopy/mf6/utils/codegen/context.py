from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
)

from flopy.mf6.utils.codegen.dfn import Dfn, Ref, Vars


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
            elif dfn.name in [
                ("gwf", "mvr"),
                ("gwf", "gnc"),
                ("gwt", "mvt"),
            ]:
                # TODO: remove special cases, deduplicate mfmvr.py/mfgwfmvr.py etc
                return [
                    Context.Name(*dfn.name),
                    Context.Name(None, dfn.name.r),
                ]
            return [Context.Name(*dfn.name)]

    name: Name
    vars: Vars
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
            yield Context(name=name, vars=dfn.data, meta=meta)
