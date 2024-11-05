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
        consists of a left term and optional right term.

        Notes
        -----
        A single definition may be associated with one or more
        contexts. For instance, a model DFN file will produce
        both a namefile package class and a model class.

        From the context name several other things are derived:

        - a description of the context
        - the input context class' name
        - the template the context will populate
        - the base class the context inherits from
        - the name of the source file the context is in
        - the name of the parent parameter in the context
        class' `__init__` method, if it can have a parent

        """

        l: str
        r: Optional[str]

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
