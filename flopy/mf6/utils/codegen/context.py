from typing import (
    Iterator,
    List,
    NamedTuple,
    Optional,
    TypedDict,
)


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
        def from_dfn(dfn: dict) -> List["Context.Name"]:
            """
            Returns a list of context names this definition produces.
            An input definition may produce one or more input contexts.
            """
            name = dfn.get("name", None)
            if not name:
                raise ValueError(f"DFN must have a 'name' entry")
            name = name.split("-")
            if name[1] == "nam":
                if name[0] == "sim":
                    return [
                        Context.Name(None, name[1]),  # nam pkg
                        Context.Name(*name),  # simulation
                    ]
                else:
                    return [
                        Context.Name(*name),  # nam pkg
                        Context.Name(name[0], None),  # model
                    ]
            elif name in [
                ("gwf", "mvr"),
                ("gwf", "gnc"),
                ("gwt", "mvt"),
            ]:
                # TODO: deduplicate mfmvr.py/mfgwfmvr.py etc and remove special cases
                return [
                    Context.Name(*name),
                    Context.Name(None, name[1]),
                ]
            return [Context.Name(*name)]

    name: Name

    @staticmethod
    def from_dfn(dfn: dict) -> Iterator["Context"]:
        """
        Extract context(s) from an input definition.
        Each definition yields one or more contexts.
        """

        def get_vars(dfn_):
            vars_ = dict()
            blocks = dfn_.get("blocks", dict())
            for block in blocks.values():
                for name, var in block.items():
                    vars_[name] = var
            return vars_

        for name in Context.Name.from_dfn(dfn):
            dfn_ = dfn.copy()
            dfn_.pop("name", None)
            vars_ = get_vars(dfn_)
            yield Context(name=name, vars=vars_, **dfn_)
