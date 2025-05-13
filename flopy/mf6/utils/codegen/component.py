from typing import (
    Iterator,
    TypedDict,
)

from flopy.mf6.utils.dfn import Dfn


def get_component_names(dfn: dict) -> list[tuple[str, str]]:
    """
    Get the names of components produced by the definition.
    A definition may produce one or more component classes.
    """
    name = dfn.get("name", None)
    if not name:
        raise ValueError(f"DFN must have a 'name' entry")
    name = name.split("-")
    if name[1] == "nam":
        if name[0] == "sim":
            return [
                (None, name[1]),  # nam pkg
                tuple([*name]),  # simulation
            ]
        else:
            return [
                tuple([*name]),  # nam pkg
                (name[0], None),  # model
            ]
    elif name in [
        ["gwf", "mvr"],
        ["gwf", "gnc"],
        ["gwt", "mvt"],
    ]:
        # TODO: deduplicate mfmvr.py/mfgwfmvr.py etc and remove special cases
        return [
            tuple([*name]),
            (None, name[1]),
        ]
    return [tuple([*name])]


class ComponentDescriptor(TypedDict):
    """
    MODFLOW 6 input component class descriptor. Each component is
    specified by a definition file. A definition file specifies 1+
    components (e.g. model DFNs yield a model and a package class).
    """

    name: tuple[str, str]

    @staticmethod
    def from_dfn(dfn: Dfn) -> Iterator["ComponentDescriptor"]:
        """
        Yield component class descriptors from an input definition.
        """
        for name in get_component_names(dfn):
            yield ComponentDescriptor(name=name, dfn=dfn)
