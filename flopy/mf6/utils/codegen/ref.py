from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from flopy.mf6.utils.codegen.dfn import Dfn


@dataclass
class Ref:
    """
    A foreign-key-like reference between a file input variable
    and another input definition. This allows an input context
    to refer to another input context, by including a filepath
    variable whose name acts as a foreign key for a different
    input context. Extra parameters are added to the referring
    context's `__init__` method, so a selected "value" variable
    defined in the referenced context can be provided directly
    as an alternative to the file path (foreign key) variable.

    Parameters
    ----------
    key : str
        The name of the foreign key file input variable.
    val : str
        The name of the selected variable in the referenced context.
    abbr : str
        An abbreviation of the referenced context's name.
    param : str
        The subpackage parameter name. TODO: explain
    parents : List[Union[str, type]]
        The subpackage's supported parent types.
    """

    key: str
    val: str
    abbr: str
    param: str
    parents: List[Union[type, str]]
    description: Optional[str]

    @classmethod
    def from_dfn(cls, dfn: Dfn) -> Optional["Ref"]:
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


Refs = Dict[str, Ref]
