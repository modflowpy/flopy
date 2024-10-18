from dataclasses import dataclass
from typing import Dict, Optional
from warnings import warn

from flopy.mf6.utils.codegen.dfn import Dfn


@dataclass
class Ref:
    """
    A foreign-key-like reference between a file input variable
    and another input definition. This allows an input context
    to refer to another input context, by including a filepath
    variable whose name acts as a foreign key for a different
    input context. The referring context's `__init__` method
    is modified such that the variable named `val` replaces
    the `key` variable.

    Notes
    -----
    This class is used to represent subpackage references.

    Parameters
    ----------
    key : str
        The name of the foreign key file input variable.
    val : str
        The name of the data variable in the referenced context.
    abbr : str
        An abbreviation of the referenced context's name.
    param : str
        The referenced parameter name.
    parents : List[str]
        The referenced context's supported parents.
    description : Optional[str]
        The reference's description.
    """

    key: str
    val: str
    abbr: str
    param: str
    parent: str
    description: Optional[str]

    @classmethod
    def from_dfn(cls, dfn: Dfn) -> Optional["Ref"]:
        """
        Try to load a reference from the definition.
        Returns `None` if the definition cannot be
        referenced by other contexts.

        """

        # TODO: all this won't be necessary once we
        # structure DFN format; we can then support
        # subpackage references directly instead of
        # by making assumptions about `dfn.meta`

        if not dfn.meta or "dfn" not in dfn.meta:
            return None

        _, meta = dfn.meta["dfn"]

        lines = {
            "subpkg": next(
                iter(
                    m
                    for m in meta
                    if isinstance(m, str) and m.startswith("subpac")
                ),
                None,
            ),
            "parent": next(
                iter(
                    m
                    for m in meta
                    if isinstance(m, str) and m.startswith("parent")
                ),
                None,
            ),
        }

        def _subpkg():
            line = lines["subpkg"]
            _, key, abbr, param, val = line.split()
            matches = [v for v in dfn.values() if v.name == val]
            if not any(matches):
                descr = None
            else:
                if len(matches) > 1:
                    warn(f"Multiple matches for referenced variable {val}")
                match = matches[0]
                descr = match.description

            return {
                "key": key,
                "val": val,
                "abbr": abbr,
                "param": param,
                "description": descr,
            }

        def _parent():
            line = lines["parent"]
            split = line.split()
            return split[1]

        return (
            cls(**_subpkg(), parent=_parent())
            if all(v for v in lines.values())
            else None
        )


Refs = Dict[str, Ref]
