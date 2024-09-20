from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from flopy.mf6.utils.codegen.dfn import Metadata
from flopy.mf6.utils.codegen.ref import Ref


class VarKind(Enum):
    """
    An input variable's kind. This is an enumeration
    of the general shapes of data MODFLOW 6 accepts.
    """

    Array = "array"
    Scalar = "scalar"
    Record = "record"
    Union = "union"
    List = "list"


@dataclass
class Var:
    """An input variable specification."""

    name: str
    _type: str
    kind: VarKind
    block: Optional[str]
    description: Optional[str]
    default: Optional[Any]
    children: Optional[Dict[str, "Var"]]
    metadata: Optional[Metadata]
    reference: Optional[Ref]

    def __init__(
        self,
        name: str,
        _type: str,
        kind: VarKind,
        block: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        children: Optional["Vars"] = None,
        metadata: Optional[Metadata] = None,
        reference: Optional[Ref] = None,
    ):
        self.name = name
        self._type = _type
        self.kind = kind
        self.block = block
        self.description = description
        self.default = default
        self.children = children
        self.metadata = metadata
        self.reference = reference


Vars = Dict[str, Var]
