import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union, get_args, get_origin

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flopy.mf6.utils.codegen.dfn import Metadata
from flopy.mf6.utils.codegen.ref import Ref


class VarKind(Enum):
    """
    An input variable's kind. This is an enumeration
    of the general shapes of data MODFLOW 6 accepts,
    convertible to/from Python primitives/composites.
    """

    Array = "array"
    Scalar = "scalar"
    Record = "record"
    Union = "union"
    List = "list"

    @classmethod
    def from_type(cls, t: type) -> Optional["VarKind"]:
        origin = get_origin(t)
        args = get_args(t)
        if origin is Union:
            if len(args) >= 2 and args[-1] is type(None):
                if len(args) > 2:
                    return VarKind.Union
                return cls.from_type(args[0])
            return VarKind.Union
        if origin is np.ndarray or origin is NDArray or origin is ArrayLike:
            return VarKind.Array
        elif origin is collections.abc.Iterable or origin is list:
            return VarKind.List
        elif origin is tuple:
            return VarKind.Record
        try:
            if issubclass(t, (bool, int, float, str)):
                return VarKind.Scalar
        except:
            pass
        return None

    def to_type(self) -> type:
        # TODO
        pass


@dataclass
class Var:
    """An input variable specification."""

    name: str
    _type: Union[type, str]
    block: Optional[str]
    description: Optional[str]
    default: Optional[Any]
    children: Optional[Dict[str, "Var"]]
    metadata: Optional[Metadata]
    reference: Optional[Ref]

    def __init__(
        self,
        name: str,
        _type: Optional[type] = None,
        block: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        children: Optional["Vars"] = None,
        metadata: Optional[Metadata] = None,
        reference: Optional[Ref] = None,
    ):
        self.name = name
        self._type = _type or Any
        self.block = block
        self.description = description
        self.default = default
        self.children = children
        self.metadata = metadata
        self.reference = reference


Vars = Dict[str, Var]
