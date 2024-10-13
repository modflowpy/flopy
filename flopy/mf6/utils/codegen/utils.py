import collections
from enum import Enum
from typing import Any, ForwardRef, Literal, Union, get_args, get_origin

import numpy as np


def _try_get_type_name(t) -> str:
    """Convert a type to a name suitable for templating."""
    origin = get_origin(t)
    args = get_args(t)
    if origin is Literal:
        args = ['"' + a + '"' for a in args]
        return f"Literal[{', '.join(args)}]"
    elif origin is Union:
        if len(args) >= 2 and args[-1] is type(None):
            if len(args) > 2:
                return f"Optional[Union[{', '.join([_try_get_type_name(a) for a in args[:-1]])}]]"
            return f"Optional[{_try_get_type_name(args[0])}]"
        return f"Union[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is tuple:
        return f"Tuple[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is collections.abc.Iterable:
        return f"Iterable[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is list:
        return f"List[{', '.join([_try_get_type_name(a) for a in args])}]"
    elif origin is np.ndarray:
        return f"NDArray[np.{_try_get_type_name(args[1].__args__[0])}]"
    elif origin is np.dtype:
        return str(t)
    elif isinstance(t, ForwardRef):
        return t.__forward_arg__
    elif t is Ellipsis:
        return "..."
    elif isinstance(t, type):
        return t.__qualname__
    return t


def _try_get_enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v
