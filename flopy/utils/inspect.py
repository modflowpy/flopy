import inspect
from collections.abc import Mapping


def get_classes(module, predicate=None) -> Mapping[str, type]:
    """Find classes in a module which satisfy a predicate."""
    classes = inspect.getmembers(module, inspect.isclass)
    return {name: cls for name, cls in classes if predicate(cls)}
