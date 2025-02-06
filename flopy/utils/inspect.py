import inspect


def get_classes(module, predicate=None):
    """Find classes in a module which satisfy a predicate."""
    classes = inspect.getmembers(module, inspect.isclass)
    return [cls for name, cls in classes if predicate(cls)]
