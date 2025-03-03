from collections.abc import Mapping


def get_classes(predicate=None) -> Mapping[str, type]:
    import flopy.mf6.modflow as modflow
    from flopy.utils.inspect import get_classes as _get_classes

    return _get_classes(
        modflow,
        lambda cls: hasattr(cls, "dfn") and (predicate(cls) if predicate else True),
    )


def get_multi_packages() -> Mapping[str, type]:
    def _filter(cls):
        return (
            len(cls.dfn) > 0
            and len(cls.dfn[0]) >= 2
            and cls.dfn[0][1] == "multi-package"
        )

    return get_classes(_filter)


def get_solution_packages() -> Mapping[str, type]:
    def _filter(cls):
        return (
            len(cls.dfn) > 0
            and len(cls.dfn[0]) >= 2
            and len(cls.dfn[0][1]) > 0
            and cls.dfn[0][1][0] == "solution_package"
        )

    return get_classes(_filter)


def get_sub_packages() -> Mapping[str, type]:
    def _filter(cls):
        return (
            len(cls.dfn) > 0
            and len(cls.dfn[0]) >= 2
            and len(cls.dfn[0][1]) > 0
            and cls.dfn[0][1][0] == "subpackage"
        )
    
    return get_classes(_filter)
