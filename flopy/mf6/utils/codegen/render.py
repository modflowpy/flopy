from dataclasses import asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

_Predicate = Callable[[Any], bool]
_Transform = Callable[[Any], Dict[str, str]]
_Pair = Tuple[str, Any]
_Pairs = Iterable[_Pair]


def _try_get_enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def renderable(
    maybe_cls=None,
    *,
    keep_none: Optional[Iterable[str]] = None,
    quote_str: Optional[Iterable[str]] = None,
    set_pairs: Optional[Iterable[Tuple[_Predicate, _Pairs]]] = None,
    transform: Optional[Iterable[Tuple[_Predicate, _Transform]]] = None,
):
    """
    Decorator for dataclasses which are meant
    to be passed into a Jinja template. The
    decorator adds a `.render()` method to
    the decorated class, which recursively
    converts the instance to a dictionary
    with (by default) the `asdict()` builtin
    `dataclasses` module function, plus a
    few modifications to make the instance
    easier to work with from the template.

    By default, attributes with value `None`
    are dropped before conversion to a `dict`.
    To specify that a given attribute should
    remain even with a `None` value, use the
    `keep_none` parameter.

    When a string value is to become the RHS
    of an assignment or an argument-passing
    expression, it needs to be wrapped with
    quotation marks before insertion into
    the template. To indicate an attribute's
    value should be wrapped with quotation
    marks, use the `quote_str` parameter.

    Arbitrary transformations can be configured
    via the `transform` parameter, which accepts
    an iterable of predicate / function tuples.
    Each of these specifies a condition in which
    an instance of a context should be modified,
    and a function to make the alteration.

    Notes
    -----
    This decorator is intended as a convenient
    way to modify dataclass instances to make
    them more palatable for templates. It also
    keeps implementation details incidental to
    the current design of MF6 input framework
    cleanly isolated from the reimplementation
    of which this code is a part, which aims
    for a more general approach.

    Jinja supports attribute- and dictionary-
    based access on arbitrary objects but does
    not support arbitrary expressions, and has
    only a limited set of custom filters; this
    can make it awkward to express some things,
    which transformations can also remedy.

    Edge cases in the MF6 classes, e.g. the logic
    determining the contents of generated classes,
    can also be implemented with transformations.
    This allows keeping the templating module as
    generic as possible and inserting "shims" to
    incrementally rewrite the existing framework.

    Because a transformation function accepts an
    instance of a dataclass and converts it to a
    dictionary, only one transformation function
    (of the first matching predicate) is applied.

    References
    ----------
    This pattern was heavily inspired by `attrs`'
    use of class decorators.
    """

    quote_str = quote_str or list()
    keep_none = keep_none or list()
    set_pairs = set_pairs or list()
    transform = transform or list()

    def __renderable(cls):
        def _render(d: dict) -> dict:
            """
            Render the dictionary recursively,
            with requested value modifications.
            """

            def _render_val(k, v):
                v = _try_get_enum_value(v)
                if (
                    k in quote_str
                    and isinstance(v, str)
                    and v[0] not in ["'", '"']
                ):
                    v = f"'{v}'"
                elif isinstance(v, dict):
                    v = _render(v)
                return v

            return {
                k: _render_val(k, v)
                for k, v in d.items()
                # drop nones except where requested
                if (k in keep_none or v is not None)
            }

        def _dict(o):
            """
            Convert the dataclass instance to a dictionary,
            applying a transformation if applicable and any
            extra key/value pairs if provided.
            """
            d = dict(o)
            for p, t in transform:
                if p(o):
                    d = t(o)
                    break

            for p, e in set_pairs:
                if not (p(d) and e):
                    continue
                for k, v in e:
                    if callable(v):
                        v = v(d)
                    d[k] = v

            return d

        def _dict_factory(o):
            return _render(_dict(o))

        def render(self) -> dict:
            """
            Recursively render the dataclass instance.
            """
            return _render(asdict(self, dict_factory=_dict_factory))

        setattr(cls, "render", render)
        return cls

    # first arg value depends on the decorator usage:
    # class if `@renderable`, `None` if `@renderable()`.
    # referenced from https://github.com/python-attrs/attrs/blob/a59c5d7292228dfec5480388b5f6a14ecdf0626c/src/attr/_next_gen.py#L405C4-L406C65
    return __renderable if maybe_cls is None else __renderable(maybe_cls)
