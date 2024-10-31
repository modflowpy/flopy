"""
This module contains a decorator intended to
allow modifying dataclass instances to make
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
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from flopy.mf6.utils.codegen.utils import try_get_enum_value

Predicate = Callable[[Any], bool]
Transform = Callable[[Any], Dict[str, str]]
Pair = Tuple[str, Any]
Pairs = Iterable[Pair]


def renderable(
    maybe_cls=None,
    *,
    keep_none: Optional[Iterable[str]] = None,
    drop_keys: Optional[Iterable[str]] = None,
    quote_str: Optional[Iterable[str]] = None,
    set_pairs: Optional[Iterable[Tuple[Predicate, Pairs]]] = None,
    transform: Optional[Iterable[Tuple[Predicate, Transform]]] = None,
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
    Because a transformation function accepts an
    instance of a dataclass and converts it to a
    dictionary, only one transformation function
    (of the first matching predicate) is applied.

    This was inspired by `attrs` class decorators.
    """

    quote_str = quote_str or list()
    keep_none = keep_none or list()
    drop_keys = drop_keys or list()
    set_pairs = set_pairs or list()
    transform = transform or list()

    def __renderable(cls):
        def _render(d: dict) -> dict:
            def _render_val(k, v):
                v = try_get_enum_value(v)
                if (
                    k in quote_str
                    and isinstance(v, str)
                    and v[0] not in ["'", '"']
                ):
                    v = f"'{v}'"
                elif isinstance(v, dict):
                    v = _render(v)
                return v

            def _keep(k, v):
                return k in keep_none or (v and not isinstance(v, bool))

            def _drop(k, v):
                return k in drop_keys

            return {
                k: _render_val(k, v)
                for k, v in d.items()
                if (_keep(k, v) and not _drop(k, v))
            }

        def _dict(o):
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
            """Recursively render the dataclass instance."""
            return _render(
                asdict(self, dict_factory=_dict_factory)
                if is_dataclass(self)
                else self
            )

        setattr(cls, "render", render)
        return cls

    # first arg value depends on the decorator usage:
    # class if `@renderable`, `None` if `@renderable()`.
    # referenced from https://github.com/python-attrs/attrs/blob/a59c5d7292228dfec5480388b5f6a14ecdf0626c/src/attr/_next_gen.py#L405C4-L406C65
    return __renderable if maybe_cls is None else __renderable(maybe_cls)
