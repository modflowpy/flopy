"""
The purpose of this module is to keep special handling
necessary to support the current `flopy.mf6` generated
classes separate from more general templating and code
generation infrastructure.
"""


def _is_context(o) -> bool:
    d = dict(o)
    return "name" in d and "vars" in d


def _replace_refs(ctx: dict, name_param: str = "val") -> dict:
    refs = ctx.get("meta", dict()).get("refs", dict())
    if any(refs):
        for key, ref in refs.items():
            key_var = ctx["vars"].get(key, None)
            if not key_var:
                continue
            ctx["vars"][key] = {
                **key_var,
                "name": ref[name_param],
                "description": (
                    f"* Contains data for the {ref['abbr']} package. Data can be "
                    f"stored in a dictionary containing data for the {ref['abbr']} "
                    "package with variable names as keys and package data as "
                    f"values. Data just for the {ref['val']} variable is also "
                    f"acceptable. See {ref['abbr']} package documentation for more "
                    "information"
                ),
                "ref": ref,
                "default": None,
                "children": None,
            }
    return ctx


def _transform_context(o):
    ctx = dict(o)
    ctx_name = ctx["name"]
    ctx_base = ctx_name.base
    if ctx_base == "MFSimulationBase":
        return _replace_refs(ctx, name_param="param")
    else:
        return _replace_refs(ctx)


SHIM = {
    "keep_none": ["default", "vars"],
    "quote_str": ["default"],
    "transform": [(_is_context, _transform_context)],
}
