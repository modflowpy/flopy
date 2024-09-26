from typing import Dict, List, Tuple, Union

Scalar = Union[bool, int, float, str]
Definition = Dict[str, Dict[str, Scalar]]


def load_dfn(f) -> Tuple[Definition, List[str]]:
    """
    Load an input definition file. Returns a tuple containing
    a dictionary variables and a list of metadata attributes.
    """
    meta = list()
    vars = dict()
    var = dict()

    for line in f:
        # remove whitespace/etc from the line
        line = line.strip()

        # record flopy metadata attributes but
        # skip all other comment lines
        if line.startswith("#"):
            _, sep, tail = line.partition("flopy")
            if sep == "flopy":
                meta.append(tail.strip())
            continue

        # if we hit a newline and the parameter dict
        # is nonempty, we've reached the end of its
        # block of attributes
        if not any(line):
            if any(var):
                vars[var["name"]] = var
                var = dict()
            continue

        # split the attribute's key and value and
        # store it in the parameter dictionary
        key, _, value = line.partition(" ")
        var[key] = value

    # add the final parameter
    if any(var):
        vars[var["name"]] = var

    return vars, meta
