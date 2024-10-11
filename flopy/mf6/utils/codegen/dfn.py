from collections import UserList
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple


class DfnName(NamedTuple):
    """
    Uniquely identifies an input definition by its name, which
    consists of a <= 3-letter left term and an optional right
    term.
    """

    l: str
    r: str


Metadata = List[str]


class Dfn(UserList):
    """
    An MF6 input definition.

    Notes
    -----
    This class is a list rather than a dictionary to
    accommodate duplicate variable names. Dictionary
    would be nicer; this constraint goes away if the
    DFN specifications become nested instead of flat.

    With conversion to a standard format we get this
    for free, and we could then drop the custom load.
    """

    name: Optional[DfnName]
    metadata: Optional[Metadata]

    def __init__(
        self,
        variables: Optional[Iterable[Tuple[str, Dict[str, Any]]]] = None,
        name: Optional[DfnName] = None,
        metadata: Optional[Metadata] = None,
    ):
        super().__init__(variables)
        self.name = name
        self.metadata = metadata or []

    @classmethod
    def load(cls, f, name: Optional[DfnName] = None) -> "Dfn":
        """
        Load an input definition from a definition file.
        """

        meta = None
        vars_ = list()
        var = dict()

        for line in f:
            # remove whitespace/etc from the line
            line = line.strip()

            # record context name and flopy metadata
            # attributes, skip all other comment lines
            if line.startswith("#"):
                _, sep, tail = line.partition("flopy")
                if sep == "flopy":
                    if meta is None:
                        meta = list()
                    tail = tail.strip()
                    if "solution_package" in tail:
                        tail = tail.split()
                        tail.pop(1)
                    meta.append(tail)
                    continue
                _, sep, tail = line.partition("package-type")
                if sep == "package-type":
                    if meta is None:
                        meta = list
                    meta.append(f"{sep} {tail.strip()}")
                    continue
                _, sep, tail = line.partition("solution_package")
                continue

            # if we hit a newline and the parameter dict
            # is nonempty, we've reached the end of its
            # block of attributes
            if not any(line):
                if any(var):
                    n = var["name"]
                    vars_.append((n, var))
                    var = dict()
                continue

            # split the attribute's key and value and
            # store it in the parameter dictionary
            key, _, value = line.partition(" ")
            if key == "default_value":
                key = "default"
            if value in ["true", "false"]:
                value = value == "true"
            var[key] = value

        # add the final parameter
        if any(var):
            n = var["name"]
            vars_.append((n, var))

        return cls(variables=vars_, name=name, metadata=meta)


Dfns = Dict[str, Dfn]
