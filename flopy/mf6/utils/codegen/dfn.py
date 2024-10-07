from collections import UserDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from boltons.dictutils import OMD


class DfnName(NamedTuple):
    """
    Uniquely identifies an input definition by its name, which
    consists of a <= 3-letter left term and an optional right
    term, also <= 3 letters.

    Notes
    -----
    A single `DefinitionName` may be associated with one or
    more `ContextName`s. For instance, a model DFN file will
    produce both a NAM package class and also a model class.
    """

    l: str
    r: str


Metadata = List[str]


@dataclass
class Dfn(UserDict):
    """
    An MF6 input definition.

    Notes
    -----
    Duplicate variable names are supported by an `OrderedMultiDict`
    this class maintains alongside a `UserDict`-managed standard
    dictionary; the former is retrievable with the `omd` property.

    This class should not be modified after loading.
    """

    name: Optional[DfnName]
    metadata: Optional[Metadata]

    def __init__(
        self,
        variables: Iterable[Tuple[str, Dict[str, Any]]],
        name: Optional[DfnName] = None,
        metadata: Optional[Metadata] = None,
    ):
        self.omd = OMD(variables)
        self.name = name
        self.metadata = metadata
        super().__init__(self.omd)


Dfns = Dict[str, Dfn]


def load_dfn(f, name: Optional[DfnName] = None) -> Dfn:
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

    return Dfn(variables=vars_, name=name, metadata=meta)
