"""
    the main entry point of utils

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """
from .mfreadnam import parsenamefile
from .util_array import Util3d, Util2d, Transient2d, read1d
from .util_list import MfList
from .binaryfile import BinaryHeader, HeadFile, UcnFile, CellBudgetFile
from .formattedfile import FormattedHeadFile
from .modpathfile import PathlineFile, EndpointFile
from .binaryswrfile import SwrObs, SwrStage, SwrBudget, SwrFlow, SwrExchange, \
    SwrStructure
from .binaryhydmodfile import HydmodObs
from .reference import SpatialReference  # , TemporalReference
from .mflistfile import MfListBudget, MfusgListBudget, SwtListBudget, \
    SwrListBudget
from .check import check, get_neighbors
