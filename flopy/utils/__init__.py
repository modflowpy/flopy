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
from .util_array import Util3d, Util2d, Transient2d, Transient3d, read1d
from .util_list import MfList
from .binaryfile import BinaryHeader, HeadFile, UcnFile, CellBudgetFile
from .formattedfile import FormattedHeadFile
from .modpathfile import PathlineFile, EndpointFile
from .swroutputfile import SwrStage, SwrBudget, SwrFlow, SwrExchange, \
    SwrStructure
from .observationfile import HydmodObs, SwrObs
from .reference import SpatialReference  # , TemporalReference
from .mflistfile import MfListBudget, MfusgListBudget, SwtListBudget, \
    SwrListBudget
from .check import check, get_neighbors
from .utils_def import FlopyBinaryData, totim_to_datetime
from .flopy_io import read_fixed_var, write_fixed_var
from .zonbud import ZoneBudget, run_zonbud
from .mfgrdfile import MfGrdFile
