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
from .utl_import import import_optional_dependency  # isort:skip
from . import get_modflow as get_modflow_module
from .binaryfile import (
    BinaryHeader,
    CellBudgetFile,
    HeadFile,
    HeadUFile,
    UcnFile,
)
from .check import check
from .flopy_io import read_fixed_var, write_fixed_var
from .formattedfile import FormattedHeadFile

get_modflow = get_modflow_module.run_main
from .gridintersect import GridIntersect, ModflowGridIndices
from .mflistfile import (
    Mf6ListBudget,
    MfListBudget,
    MfusgListBudget,
    SwrListBudget,
    SwtListBudget,
)
from .mfreadnam import parsenamefile
from .modpathfile import EndpointFile, PathlineFile, TimeseriesFile
from .mtlistfile import MtListBudget
from .observationfile import HydmodObs, Mf6Obs, SwrObs
from .optionblock import OptionBlock
from .postprocessing import get_specific_discharge, get_transmissivities
from .rasters import Raster
from .recarray_utils import create_empty_recarray, ra_slice
from .reference import TemporalReference
from .sfroutputfile import SfrFile
from .swroutputfile import (
    SwrBudget,
    SwrExchange,
    SwrFlow,
    SwrStage,
    SwrStructure,
)
from .util_array import Transient2d, Transient3d, Util2d, Util3d, read1d
from .util_list import MfList
from .utils_def import FlopyBinaryData, totim_to_datetime
from .zonbud import ZBNetOutput, ZoneBudget, ZoneBudget6, ZoneFile6
