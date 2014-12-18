"""
    the main entry point of the flopy library

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

#--version number
__name__='flopy'
__author__='Mark Bakker, Vincent Post, Chris Langevin, Joe Hughes, Jeremy White, Alain Frances, and Jeffrey Starn'
from version import __version__, __build__
#--modflow
import modflow
#--mt3d
import mt3d
#--seawat
import seawat
#--modpath
import modpath
#--utils
import utils



