"""
Deprecated version of module to read MODFLOW 6 binary grid files (*.grb) that
define the model grid binary output files. This function imports the current
version of MfGrdFile from ..mf6.utils and returns the instantiated object.
This version is deprecated and will be removed.

"""

import warnings

warnings.simplefilter("always", DeprecationWarning)


def MfGrdFile(filename, precision="double", verbose=False):
    """
    The deprecated MfGrdFile class.

    Parameters
    ----------
    filename : str
        Name of the MODFLOW 6 binary grid file
    precision : string
        'single' or 'double'.  Default is 'double'.
    verbose : bool
        Write information to the screen.  Default is False.

    Returns
    -------
    mfg : MfGrdFile
        MfGrdFile object instantiated using flopy.mf6.utils.MfGrdFile

    """

    warnings.warn(
        "flopy.utils.MfGrdFile has been deprecated and will be "
        "removed in version 3.3.5. Use flopy.mf6.utils.MfGrdFile instead.",
        category=DeprecationWarning,
    )

    from ..mf6.utils import MfGrdFile as deprecated_MfGrdFile

    return deprecated_MfGrdFile(filename, precision=precision, verbose=verbose)
