from __future__ import print_function
import numpy as np


class Params(object):
    """
    Class to define parameters that will be estimated using PEST.

    Parameters
    ----------
    mfpackage : str
        The Modflow package type to associated with this parameter.
        'LPF' is one package that is working now.
    partype : str
        The parameter type, such as 'hk'.  This must be a valid attribute
        in the mfpackage.
    parname : str
        The parameter name, such as 'HK_1'.
    startvalue : float
        The starting value for the parameter.
    lbound : float
        The lower bound for the parameter.
    ubound : float
        The upper bound for the parameter.
    span : dict
        The span over which the parameter applies.  The span depends on the
        type of array that the parameter applies to.  For 3d arrays, span
        should have either 'idx' or 'layers' keys.  span['layers'] should
        be a list of layer to for which parname will be applied as a
        multiplier.
        idx is a tuple, which contains the indices to which this parameter
        applies.  For example, if the parameter applies to a part of a 3D
        MODFLOW array, then idx can be a tuple of layer, row, and column
        indices (e.g. (karray, iarray, jarray).
        This idx variable could also be a 3D bool array.  It is ultimately
        used to assign parameter to the array using arr[idx] = parname.
        For transient 2d arrays, span must include a 'kpers' key such that
        span['kpers'] is a list of stress period to which parname will be
        applied as a multiplier.
    transform : Parameter transformation type.
    """
    def __init__(self, mfpackage, partype, parname,
                 startvalue, lbound, ubound, span, transform='log'):
        self.name = parname
        self.type = partype
        self.mfpackage = mfpackage
        self.startvalue = startvalue
        self.lbound = lbound
        self.ubound = ubound
        self.transform = transform
        self.span = span
        return


def zonearray2params(mfpackage, partype, parzones, lbound, ubound,
                     parvals, transform, zonearray):
    """
    Helper function to create a list of flopy parameters from a zone array
    and list of parameter zone numbers.

    The parameter name is set equal to the parameter type and the parameter
    zone value, separated by an underscore.
    """
    plist = []
    for i, iz in enumerate(parzones):
        span = {}
        span['idx'] = np.where(zonearray == iz)
        parname = partype + '_' + str(iz)
        startvalue = parvals[i]
        p = Params(mfpackage, partype, parname, startvalue, lbound,
                   ubound, span, transform)
        plist.append(p)
    return plist


