from __future__ import print_function
import numpy as np


class Params(object):
    """
    Class to define parameters that will be estimated using PEST.

    Parameters
    ----------
    mfpackage : The Modflow package type to associated with this parameter.
        'LPF' is one package that is working now.
    partype : The parameter type, such as 'hk'.  This must be a valid attribute
        in the mfpackage.
    parname : The parameter name, such as 'HK_1'.
    startvalue : The starting value for the parameter.
    lbound : The lower bound for the parameter.
    ubound : The upper bound for the parameter.
    idx : The indices to which this parameter applies.  For example, if the
        parameter applies to a part of a 3D MODFLOW array, then idx can be a
        tuple of layer, row, and column indices (e.g. (karray, iarray, jarray).
        This idx variable could also be a 3D bool array.  It is ultimately
        used to assign parameter to the array using arr[idx] = parname.
    transform : Parameter transformation type.
    """
    def __init__(self, mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform='log'):
        self.name = parname
        self.type = partype
        self.mfpackage = mfpackage
        self.startvalue = startvalue
        self.lbound = lbound
        self.ubound = ubound
        self.transform = transform
        self.idx = idx


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
        idx = np.where(zonearray == iz)
        parname = partype + '_' + str(iz)
        startvalue = parvals[i]
        p = Params(mfpackage, partype, parname, startvalue, lbound,
                   ubound, idx, transform)
        plist.append(p)
    return plist


