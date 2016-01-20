__author__ = 'emorway'

import sys
import numpy as np

from ..pakbase import Package
from flopy.utils import Util2d, Util3d, read1d, MfList
class Mt3dSft(Package):
    """
    MT3D-USGS Contaminant Treatment System package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    mxcts : int
        The maximum number of contaminant transport systems implemented in a 
        simulation.
    ictsout : int
        The unit number on which well-by-well output information is written. 
        The default file extension assigned to the output file is TSO
    mxext : int
        The maximum number of extraction wells specified as part of a 
        contaminant treatment system
    mxinj: int
        The maximum number of injection wells specified as part of a 
        contaminant treatment system
    mxwel : int
        The maximum number of wells in the flow model. MXWEL is recommended 
        to be set equal to MXWEL as specified in the WEL file
    iforce : int
        A flag to force concentration in treatment systems to satisfy 
        specified concentration/mass values based on the treatment option 
        selected without considering whether treatment is necessary or not. 
        This flag is ignored if 'no treatment' option is selected.
           0   Concentration for all injection wells is set to satisfy 
               treatment levels only if blended concentration exceeds 
               the desired concentration/mass level for a treatment system. 
               If the blended concentration in a treatment system is less 
               than the specified concentration/mass level, then injection 
               wells inject water with blended concentrations. 
           1   Concentration for all injection wells is forced to satisfy 
               specified concentration/mass values.
    ncts
