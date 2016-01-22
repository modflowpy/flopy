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
    ncts : int
        The number of contaminant treatment systems.  If NCTS >= 0, 
        NCTS is the number of contaminant treatment systems. If NCTS = -1, 
        treatment system information from the previous stress period is reused 
        for the current stress period.
    icts : int
        The contaminant treatment system index number.
    next : int
        The number of extraction wells for the treatment system number ICTS.
    ninj : int
        The number of injection wells for the treatment system number ICTS.
    itrtinj : int
        Is the level of treatment provided for the treatment system number 
        ICTS. Each treatment system blends concentration collected from all 
        extraction wells contributing to the treatment system and assigns a 
        treated concentration to all injection wells associated with that 
        treatment system based on the treatment option selected
            0   no treatment is provided
            1   same level of treatment is provided to all injection wells. 
            2   different level of treatment can be provided to each 
                individual injection well.
    qincts : float
        The external flow entering a treatment system. External flow may be 
        flow entering a treatment system that is not a part of the model 
        domain but plays an important role in influencing the blended 
        concentration of a treatment system
    cincts : float
        The concentration with which the external flow enters a treatment 
        system
    ioptinj : int
        Is a treatment option. Negative values indicate removal of 
        concentration/mass and positive values indicate addition of 
        concentration/mass.
            1   Percentage concentration/mass addition/removal is performed. 
                Percentages must be specified as fractions. Example, for 50% 
                concentration/mass removal is desired, -0.5 must be specified.
            2   Concentration is added/removed from the blended concentration. 
                Specified concentration CMCHGINJ is added to the blended 
                concentration. If the specified concentration removal, 
                CMCHGINJ, is greater than the blended concentration, the 
                treated concentration is set to zero.
            3   Mass is added/removed from the blended concentration. 
                Specified mass CMCHGINJ is added to the blended concentration. 
                If the specified mass removal, CMCHGINJ, is greater than the 
                blended total mass, the treated concentration is set to zero.
            4   Specified concentration is set equal to the entered value 
                CMCHGINJ. A positive value is expected for CMCHGINJ with this 
                option. 
    cmchginj : float
        Is the addition, removal, or specified concentration/mass values set 
        for the treatment system. Concentration/mass is added, removed, or 
        used as specified concentrations depending on the treatment option 
        IOPTINJ.
        Note that concentration/mass values as specified by CMCHGINJ are 
        enforced if the option IFORCE is set to 1. If IFORCE is set to 0, 
        then CMCHGINJ is enforced only when the blended concentration exceeds 
        the specified concentration CNTE. 
    cnte : float
        The concentration that is not to be exceeded for a treatment system. 
        Treatment is applied to blended concentration only if it exceeds 
        CNTE, when IFORCE is set to 0.
    kinj : int
        Layer index for a CTS injection well
    iinj : int
        Row index for a CTS injection well
    jinj : int
        Column index for a CTS injection well
    iwinj : int
        The well index number. This number corresponds to the well number as 
        it appears in the WEL file of the flow model.
    qoutcts : float
        the flow rate of outflow from a treatment system to an external sink. 
        This flow rate must be specified to maintain an overall treatment 
        system mass balance. QOUTCTS must be set equal to total inflow into a 
        treatment system minus total outflow to all injection wells for a 
        treatment system

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>>
    >>>
    >>>
    >>>

    """

    unitnumber