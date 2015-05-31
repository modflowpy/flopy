# from numpy import empty, zeros, ones, where
import numpy as np
from flopy.mbase import Package


class ModflowMnw2(Package):
    """
    'Multi-Node Well 2 Package Class'

    Parameters
    ----------
    model : model object
        The model object (of type :class:'flopy.modflow.mf.Modflow') to which
        this package will be added.
    mnwmax : integer
        Maximum number of multi-node wells (MNW) to be simulated (default is 0)
    iwl2cb : integer
        A flag and a unit number
            if IWL2CB > 0, then it is the unit number to which MNW cell-by-cell flow terms will be
                recorded whenever cell-by-cell budget data are written to a file (as determined by the
                output control options of MODFLOW).
            if IWL2CB = 0, then MNW cell-by-cell flow terms will not be printed or recorded.
            if IWL2CB < 0, then well injection or withdrawal rates and water levels in the well and
                its multiple cells will be printed in the main MODFLOW listing (output) file whenever cell-by-cell
                budget data are written to a file (as determined by the output control options of MODFLOW).
        (default is -1)
    mnwprnt : integer
        Flag controlling the level of detail of information about multi-node wells to be written to the
        main MODFLOW listing (output) file. If MNWPRNT = 0, then only basic well information will be
        printed in the main MODFLOW output file; increasing the value of MNWPRNT yields more information,
        up to a maximum level of detail corresponding with MNWPRNT = 2.
        (default is 0)
    aux : list of strings
    (note: not sure if the words AUX or AUXILIARY are necessary)
        in the style of AUXILIARY abc or AUX abc where
        abc is the name of an auxiliary parameter to be read for each multi-node well as part
        of dataset 4a. Up to five parameters can be specified, each of which must be preceded
        by AUXILIARY or AUX. These parameters will not be used by the MNW2 Package, but
        they will be available for use by other packages.
        (default is None)
    wellid : array of strings (shape = (MNWMAX))
        The name of the wells. This is a unique identification label for each well.
        The text string is limited to 20 alphanumeric characters. If the name of the well includes
        spaces, then enclose the name in quotes.
        (default is None)
    nnodes : integers
        The number of cells (nodes) associated with this well. NNODES normally is > 0, but for the
        case of a vertical borehole, setting NNODES < 0 will allow the user to specify the elevations
        of the tops and bottoms of well screens or open intervals (rather than grid layer numbers),
        and the absolute value of NNODES equals the number of open intervals (or well screens) to be
        specified in dataset 2d. If this option is used, then the model will compute the layers in which
        the open intervals occur, the lengths of the open intervals, and the relative vertical position
        of the open interval within a model layer (for example, see figure 14 and related discussion)
        (default is None)
    losstype : string
        The user-specified model for well loss. The following loss types are currently supported.
            NONE there are no well corrections and the head in the well is assumed to equal the head
                in the cell. This option (hWELL = hn) is only valid for a single-node well (NNODES = 1).
                (This is equivalent to using the original WEL Package of MODFLOW, but specifying the single-node
                well within the MNW2 Package enables the use of constraints.)
            THIEM this option allows for only the cell-to-well correction at the well based on the Thiem (1906)
                equation; head in the well is determined from equation 2 as (hWELL = hn + AQn), and the model
                computes A on the basis of the user-specified well radius (Rw) and previously defined values of
                cell transmissivity and grid spacing. Coefficients B and C in equation 2 are automatically
                set = 0.0. User must define Rw in dataset 2c or 2d.
            SKIN this option allows for formation damage or skin corrections at the well:
                hWELL = hn + AQn + BQn (from equation 2), where A is determined by the model from the value of
                Rw, and B is determined by the model from Rskin and Kskin. User must define Rw, Rskin, and Kskin in
                dataset 2c or 2d.
            GENERAL head loss is defined with coefficients A, B, and C and power exponent P
                (hWELL = hn + AQn + BQn + CQnP). A is determined by the model from the value of Rw.
                must define Rw, B, C, and P in dataset 2c or 2d. A value of P = 2.0 is suggested if no other
                data are available (the model allows 1.0 <= P <= 3.5). Entering a value of C = 0 will result
                in a linear model in which the value of B is entered directly (rather than entering properties
                of the skin, as with the SKIN option).
            SPECIFYcwc the user specifies an effective conductance value (equivalent to the combined
                effects of the A, B, and C well-loss coefficients expressed in equation 15) between the well and
                the cell representing the aquifer, CWC. User must define CWC in dataset 2c or 2d. If there are
                multiple screens within the grid cell or if partial penetration corrections are to be made, then
                the effective value of CWC for the node may be further adjusted automatically by MNW2.
                (default is None)
    pumploc : integer
        The location along the borehole of the pump intake (if any).
        If PUMPLOC = 0, then either there is no pump or the intake location (or discharge point for an
        injection well) is assumed to occur above the first active node associated with the multi-node well
        (that is, the node closest to the land surface or to the wellhead). If PUMPLOC > 0, then the cell in
        which the intake (or outflow) is located will be specified in dataset 2e as a LAY-ROW-COL grid location.
        For a vertical well only, specifying PUMPLOC < 0, will enable the option to define the vertical position of
        the pump intake (or outflow) as an elevation in dataset 2e (for the given spatial grid location [ROW-COL]
        defined for this well in 2d).
        (default is 0)
    qlimit : integer
        Indicates whether the water level (head) in the well will be used to constrain
        the pumping rate. If Qlimit = 0, then there are no constraints for this well. If Qlimit > 0, then
        pumpage will be limited (constrained) by the water level in the well, and relevant parameters are
        constant in time and defined below in dataset 2f. If Qlimit < 0, then pumpage will be limited
        (constrained) by the water level in the well, and relevant parameters can vary with time and are
        defined for every stress period in dataset 4b.
        (default is 0)
    ppflag : integer
        Flag that determines whether the calculated head in the well will be corrected for the
        effect of partial penetration of the well screen in the cell. If PPFLAG = 0, then the head in the
        well will not be adjusted for the effects of partial penetration. If PPFLAG > 0, then the head in
        the well will be adjusted for the effects of partial penetration if the section of well containing
        the well screen is vertical (as indicated by identical row-column locations in the grid). If
        NNODES < 0 (that is, the open intervals of the well are defined by top and bottom elevations),
        then the model will automatically calculate the fraction of penetration for each node and the
        relative vertical position of the well screen. If NNODES > 0, then the fraction of penetration for
        each node must be defined in dataset 2d (see below) and the well screen will be assumed to be
        centered vertically within the thickness of the cell (except if the well is located in the uppermost
        model layer that is under unconfined conditions, in which case the bottom of the well screen will be
        assumed to be aligned with the bottom boundary of the cell and the assumed length of well screen will
        be based on the initial head in that cell).
        (default is 0)
    pumpcap : integer
        A flag and value that determines whether the discharge of a pumping (withdrawal) well (Q < 0.0)
        will be adjusted for changes in the lift (or total dynamic head) with time. If PUMPCAP = 0,
        then the discharge from the well will not be adjusted on the basis of changes in lift. If PUMPCAP > 0
        for a withdrawal well, then the discharge from the well will be adjusted on the basis of the lift, as
        calculated from the most recent water level in the well. In this case, data describing the head-capacity
        relation for the pump must be listed in datasets 2g and 2h, and the use of that relation can be switched
        on or off for each stress period using a flag in dataset 4a. The number of entries (lines) in dataset 2h
        corresponds to the value of PUMPCAP. If PUMPCAP does not equal 0, it must be set to an integer value of
        between 1 and 25, inclusive.
        (default is 0)
    lay_row_col : list of arrays (shape = (NNODES,3), length = MNWMAX)
        Layer, row, and column numbers of each model cell (node) for the current well. If NNODES > 0,
        then a total of NNODES model cells (nodes) must be specified for each well (and dataset 2d must
        contain NNODES records). In the list of nodes defining the multi-node well, the data list must be
        constructed and ordered so that the first node listed represents the node closest to the wellhead,
        the last node listed represents the node furthest from the wellhead, and all nodes are listed in
        sequential order from the top to the bottom of the well (corresponding to the order of first to
        last well nodes). A particular node in the grid can be associated with more than one multi-node well.
        (default is None)
    ztop_zbotm_row_col : list of arrays (shape = (abs(NNODES),2), length = MNWMAX)
        The top and bottom elevations of the open intervals (or screened intervals) of a vertical well.
        These values are only read if NNODES < 0 in dataset 2a. The absolute value of NNODES indicates
        how many open intervals are to be defined, and so must correspond exactly to the number of records
        in dataset 2d for this well. In the list of intervals defining the multi-node well, the data list
        must be constructed and ordered so that the first interval listed represents the shallowest one,
        the last interval listed represents the deepest one, and all intervals are listed in sequential
        order from the top to the bottom of the well. If an interval partially or fully intersects a model
        layer, then a node will be defined in that cell. If more than one open interval intersects a
        particular layer, then a length-weighted average of the cell-to-well conductances will be used to
        define the well-node characteristics; for purposes of calculating effects of partial penetration,
        the cumulative length of well screens will be assumed to be centered vertically within the thickness
        of the cell. If the well is a single-node well by definition of LOSSTYPE = NONE and the defined open
        interval straddles more than one model layer, then the well will be associated with the cell where
        the center of the open interval exists.
        (default is None)

        if losstype != None (see losstype for definitions)
            rw : float
                (default is 0)
            rskin : float
                (default is 0)
            kskin : float
                (default is 0)
            b : float
                (default is 0)
            c : float
                (default is 0)
            p : float
                (default is 0)
            cwc  float
                (default is 0)
    pp : float
        the fraction of partial penetration for this cell
        (see PPFLAG in dataset 2b). Only specify if PPFLAG > 0 and NNODES > 0.
        (default is 1)
    itmp : integer
        For reusing or reading multi-node well data; it can change each stress period.
        ITMP must be >= 0 for the first stress period of a simulation.
        if ITMP > 0, then ITMP is the total number of active multi-node wells simulated during the stress period,
        and only wells listed in dataset 4a will be active during the stress period. Characteristics of each
        well are defined in datasets 2 and 4.
        if ITMP = 0, then no multi-node wells are active for the stress period and the following dataset is
        skipped.
        if ITMP < 0, then the same number of wells and well information will be reused from the previous stress
        period and dataset 4 is skipped.
        (default is 0)
    wellid_qdes : list of arrays (shape = (NPER,MNWMAX,2))
        the actual (or maximum desired, if constraints are to be applied) volumetric pumping rate
        (negative for withdrawal or positive for injection) at the well (L3/T). Qdes should be set to 0
        for nonpumping wells. If constraints are applied, then the calculated volumetric withdrawal or
        injection rate may be adjusted to range from 0 to Qdes and is not allowed to switch directions
        between withdrawal and injection conditions during any stress period. When PUMPCAP > 0, in the
        first stress period in which Qdes is specified with a negative value, Qdes represents the maximum
        operating discharge for the pump; in subsequent stress periods, any different negative values of
        Qdes are ignored, although values are subject to adjustment for CapMult. If Qdes >= 0.0, then
        pump-capacity adjustments are not applied.
        (default is None)
    extension : string
        Filename extension (default is 'mnw2')
    unitnumber : int
        File unit number (default is 34).

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    This implementation does not allow well loss parameters {Rw,Rskin,Kskin,B,C,P,CWC,PP} to vary along the length
    of a given well. It also does not currently support data sections 2e, 2f, 2g, 2h, or 4b as defined in the data
    input instructions for the MNW2 package.

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> mnw2 = flopy.modflow.ModflowMnw2(ml, ...)

    """

    def __init__(self, model, mnwmax=0, iwl2cb=-1, mnwprnt=0, aux=None,
                 wellid=None, nnodes=None, losstype=None, pumploc=0, qlimit=0, ppflag=0, pumpcap=0,
                 lay_row_col=None, ztop_zbotm_row_col=None, rw=0, rskin=0, kskin=0, b=0, c=0, p=0, cwc=0, pp=1,
                 itmp=0, wellid_qdes=None,
                 extension='mnw2', unitnumber=34):
        """
        Package constructor
        """
        Package.__init__(self, model, extension, 'MNW2',
                         unitnumber)  # Call ancestor's init to set self.parent, extension, name, and unit number

        self.url = 'mnw2.htm'
        self.nper = self.parent.nrow_ncol_nlay_nper[-1]
        self.heading = '# Multi-node well 2 (MNW2) file for MODFLOW, generated by Flopy'
        self.mnwmax = int(mnwmax)  # -maximum number of multi-node wells to be simulated
        self.iwl2cb = int(iwl2cb)  # -flag and unit number
        self.mnwprnt = int(mnwprnt)  # -verbosity flag
        self.aux = aux  # -list of optional auxilary parameters
        self.wellid = wellid  # -array containing well id's (shape = (MNWMAX))

        self.lay_row_col = lay_row_col  # -list of arrays containing lay, row, and col for all well nodes [NNODES > 0](shape = (NNODES,3), length = MNWMAX)
        self.ztop_zbotm_row_col = ztop_zbotm_row_col  # -list of arrays containing top and botm elevation of all open intervals [NNODES < 0](shape = (abs(NNODES),2), length = MNWMAX)

        self.wellid_qdes = wellid_qdes  #-list of arrays containing desired Q for each well in each stress period (shape = (NPER,MNWMAX,2))

        #-create empty arrays of the correct size
        '''
        NOTE: some arrays are not pre-formatted here as their shapes vary from well to well and from period to period.
        '''
        self.wellid = np.empty((self.mnwmax), dtype='S25')
        self.nnodes = np.zeros((self.mnwmax), dtype=np.int32)
        self.losstype = np.empty((self.mnwmax), dtype='S25')
        self.pumploc = np.zeros((self.mnwmax), dtype='int32')
        self.qlimit = np.zeros((self.mnwmax), dtype='int32')
        self.ppflag = np.zeros((self.mnwmax), dtype='int32')
        self.pumpcap = np.zeros((self.mnwmax), dtype='int32')
        self.rw = np.zeros(self.mnwmax, dtype='float32')
        self.rskin = np.zeros(self.mnwmax, dtype='float32')
        self.kskin = np.zeros(self.mnwmax, dtype='float32')
        self.b = np.zeros(self.mnwmax, dtype='float32')
        self.c = np.zeros(self.mnwmax, dtype='float32')
        self.p = np.zeros(self.mnwmax, dtype='float32')
        self.cwc = np.zeros(self.mnwmax, dtype='float32')
        self.pp = np.zeros(self.mnwmax, dtype='float32')

        self.itmp = np.zeros(self.nper, dtype='int32')

        #-assign values to arrays        
        self.wellid[:] = np.array(wellid, dtype='S25')
        self.nnodes[:] = np.array(nnodes, dtype=np.int)
        self.losstype[:] = np.array(losstype, dtype='S25')
        self.pumploc[:] = np.array(pumploc, dtype=np.int32)
        self.qlimit[:] = np.array(qlimit, dtype=np.int32)
        self.ppflag[:] = np.array(ppflag, dtype=np.int32)
        self.pumpcap[:] = np.array(pumpcap, dtype=np.int32)
        self.rw[:] = np.array(rw, dtype=np.float32)
        self.rskin[:] = np.array(rskin, dtype=np.float32)
        self.kskin[:] = np.array(kskin, dtype=np.float32)
        self.b[:] = np.array(b, dtype=np.float32)
        self.c[:] = np.array(c, dtype=np.float32)
        self.p[:] = np.array(p, dtype=np.float32)
        self.cwc[:] = np.array(cwc, dtype=np.float32)
        self.pp[:] = np.array(pp, dtype=np.float32)

        self.itmp[:] = np.array(itmp, dtype=np.int32)

        #-input format checks:
        lossTypes = ['NONE', 'THIEM', 'SKIN', 'GENERAL', 'SPECIFYcwc']
        for i in range(mnwmax):
            assert len(self.wellid[i].split(' ')) == 1, 'WELLID (%s) must not contain spaces' % self.wellid[i]
            assert self.losstype[
                       i] in lossTypes, 'LOSSTYPE (%s) must be one of the following: NONE, THIEM, SKIN, GENERAL, or SPECIFYcwc' % \
                                        self.losstype[i]
        assert self.itmp[0] >= 0, 'ITMP must be greater than or equal to zero for the first time step.'
        assert self.itmp.max() <= self.mnwmax, 'ITMP cannot exceed maximum number of wells to be simulated.'

        self.parent.add_package(self)

    def write_file(self):
        """
        Write the file.

        """
        # -open file for writing
        f = open(self.fn_path, 'w')

        # -write header
        f.write('{}\n'.format(self.heading))

        # -Section 1 - MNWMAX, IWL2CB, MNWPRNT {OPTION}
        auxParamString = ''
        if self.aux != None:
            for param in self.aux:
                auxParamString = auxParamString + 'AUX %s ' % param
        f.write('{:10d}{:10d}{:10d} {}\n'.format(self.mnwmax,
                                                 self.iwl2cb,
                                                 self.mnwprnt,
                                                 auxParamString))

        # -Section 2 - Repeat this section MNWMAX times (once for each well)
        for i in range(self.mnwmax):
            #-Section 2a - WELLID, NNODES
            f.write('{}{:10d}\n'.format(self.wellid[i], self.nnodes[i]))
            #-Section 2b - LOSSTYPE, PUMPLOC, Qlimit, PPFLAG, PUMPCAP
            f.write('{} {:10d}{:10d}{:10d}{:10d}\n'.format(self.losstype[i],
                                                           self.pumploc[i],
                                                           self.qlimit[i],
                                                           self.ppflag[i],
                                                           self.pumpcap[i]))
            #-Section 2c - {Rw, Rskin, Kskin, B, C, P, CWC}
            if self.losstype[i] == 'THIEM':
                f.write('{:10.4g}\n'.format(self.rw[i]))
            elif self.losstype[i] == 'SKIN':
                f.write('{:10.4g}{:10.4g}{:10.4g}\n'.format(self.rw[i],
                                                            self.rskin[i],
                                                            self.kskin[i]))
            elif self.losstype[i] == 'GENERAL':
                f.write('{:10.4g}{:10.4g}{:10.4g}{:10.4g}\n'.format(self.rw[i],
                                                                    self.b[i],
                                                                    self.c[i],
                                                                    self.p[i]))
            elif self.losstype[i] == 'SPECIFYcwc':
                f.write('{:10.4g}\n'.format(self.cwc[i]))

            #-Section 2d - Repeat sections 2d-1 or 2d-2 once for each open interval
            #-Section 2d-1 - NNODES > 0; LAY, ROW, COL {Rw, Rskin, Kskin, B, C, P, CWC, PP}
            absNnodes = abs(self.nnodes[i])
            if self.nnodes[i] > 0:
                for n in range(absNnodes):
                    f.write('{:10d}{:10d}{:10d}\n'.format(self.lay_row_col[i][n, 0]+1,
                                                          self.lay_row_col[i][n, 1]+1,
                                                          self.lay_row_col[i][n, 2]+1))
            #-Section 2d-2 - NNODES < 0; Ztop, Zbotm, ROW, COL {Rw, Rskin, Kskin, B, C, P, CWC, PP}
            elif self.nnodes[i] < 0:
                for n in range(absNnodes):
                    #print i, n
                    #print self.ztop_zbotm_row_col
                    f.write('{:10.4g} {:10.4g} {:10d} {:10d}\n'.format(self.ztop_zbotm_row_col[i][n, 0],
                                                                       self.ztop_zbotm_row_col[i][n, 1],
                                                                       int(self.ztop_zbotm_row_col[i][n, 2])+1,
                                                                       int(self.ztop_zbotm_row_col[i][n, 3])+1))

        #-Section 3 - Repeat this section NPER times (once for each stress period)
        for p in range(self.nper):
            f.write('{:10d}\n'.format(self.itmp[p]))

            #-Section 4 - Repeat this section ITMP times (once for each well to be simulated in current stress period)
            if self.itmp[p] > 0:
                '''
                Create an array that will hold well names to be simulated during this stress period and find their corresponding
                index number in the "wellid" array so the right parameters (Hlim Qcut {Qfrcmn Qfrcmx}) are accessed.
                '''
                itmp_wellid_index_array = np.empty((self.itmp[p], 2), dtype='object')
                for well in range(self.itmp[p]):
                    itmp_wellid_index_array[well, 0] = self.wellid_qdes[p][well, 0]
                    itmp_wellid_index_array[well, 1] = np.where(self.wellid == self.wellid_qdes[p][well, 0])

                for j in range(self.itmp[p]):
                    #-Section 4a - WELLID Qdes {CapMult} {Cprime} {xyz}
                    assert self.wellid_qdes[p][j, 0] in self.wellid, \
                        'WELLID for pumping well is not present in "wellid" array'

                    #print self.wellid_qdes[p][j, 0], self.wellid_qdes[p][j, 1]

                    f.write('{} {:10.4g}\n'.format(self.wellid_qdes[p][j, 0],
                                                   float(self.wellid_qdes[p][j, 1])))

        f.close()

