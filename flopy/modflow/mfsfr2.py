# from numpy import empty, zeros, ones, where
import os
import numpy as np
from flopy.mbase import Package


class ModflowSfr2(Package):
    """
    'Streamflow-Routing (SFR2) Package Class'

    Parameters
    ----------
    model : model object
        The model object (of type :class:'flopy.modflow.mf.Modflow') to which
        this package will be added.
    nstrm : integer
        An integer value that can be specified to be positive or negative. The absolute value of NSTRM is equal to
        the number of stream reaches (finite-difference cells) that are active during the simulation and the number of
        lines of data to be included in Item 2, described below. When NSTRM is specified to be a negative integer,
        it is also used as a flag for changing the format of the data input, for simulating unsaturated flow beneath
        streams, and (or) for simulating transient streamflow routing (for MODFLOW-2005 simulations only), depending
        on the values specified for variables ISFROPT and IRTFLG, as described below. When NSTRM is negative, NSFRPAR
        must be set to zero, which means that parameters cannot be specified.
    nss : integer
        An integer value equal to the number of stream segments (consisting of one or more reaches) that are used
        to define the complete stream network. The value of NSS represents the number of segments that must be
        defined through a combination of parameters and variables in Item 4 or variables in Item 6.
    nparseg : integer
        An integer value equal to (or exceeding) the number of stream-segment definitions associated with all
        parameters. This number can be more than the total number of segments (NSS) in the stream network because
        the same segment can be defined in multiple parameters, and because parameters can be time-varying. NPARSEG
        must equal or exceed the sum of NLST x N for all parameters, where N is the greater of 1 and NUMINST;
        that is, NPARSEG must equal or exceed the total number of repetitions of item 4b. This variable must be zero
        when NSTRM is negative.
    const : float
        A real value (or conversion factor) used in calculating stream depth for stream reach. If stream depth is
        not calculated using Manning's equation for any stream segment (that is, ICALC does not equal 1 or 2), then
        a value of zero can be entered. If Manning's equation is used, a constant of 1.486 is used for flow units of
        cubic feet per second, and a constant of 1.0 is used for units of cubic meters per second. The constant must
        be multiplied by 86,400 when using time units of days in the simulation. An explanation of time units used
        in MODFLOW is given by Harbaugh and others (2000, p. 10).
    dleak : float
        A real value equal to the tolerance level of stream depth used in computing leakage between each stream
        reach and active model cell. Value is in units of length. Usually a value of 0.0001 is sufficient when units
        of feet or meters are used in model.
    istcsb1 : integer
        An integer value used as a flag for writing stream-aquifer leakage values. If ISTCB1 > 0, it is the unit
        number to which unformatted leakage between each stream reach and corresponding model cell will be saved to
        a file whenever the cell-by-cell budget has been specified in Output Control (see Harbaugh and others, 2000,
        pages 52-55). If ISTCB1 = 0, leakage values will not be printed or saved. If ISTCB1 < 0, all information on
        inflows and outflows from each reach; on stream depth, width, and streambed conductance; and on head difference
        and gradient across the streambed will be printed in the main listing file whenever a cell-by-cell budget has
        been specified in Output Control.
    istcsb2 : integer
        An integer value used as a flag for writing to a separate formatted file all information on inflows and
        outflows from each reach; on stream depth, width, and streambed conductance; and on head difference and
        gradient across the streambed. If ISTCB2 > 0, then ISTCB2 also represents the unit number to which all
        information for each stream reach will be saved to a separate file when a cell-by-cell budget has been
        specified in Output Control. If ISTCB2 < 0, it is the unit number to which unformatted streamflow out of
        each reach will be saved to a file whenever the cell-by-cell budget has been specified in Output Control.
    isfropt : integer
        An integer value that defines the format of the input data and whether or not unsaturated flow is simulated
        beneath streams. Values of ISFROPT are defined as follows:
        0   No vertical unsaturated flow beneath streams. Streambed elevations, stream slope, streambed thickness,
            and streambed hydraulic conductivity are read for each stress period using variables defined in Items 6b
            and 6c; the optional variables in Item 2 are not used.
        1   No vertical unsaturated flow beneath streams. Streambed elevation, stream slope, streambed thickness,
            and streambed hydraulic conductivity are read for each reach only once at the beginning of the simulation
            using optional variables defined in Item 2; Items 6b and 6c are used to define stream width and depth for
            ICALC = 0 and stream width for ICALC = 1.
        2   Streambed and unsaturated-zone properties are read for each reach only once at the beginning of the
            simulation using optional variables defined in Item 2; Items 6b and 6c are used to define stream width and
            depth for ICALC = 0 and stream width for ICALC = 1. When using the LPF Package, saturated vertical
            hydraulic conductivity for the unsaturated zone is the same as the vertical hydraulic conductivity of the
            corresponding layer in LPF and input variable UHC is not read.
        3   Same as 2 except saturated vertical hydraulic conductivity for the unsaturated zone (input variable UHC)
            is read for each reach.
        4   Streambed and unsaturated-zone properties are read for the beginning and end of each stream segment using
            variables defined in Items 6b and 6c; the optional variables in Item 2 are not used. Streambed properties
            can vary each stress period. When using the LPF Package, saturated vertical hydraulic conductivity for the
            unsaturated zone is the same as the vertical hydraulic conductivity of the corresponding layer in LPF
            and input variable UHC1 is not read.
        5   Same as 4 except saturated vertical hydraulic conductivity for the unsaturated zone (input variable UHC1)
            is read for each segment at the beginning of the first stress period only.
    nstrail : integer
        An integer value that is the number of trailing wave increments used to represent a trailing wave. Trailing
        waves are used to represent a decrease in the surface infiltration rate. The value can be increased to improve
        mass balance in the unsaturated zone. Values between 10 and 20 work well and result in unsaturated-zone mass
        balance errors beneath streams ranging between 0.001 and 0.01 percent. Please see Smith (1983) for further
        details. (default is 10; for MODFLOW-2005 simulations only when isfropt > 1)
    isuzn : integer
        An integer value that is the maximum number of vertical cells used to define the unsaturated zone beneath a
        stream reach. If ICALC is 1 for all segments then ISUZN should be set to 1.
        (default is 1; for MODFLOW-2005 simulations only when isfropt > 1)
    nsfrsets : integer
        An integer value that is the maximum number of different sets of trailing waves used to allocate arrays.
        Arrays are allocated by multiplying NSTRAIL by NSFRSETS. A value of 30 is sufficient for problems where the
        stream depth varies often. NSFRSETS does not affect model run time.
        (default is 30; for MODFLOW-2005 simulations only when isfropt > 1)
    irtflg : integer
        An integer value that indicates whether transient streamflow routing is active. IRTFLG must be specified
        if NSTRM < 0. If IRTFLG > 0, streamflow will be routed using the kinematic-wave equation (see USGS Techniques
        and Methods 6-D1, p. 68-69); otherwise, IRTFLG should be specified as 0. Transient streamflow routing is only
        available for MODFLOW-2005; IRTFLG can be left blank for MODFLOW-2000 simulations.
        (default is 1)
    numtim : integer
        An integer value equal to the number of sub time steps used to route streamflow. The time step that will be
        used to route streamflow will be equal to the MODFLOW time step divided by NUMTIM.
        (default is 2; for MODFLOW-2005 simulations only when irtflg > 0)
    weight : float
        A real number equal to the time weighting factor used to calculate the change in channel storage. WEIGHT has
        a value between 0.5 and 1. Please refer to equation 83 in USGS Techniques and Methods 6-D1 for further
        details. (default is 0.75; for MODFLOW-2005 simulations only when irtflg > 0)
    flwtol : float
        A real number equal to the streamflow tolerance for convergence of the kinematic wave equation used for
        transient streamflow routing. A value of 0.00003 cubic meters per second has been used successfully in test
        simulations (and would need to be converted to whatever units are being used in the particular simulation).
        (default is 0.0001; for MODFLOW-2005 simulations only when irtflg > 0)
    reach_data :
    segment_data :
    itmp : list of integers (len = NPER)
        For each stress period, an integer value for reusing or reading stream segment data that can change each
        stress period. If ITMP = 0 then all stream segment data are defined by Item 4 (NSFRPAR > 0; number of stream
        parameters is greater than 0). If ITMP > 0, then stream segment data are not defined in Item 4 and must be
        defined in Item 6 below for a number of segments equal to the value of ITMP. If ITMP < 0, then stream segment
        data not defined in Item 4 will be reused from the last stress period (Item 6 is not read for the current
        stress period). ITMP must be defined >= 0 for the first stress period of a simulation.
    irdflag : list of integers (len = NPER)
        For each stress period, an integer value for printing input data specified for this stress period.
        If IRDFLG = 0, input data for this stress period will be printed. If IRDFLG > 0, then input data for this
        stress period will not be printed.
    iptflag : list of integers (len = NPER)
        For each stress period, an integer value for printing streamflow-routing results during this stress period.
        If IPTFLG = 0, or whenever the variable ICBCFL or "Save Budget" is specified in Output Control, the results
        for specified time steps during this stress period will be printed. If IPTFLG > 0, then the results during
        this stress period will not be printed.
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

    MODFLOW-OWHM is not supported.

    The Ground-Water Transport (GWT) process is not supported.

    Limitations on which features are supported...

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> sfr2 = flopy.modflow.ModflowSfr2(ml, ...)

    """

    nsfrpar = 0
    heading = '# Streamflow-Routing (SFR2) file for MODFLOW, generated by Flopy'

    def __init__(self, model, nstrm=0, nss=0, nsfrpar=0, nparseg=0, const=128390.4, dleak=0.0001, istcb1=50, istcb2=66, isfropt=0,
                 nstrail=10, isuzn=1, nsfrsets=30, irtflg=1, numtim=2, weight=0.75, flwtol=0.0001,
                 reach_data=None,
                 segment_data=None,
                 channel_geometry_data=None,
                 channel_flow_data=None,
                 dataset_5=None,
                 reachinput=False, transroute=False,
                 tabfiles=False, tabfiles_dict=None,
                 extension='sfr', unitnumber=14):

        """
        Package constructor
        """
        Package.__init__(self, model, extension, 'SFR2',
                         unitnumber)  # Call ancestor's init to set self.parent, extension, name, and unit number

        self.url = 'sfr2.htm'
        self.nper = self.parent.nrow_ncol_nlay_nper[-1]

        # Dataset 0 -----------------------------------------------------------------------
        self.heading = '# SFR2 for MODFLOW, generated by Flopy.'

        # Dataset 1a and 1b. -----------------------------------------------------------------------
        self.reachinput = reachinput
        self.transroute = transroute
        self.tabfiles = tabfiles
        self.tabfiles_dict = tabfiles_dict
        self.numtab = len(tabfiles_dict)
        self.maxval = np.max([tb['numval'] for tb in tabfiles_dict.values()]) if len(tabfiles_dict) > 0 else 0

        # Dataset 1c. ----------------------------------------------------------------------
        self.nstrm = nstrm # number of reaches, negative value is flag for unsat. flow beneath streams and/or transient routing
        self.nss = nss # number of stream segments
        self.nsfrpar = nsfrpar
        self.nparseg = nparseg
        self.const = const #conversion factor used in calculating stream depth for stream reach (icalc = 1 or 2)
        self.dleak = dleak # tolerance level of stream depth used in computing leakage
        self.istcb1 = istcb1 # flag; unit number for stream leakage output
        self.istcb2 = istcb2 # flag; unit number for writing table of SFR output to text file

        # if nstrm < 0
        self.isfropt = isfropt # defines the format of the input data and whether or not unsaturated flow is simulated

        # if isfropt > 1
        self.nstrail = nstrail # number of trailing wave increments
        self.isuzn = isuzn # max number of vertical cells used to define unsat. zone
        self.nsfrsets = nsfrsets # max number trailing waves sets

        # if nstrm < 0 (MF-2005 only)
        self.irtflag = irtflg # switch for transient streamflow routing (> 0 = kinematic wave)
        # if irtflag > 0
        self.numtim = numtim # number of subtimesteps used for routing
        self.weight = weight # time weighting factor used to calculate the change in channel storage
        self.flwtol = flwtol # streamflow tolerance for convergence of the kinematic wave equation

        # Dataset 2. -----------------------------------------------------------------------
        # columns:  KRCH IRCH JRCH ISEG IREACH RCHLEN {STRTOP} {SLOPE} {STRTHICK} {STRHC1}
        #           {THTS} {THTI} {EPS} {UHC}

        self.reach_data = reach_data

        # Datasets 4 and 6. -----------------------------------------------------------------------
        self.segment_data = segment_data
        self.channel_geometry_data = channel_geometry_data
        self.channel_flow_data = channel_flow_data

        # Dataset 5 -----------------------------------------------------------------------
        self.dataset_5 = dataset_5

        #-input format checks:
        '''
        lossTypes = ['NONE', 'THIEM', 'SKIN', 'GENERAL', 'SPECIFYcwc']
        for i in range(mnwmax):
            assert len(self.wellid[i].split(' ')) == 1, 'WELLID (%s) must not contain spaces' % self.wellid[i]
            assert self.losstype[
                       i] in lossTypes, 'LOSSTYPE (%s) must be one of the following: NONE, THIEM, SKIN, GENERAL, or SPECIFYcwc' % \
                                        self.losstype[i]
        assert self.itmp[0] >= 0, 'ITMP must be greater than or equal to zero for the first time step.'
        assert self.itmp.max() <= self.mnwmax, 'ITMP cannot exceed maximum number of wells to be simulated.'
        '''
        self.parent.add_package(self)

    def __repr__(self):
        return 'SFR2 class'

    @staticmethod
    def get_empty_reach_data(nreaches=0, aux_names=None, structured=True):
        # get an empty recarray that correponds to dtype
        dtype = ModflowSfr2.get_default_reach_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((nreaches, len(dtype)), dtype=dtype)
        d[:, :] = 0.0 #-1.0E+10
        return np.core.records.fromarrays(d.transpose(), dtype=dtype)

    @staticmethod
    def get_empty_segment_data(nsegments=0, aux_names=None):
        # get an empty recarray that correponds to dtype
        dtype = ModflowSfr2.get_default_segment_dtype()
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((nsegments, len(dtype)), dtype=dtype)
        d[:, :] = 0.0 #-1.0E+10
        return np.core.records.fromarrays(d.transpose(), dtype=dtype)

    @staticmethod
    def get_default_reach_dtype(structured=True):
        if structured:
            return np.dtype([('krch', np.int),
                              ('irch', np.int), 
                              ('jrch', np.int), 
                              ('iseg', np.int), 
                              ('ireach', np.int), 
                              ('rchlen', np.float32),
                              ('strtop', np.float32), 
                              ('slope', np.float32), 
                              ('strthick', np.float32), 
                              ('strhc1', np.float32),
                              ('thts', np.int), 
                              ('thti', np.float32), 
                              ('eps', np.float32), 
                              ('uhc', np.float32)])
        else:
            return np.dtype([('node', np.int)
                              ('iseg', np.int), 
                              ('ireach', np.int), 
                              ('rchlen', np.float32),
                              ('strtop', np.float32), 
                              ('slope', np.float32), 
                              ('strthick', np.float32), 
                              ('strhc1', np.float32),
                              ('thts', np.int), 
                              ('thti', np.float32), 
                              ('eps', np.float32), 
                              ('uhc', np.float32)])        

    @staticmethod
    def get_default_segment_dtype():
        return np.dtype([('nseg', np.int),
                          ('icalc', np.int),
                          ('outseg', np.int),
                          ('iupseg', np.int),
                          ('iprior', np.int), 
                          ('nstrpts', np.int), 
                          ('flow', np.float32), 
                          ('runoff', np.float32),
                          ('etsw', np.float32), 
                          ('pptsw', np.float32), 
                          ('roughch', np.float32), 
                          ('roughbk', np.float32),
                          ('cdpth', np.float32), 
                          ('fdpth', np.float32), 
                          ('awdth', np.float32), 
                          ('bwdth', np.float32),
                          ('hcond1', np.float32), 
                          ('thickm1', np.float32), 
                          ('elevup1', np.float32), 
                          ('width1', np.float32), 
                          ('depth1', np.float32),
                          ('thts1', np.float32), 
                          ('thti1', np.float32), 
                          ('eps1', np.float32), 
                          ('uhc1', np.float32),
                          ('hcond2', np.float32), 
                          ('thickm2', np.float32), 
                          ('elevup2', np.float32), 
                          ('width2', np.float32), 
                          ('depth2', np.float32),
                          ('thts2', np.float32), 
                          ('thti2', np.float32), 
                          ('eps2', np.float32), 
                          ('uhc2', np.float32)])       

    @staticmethod
    def load(f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):

        tabfiles = False
        tabfiles_dict = {}
        transroute = False
        reachinput = False
        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        # Item 0 -- header
        while True:
            line = next(f)
            if line[0] != '#':
                break
        # Item 1
        if "reachinput" in line.lower():
            """
            When REACHINPUT is specified, variable ISFROPT is read in data set 1c.
            ISFROPT can be used to change the default format for entering reach and segment data
            or to specify that unsaturated flow beneath streams will be simulated.
            """
            reachinput = True
        if "transroute" in line.lower():
            """When TRANSROUTE is specified, optional variables IRTFLG, NUMTIM, WEIGHT, and FLWTOL
            also must be specified in Item 1c.
            """
            transroute = True
        if transroute or reachinput:
            line = next(f)
        if "tabfiles" in line.lower():
            """
            tabfiles
            An optional character variable that is a flag to indicate that inflows to one or more stream
            segments will be specified with tabular inflow files.
            numtab
            An integer value equal to the number of tabular inflow files that will be read if TABFILES
            is specified. A separate input file is required for each segment that receives specified inflow.
            Thus, the maximum value of NUMTAB that can be specified is equal to the total number of
            segments specified in Item 1c with variables NSS. The name (Fname) and unit number (Nunit)
            of each tabular file must be specified in the MODFLOW-2005 Name File using tile type (Ftype) DATA.
            maxval

            """
            tabfiles, numtab, maxval = line.strip().split()
            numtab, maxval = int(numtab), int(maxval)
            line = next(f)
        # NSTRM NSS NSFRPAR NPARSEG CONST DLEAK ISTCB1  ISTCB2
        # [ISFROPT] [NSTRAIL] [ISUZN] [NSFRSETS] [IRTFLG] [NUMTIM] [WEIGHT] [FLWTOL]
        '''
        item1cvalues = [0,0,0,0,0,0,0,0, \
                        0,10,1,30,1, 2, 0.75, 0.0001]
        tmp = [s for s in line.strip().split() if s.isnumeric()]
        nvars = len(tmp)
        item1cvalues[:nvars] = tmp
        item1cvalues = list(map(float, item1cvalues))
        '''
        item1cvalues = _get_dataset(line, [0, 0, 0, 0, 0, 0, 0, 0, \
                                           0, 10, 1, 30, 1, 2, 0.75, 0.0001])
        nstrm, nss, nsfrpar, nparseg = map(int, item1cvalues[0:4])
        if reachinput:
            nstrm = abs(nstrm) # see explanation for dataset 1c in online guide
        const, dleak = item1cvalues[4:6]
        istcb1, istcb2, isfropt, nstrail, isuzn, nsfrsets, irtflg, numtim = map(int, item1cvalues[6:14])
        weight, flwtol = item1cvalues[14:]

        # item 2
        # set column names, dtypes
        #names = [('krch', int), ('irch', int), ('jrch', int), ('iseg', int), ('ireach', int), ('rchlen', float),
        #         ('strtop', float), ('slope', float), ('strthick', float), ('strhc1', float),
        #         ('thts', int), ('thti', float), ('eps', float), ('uhc', float)]
        names = ModflowSfr2.get_default_reach_dtype().descr

        lines = []
        for i in range(abs(nstrm)):
            ireach = tuple(map(float, next(f).strip().split()))
            lines.append(ireach)
        ncols = len(lines[0])
        reach_data = np.array(lines, dtype=names[:ncols])
        reach_data = reach_data.view(np.recarray)

        # zero-based convention
        _markitzero(reach_data, ['krch', 'irch', 'jrch', 'node'])

        # items 3 and 4 are skipped (parameters not supported)
        # item 5

        segment_data = {}
        channel_geometry_data = {}
        channel_flow_data = {}
        dataset_5 = {}
        for i in range(0, nper):
            # Dataset 5
            dataset_5[i] = _get_dataset(next(f), [1, 0, 0, 0])
            itmp = dataset_5[i][0]
            if itmp > 0:
                # Item 6
                #current = np.recarray((itmp,),
                #        dtype=[('nseg', int), ('icalc', float), ('outseg', int), ('iupseg', int),
                #               ('iprior', int), ('nstrpts', int), ('flow', float), ('runoff', float),
                #               ('etsw', float), ('pptsw', float), ('roughch', float), ('roughbk', float),
                #               ('cdpth', float), ('fdpth', float), ('awdth', float), ('bwdth', float),
                #               ('hcond1', float), ('thickm1', float), ('elevup1', float), ('width1', float), ('depth1', float),
                #               ('thts1', float), ('thti1', float), ('eps1', float), ('uhc1', float),
                #               ('hcond2', float), ('thickm2', float), ('elevup2', float), ('width2', float), ('depth2', float),
                #               ('thts2', float), ('thti2', float), ('eps2', float), ('uhc2', float)])
                current = ModflowSfr2.get_empty_segment_data(nsegments=itmp)

                current_6d = {} # these could also be implemented as structured arrays with a column for segment number
                current_6e = {}
                for j in range(itmp):

                    dataset_6a = parse_6a(next(f))
                    icalc = dataset_6a[1]
                    dataset_6b = parse_6bc(next(f), icalc, nstrm, isfropt, reachinput, per=i)
                    dataset_6c = parse_6bc(next(f), icalc, nstrm, isfropt, reachinput, per=i)

                    current[j] = dataset_6a + dataset_6b + dataset_6c

                    if icalc == 2:
                        # ATL: not sure exactly how isfropt logic functions for this
                        # dataset 6d description suggests that this line isn't read for isfropt > 1
                        # but description of icalc suggest that icalc=2 (8-point channel) can be used with any isfropt
                        if i == 0 or nstrm > 0 and not reachinput: # or isfropt <= 1:
                            dataset_6d = []
                            for k in range(2):
                                dataset_6d.append(_get_dataset(next(f), [0.0] * 8))
                                #dataset_6d.append(list(map(float, next(f).strip().split())))
                            current_6d[j] = dataset_6d
                    if icalc == 4:
                        nstrpts = dataset_6a[5]
                        dataset_6e = []
                        for k in range(3):
                            dataset_6e.append(_get_dataset(next(f), [0.0] * nstrpts))
                        current_6e[j] = dataset_6e

                segment_data[i] = current
                if len(current_6d) > 0:
                    channel_geometry_data[i] = current_6d
                if len(current_6e) > 0:
                    channel_flow_data[i] = current_6e

            if tabfiles and i == 0:
                for j in range(numtab):
                    segnum, numval, iunit = map(int, next(f).strip().split())
                    tabfiles_dict[segnum] = {'numval': numval, 'inuit': iunit}

            else:
                continue

        return ModflowSfr2(model, nstrm=nstrm, nss=nss, nsfrpar=nsfrpar, nparseg=nparseg, const=const, dleak=dleak, istcb1=istcb1, istcb2=istcb2,
                          isfropt=isfropt, nstrail=nstrail, isuzn=isuzn, nsfrsets=nsfrsets, irtflg=irtflg,
                          numtim=numtim, weight=weight, flwtol=flwtol,
                          reach_data=reach_data,
                          segment_data=segment_data,
                          dataset_5=dataset_5,
                          channel_geometry_data=channel_geometry_data,
                          channel_flow_data=channel_flow_data,
                          reachinput=reachinput, transroute=transroute,
                          tabfiles=tabfiles, tabfiles_dict=tabfiles_dict
                          )

    def _write_1c(self, f_sfr):

        # NSTRM NSS NSFRPAR NPARSEG CONST DLEAK ISTCB1  ISTCB2
        # [ISFROPT] [NSTRAIL] [ISUZN] [NSFRSETS] [IRTFLG] [NUMTIM] [WEIGHT] [FLWTOL]
        f_sfr.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.8f} {:.8f} {:.0f} {:.0f} '
                    .format(self.nstrm, self.nss, self.nsfrpar, self.nparseg,
                            self.const, self.dleak, self.istcb1, self.istcb2))
        if self.reachinput:
            self.nstrm = abs(self.nstrm) # see explanation for dataset 1c in online guide
        if self.nstrm < 0:
            f_sfr.write('{:.0f} '.format(self.isfropt))
        if self.isfropt > 1:
            f_sfr.write('{:.0f} {:.0f} {:.0f} '.format(self.nstrail, self.isuzn, self.nsfrsets))
        if self.nstrm < 0:
            f_sfr.write('{:.0f} '.format(self.irtflag))
        if self.irtflag < 0:
            f_sfr.write('{:.0f} {:.8f} {:.8f} '.format(self.numtim, self.weight, self.flwtol))
        f_sfr.write('\n')

    def _write_reach_data(self, f_sfr):

        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(self.reach_data, np.recarray), "mflist.__tofile() data arg " + \
                                              "not a recarray"

        # Add one to the kij indices
        names = self.reach_data.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        # --make copy of data for multiple calls
        d = np.recarray.copy(self.reach_data)
        for idx in ['krch', 'irch', 'jrch', 'node']:
            if (idx in lnames):
                d[idx] += 1
        formats = _fmt_string(d)[:-1] + '\n'
        for i in range(len(d)):
            f_sfr.write(formats.format(*d[i]))

    def _write_segment_data(self, i, j, f_sfr):
        cols = ['nseg', 'icalc', 'outseg', 'iupseg', 'iprior', 'nstrpts', 'flow', 'runoff',
                'etsw', 'pptsw', 'roughch', 'roughbk', 'cdpth', 'fdpth', 'awdth', 'bwdth']
        fmts = _fmt_string_list(self.segment_data[i][cols][j])

        nseg, icalc, outseg, iupseg, iprior, nstrpts, flow, runoff, etsw, \
        pptsw, roughch, roughbk, cdpth, fdpth, awdth, bwdth = self.segment_data[i][cols][j]

        f_sfr.write(' '.join(fmts[0:4]).format(nseg, icalc, outseg, iupseg) + ' ')

        if iupseg != 0:
            f_sfr.write(fmts[4].format(iprior) + ' ')
        if icalc == 4:
            f_sfr.write(fmts[5].format(nstrpts) + ' ')

        f_sfr.write(' '.join(fmts[6:10]).format(flow, runoff, etsw, pptsw) + ' ')

        if icalc in [1, 2]:
            f_sfr.write(fmts[10].format(roughch) + ' ')
        if icalc == 2:
            f_sfr.write(fmts[11].format(roughbk) + ' ')

        if icalc == 3:
            f_sfr.write(' '.join(fmts[12:16]).format(cdpth, fdpth, awdth, bwdth) + ' ')
        f_sfr.write('\n')

        self._write_6bc(i, j, f_sfr, cols=['hcond1', 'thickm1', 'elevup1', 'width1', 'depth1', 'thts1', 'thti1',
                                           'eps1', 'uhc1'])
        self._write_6bc(i, j, f_sfr, cols=['hcond2', 'thickm2', 'elevup2', 'width2', 'depth2', 'thts2', 'thti2',
                                           'eps2', 'uhc2'])

    def _write_6bc(self, i, j, f_sfr, cols=[]):

        icalc = self.segment_data[i][j][1]
        fmts = _fmt_string_list(self.segment_data[i][cols][j])
        hcond, thickm, elevup, width, depth, thts, thti, eps, uhc = self.segment_data[i][cols][j]

        if self.isfropt in [0, 4, 5] and icalc <= 0:
            f_sfr.write(' '.join(fmts[0:5]).format(hcond, thickm, elevup, width, depth) + ' ')

        elif self.isfropt in [0, 4, 5] and icalc == 1:
            f_sfr.write(fmts[0].format(hcond) + ' ')

            if i == 0:
                f_sfr.write(' '.join(fmts[1:8]).format(thickm, elevup, width, depth, thts, thti, eps) + ' ')
                if self.isfropt == 5:
                    f_sfr.write(fmts[8].format(uhc) + ' ')
        elif self.isfropt in [0, 4, 5] and icalc >= 2:
            f_sfr.write(fmts[0].format(hcond) + ' ')

            if self.isfropt in [4, 5] and i > 0 and icalc == 2:
                pass
            else:
                f_sfr.write(' '.join(fmts[1:3]).format(thickm, elevup))

                if self.isfropt in [4, 5] and icalc == 2 and i == 0:
                    f_sfr.write(' '.join(fmts[3:6]).format(thts, thti, eps))

                    if self.isfropt == 5:
                        f_sfr.write(fmts[8].format(uhc) + ' ')
                else:
                    pass
        elif self.isfropt == 1 and icalc <= 1:
            f_sfr.write(fmts[3].format(width) + ' ')
            if icalc <= 0:
                f_sfr.write(fmts[4].format(depth) + ' ')
        elif self.isfropt in [2, 3] and icalc <= 1:
            if i > 0:
                pass
            else:
                f_sfr.write(fmts[3].format(width) + ' ')
                if icalc <= 0:
                    f_sfr.write(fmts[4].format(depth) + ' ')
        else:
            pass
        f_sfr.write('\n')

    def write(self, filename=None):

        #tabfiles = False
        #tabfiles_dict = {}
        #transroute = False
        #reachinput = False
        if filename is not None:
            self.fn_path = filename

        f_sfr = open(self.fn_path, 'w')
        '''
        line = '{0:10d}{1:10d}'.format(self.stress_period_data.mxact, self.ipakcb)
        for opt in self.options:
            line += ' ' + str(opt)
        line += '\n'
        f_riv.write(line)
        self.stress_period_data.write_transient(f_riv)
        f_riv.close()
        
        self.reachinput = reachinput
        self.transroute = transroute
        self.tabfiles = tabfiles
        self.tabfiles_dict = tabfiles_dict
        self.numtab = len(tabfiles_dict)
        self.maxval = np.max([tb['numval'] for tb in tabfiles_dict.values()]) if len(tabfiles_dict) > 0 else 0
        '''
        # Item 0 -- header
        f_sfr.write('{0}\n'.format(self.heading))

        # Item 1
        if self.reachinput:
            
            """
            When REACHINPUT is specified, variable ISFROPT is read in data set 1c.
            ISFROPT can be used to change the default format for entering reach and segment data
            or to specify that unsaturated flow beneath streams will be simulated.
            """
            f_sfr.write('reachinput ')
        if self.transroute:
            """When TRANSROUTE is specified, optional variables IRTFLG, NUMTIM, WEIGHT, and FLWTOL
            also must be specified in Item 1c.
            """
            f_sfr.write('transroute')
        if self.transroute or self.reachinput:
            f_sfr.write('\n')        
        if self.tabfiles:
            """
            tabfiles
            An optional character variable that is a flag to indicate that inflows to one or more stream
            segments will be specified with tabular inflow files.
            numtab
            An integer value equal to the number of tabular inflow files that will be read if TABFILES
            is specified. A separate input file is required for each segment that receives specified inflow.
            Thus, the maximum value of NUMTAB that can be specified is equal to the total number of
            segments specified in Item 1c with variables NSS. The name (Fname) and unit number (Nunit)
            of each tabular file must be specified in the MODFLOW-2005 Name File using tile type (Ftype) DATA.
            maxval

            """
            f_sfr.write('{} {} {}\n'.format(self.tabfiles, self.numtab, self.maxval))

        self._write_1c(f_sfr)

        # item 2
        self._write_reach_data(f_sfr)

        # items 3 and 4 are skipped (parameters not supported)

        for i in range(0, self.nper):

            # item 5
            itmp = self.dataset_5[i][0]
            f_sfr.write(' '.join(map(str, self.dataset_5[i])) + '\n')
            if itmp > 0:

                # Item 6
                for j in range(itmp):

                    # write datasets 6a, 6b and 6c
                    self._write_segment_data(i, j, f_sfr)

                    icalc = self.segment_data[i][j][1]
                    if icalc == 2:
                        if i == 0 or self.nstrm > 0 and not self.reachinput: # or isfropt <= 1:
                            for k in range(2):
                                for d in self.channel_geometry_data[i][j][k]:
                                    f_sfr.write('{:.2f} '.format(d))
                                f_sfr.write('\n')

                    if icalc == 4:
                        #nstrpts = self.segment_data[i][j][5]
                        for k in range(3):
                            for d in self.channel_flow_data[i][j][k]:
                                f_sfr.write('{:.2f} '.format(d))
                            f_sfr.write('\n')
            if self.tabfiles and i == 0:
                for j in sorted(self.tabfiles_dict.keys()):
                    f_sfr.write('{:.0f} {:.0f} {:.0f}\n'.format(j,
                                                                self.tabfiles_dict[j]['numval'],
                                                                self.tabfiles_dict[j]['inuit']))
            else:
                continue
        f_sfr.close()

def _markitzero(recarray, inds):
    """subtracts 1 from columns specified in inds argument, to convert from 1 to 0-based indexing
    """
    lnames = [n.lower() for n in recarray.dtype.names]
    for idx in inds:
        if (idx in lnames):
            recarray[idx] -= 1

def _pop_item(line):
    if len(line) > 0:
        return line.pop(0)
    return 0

def _get_dataset(line, dataset):
    tmp = []
    # interpret number supplied with decimal points as floats, rest as ints
    # this could be a bad idea (vs. explicitly formatting values for each dataset)
    for i, s in enumerate(line.strip().split()):
        if s.isnumeric():
            dataset[i] = int(s)
        else:
            try:
                dataset[i] = float(s)
            except:
                break
    return dataset

def _fmt_string(array):
    fmt_string = ''
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if (vtype == 'i'):
            fmt_string += '{:.0f} '
        elif (vtype == 'f'):
            fmt_string += '{:.6f} '
        elif (vtype == 'o'):
            fmt_string += '{} '
        elif (vtype == 's'):
            raise Exception("mflist error: '\str\' type found it dtype." + \
                            " This gives unpredictable results when " + \
                            "recarray to file - change to \'object\' type")
        else:
            raise Exception("mflist.fmt_string error: unknown vtype " + \
                            "in dtype:" + vtype)
    return fmt_string

def _fmt_string_list(array):
    fmt_string = []
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if (vtype == 'i'):
            fmt_string += ['{:.0f}']
        elif (vtype == 'f'):
            fmt_string += ['{:.6f}']
        elif (vtype == 'o'):
            fmt_string += ['{}']
        elif (vtype == 's'):
            raise Exception("mflist error: '\str\' type found it dtype." + \
                            " This gives unpredictable results when " + \
                            "recarray to file - change to \'object\' type")
        else:
            raise Exception("mflist.fmt_string error: unknown vtype " + \
                            "in dtype:" + vtype)
    return fmt_string

def parse_6a(line):
    """Parse Data Set 6a for SFR2 package.
    See http://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?sfr.htm for more info

    Parameters
    ----------
    line : str
        line read from SFR package input file

    Returns
    -------
        a list of length 13 containing all variables for Data Set 6a
    """
    na = 0
    #line = [s for s in line.strip().split() if s.isnumeric()]
    line = _get_dataset(line, [0] * len(line.strip().split()))

    nseg = int(line.pop(0))
    icalc = int(line.pop(0))
    outseg = int(line.pop(0))
    iupseg = int(line.pop(0))
    iprior = na
    nstrpts = na

    if iupseg !=0:
        iprior = int(line.pop(0))
    if icalc == 4:
        nstrpts = int(line.pop(0))

    flow = float(line.pop(0))
    runoff = float(line.pop(0))
    etsw = float(line.pop(0))
    pptsw = float(line.pop(0))
    roughch = na
    roughbk = na

    if icalc in [1, 2]:
        roughch = float(line.pop(0))
    if icalc == 2:
        roughbk = float(line.pop(0))

    cdpth, fdpth, awdth, bwdth = na, na, na, na
    if icalc == 3:
        cdpth, fdpth, awdth, bwdth = map(float, line)
    return nseg, icalc, outseg, iupseg, iprior, nstrpts, flow, runoff, etsw, \
           pptsw, roughch, roughbk, cdpth, fdpth, awdth, bwdth


def parse_6bc(line, icalc, nstrm, isfropt, reachinput, per=0):
    """Parse Data Set 6b for SFR2 package.
    See http://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?sfr.htm for more info

    Parameters
    ----------
    line : str
        line read from SFR package input file

    Returns
    -------
        a list of length 9 containing all variables for Data Set 6b
    """
    na = 0
    #line = [s for s in line.strip().split() if s.isnumeric()]
    #line = list(map(float, line.strip().split()))
    line = _get_dataset(line, [0] * len(line.strip().split()))

    hcond, thickm, elevup, width, depth, thts, thti, eps, uhc = [0.0] * 9

    if isfropt in [0, 4, 5] and icalc <=0:
        hcond = line.pop(0)
        thickm = line.pop(0)
        elevup = line.pop(0)
        width = line.pop(0)
        depth = line.pop(0)
    elif isfropt in [0, 4, 5] and icalc == 1:
        hcond = line.pop(0)
        if per == 0:
            thickm = line.pop(0)
            elevup = line.pop(0)
            width = line.pop(0)
            thts = _pop_item(line)
            thti = _pop_item(line)
            eps = _pop_item(line)
            if isfropt == 5:
                uhc = line.pop(0)
    elif isfropt in [0, 4, 5] and icalc >= 2:
        hcond = line.pop(0)
        if isfropt in [4, 5] and per > 0 and icalc == 2:
            pass
        else:
            thickm = line.pop(0)
            elevup = line.pop(0)
            if isfropt in [4, 5] and icalc == 2 and per == 0:
                thts = line.pop(0)
                thti = line.pop(0)
                eps = line.pop(0)
                if isfropt == 5:
                    uhc = line.pop(0)
            else:
                pass
    elif isfropt == 1 and icalc <= 1:
        width = line.pop(0)
        if icalc <= 0:
            depth = line.pop(0)
    elif isfropt in [2, 3] and icalc <= 1:
        if per > 0:
            pass
        else:
            width = line.pop(0)
            if icalc <= 0:
                depth = line.pop(0)
    else:
        pass
    return hcond, thickm, elevup, width, depth, thts, thti, eps, uhc