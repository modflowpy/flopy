__author__ = 'aleaf'

import sys

sys.path.insert(0, '..')
import os
import numpy as np
from numpy.lib import recfunctions
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
    reach_data : recarray
        Numpy record array of length equal to nstrm, with columns for each variable entered in item 2
        (see SFR package input instructions). In following flopy convention, layer, row, column and node number
        (for unstructured grids) are zero-based; segment and reach are one-based.
    segment_data : recarray
        Numpy record array of length equal to nss, with columns for each variable entered in items 6a, 6b and 6c
        (see SFR package input instructions). Segment numbers are one-based.
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
    outlets : nested dictionary
        Contains the outlet for each SFR segment; format is {per: {segment: outlet}}
        This attribute is created by the get_outlets() method.
    outsegs : dictionary of arrays
        Each array is of shape nss rows x maximum of nss columns. The first column contains the SFR segments,
        the second column contains the outsegs of those segments; the third column the outsegs of the outsegs,
        and so on, until all outlets have been encountered, or nss is reached. The latter case indicates
        circular routing. This attribute is created by the get_outlets() method.

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
    default_value = -1.0E+10

    def __init__(self, model, nstrm=0, nss=0, nsfrpar=0, nparseg=0, const=128390.4, dleak=0.0001, istcb1=50, istcb2=66,
                 isfropt=0,
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
        self.numtab = 0 if not tabfiles else len(tabfiles_dict)
        self.maxval = np.max([tb['numval'] for tb in tabfiles_dict.values()]) if self.numtab > 0 else 0

        # Dataset 1c. ----------------------------------------------------------------------
        self.nstrm = nstrm  # number of reaches, negative value is flag for unsat. flow beneath streams and/or transient routing
        self.nss = nss  # number of stream segments
        self.nsfrpar = nsfrpar
        self.nparseg = nparseg
        self.const = const  # conversion factor used in calculating stream depth for stream reach (icalc = 1 or 2)
        self.dleak = dleak  # tolerance level of stream depth used in computing leakage
        self.istcb1 = istcb1  # flag; unit number for stream leakage output
        self.istcb2 = istcb2  # flag; unit number for writing table of SFR output to text file

        # if nstrm < 0
        self.isfropt = isfropt  # defines the format of the input data and whether or not unsaturated flow is simulated

        # if isfropt > 1
        self.nstrail = nstrail  # number of trailing wave increments
        self.isuzn = isuzn  # max number of vertical cells used to define unsat. zone
        self.nsfrsets = nsfrsets  # max number trailing waves sets

        # if nstrm < 0 (MF-2005 only)
        self.irtflag = irtflg  # switch for transient streamflow routing (> 0 = kinematic wave)
        # if irtflag > 0
        self.numtim = numtim  # number of subtimesteps used for routing
        self.weight = weight  # time weighting factor used to calculate the change in channel storage
        self.flwtol = flwtol  # streamflow tolerance for convergence of the kinematic wave equation

        # Dataset 2. -----------------------------------------------------------------------
        self.reach_data = reach_data
        # assign node numbers if there are none (structured grid)
        if np.diff(self.reach_data.node).max() == 0 and 'DIS' in self.parent.get_package_list():
            # first make kij list
            lrc = self.reach_data[['krch', 'irch', 'jrch']]
            lrc = (lrc.view((int, len(lrc.dtype.names))) + 1).tolist()
            self.reach_data['node'] = self.parent.dis.get_node(lrc)
        # assign unique ID and outreach columns to each reach
        self.reach_data.sort(order=['iseg', 'ireach'])
        new_cols = {'reachID': np.arange(1, len(self.reach_data) + 1),
                    'outreach': np.zeros(len(self.reach_data))}
        for k, v in new_cols.items():
            if k not in self.reach_data.dtype.names:
                recfunctions.append_fields(self.reach_data, names=k, data=v, asrecarray=True)

        # Datasets 4 and 6. -----------------------------------------------------------------------
        self.segment_data = segment_data
        self.channel_geometry_data = channel_geometry_data
        self.channel_flow_data = channel_flow_data

        # Dataset 5 -----------------------------------------------------------------------
        self.dataset_5 = dataset_5

        # Attributes not included in SFR package input
        self.outsegs = {}  # dictionary of arrays; see Attributes section of documentation
        self.outlets = {}  # nested dictionary of format {per: {segment: outlet}}
        # -input format checks:
        assert isfropt in [0, 1, 2, 3, 4, 5]

        self.parent.add_package(self)

    def __repr__(self):
        return 'SFR2 class'

    @staticmethod
    def get_empty_reach_data(nreaches=0, aux_names=None, structured=True, default_value=-1.0E+10):
        # get an empty recarray that correponds to dtype
        dtype = ModflowSfr2.get_default_reach_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((nreaches, len(dtype)), dtype=dtype)
        d[:, :] = default_value
        d = np.core.records.fromarrays(d.transpose(), dtype=dtype)
        d['reachID'] = np.arange(1, nreaches + 1)
        return d

    @staticmethod
    def get_empty_segment_data(nsegments=0, aux_names=None, default_value=-1.0E+10):
        # get an empty recarray that correponds to dtype
        dtype = ModflowSfr2.get_default_segment_dtype()
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((nsegments, len(dtype)), dtype=dtype)
        d[:, :] = default_value
        return np.core.records.fromarrays(d.transpose(), dtype=dtype)

    @staticmethod
    def get_default_reach_dtype(structured=True):
        if structured:
            # include node column for structured grids (useful for indexing)
            return np.dtype([('node', np.int),
                             ('krch', np.int),
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
                             ('uhc', np.float32),
                             ('reachID', np.int),
                             ('outreach', np.int)])
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
                             ('uhc', np.float32),
                             ('reachID', np.int),
                             ('outreach', np.int)])

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
                         ('elevup', np.float32),
                         ('width1', np.float32),
                         ('depth1', np.float32),
                         ('thts1', np.float32),
                         ('thti1', np.float32),
                         ('eps1', np.float32),
                         ('uhc1', np.float32),
                         ('hcond2', np.float32),
                         ('thickm2', np.float32),
                         ('elevdn', np.float32),
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
        structured = model.structured
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

        # item 1c
        nstrm, nss, nsfrpar, nparseg, const, dleak, istcb1, istcb2, \
        isfropt, nstrail, isuzn, nsfrsets, \
        irtflg, numtim, weight, flwtol, option = parse_1c(line, reachinput=reachinput, transroute=transroute)

        # item 2
        # set column names, dtypes
        names = _get_item2_names(nstrm, reachinput, isfropt, structured)
        dtypes = [d for d in ModflowSfr2.get_default_reach_dtype().descr
                  if d[0] in names]

        lines = []
        for i in range(abs(nstrm)):
            ireach = tuple(map(float, next(f).strip().split()))
            lines.append(ireach)

        tmp = np.array(lines, dtype=dtypes)
        # initialize full reach_data array with all possible columns
        reach_data = ModflowSfr2.get_empty_reach_data(len(lines))
        for n in names:
            reach_data[n] = tmp[n]  # not sure if there's a way to assign multiple columns

        # zero-based convention
        inds = ['krch', 'irch', 'jrch'] if structured else ['node']
        _markitzero(reach_data, inds)

        # items 3 and 4 are skipped (parameters not supported)
        # item 5
        segment_data = {}
        channel_geometry_data = {}
        channel_flow_data = {}
        dataset_5 = {}
        aux_variables = {}  # not sure where the auxillary variables are supposed to go
        for i in range(0, nper):
            # Dataset 5
            dataset_5[i] = _get_dataset(next(f), [1, 0, 0, 0])
            itmp = dataset_5[i][0]
            if itmp > 0:
                # Item 6
                current = ModflowSfr2.get_empty_segment_data(nsegments=itmp, aux_names=option)
                current_aux = {}  # container to hold any auxillary variables
                current_6d = {}  # these could also be implemented as structured arrays with a column for segment number
                current_6e = {}
                for j in range(itmp):

                    dataset_6a = parse_6a(next(f), option)
                    current_aux[j] = dataset_6a[-1]
                    dataset_6a = dataset_6a[:-1]  # drop xyz
                    icalc = dataset_6a[1]
                    dataset_6b = parse_6bc(next(f), icalc, nstrm, isfropt, reachinput, per=i)
                    dataset_6c = parse_6bc(next(f), icalc, nstrm, isfropt, reachinput, per=i)

                    current[j] = dataset_6a + dataset_6b + dataset_6c

                    if icalc == 2:
                        # ATL: not sure exactly how isfropt logic functions for this
                        # dataset 6d description suggests that this line isn't read for isfropt > 1
                        # but description of icalc suggest that icalc=2 (8-point channel) can be used with any isfropt
                        if i == 0 or nstrm > 0 and not reachinput:  # or isfropt <= 1:
                            dataset_6d = []
                            for k in range(2):
                                dataset_6d.append(_get_dataset(next(f), [0.0] * 8))
                                # dataset_6d.append(list(map(float, next(f).strip().split())))
                            current_6d[j + 1] = dataset_6d
                    if icalc == 4:
                        nstrpts = dataset_6a[5]
                        dataset_6e = []
                        for k in range(3):
                            dataset_6e.append(_get_dataset(next(f), [0.0] * nstrpts))
                        current_6e[j + 1] = dataset_6e

                segment_data[i] = current
                aux_variables[j + 1] = current_aux
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

        return ModflowSfr2(model, nstrm=nstrm, nss=nss, nsfrpar=nsfrpar, nparseg=nparseg, const=const, dleak=dleak,
                           istcb1=istcb1, istcb2=istcb2,
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

    def check(self, f=None, verbose=True, level=1):
        """
        Check sfr2 package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a sting is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.sfr2.check()
        """
        chk = check(self, verbose=verbose, level=level)
        chk.numbering()
        chk.routing()
        chk.overlapping_conductance()
        chk.elevations()
        chk.slope()

        if f is not None:
            if isinstance(f, str):
                pth = os.path.join(self.parent.model_ws, f)
                f = open(pth, 'w')
            f.write('{}\n'.format(chk.txt))
            f.close()
        return chk

    def get_outlets(self, level=0, verbose=True):
        """Traces all routing connections from each headwater to the outlet.
        """
        txt = ''
        for per in range(self.nper):
            if per > 0 > self.dataset_5[per][0]:  # skip stress periods where seg data not defined
                continue
            segments = self.segment_data[per].nseg
            outsegs = self.segment_data[per].outseg
            all_outsegs = np.vstack([segments, outsegs])
            max_outseg = all_outsegs[-1].max()
            knt = 1
            while max_outseg > 0:

                nextlevel = np.array([outsegs[s - 1] if s > 0 and s < 999999 else 0
                                      for s in all_outsegs[-1]])

                all_outsegs = np.vstack([all_outsegs, nextlevel])
                max_outseg = nextlevel.max()
                if max_outseg == 0:
                    break
                knt += 1
                if knt > self.nss:
                    # subset outsegs map to only include rows with outseg number > 0 in last column
                    circular_segs = all_outsegs.T[all_outsegs[-1] > 0]

                    # only retain one instance of each outseg number at iteration 1000
                    vals = []  # append outseg values to vals after they've appeared once
                    mask = [(True, vals.append(v))[0]
                            if v not in vals
                            else False for v in circular_segs[-1]]
                    circular_segs = circular_segs[np.array(mask)]

                    # cull the circular segments array to remove duplicate instances of routing circles
                    circles = []
                    duplicates = []
                    for i in range(np.shape(circular_segs)[0]):
                        # find where values in the row equal the last value;
                        # record the index of the second to last instance of last value
                        repeat_start_ind = np.where(circular_segs[i] == circular_segs[i, -1])[0][-2:][0]
                        # use that index to slice out the repeated segment sequence
                        circular_seq = circular_segs[i, repeat_start_ind:].tolist()
                        # keep track of unique sequences of repeated segments
                        if set(circular_seq) not in circles:
                            circles.append(set(circular_seq))
                            duplicates.append(False)
                        else:
                            duplicates.append(True)
                    circular_segs = circular_segs[~np.array(duplicates), :]

                    txt += '{0} instances where an outlet was not found after {1} consecutive segments!\n' \
                        .format(len(circular_segs), self.nss)
                    if level == 1:
                        txt += '\n'.join([' '.join(map(str, row)) for row in circular_segs]) + '\n'
                    else:
                        f = 'circular_routing.csv'
                        np.savetxt(f, circular_segs, fmt='%d', delimiter=',', header=txt)
                        txt += 'See {} for details.'.format(f)
                    if verbose:
                        print(txt)
                    break

            # the array of segment sequence is useful for other other operations,
            # such as plotting elevation profiles
            self.outsegs[per] = all_outsegs

            # create a dictionary listing outlets associated with each segment
            # outlet is the last value in each row of outseg array that is != 0 or 999999
            self.outlets[per] = {i + 1: r[(r != 0) & (r != 999999)][-1]
            if len(r[(r != 0) & (r != 999999)]) > 0
            else i + 1
                                 for i, r in enumerate(all_outsegs.T)}
        return txt

    def get_outreaches(self):
        """Determine the outreach for each SFR reach (requires a reachID column in reach_data).
        Uses the segment routing specified for the first stress period to route reaches between segments.
        """
        self.reach_data.sort(order=['iseg', 'ireach'])
        reach_data = self.reach_data
        segment_data = self.segment_data[0]
        # this vectorized approach is more than an order of magnitude faster than a list comprehension
        first_reaches = reach_data[reach_data.ireach == 1]
        last_reaches = np.append((np.diff(reach_data.iseg) == 1), True)
        reach_data.outreach = np.append(reach_data.reachID[1:], 0)
        reach_data.outreach[last_reaches] = first_reaches.reachID[segment_data.outseg - 1]
        self.reach_data['outreach'] = reach_data.outreach

    def _interpolate_to_reaches(self, segvar1, segvar2, per=0):
        """Interpolate values in datasets 6b and 6c to each reach in stream segment

        Parameters
        ----------
        segvar1 : str
            Column/variable name in segment_data array for representing start of segment
            (e.g. hcond1 for hydraulic conductivity)
            For segments with icalc=2 (specified channel geometry); if width1 is given,
            the eigth distance point (XCPT8) from dataset 6d will be used as the stream width.
            For icalc=3, an abitrary width of 5 is assigned.
            For icalc=4, the mean value for width given in item 6e is used.
        segvar2 : str
            Column/variable name in segment_data array for representing start of segment
            (e.g. hcond2 for hydraulic conductivity)
        per : int
            Stress period with segment data to interpolate

        Returns
        -------
        reach_values : 1D array
            One dimmensional array of interpolated values of same length as reach_data array.
            For example, hcond1 and hcond2 could be entered as inputs to get values for the
            strhc1 (hydraulic conductivity) column in reach_data.

        """
        reach_data = self.reach_data
        segment_data = self.segment_data[per]
        reach_values = []
        for seg in segment_data.nseg:
            reaches = reach_data[reach_data.iseg == seg]
            dist = np.cumsum(reaches.rchlen) - 0.5 * reaches.rchlen
            icalc = segment_data.icalc[segment_data.nseg == seg]
            if 'width' in segvar1 and icalc == 2:  # get width from channel cross section length
                channel_geometry_data = self.channel_geometry_data[per]
                reach_values += list(np.ones(len(reaches)) * channel_geometry_data[seg][0][-1])
            elif 'width' in segvar1 and icalc == 3:  # assign arbitrary width since width is based on flow
                reach_values += list(np.ones(len(reaches)) * 5)
            elif 'width' in segvar1 and icalc == 4:  # assume width to be mean from streamflow width/flow table
                channel_flow_data = self.channel_flow_data[per]
                reach_values += list(np.ones(len(reaches)) * np.mean(channel_flow_data[seg][2]))
            else:
                fp = [segment_data[segment_data['nseg'] == seg][segvar1][0],
                      segment_data[segment_data['nseg'] == seg][segvar2][0]]
                xp = [dist[0], dist[-1]]
                reach_values += np.interp(dist, xp, fp).tolist()
        return np.array(reach_values)

    def _write_1c(self, f_sfr):

        # NSTRM NSS NSFRPAR NPARSEG CONST DLEAK ISTCB1  ISTCB2
        # [ISFROPT] [NSTRAIL] [ISUZN] [NSFRSETS] [IRTFLG] [NUMTIM] [WEIGHT] [FLWTOL]
        f_sfr.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.8f} {:.8f} {:.0f} {:.0f} '
                    .format(self.nstrm, self.nss, self.nsfrpar, self.nparseg,
                            self.const, self.dleak, self.istcb1, self.istcb2))
        if self.reachinput:
            self.nstrm = abs(self.nstrm)  # see explanation for dataset 1c in online guide
            f_sfr.write('{:.0f} '.format(self.isfropt))
            if self.isfropt > 1:
                f_sfr.write('{:.0f} {:.0f} {:.0f} '.format(self.nstrail,
                                                           self.isuzn,
                                                           self.nsfrsets))
        if self.nstrm < 0:
            f_sfr.write('{:.0f} {:.0f} {:.0f} {:.0f} '.format(self.isfropt,
                                                              self.nstrail,
                                                              self.isuzn,
                                                              self.nsfrsets))
        if self.nstrm < 0 or self.transroute:
            f_sfr.write('{:.0f} '.format(self.irtflag))
            if self.irtflag < 0:
                f_sfr.write('{:.0f} {:.8f} {:.8f} '.format(self.numtim,
                                                           self.weight,
                                                           self.flwtol))
        f_sfr.write('\n')

    def _write_reach_data(self, f_sfr):

        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(self.reach_data, np.recarray), "mflist.__tofile() data arg " + \
                                                         "not a recarray"

        # decide which columns to write
        columns = self._get_item2_names()

        # Add one to the kij indices
        names = self.reach_data.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        # --make copy of data for multiple calls
        d = np.recarray.copy(self.reach_data[columns])
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

        self._write_6bc(i, j, f_sfr, cols=['hcond1', 'thickm1', 'elevup', 'width1', 'depth1', 'thts1', 'thti1',
                                           'eps1', 'uhc1'])
        self._write_6bc(i, j, f_sfr, cols=['hcond2', 'thickm2', 'elevdn', 'width2', 'depth2', 'thts2', 'thti2',
                                           'eps2', 'uhc2'])

    def _write_6bc(self, i, j, f_sfr, cols=[]):

        icalc = self.segment_data[i][j][1]
        fmts = _fmt_string_list(self.segment_data[i][cols][j])
        hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc = self.segment_data[i][cols][j]

        if self.isfropt in [0, 4, 5] and icalc <= 0:
            f_sfr.write(' '.join(fmts[0:5]).format(hcond, thickm, elevupdn, width, depth) + ' ')

        elif self.isfropt in [0, 4, 5] and icalc == 1:
            f_sfr.write(fmts[0].format(hcond) + ' ')

            if i == 0:
                f_sfr.write(' '.join(fmts[1:4]).format(thickm, elevupdn, width) + ' ')
                f_sfr.write(' '.join(fmts[5:8]).format(thts, thti, eps) + ' ')
                if self.isfropt == 5:
                    f_sfr.write(fmts[8].format(uhc) + ' ')
        elif self.isfropt in [0, 4, 5] and icalc >= 2:
            f_sfr.write(fmts[0].format(hcond) + ' ')

            if self.isfropt in [4, 5] and i > 0 and icalc == 2:
                pass
            else:
                f_sfr.write(' '.join(fmts[1:3]).format(thickm, elevupdn) + ' ')

                if self.isfropt in [4, 5] and icalc == 2 and i == 0:
                    f_sfr.write(' '.join(fmts[3:6]).format(thts, thti, eps) + ' ')

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

        # tabfiles = False
        # tabfiles_dict = {}
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
                        if i == 0 or self.nstrm > 0 and not self.reachinput:  # or isfropt <= 1:
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


class check:
    """
    Check SFR2 package for common errors

    Parameters
    ----------
    sfrpackage : object
        Instance of Flopy ModflowSfr2 class.
    verbose : bool
        Boolean flag used to determine if check method results are
        written to the screen
    level : int
        Check method analysis level. If level=0, summary checks are
        performed. If level=1, full checks are performed.

    Notes
    -----

    Daniel Feinstein's top 10 SFR problems (7/16/2014):
    1) cell gaps btw adjacent reaches in a single segment
    2) cell gaps btw routed segments. possibly because of re-entry problems at domain edge
    3) adjacent reaches with STOP sloping the wrong way
    4) routed segments with end/start sloping the wrong way
    5) STOP>TOP1 violations, i.e.,floaters
    6) STOP<<TOP1 violations, i.e., exaggerated incisions
    7) segments that end within one diagonal cell distance from another segment, inviting linkage
    8) circular routing of segments
    9) multiple reaches with non-zero conductance in a single cell
    10) reaches in inactive cells

    Also after running the model they will want to check for backwater effects.
    """

    def __init__(self, sfrpackage, verbose=True, level=1):
        self.sfr = sfrpackage
        self.reach_data = sfrpackage.reach_data
        self.segment_data = sfrpackage.segment_data
        self.verbose = verbose
        self.level = level
        self.passed = []
        self.failed = []
        self.txt = '\n{} ERRORS:\n'.format(self.sfr.name[0])

    def _txt_footer(self, headertxt, txt, testname, passed=False):
        if len(txt) == 0 or passed:
            txt += 'passed.'
            self.passed.append(testname)
        else:
            self.failed.append(testname)
        if self.verbose:
            print(txt + '\n')
        self.txt += headertxt + txt + '\n'

    def run_all(self):
        return self.sfr.check()

    def numbering(self):
        """checks for continuity in segment and reach numbering
        """

        headertxt = 'Checking for continuity in segment and reach numbering...\n'
        if self.verbose:
            print(headertxt.strip())
        txt = ''
        for per in range(self.sfr.nper):
            if per > 0 > self.sfr.dataset_5[per][0]:
                continue
            # check segment numbering
            txt += _check_numbers(self.sfr.nss,
                                  self.segment_data[per]['nseg'],
                                  level=self.level,
                                  datatype='segment')

        # check reach numbering
        for segment in np.arange(1, self.sfr.nss + 1):
            reaches = self.reach_data.ireach[self.reach_data.iseg == segment]
            t = _check_numbers(len(reaches),
                               reaches,
                               level=self.level,
                               datatype='reach')
            if len(t) > 0:
                txt += 'Segment {} has {}'.format(segment, t)
        self._txt_footer(headertxt, txt, 'continuity in segment and reach numbering')

    def routing(self):
        """checks for breaks in routing and does comprehensive check for circular routing
        """
        headertxt = 'Checking for circular routing...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        txt += self.sfr.get_outlets(level=self.level, verbose=False)  # will print twice if verbose=True
        self._txt_footer(headertxt, txt, 'circular routing')

    def overlapping_conductance(self, tol=1e-6):
        """checks for multiple SFR reaches in one cell; and whether more than one reach has Cond > 0
        """
        headertxt = 'Checking for model cells with multiple non-zero SFR conductances...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        # make nreach vectors of each conductance parameter
        reach_data = self.reach_data.copy()

        K = reach_data.strhc1
        if K.max() == 0:
            K = self.sfr._interpolate_to_reaches('hcond1', 'hcond2')
        b = reach_data.strthick
        if b.max() == 0:
            b = self.sfr._interpolate_to_reaches('thickm1', 'thickm2')
        L = reach_data.rchlen
        w = self.sfr._interpolate_to_reaches('width1', 'width2')

        # Calculate SFR conductance for each reach
        Cond = K * w * L / b

        shared_cells = _get_duplicates(reach_data.node)

        nodes_with_multiple_conductance = set()
        for node in shared_cells:

            # select the collocated reaches for this cell
            conductances = Cond[reach_data.node == node].copy()
            conductances.sort()

            # list nodes with multiple non-zero SFR reach conductances
            if conductances[0] / conductances[-1] > tol:
                nodes_with_multiple_conductance.update({node})

        if len(nodes_with_multiple_conductance) > 0:
            txt += '{} model cells with multiple non-zero SFR conductances found.\n' \
                   'This may lead to circular routing between collocated reaches.\n' \
                .format(len(nodes_with_multiple_conductance))
            if self.level == 1:
                txt += 'Nodes with overlapping conductances:\n'

                reach_data['strthick'] = b
                reach_data['strhc1'] = K

                cols = [c for c in reach_data.dtype.names if c in \
                        ['node', 'krch', 'irch', 'jrch', 'iseg', 'ireach', 'rchlen', 'strthick', 'strhc1']]

                recfunctions.append_fields(reach_data,
                                           names=['width', 'conductance'],
                                           data=[w, Cond],
                                           asrecarray=True)
                has_multiple = np.array([True if n in nodes_with_multiple_conductance
                                         else False for n in reach_data.node])
                txt += _print_rec_array(reach_data[cols][has_multiple])

        self._txt_footer(headertxt, txt, 'overlapping conductance')

    def elevations(self):
        """checks for multiple SFR reaches in one cell; and whether more than one reach has Cond > 0
        """
        headertxt = 'Checking segment_data for downstream rises in streambed elevation...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        # decide whether to check elevup and elevdn from items 6b/c
        # (see online guide to SFR input; Data Set 6b description)
        passed = False
        if self.sfr.isfropt in [0, 4, 5]:
            pers = sorted(self.segment_data.keys())
            for per in pers:
                segment_data = self.segment_data[per][self.segment_data[per].elevup > -999999]

                # enforce consecutive increasing segment numbers (for indexing)
                segment_data.sort(order='nseg')
                t = _check_numbers(len(segment_data), segment_data.nseg, level=1, datatype='Segment')
                if len(t) > 0:
                    raise Exception('Elevation check requires consecutive segment numbering.')

                # first check for segments where elevdn > elevup
                backwards_segments = (segment_data.elevdn - segment_data.elevup) > 0
                if np.any(backwards_segments):
                    backwards_info = segment_data[['nseg', 'outseg', 'elevup', 'elevdn']][backwards_segments]
                    txt += 'Stress Period {}: {} segments encountered with elevdn > elevup.\n' \
                        .format(per + 1, len(backwards_info))
                    if self.level == 1:
                        txt += 'Backwards segments:\n'
                        txt += _print_rec_array(backwards_info)
                    txt += '\n'

                # next check for rises between segments
                outseg_elevup = np.array([segment_data.elevup[o - 1] for o in segment_data.outseg])
                backwards_connections = ((outseg_elevup - segment_data.elevdn) > 0) & \
                                        (segment_data.outseg != 0)
                if np.any(backwards_connections):
                    backwards_info = segment_data[['nseg', 'outseg', 'elevdn']]
                    recfunctions.append_fields(backwards_info,
                                               names='outseg_elevup',
                                               data=outseg_elevup,
                                               asrecarray=True)
                    backwards_info = backwards_info[backwards_connections]
                    txt += 'Stress Period {}: {} segments encountered with outseg elevup > elevdn.\n' \
                        .format(per + 1, len(backwards_info))
                    if self.level == 1:
                        txt += 'Backwards segment connections:\n'
                        txt += _print_rec_array(backwards_info)
                    txt += '\n'
                if len(txt) == 0:
                    passed = True
        else:
            txt += 'Segment elevup and elevdn not specified for nstrm={} and isfropt={}\n' \
                .format(self.sfr.nstrm, self.sfr.isfropt)
            passed = True
        self._txt_footer(headertxt, txt, 'segment elevations')

        headertxt = 'Checking reach_data for downstream rises in streambed elevation...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        if self.sfr.nstrm < 0 or self.sfr.reachinput and self.sfr.isfropt in [1, 2, 3]:  # see SFR input instructions
            # first get an outreach for each reach
            if np.diff(self.sfr.reach_data.outreach).max() == 0: # not sure if this is the best test
                self.sfr.get_outreaches()
            reach_data = self.sfr.reach_data  # inconsistent with other checks that work with
            # reach_data attribute of check class. Want to have get_outreaches as a method of sfr class
            # (for other uses). Not sure if other check methods should also copy reach_data directly from
            # SFR package instance for consistency.

            # use outreach values to get downstream elevations
            outreach_elevdn = np.array([reach_data.strtop[o-1] for o in reach_data.outreach])
            reach_data = recfunctions.append_fields(reach_data,
                                                    names='strtopdn',
                                                    data=outreach_elevdn,
                                                    asrecarray=True)
            elevation_rises = (reach_data.strtopdn - reach_data.strtop) > 0
            if np.any(elevation_rises):
                backwards_info = reach_data[['krch', 'irch', 'jrch', 'iseg', 'strtop',
                                             'strtopdn', 'reachID', 'outreach']][elevation_rises]
                txt += '{} reaches encountered with strtop > strtop of downstream reach.\n' \
                    .format(len(backwards_info))
                if self.level == 1:
                    txt += 'Elevation rises:\n'
                    txt += _print_rec_array(backwards_info)
                txt += '\n'
                passed = False
        else:
            txt += 'Reach strtop not specified for nstrm={}, reachinput={} and isfropt={}\n' \
                .format(self.sfr.nstrm, self.sfr.reachinput, self.sfr.isfropt)
            passed = True
        self._txt_footer(headertxt, txt, 'reach elevations', passed)

        headertxt = 'Checking reach_data for inconsistencies between streambed elevations and the model grid...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())
        passed = False
        if self.sfr.nstrm < 0 or self.sfr.reachinput and self.sfr.isfropt in [1, 2, 3]:  # see SFR input instructions
            reach_data = self.reach_data
            i, j, k = reach_data.irch, reach_data.jrch, reach_data.krch

            # check streambed bottoms in relation to respective cell bottoms
            bots = self.sfr.parent.dis.botm.array[k, i, j]
            recfunctions.append_fields(reach_data, names='layerbot', data=bots, asrecarray=True)
            streambed_bots = reach_data.strtop - reach_data.strthick
            below_layer_bottoms = streambed_bots < bots
            if np.any(below_layer_bottoms):
                below_info = reach_data[['krch', 'irch', 'jrch', 'iseg', 'strtop',
                                             'strthick', 'layerbot', 'reachID']][below_layer_bottoms]
                txt += '{} reaches encountered with streambed bottom below layer bottom.\n' \
                    .format(len(below_info))
                if self.level == 1:
                    txt += 'Layer bottom violations:\n'
                    txt += _print_rec_array(below_info)
                txt += '\n'

            # check streambed elevations in relation to model top
            tops = self.sfr.parent.dis.top.array[i, j]
            recfunctions.append_fields(reach_data, names='modeltop', data=tops, asrecarray=True)
            above_model_top = reach_data.strtop > tops
            if np.any(above_model_top):
                above_info = reach_data[['krch', 'irch', 'jrch', 'iseg', 'strtop',
                                             'modeltop', 'reachID']][above_model_top]
                txt += '{} reaches encountered with streambed above model top.\n' \
                    .format(len(below_info))
                if self.level == 1:
                    txt += 'Model top violations:\n'
                    txt += _print_rec_array(above_info)
                txt += '\n'
            if len(txt) == 0:
                passed = True
        else:
            txt += 'Reach strtop, strthick not specified for nstrm={}, reachinput={} and isfropt={}\n' \
                .format(self.sfr.nstrm, self.sfr.reachinput, self.sfr.isfropt)
            passed = True
        self._txt_footer(headertxt, txt, 'reach elevations vs. grid elevations', passed)

        # In cases where segment end elevations/thicknesses are used,
        # do these need to be checked for consistency with layer bottoms?

        headertxt = 'Checking segment_data for inconsistencies between segment end elevations and the model grid...\n'
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        if self.sfr.isfropt in [0, 4, 5]:
            reach_data = self.reach_data
            pers = sorted(self.segment_data.keys())
            for per in pers:
                segment_data = self.segment_data[per][self.segment_data[per].elevup > -999999]

                # enforce consecutive increasing segment numbers (for indexing)
                segment_data.sort(order='nseg')
                t = _check_numbers(len(segment_data), segment_data.nseg, level=1, datatype='Segment')
                if len(t) > 0:
                    raise Exception('Elevation check requires consecutive segment numbering.')

            first_reaches = reach_data[reach_data.ireach == 1].copy()
            last_reaches = reach_data[np.append((np.diff(reach_data.iseg) == 1), True)].copy()
            segment_ends = recfunctions.stack_arrays([first_reaches, last_reaches],
                                                     asrecarray=True, usemask=False)
            segment_ends['strtop'] = np.append(segment_data.elevup, segment_data.elevdn)
            i, j = segment_ends.irch, segment_ends.jrch
            tops = self.sfr.parent.dis.top.array[i, j]
            segment_ends = recfunctions.append_fields(segment_ends,
                                                      names='modeltop',
                                                      data=tops,
                                                      asrecarray=True)
            above_model_top = (tops - segment_ends.strtop) < 0
            if np.any(above_model_top):
                above_info = segment_ends[['krch', 'irch', 'jrch', 'iseg', 'strtop',
                                             'modeltop', 'reachID']][above_model_top]
                txt += '{} reaches encountered with streambed above model top.\n' \
                    .format(len(above_info))
                if self.level == 1:
                    txt += 'Model top violations:\n'
                    txt += _print_rec_array(above_info)
                txt += '\n'
                passed = False
        else:
            txt += 'Segment elevup and elevdn not specified for nstrm={} and isfropt={}\n' \
                .format(self.sfr.nstrm, self.sfr.isfropt)
            passed = True
        self._txt_footer(headertxt, txt, 'segment elevations vs. model grid', passed)

    def slope(self, minimum_slope=1e-4):
        """Checks that streambed slopes are greater than or equal to a specified minimum value.
            Low slope values can cause "backup" or unrealistic stream stages with icalc options
            where stage is computed.
            """
        headertxt = 'Checking for streambed slopes of less than {}...\n'.format(minimum_slope)
        txt = ''
        if self.verbose:
            print(headertxt.strip())

        if self.sfr.isfropt in [1, 2, 3]:
            is_less = self.reach_data.slope < minimum_slope
            if np.any(is_less):
                below_minimum = self.reach_data[is_less]
                txt += '{} instances of streambed slopes below minimum found.'.format(len(below_minimum))
                if self.level == 1:
                    txt += 'Reaches with low slopes:\n'
                    txt += _print_rec_array(below_minimum)
                passed = False
        else:
            txt += 'slope not specified for isfropt={}\n'.format(self.sfr.isfropt)
            passed = True
        self._txt_footer(headertxt, txt, 'slope', passed)


def _check_numbers(n, numbers, level=1, datatype='reach'):
    """Check that a sequence of numbers is consecutive
    (that the sequence is equal to the range from 1 to n+1, where n is the expected length of the sequence).

    Parameters
    ----------
    n : int
        Expected length of the sequence (i.e. number of stream segments)
    numbers : array
        Sequence of numbers (i.e. 'nseg' column from the segment_data array)
    level : int
        Check method analysis level. If level=0, summary checks are
        performed. If level=1, full checks are performed.
    datatype : str, optional
        Only used for reporting.
    """
    txt = ''
    num_range = np.arange(1, n + 1)
    if not np.array_equal(num_range, numbers):
        txt += 'Invalid {} numbering\n'.format(datatype)
        if level == 1:
            gaps = num_range[np.diff(numbers) != 1] + 1
            if len(gaps) > 0:
                gapstr = ' '.join(map(str, gaps))
                txt += 'Gaps in numbering at positions {}\n'.format(gapstr)
    return txt


def _isnumeric(str):
    try:
        float(str)
        return True
    except:
        return False


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
        try:
            n = int(s)
        except:
            try:
                n = float(s)
            except:
                break
        dataset[i] = n
    return dataset


def _get_duplicates(a):
    """Returns duplcate values in an array, similar to pandas .duplicated() method
    http://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
    """
    s = np.sort(a, axis=None)
    return s[s[1:] == s[:-1]]


def _get_item2_names(nstrm, reachinput, isfropt, structured=False):
    """Determine which variables should be in item 2, based on model grid type,
    reachinput specification, and isfropt.

    Returns
    -------
    names : list of str
        List of names (same as variables in SFR Package input instructions) of columns
        to assign (upon load) or retain (upon write) in reach_data array.

    Note
    ----
    Lowercase is used for all variable names.
    """
    names = []
    if structured:
        names += ['krch', 'irch', 'jrch']
    else:
        names += ['node']
    names += ['iseg', 'ireach', 'rchlen']
    if nstrm < 0 or reachinput:
        if isfropt in [1, 2, 3]:
            names += ['strtop', 'slope', 'strthick', 'strhc1']
            if isfropt in [2, 3]:
                names += ['thts', 'thti', 'eps']
                if isfropt == 3:
                    names += ['uhc']
    return names


def _fmt_string(array):
    fmt_string = ''
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if (vtype == 'i'):
            fmt_string += '{:.0f} '
        elif (vtype == 'f'):
            fmt_string += '{} '
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
            fmt_string += ['{}']
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


def _print_rec_array(array, cols=None, delimiter=' '):
    """Print out a numpy record array to string, with column names.

    Parameters
    ----------
    cols : list of strings
        List of columns to print.
    delimiter : string
        Delimited to use.

    Returns
    -------
    txt : string
        Text string of array.
    """
    txt = ''
    if cols is not None:
        cols = [c for c in array.dtype.names if c in cols]
    else:
        cols = list(array.dtype.names)
    txt += delimiter.join(cols) + '\n'
    txt += '\n'.join([delimiter.join(map(str, r)) for r in array[cols].tolist()])
    return txt


def parse_1c(line, reachinput, transroute):
    """Parse Data Set 1c for SFR2 package.
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
    # line = _get_dataset(line, [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 30, 1, 2, 0.75, 0.0001, []])
    line = line.strip().split()

    nstrm = int(line.pop(0))
    nss = int(line.pop(0))
    nsfrpar = int(line.pop(0))
    nparseg = int(line.pop(0))
    const = float(line.pop(0))
    dleak = float(line.pop(0))
    istcb1 = int(line.pop(0))
    istcb2 = int(line.pop(0))

    isfropt, nstrail, isuzn, nsfrsets = na, na, na, na
    if reachinput:
        nstrm = abs(nstrm)  # see explanation for dataset 1c in online guide
        isfropt = int(line.pop(0))
        if isfropt > 1:
            nstrail = int(line.pop(0))
            isuzn = int(line.pop(0))
            nsfrsets = int(line.pop(0))
    if nstrm < 0:
        isfropt = int(line.pop(0))
        nstrail = int(line.pop(0))
        isuzn = int(line.pop(0))
        nsfrsets = int(line.pop(0))

    irtflg, numtim, weight, flwtol = na, na, na, na
    if nstrm < 0 or transroute:
        irtflg = int(line.pop(0))
        if irtflg > 0:
            numtim = int(line.pop(0))
            weight = int(line.pop(0))
            flwtol = int(line.pop(0))

    # auxillary variables (MODFLOW-LGR)
    option = [line[i] for i in np.arange(1, len(line)) if 'aux' in line[i - 1].lower()]

    return nstrm, nss, nsfrpar, nparseg, const, dleak, istcb1, istcb2, \
           isfropt, nstrail, isuzn, nsfrsets, irtflg, numtim, weight, flwtol, option


def parse_6a(line, option):
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
    line = line.strip().split()

    xyz = []
    # handle any aux variables at end of line
    for i, s in enumerate(line):
        if s.lower() in option:
            xyz.append(s.lower())

    na = 0
    nvalues = sum([_isnumeric(s) for s in line])
    # line = _get_dataset(line, [0] * nvalues)

    nseg = int(line.pop(0))
    icalc = int(line.pop(0))
    outseg = int(line.pop(0))
    iupseg = int(line.pop(0))
    iprior = na
    nstrpts = na

    if iupseg != 0:
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
           pptsw, roughch, roughbk, cdpth, fdpth, awdth, bwdth, xyz


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
    # line = [s for s in line.strip().split() if s.isnumeric()]
    nvalues = sum([_isnumeric(s) for s in line.strip().split()])
    line = _get_dataset(line, [0] * nvalues)

    hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc = [0.0] * 9

    if isfropt in [0, 4, 5] and icalc <= 0:
        hcond = line.pop(0)
        thickm = line.pop(0)
        elevupdn = line.pop(0)
        width = line.pop(0)
        depth = line.pop(0)
    elif isfropt in [0, 4, 5] and icalc == 1:
        hcond = line.pop(0)
        if per == 0:
            thickm = line.pop(0)
            elevupdn = line.pop(0)
            width = line.pop(0)  # depth is not read if icalc == 1; see table in online guide
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
            elevupdn = line.pop(0)
            if isfropt in [4, 5] and icalc == 2 and per == 0:
                # table in online guide suggests that the following items should be present in this case
                # but in the example
                thts = _pop_item(line)
                thti = _pop_item(line)
                eps = _pop_item(line)
                if isfropt == 5:
                    uhc = _pop_item(line)
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
    return hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc
