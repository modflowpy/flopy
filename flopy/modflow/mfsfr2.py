__author__ = "aleaf"

import sys
import os
import numpy as np
import warnings
import copy
from numpy.lib import recfunctions
from ..pakbase import Package
from ..utils import MfList
from ..utils.flopy_io import line_parse
from ..utils.recarray_utils import create_empty_recarray
from ..utils.optionblock import OptionBlock
from collections import OrderedDict

try:
    import pandas as pd
except:
    pd = False

try:
    from numpy.lib import NumpyVersion

    numpy114 = NumpyVersion(np.__version__) >= "1.14.0"
except ImportError:
    numpy114 = False
if numpy114:
    # use numpy's floating-point formatter (Dragon4)
    default_float_format = "{!s}"
else:
    # single-precision floats have ~7.2 decimal digits
    default_float_format = "{:.8g}"


class ModflowSfr2(Package):
    """
    Streamflow-Routing (SFR2) Package Class

    Parameters
    ----------
    model : model object
        The model object (of type :class:'flopy.modflow.mf.Modflow') to which
        this package will be added.
    nstrm : integer
        An integer value that can be specified to be positive or negative. The
        absolute value of NSTRM is equal to the number of stream reaches
        (finite-difference cells) that are active during the simulation and
        the number of lines of data to be included in Item 2, described below.
        When NSTRM is specified to be a negative integer, it is also used as a
        flag for changing the format of the data input, for simulating
        unsaturated flow beneath streams, and (or) for simulating transient
        streamflow routing (for MODFLOW-2005 simulations only), depending
        on the values specified for variables ISFROPT and IRTFLG, as described
        below. When NSTRM is negative, NSFRPAR must be set to zero, which means
        that parameters cannot be specified. By default, nstrm is set to
        negative.
    nss : integer
        An integer value equal to the number of stream segments (consisting of
        one or more reaches) that are used to define the complete stream
        network. The value of NSS represents the number of segments that must
        be defined through a combination of parameters and variables in Item 4
        or variables in Item 6.
    nparseg : integer
        An integer value equal to (or exceeding) the number of stream-segment
        definitions associated with all parameters. This number can be more
        than the total number of segments (NSS) in the stream network because
        the same segment can be defined in multiple parameters, and because
        parameters can be time-varying. NPARSEG must equal or exceed the sum
        of NLST x N for all parameters, where N is the greater of 1 and
        NUMINST; that is, NPARSEG must equal or exceed the total number of
        repetitions of item 4b. This variable must be zero when NSTRM is
        negative.
    const : float
        A real value (or conversion factor) used in calculating stream depth
        for stream reach. If stream depth is not calculated using Manning's
        equation for any stream segment (that is, ICALC does not equal 1 or 2),
        then a value of zero can be entered. If Manning's equation is used, a
        constant of 1.486 is used for flow units of cubic feet per second, and
        a constant of 1.0 is used for units of cubic meters per second. The
        constant must be multiplied by 86,400 when using time units of days in
        the simulation. An explanation of time units used in MODFLOW is given
        by Harbaugh and others (2000, p. 10).
    dleak : float
        A real value equal to the tolerance level of stream depth used in
        computing leakage between each stream reach and active model cell.
        Value is in units of length. Usually a value of 0.0001 is sufficient
        when units of feet or meters are used in model.
    ipakcb : integer
        An integer value used as a flag for writing stream-aquifer leakage
        values. If ipakcb > 0, unformatted leakage between each stream reach
        and corresponding model cell will be saved to the main cell-by-cell
        budget file whenever when a cell-by-cell budget has been specified in
        Output Control (see Harbaugh and others, 2000, pages 52-55). If
        ipakcb = 0, leakage values will not be printed or saved. Printing to
        the listing file (ipakcb < 0) is not supported.
    istcb2 : integer
        An integer value used as a flag for writing to a separate formatted
        file all information on inflows and outflows from each reach; on
        stream depth, width, and streambed conductance; and on head difference
        and gradient across the streambed. If ISTCB2 > 0, then ISTCB2 also
        represents the unit number to which all information for each stream
        reach will be saved to a separate file when a cell-by-cell budget has
        been specified in Output Control. If ISTCB2 < 0, it is the unit number
        to which unformatted streamflow out of each reach will be saved to a
        file whenever the cell-by-cell budget has been specified in Output
        Control. Unformatted output will be saved to <model name>.sfq.
    isfropt : integer
        An integer value that defines the format of the input data and whether
        or not unsaturated flow is simulated beneath streams. Values of ISFROPT
        are defined as follows

        0   No vertical unsaturated flow beneath streams. Streambed elevations,
            stream slope, streambed thickness, and streambed hydraulic
            conductivity are read for each stress period using variables
            defined in Items 6b and 6c; the optional variables in Item 2 are
            not used.
        1   No vertical unsaturated flow beneath streams. Streambed elevation,
            stream slope, streambed thickness, and streambed hydraulic
            conductivity are read for each reach only once at the beginning of
            the simulation using optional variables defined in Item 2; Items 6b
            and 6c are used to define stream width and depth for ICALC = 0 and
            stream width for ICALC = 1.
        2   Streambed and unsaturated-zone properties are read for each reach
            only once at the beginning of the simulation using optional
            variables defined in Item 2; Items 6b and 6c are used to define
            stream width and depth for ICALC = 0 and stream width for
            ICALC = 1. When using the LPF Package, saturated vertical
            hydraulic conductivity for the unsaturated zone is the same as
            the vertical hydraulic conductivity of the corresponding layer in
            LPF and input variable UHC is not read.
        3   Same as 2 except saturated vertical hydraulic conductivity for the
            unsaturated zone (input variable UHC) is read for each reach.
        4   Streambed and unsaturated-zone properties are read for the
            beginning and end of each stream segment using variables defined
            in Items 6b and 6c; the optional variables in Item 2 are not used.
            Streambed properties can vary each stress period. When using the
            LPF Package, saturated vertical hydraulic conductivity for the
            unsaturated zone is the same as the vertical hydraulic conductivity
            of the corresponding layer in LPF and input variable UHC1 is not
            read.
        5   Same as 4 except saturated vertical hydraulic conductivity for the
            unsaturated zone (input variable UHC1) is read for each segment at
            the beginning of the first stress period only.

    nstrail : integer
        An integer value that is the number of trailing wave increments used to
        represent a trailing wave. Trailing waves are used to represent a
        decrease in the surface infiltration rate. The value can be increased
        to improve mass balance in the unsaturated zone. Values between 10 and
        20 work well and result in unsaturated-zone mass balance errors beneath
        streams ranging between 0.001 and 0.01 percent. Please see Smith (1983)
        for further details. (default is 10; for MODFLOW-2005 simulations only
        when isfropt > 1)
    isuzn : integer
        An integer value that is the maximum number of vertical cells used to
        define the unsaturated zone beneath a stream reach. If ICALC is 1 for
        all segments then ISUZN should be set to 1. (default is 1; for
        MODFLOW-2005 simulations only when isfropt > 1)
    nsfrsets : integer
        An integer value that is the maximum number of different sets of
        trailing waves used to allocate arrays. Arrays are allocated by
        multiplying NSTRAIL by NSFRSETS. A value of 30 is sufficient for
        problems where the stream depth varies often. NSFRSETS does not affect
        model run time. (default is 30; for MODFLOW-2005 simulations only
        when isfropt > 1)
    irtflg : integer
        An integer value that indicates whether transient streamflow routing is
        active. IRTFLG must be specified if NSTRM < 0. If IRTFLG > 0,
        streamflow will be routed using the kinematic-wave equation (see USGS
        Techniques and Methods 6-D1, p. 68-69); otherwise, IRTFLG should be
        specified as 0. Transient streamflow routing is only available for
        MODFLOW-2005; IRTFLG can be left blank for MODFLOW-2000 simulations.
        (default is 1)
    numtim : integer
        An integer value equal to the number of sub time steps used to route
        streamflow. The time step that will be used to route streamflow will
        be equal to the MODFLOW time step divided by NUMTIM. (default is 2;
        for MODFLOW-2005 simulations only when irtflg > 0)
    weight : float
        A real number equal to the time weighting factor used to calculate the
        change in channel storage. WEIGHT has a value between 0.5 and 1. Please
        refer to equation 83 in USGS Techniques and Methods 6-D1 for further
        details. (default is 0.75; for MODFLOW-2005 simulations only when
        irtflg > 0)
    flwtol : float
        A real number equal to the streamflow tolerance for convergence of the
        kinematic wave equation used for transient streamflow routing. A value
        of 0.00003 cubic meters per second has been used successfully in test
        simulations (and would need to be converted to whatever units are being
        used in the particular simulation). (default is 0.0001; for
        MODFLOW-2005 simulations only when irtflg > 0)
    reach_data : recarray
        Numpy record array of length equal to nstrm, with columns for each
        variable entered in item 2 (see SFR package input instructions). In
        following flopy convention, layer, row, column and node number
        (for unstructured grids) are zero-based; segment and reach are
        one-based.
    segment_data : recarray
        Numpy record array of length equal to nss, with columns for each
        variable entered in items 6a, 6b and 6c (see SFR package input
        instructions). Segment numbers are one-based.
    dataset_5 : dict of lists
        Optional; will be built automatically from segment_data unless
        specified. Dict of lists, with key for each stress period. Each list
        contains the variables [itmp, irdflag, iptflag]. (see SFR documentation
        for more details):
    itmp : list of integers (len = NPER)
        For each stress period, an integer value for reusing or reading stream
        segment data that can change each stress period. If ITMP = 0 then all
        stream segment data are defined by Item 4 (NSFRPAR > 0; number of
        stream parameters is greater than 0). If ITMP > 0, then stream segment
        data are not defined in Item 4 and must be defined in Item 6 below for
        a number of segments equal to the value of ITMP. If ITMP < 0, then
        stream segment data not defined in Item 4 will be reused from the last
        stress period (Item 6 is not read for the current stress period). ITMP
        must be defined >= 0 for the first stress period of a simulation.
    irdflag : int or list of integers (len = NPER)
        For each stress period, an integer value for printing input data
        specified for this stress period. If IRDFLG = 0, input data for this
        stress period will be printed. If IRDFLG > 0, then input data for this
        stress period will not be printed.
    iptflag : int or list of integers (len = NPER)
        For each stress period, an integer value for printing streamflow-
        routing results during this stress period. If IPTFLG = 0, or whenever
        the variable ICBCFL or "Save Budget" is specified in Output Control,
        the results for specified time steps during this stress period will be
        printed. If IPTFLG > 0, then the results during this stress period will
        not be printed.
    extension : string
        Filename extension (default is 'sfr')
    unit_number : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output and sfr output name will be
        created using the model name and .cbc the .sfr.bin/.sfr.out extensions
        (for example, modflowtest.cbc, and modflowtest.sfr.bin), if ipakcbc and
        istcb2 are numbers greater than zero. If a single string is passed the
        package name will be set to the string and other uzf output files will
        be set to the model name with the appropriate output file extensions.
        To define the names for all package files (input and output) the
        length of the list of strings should be 3. Default is None.

    Attributes
    ----------
    outlets : nested dictionary
        Contains the outlet for each SFR segment; format is
        {per: {segment: outlet}} This attribute is created by the
        get_outlets() method.
    outsegs : dictionary of arrays
        Each array is of shape nss rows x maximum of nss columns. The first
        column contains the SFR segments, the second column contains the
        outsegs of those segments; the third column the outsegs of the outsegs,
        and so on, until all outlets have been encountered, or nss is reached.
        The latter case indicates circular routing. This attribute is created
        by the get_outlets() method.

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

    _options = OrderedDict(
        [
            ("reachinput", OptionBlock.simple_flag),
            ("transroute", OptionBlock.simple_flag),
            ("tabfiles", OptionBlock.simple_tabfile),
            (
                "lossfactor",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 1,
                    OptionBlock.vars: {"factor": OptionBlock.simple_float},
                },
            ),
            (
                "strhc1kh",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 1,
                    OptionBlock.vars: {"factorkh": OptionBlock.simple_float},
                },
            ),
            (
                "strhc1kv",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 1,
                    OptionBlock.vars: {"factorkv": OptionBlock.simple_float},
                },
            ),
        ]
    )

    nsfrpar = 0
    heading = (
        "# Streamflow-Routing (SFR2) file for MODFLOW, generated by Flopy"
    )
    default_value = 0.0
    # LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}
    len_const = {1: 1.486, 2: 1.0, 3: 100.0}
    # {"u": 0, "s": 1, "m": 2, "h": 3, "d": 4, "y": 5}
    time_const = {1: 1.0, 2: 60.0, 3: 3600.0, 4: 86400.0, 5: 31557600.0}

    def __init__(
        self,
        model,
        nstrm=-2,
        nss=1,
        nsfrpar=0,
        nparseg=0,
        const=None,
        dleak=0.0001,
        ipakcb=None,
        istcb2=None,
        isfropt=0,
        nstrail=10,
        isuzn=1,
        nsfrsets=30,
        irtflg=0,
        numtim=2,
        weight=0.75,
        flwtol=0.0001,
        reach_data=None,
        segment_data=None,
        channel_geometry_data=None,
        channel_flow_data=None,
        dataset_5=None,
        irdflag=0,
        iptflag=0,
        reachinput=False,
        transroute=False,
        tabfiles=False,
        tabfiles_dict=None,
        extension="sfr",
        unit_number=None,
        filenames=None,
        options=None,
    ):

        """
        Package constructor
        """
        # set default unit number of one is not specified
        if unit_number is None:
            unit_number = ModflowSfr2._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 3:
                for _ in range(len(filenames), 3):
                    filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                ipakcb, fname=fname, package=ModflowSfr2._ftype()
            )
        else:
            ipakcb = 0

        # add sfr flow output file
        if istcb2 is not None:
            if abs(istcb2) > 0:
                binflag = False
                ext = "out"
                if istcb2 < 0:
                    binflag = True
                    ext = "bin"
                fname = filenames[2]
                if fname is None:
                    fname = model.name + ".sfr.{}".format(ext)
                model.add_output_file(
                    abs(istcb2),
                    fname=fname,
                    binflag=binflag,
                    package=ModflowSfr2._ftype(),
                )
        else:
            istcb2 = 0

        # Fill namefile items
        name = [ModflowSfr2._ftype()]
        units = [unit_number]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.url = "sfr2.htm"
        self._graph = None  # dict of routing connections

        # Dataset 0
        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )

        # Dataset 1a and 1b
        self.reachinput = reachinput
        self.transroute = transroute
        self.tabfiles = tabfiles
        self.tabfiles_dict = tabfiles_dict
        self.numtab = 0 if not tabfiles else len(tabfiles_dict)
        self.maxval = (
            np.max([tb["numval"] for tb in tabfiles_dict.values()])
            if self.numtab > 0
            else 0
        )

        if options is None:
            if (reachinput, transroute, tabfiles) != (False, False, False):
                options = OptionBlock("", ModflowSfr2, block=False)
                if "nwt" in self.parent.version:
                    options.block = True

        self.options = options

        # Dataset 1c.
        # number of reaches, negative value is flag for unsat.
        # flow beneath streams and/or transient routing
        self._nstrm = (
            np.sign(nstrm) * len(reach_data)
            if reach_data is not None
            else nstrm
        )
        if segment_data is not None:
            # segment_data is a zero-d array
            if not isinstance(segment_data, dict):
                if len(segment_data.shape) == 0:
                    segment_data = np.atleast_1d(segment_data)
                nss = len(segment_data)
                segment_data = {0: segment_data}
            nss = len(set(reach_data["iseg"]))
        else:
            pass
        # use atleast_1d for length since segment_data might be a 0D array
        # this seems to be OK, because self.segment_data is produced by the constructor (never 0D)
        self.nsfrpar = nsfrpar
        self.nparseg = nparseg
        # conversion factor used in calculating stream depth for stream reach (icalc = 1 or 2)
        self._const = const if const is not None else None
        self.dleak = (
            dleak  # tolerance level of stream depth used in computing leakage
        )

        self.ipakcb = ipakcb
        # flag; unit number for writing table of SFR output to text file
        self.istcb2 = istcb2

        # if nstrm < 0
        # defines the format of the input data and whether or not unsaturated flow is simulated
        self.isfropt = isfropt

        # if isfropt > 1
        # number of trailing wave increments
        self.nstrail = nstrail
        # max number of vertical cells used to define unsat. zone
        self.isuzn = isuzn
        # max number trailing waves sets
        self.nsfrsets = nsfrsets

        # if nstrm < 0 (MF-2005 only)
        # switch for transient streamflow routing (> 0 = kinematic wave)
        self.irtflg = irtflg
        # if irtflg > 0
        # number of subtimesteps used for routing
        self.numtim = numtim
        # time weighting factor used to calculate the change in channel storage
        self.weight = weight
        # streamflow tolerance for convergence of the kinematic wave equation
        self.flwtol = flwtol

        # Dataset 2.
        self.reach_data = self.get_empty_reach_data(np.abs(self._nstrm))
        if reach_data is not None:
            for n in reach_data.dtype.names:
                self.reach_data[n] = reach_data[n]

        # assign node numbers if there are none (structured grid)
        if np.diff(
            self.reach_data.node
        ).max() == 0 and self.parent.has_package("DIS"):
            # first make kij list
            lrc = np.array(self.reach_data)[["k", "i", "j"]].tolist()
            self.reach_data["node"] = self.parent.dis.get_node(lrc)
        # assign unique ID and outreach columns to each reach
        self.reach_data.sort(order=["iseg", "ireach"])
        new_cols = {
            "reachID": np.arange(1, len(self.reach_data) + 1),
            "outreach": np.zeros(len(self.reach_data)),
        }
        for k, v in new_cols.items():
            if k not in self.reach_data.dtype.names:
                recfunctions.append_fields(
                    self.reach_data, names=k, data=v, asrecarray=True
                )
        # create a stress_period_data attribute to enable parent functions (e.g. plot)
        self.stress_period_data = MfList(
            self, self.reach_data, dtype=self.reach_data.dtype
        )

        # Datasets 4 and 6.

        # list of values that indicate segments outside of the model
        # (depending on how SFR package was constructed)
        self.not_a_segment_values = [999999]

        self._segments = None
        self.segment_data = {0: self.get_empty_segment_data(nss)}
        if segment_data is not None:
            for i in segment_data.keys():
                nseg = len(segment_data[i])
                self.segment_data[i] = self.get_empty_segment_data(nseg)
                for n in segment_data[i].dtype.names:
                    # inds = (segment_data[i]['nseg'] -1).astype(int)
                    self.segment_data[i][n] = segment_data[i][n]
        # compute outreaches if nseg and outseg columns have non-default values
        if (
            np.diff(self.reach_data.iseg).max() != 0
            and np.max(list(set(self.graph.keys()))) != 0
            and np.max(list(set(self.graph.values()))) != 0
        ):
            if len(self.graph) == 1:
                self.segment_data[0]["nseg"] = 1
                self.reach_data["iseg"] = 1

            consistent_seg_numbers = (
                len(
                    set(self.reach_data.iseg).difference(
                        set(self.graph.keys())
                    )
                )
                == 0
            )
            if not consistent_seg_numbers:
                warnings.warn(
                    "Inconsistent segment numbers of reach_data and segment_data"
                )

            # first convert any not_a_segment_values to 0
            for v in self.not_a_segment_values:
                self.segment_data[0].outseg[
                    self.segment_data[0].outseg == v
                ] = 0
            self.set_outreaches()
        self.channel_geometry_data = channel_geometry_data
        self.channel_flow_data = channel_flow_data

        # Dataset 5
        # set by property from segment_data unless specified manually
        self._dataset_5 = dataset_5
        self.irdflag = irdflag
        self.iptflag = iptflag

        # Attributes not included in SFR package input
        # dictionary of arrays; see Attributes section of documentation
        self.outsegs = {}
        # nested dictionary of format {per: {segment: outlet}}
        self.outlets = {}
        # input format checks:
        assert isfropt in [0, 1, 2, 3, 4, 5]

        # derived attributes
        self._paths = None

        self.parent.add_package(self)

    def __setattr__(self, key, value):
        if key == "nstrm":
            super().__setattr__("_nstrm", value)
        elif key == "dataset_5":
            super().__setattr__("_dataset_5", value)
        elif key == "segment_data":
            super().__setattr__("segment_data", value)
            self._dataset_5 = None
        elif key == "const":
            super().__setattr__("_const", value)
        else:  # return to default behavior of pakbase
            super().__setattr__(key, value)

    @property
    def const(self):
        if self._const is None:
            const = (
                self.len_const[self.parent.dis.lenuni]
                * self.time_const[self.parent.dis.itmuni]
            )
        else:
            const = self._const
        return const

    @property
    def nss(self):
        # number of stream segments
        return len(set(self.reach_data["iseg"]))

    @property
    def nstrm(self):
        return np.sign(self._nstrm) * len(self.reach_data)

    @property
    def nper(self):
        nper = self.parent.nrow_ncol_nlay_nper[-1]
        nper = (
            1 if nper == 0 else nper
        )  # otherwise iterations from 0, nper won't run
        return nper

    @property
    def dataset_5(self):
        """
        auto-update itmp so it is consistent with segment_data.
        """
        ds5 = self._dataset_5
        nss = self.nss
        if ds5 is None:
            irdflag = self._get_flag("irdflag")
            iptflag = self._get_flag("iptflag")
            ds5 = {0: [nss, irdflag[0], iptflag[0]]}
            for per in range(1, self.nper):
                sd = self.segment_data.get(per, None)
                if sd is None:
                    ds5[per] = [-nss, irdflag[per], iptflag[per]]
                else:
                    ds5[per] = [len(sd), irdflag[per], iptflag[per]]
        return ds5

    @property
    def graph(self):
        """Dictionary of routing connections between segments."""
        if self._graph is None:
            self._graph = self._make_graph()
        return self._graph

    @property
    def paths(self):
        if self._paths is None:
            self._set_paths()
            return self._paths
        # check to see if routing in segment data was changed
        nseg = np.array(sorted(self._paths.keys()), dtype=int)
        nseg = nseg[nseg > 0].copy()
        outseg = np.array([self._paths[k][1] for k in nseg])
        existing_nseg = sorted(list(self.graph.keys()))
        existing_outseg = [self.graph[k] for k in existing_nseg]
        if not np.array_equal(nseg, existing_nseg) or not np.array_equal(
            outseg, existing_outseg
        ):
            self._set_paths()
        return self._paths

    @property
    def df(self):
        if pd:
            return pd.DataFrame(self.reach_data)
        else:
            msg = "ModflowSfr2.df: pandas not available"
            raise ImportError(msg)

    def _make_graph(self):
        # get all segments and their outseg
        graph = {}
        for recarray in self.segment_data.values():
            graph.update(dict(zip(recarray["nseg"], recarray["outseg"])))

        outlets = set(graph.values()).difference(
            set(graph.keys())
        )  # including lakes
        graph.update({o: 0 for o in outlets if o != 0})
        return graph

    def _set_paths(self):
        graph = self.graph
        self._paths = {seg: find_path(graph, seg) for seg in graph.keys()}

    def _get_flag(self, flagname):
        """
        populate values for each stress period
        """
        flg = self.__dict__[flagname]
        flg = [flg] if np.isscalar(flg) else flg
        if len(flg) < self.nper:
            return flg + [flg[-1]] * (self.nper - len(flg))
        return flg

    @staticmethod
    def get_empty_reach_data(
        nreaches=0, aux_names=None, structured=True, default_value=0.0
    ):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowSfr2.get_default_reach_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = create_empty_recarray(nreaches, dtype, default_value=default_value)
        d["reachID"] = np.arange(1, nreaches + 1)
        return d

    @staticmethod
    def get_empty_segment_data(nsegments=0, aux_names=None, default_value=0.0):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowSfr2.get_default_segment_dtype()
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = create_empty_recarray(
            nsegments, dtype, default_value=default_value
        )
        return d

    @staticmethod
    def get_default_reach_dtype(structured=True):
        if structured:
            # include node column for structured grids (useful for indexing)
            return np.dtype(
                [
                    ("node", int),
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("iseg", int),
                    ("ireach", int),
                    ("rchlen", np.float32),
                    ("strtop", np.float32),
                    ("slope", np.float32),
                    ("strthick", np.float32),
                    ("strhc1", np.float32),
                    ("thts", np.float32),
                    ("thti", np.float32),
                    ("eps", np.float32),
                    ("uhc", np.float32),
                    ("reachID", int),
                    ("outreach", int),
                ]
            )
        else:
            return np.dtype(
                [
                    ("node", int),
                    ("iseg", int),
                    ("ireach", int),
                    ("rchlen", np.float32),
                    ("strtop", np.float32),
                    ("slope", np.float32),
                    ("strthick", np.float32),
                    ("strhc1", np.float32),
                    ("thts", np.float32),
                    ("thti", np.float32),
                    ("eps", np.float32),
                    ("uhc", np.float32),
                    ("reachID", int),
                    ("outreach", int),
                ]
            )

    @staticmethod
    def get_default_segment_dtype():
        return np.dtype(
            [
                ("nseg", int),
                ("icalc", int),
                ("outseg", int),
                ("iupseg", int),
                ("iprior", int),
                ("nstrpts", int),
                ("flow", np.float32),
                ("runoff", np.float32),
                ("etsw", np.float32),
                ("pptsw", np.float32),
                ("roughch", np.float32),
                ("roughbk", np.float32),
                ("cdpth", np.float32),
                ("fdpth", np.float32),
                ("awdth", np.float32),
                ("bwdth", np.float32),
                ("hcond1", np.float32),
                ("thickm1", np.float32),
                ("elevup", np.float32),
                ("width1", np.float32),
                ("depth1", np.float32),
                ("thts1", np.float32),
                ("thti1", np.float32),
                ("eps1", np.float32),
                ("uhc1", np.float32),
                ("hcond2", np.float32),
                ("thickm2", np.float32),
                ("elevdn", np.float32),
                ("width2", np.float32),
                ("depth2", np.float32),
                ("thts2", np.float32),
                ("thti2", np.float32),
                ("eps2", np.float32),
                ("uhc2", np.float32),
            ]
        )

    @classmethod
    def load(cls, f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):

        if model.verbose:
            sys.stdout.write("loading sfr2 package file...\n")

        tabfiles = False
        tabfiles_dict = {}
        transroute = False
        reachinput = False
        structured = model.structured
        if nper is None:
            nper = model.nper
            nper = (
                1 if nper == 0 else nper
            )  # otherwise iterations from 0, nper won't run

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Item 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        options = None
        if model.version == "mfnwt" and "options" in line.lower():
            options = OptionBlock.load_options(f, ModflowSfr2)

        else:
            query = (
                "reachinput",
                "transroute",
                "tabfiles",
                "lossfactor",
                "strhc1kh",
                "strhc1kv",
            )
            for i in query:
                if i in line.lower():
                    options = OptionBlock(
                        line.lower().strip(), ModflowSfr2, block=False
                    )
                    break

        if options is not None:
            line = f.readline()
            # check for 1b in modflow-2005
            if "tabfile" in line.lower():
                t = line.strip().split()
                options.tabfiles = True
                options.numtab = int(t[1])
                options.maxval = int(t[2])
                line = f.readline()

            # set varibles to be passed to class args
            transroute = options.transroute
            reachinput = options.reachinput
            tabfiles = isinstance(options.tabfiles, np.ndarray)
            numtab = options.numtab if tabfiles else 0

        # item 1c
        (
            nstrm,
            nss,
            nsfrpar,
            nparseg,
            const,
            dleak,
            ipakcb,
            istcb2,
            isfropt,
            nstrail,
            isuzn,
            nsfrsets,
            irtflg,
            numtim,
            weight,
            flwtol,
            option,
        ) = _parse_1c(line, reachinput=reachinput, transroute=transroute)

        # item 2
        # set column names, dtypes
        names = _get_item2_names(nstrm, reachinput, isfropt, structured)
        dtypes = [
            d
            for d in ModflowSfr2.get_default_reach_dtype().descr
            if d[0] in names
        ]

        lines = []
        for i in range(abs(nstrm)):
            line = f.readline()
            line = line_parse(line)
            ireach = tuple(map(float, line[: len(dtypes)]))
            lines.append(ireach)

        tmp = np.array(lines, dtype=dtypes)
        # initialize full reach_data array with all possible columns
        reach_data = ModflowSfr2.get_empty_reach_data(len(lines))
        for n in names:
            reach_data[n] = tmp[
                n
            ]  # not sure if there's a way to assign multiple columns

        # zero-based convention
        inds = ["k", "i", "j"] if structured else ["node"]
        _markitzero(reach_data, inds)

        # items 3 and 4 are skipped (parameters not supported)
        # item 5
        segment_data = {}
        channel_geometry_data = {}
        channel_flow_data = {}
        dataset_5 = {}
        aux_variables = (
            {}
        )  # not sure where the auxiliary variables are supposed to go
        for i in range(0, nper):
            # Dataset 5
            dataset_5[i] = _get_dataset(f.readline(), [-1, 0, 0, 0])
            itmp = dataset_5[i][0]
            if itmp > 0:
                # Item 6
                current = ModflowSfr2.get_empty_segment_data(
                    nsegments=itmp, aux_names=option
                )
                # container to hold any auxiliary variables
                current_aux = {}
                # these could also be implemented as structured arrays with a column for segment number
                current_6d = {}
                current_6e = {}
                # print(i,icalc,nstrm,isfropt,reachinput)
                for j in range(itmp):
                    dataset_6a = _parse_6a(f.readline(), option)
                    current_aux[j] = dataset_6a[-1]
                    dataset_6a = dataset_6a[:-1]  # drop xyz
                    icalc = dataset_6a[1]
                    # link dataset 6d, 6e by nseg of dataset_6a
                    temp_nseg = dataset_6a[0]
                    # datasets 6b and 6c aren't read under the conditions below
                    # see table under description of dataset 6c,
                    # in the MODFLOW Online Guide for a description
                    # of this logic
                    # https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/sfr.htm
                    dataset_6b, dataset_6c = (0,) * 9, (0,) * 9
                    if not (
                        isfropt in [2, 3] and icalc == 1 and i > 1
                    ) and not (isfropt in [1, 2, 3] and icalc >= 2):
                        dataset_6b = _parse_6bc(
                            f.readline(),
                            icalc,
                            nstrm,
                            isfropt,
                            reachinput,
                            per=i,
                        )
                        dataset_6c = _parse_6bc(
                            f.readline(),
                            icalc,
                            nstrm,
                            isfropt,
                            reachinput,
                            per=i,
                        )
                    current[j] = dataset_6a + dataset_6b + dataset_6c

                    if icalc == 2:
                        # ATL: not sure exactly how isfropt logic functions for this
                        # dataset 6d description suggests that this line isn't read for isfropt > 1
                        # but description of icalc suggest that icalc=2 (8-point channel) can be used with any isfropt
                        if (
                            i == 0
                            or nstrm > 0
                            and not reachinput
                            or isfropt <= 1
                        ):
                            dataset_6d = []
                            for _ in range(2):
                                dataset_6d.append(
                                    _get_dataset(f.readline(), [0.0] * 8)
                                )
                                # dataset_6d.append(list(map(float, f.readline().strip().split())))
                            current_6d[temp_nseg] = dataset_6d
                    if icalc == 4:
                        nstrpts = dataset_6a[5]
                        dataset_6e = []
                        for _ in range(3):
                            dataset_6e.append(
                                _get_dataset(f.readline(), [0.0] * nstrpts)
                            )
                        current_6e[temp_nseg] = dataset_6e

                segment_data[i] = current
                aux_variables[j + 1] = current_aux
                if len(current_6d) > 0:
                    channel_geometry_data[i] = current_6d
                if len(current_6e) > 0:
                    channel_flow_data[i] = current_6e

            if tabfiles and i == 0:
                for j in range(numtab):
                    segnum, numval, iunit = map(
                        int, f.readline().strip().split()
                    )
                    tabfiles_dict[segnum] = {"numval": numval, "inuit": iunit}

            else:
                continue

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None, None]
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowSfr2._ftype():
                    unitnumber = key
                    filenames[0] = os.path.basename(value.filename)

                if ipakcb > 0:
                    if key == ipakcb:
                        filenames[1] = os.path.basename(value.filename)
                        model.add_pop_key_list(key)

                if abs(istcb2) > 0:
                    if key == abs(istcb2):
                        filenames[2] = os.path.basename(value.filename)
                        model.add_pop_key_list(key)

        return cls(
            model,
            nstrm=nstrm,
            nss=nss,
            nsfrpar=nsfrpar,
            nparseg=nparseg,
            const=const,
            dleak=dleak,
            ipakcb=ipakcb,
            istcb2=istcb2,
            isfropt=isfropt,
            nstrail=nstrail,
            isuzn=isuzn,
            nsfrsets=nsfrsets,
            irtflg=irtflg,
            numtim=numtim,
            weight=weight,
            flwtol=flwtol,
            reach_data=reach_data,
            segment_data=segment_data,
            dataset_5=dataset_5,
            channel_geometry_data=channel_geometry_data,
            channel_flow_data=channel_flow_data,
            reachinput=reachinput,
            transroute=transroute,
            tabfiles=tabfiles,
            tabfiles_dict=tabfiles_dict,
            unit_number=unitnumber,
            filenames=filenames,
            options=options,
        )

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check sfr2 package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
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
        self._graph = None  # remake routing graph from segment data
        chk = check(self, verbose=verbose, level=level)
        chk.for_nans()
        chk.numbering()
        chk.routing()
        chk.overlapping_conductance()
        chk.elevations()
        chk.slope()

        if f is not None:
            if isinstance(f, str):
                pth = os.path.join(self.parent.model_ws, f)
                f = open(pth, "w")
            f.write("{}\n".format(chk.txt))
            # f.close()
        return chk

    def assign_layers(self, adjust_botms=False, pad=1.0):
        """
        Assigns the appropriate layer for each SFR reach,
        based on cell bottoms at location of reach.

        Parameters
        ----------
        adjust_botms : bool
            Streambed bottom elevations below the model bottom
            will cause an error in MODFLOW. If True, adjust
            bottom elevations in lowest layer of the model
            so they are at least pad distance below any co-located
            streambed elevations.
        pad : scalar
            Minimum distance below streambed bottom to set
            any conflicting model bottom elevations.

        Notes
        -----
        Streambed bottom = strtop - strthick
        This routine updates the elevations in the botm array
        of the flopy.model.ModflowDis instance. To produce a
        new DIS package file, model.write() or flopy.model.ModflowDis.write()
        must be run.

        """
        streambotms = self.reach_data.strtop - self.reach_data.strthick
        i, j = self.reach_data.i, self.reach_data.j
        layers = self.parent.dis.get_layer(i, j, streambotms)

        # check against model bottom
        logfile = "sfr_botm_conflicts.chk"
        mbotms = self.parent.dis.botm.array[-1, i, j]
        below = streambotms <= mbotms
        below_i = self.reach_data.i[below]
        below_j = self.reach_data.j[below]
        l = []
        header = ""
        if np.any(below):
            print(
                "Warning: SFR streambed elevations below model bottom. "
                "See sfr_botm_conflicts.chk"
            )
            if not adjust_botms:
                l += [below_i, below_j, mbotms[below], streambotms[below]]
                header += "i,j,model_botm,streambed_botm"
            else:
                print("Fixing elevation conflicts...")
                botm = self.parent.dis.botm.array.copy()
                for ib, jb in zip(below_i, below_j):
                    inds = (self.reach_data.i == ib) & (
                        self.reach_data.j == jb
                    )
                    botm[-1, ib, jb] = streambotms[inds].min() - pad
                    # l.append(botm[-1, ib, jb])
                # botm[-1, below_i, below_j] = streambotms[below] - pad
                l.append(botm[-1, below_i, below_j])
                header += ",new_model_botm"
                self.parent.dis.botm = botm
                mbotms = self.parent.dis.botm.array[-1, i, j]
                assert not np.any(streambotms <= mbotms)
                print(
                    "New bottom array assigned to Flopy DIS package "
                    "instance.\nRun flopy.model.write() or "
                    "flopy.model.ModflowDis.write() to write new DIS file."
                )
            header += "\n"

            with open(logfile, "w") as log:
                log.write(header)
                a = np.array(l).transpose()
                for line in a:
                    log.write(",".join(map(str, line)) + "\n")
        self.reach_data["k"] = layers

    def deactivate_ibound_above(self):
        """
        Sets ibound to 0 for all cells above active SFR cells.

        Parameters
        ----------
        none

        Notes
        -----
        This routine updates the ibound array of the flopy.model.ModflowBas6
        instance. To produce a new BAS6 package file, model.write() or
        flopy.model.ModflowBas6.write() must be run.

        """
        ib = self.parent.bas6.ibound.array
        deact_lays = [list(range(i)) for i in self.reach_data.k]
        for ks, i, j in zip(deact_lays, self.reach_data.i, self.reach_data.j):
            for k in ks:
                ib[k, i, j] = 0
        self.parent.bas6.ibound = ib

    def get_outlets(self, level=0, verbose=True):
        """
        Traces all routing connections from each headwater to the outlet.
        """
        txt = ""
        for per in range(self.nper):
            if (
                per > 0 > self.dataset_5[per][0]
            ):  # skip stress periods where seg data not defined
                continue
            # segments = self.segment_data[per].nseg
            # outsegs = self.segment_data[per].outseg
            #
            # all_outsegs = np.vstack([segments, outsegs])
            # max_outseg = all_outsegs[-1].max()
            # knt = 1
            # while max_outseg > 0:
            #
            #     nextlevel = np.array([outsegs[s - 1] if s > 0 and s < 999999 else 0
            #                           for s in all_outsegs[-1]])
            #
            #     all_outsegs = np.vstack([all_outsegs, nextlevel])
            #     max_outseg = nextlevel.max()
            #     if max_outseg == 0:
            #         break
            #     knt += 1
            #     if knt > self.nss:
            #         # subset outsegs map to only include rows with outseg number > 0 in last column
            #         circular_segs = all_outsegs.T[all_outsegs[-1] > 0]
            #
            #         # only retain one instance of each outseg number at iteration=nss
            #         vals = []  # append outseg values to vals after they've appeared once
            #         mask = [(True, vals.append(v))[0]
            #                 if v not in vals
            #                 else False for v in circular_segs[-1]]
            #         circular_segs = circular_segs[:, np.array(mask)]
            #
            #         # cull the circular segments array to remove duplicate instances of routing circles
            #         circles = []
            #         duplicates = []
            #         for i in range(np.shape(circular_segs)[0]):
            #             # find where values in the row equal the last value;
            #             # record the index of the second to last instance of last value
            #             repeat_start_ind = np.where(circular_segs[i] == circular_segs[i, -1])[0][-2:][0]
            #             # use that index to slice out the repeated segment sequence
            #             circular_seq = circular_segs[i, repeat_start_ind:].tolist()
            #             # keep track of unique sequences of repeated segments
            #             if set(circular_seq) not in circles:
            #                 circles.append(set(circular_seq))
            #                 duplicates.append(False)
            #             else:
            #                 duplicates.append(True)
            #         circular_segs = circular_segs[~np.array(duplicates), :]
            #
            #         txt += '{0} instances where an outlet was not found after {1} consecutive segments!\n' \
            #             .format(len(circular_segs), self.nss)
            #         if level == 1:
            #             txt += '\n'.join([' '.join(map(str, row)) for row in circular_segs]) + '\n'
            #         else:
            #             f = 'circular_routing.csv'
            #             np.savetxt(f, circular_segs, fmt='%d', delimiter=',', header=txt)
            #             txt += 'See {} for details.'.format(f)
            #         if verbose:
            #             print(txt)
            #         break
            # # the array of segment sequence is useful for other other operations,
            # # such as plotting elevation profiles
            # self.outsegs[per] = all_outsegs
            #
            # use graph instead of above loop
            nrow = len(self.segment_data[per].nseg)
            ncol = np.max(
                [len(v) if v is not None else 0 for v in self.paths.values()]
            )
            all_outsegs = np.zeros((nrow, ncol), dtype=int)
            for i, (k, v) in enumerate(self.paths.items()):
                if k > 0:
                    all_outsegs[i, : len(v)] = v
            all_outsegs.sort(axis=0)
            self.outsegs[per] = all_outsegs
            # create a dictionary listing outlets associated with each segment
            # outlet is the last value in each row of outseg array that is != 0 or 999999
            # self.outlets[per] = {i + 1: r[(r != 0) & (r != 999999)][-1]
            # if len(r[(r != 0) & (r != 999999)]) > 0
            # else i + 1
            #                     for i, r in enumerate(all_outsegs.T)}
            self.outlets[per] = {
                k: self.paths[k][-1] if k in self.paths else k
                for k in self.segment_data[per].nseg
            }
        return txt

    def reset_reaches(self):
        self.reach_data.sort(order=["iseg", "ireach"])
        reach_data = self.reach_data
        segment_data = list(set(self.reach_data.iseg))  # self.segment_data[0]
        reach_counts = np.bincount(reach_data.iseg)[1:]
        reach_counts = dict(zip(range(1, len(reach_counts) + 1), reach_counts))
        ireach = [list(range(1, reach_counts[s] + 1)) for s in segment_data]
        ireach = np.concatenate(ireach)
        self.reach_data["ireach"] = ireach

    def set_outreaches(self):
        """
        Determine the outreach for each SFR reach (requires a reachID
        column in reach_data). Uses the segment routing specified for the
        first stress period to route reaches between segments.
        """
        self.reach_data.sort(order=["iseg", "ireach"])
        # ensure that each segment starts with reach 1
        self.reset_reaches()
        # ensure that all outsegs are segments, outlets, or negative (lakes)
        self.repair_outsegs()
        rd = self.reach_data
        outseg = self.graph
        reach1IDs = dict(
            zip(rd[rd.ireach == 1].iseg, rd[rd.ireach == 1].reachID)
        )
        outreach = []
        for i in range(len(rd)):
            # if at the end of reach data or current segment
            if i + 1 == len(rd) or rd.ireach[i + 1] == 1:
                nextseg = outseg[rd.iseg[i]]  # get next segment
                if nextseg > 0:  # current reach is not an outlet
                    nextrchid = reach1IDs[
                        nextseg
                    ]  # get reach 1 of next segment
                else:
                    nextrchid = 0
            else:  # otherwise, it's the next reachID
                nextrchid = rd.reachID[i + 1]
            outreach.append(nextrchid)
        self.reach_data["outreach"] = outreach

    def get_slopes(
        self, default_slope=0.001, minimum_slope=0.0001, maximum_slope=1.0
    ):
        """
        Compute slopes by reach using values in strtop (streambed top)
        and rchlen (reach length) columns of reach_data. The slope for a
        reach n is computed as strtop(n+1) - strtop(n) / rchlen(n).
        Slopes for outlet reaches are set equal to a default value
        (default_slope). Populates the slope column in reach_data.

        Parameters
        ----------
        default_slope : float
            Slope value applied to outlet reaches
            (where water leaves the model). Default value is 0.001
        minimum_slope : float
            Assigned to reaches with computed slopes less than this value.
            This ensures that the Manning's equation won't produce unreasonable
            values of stage (in other words, that stage is consistent with
            assumption that streamflow is primarily drive by the streambed
            gradient). Default value is 0.0001.
        maximum_slope : float
            Assigned to reaches with computed slopes more than this value.
            Default value is 1.

        """
        # compute outreaches if they aren't there already
        if np.diff(self.reach_data.outreach).max() == 0:
            self.set_outreaches()
        rd = self.reach_data
        elev = dict(zip(rd.reachID, rd.strtop))
        dist = dict(zip(rd.reachID, rd.rchlen))
        dnelev = {
            rid: elev[rd.outreach[i]] if rd.outreach[i] != 0 else -9999
            for i, rid in enumerate(rd.reachID)
        }
        slopes = np.array(
            [
                (elev[i] - dnelev[i]) / dist[i]
                if dnelev[i] != -9999
                else default_slope
                for i in rd.reachID
            ]
        )
        slopes[slopes < minimum_slope] = minimum_slope
        slopes[slopes > maximum_slope] = maximum_slope
        self.reach_data["slope"] = slopes

    def get_upsegs(self):
        """
        From segment_data, returns nested dict of all upstream segments by
        segment, by stress period.

        Returns
        -------
        all_upsegs : dict
            Nested dictionary of form
            {stress period: {segment: [list of upsegs]}}

        Notes
        -----
        This method will not work if there are instances of circular routing.

        """
        all_upsegs = {}
        for per in range(self.nper):
            if (
                per > 0 > self.dataset_5[per][0]
            ):  # skip stress periods where seg data not defined
                continue
            segment_data = self.segment_data[per]

            # make a list of adjacent upsegments keyed to outseg list in Mat2
            upsegs = {
                o: segment_data.nseg[segment_data.outseg == o].tolist()
                for o in np.unique(segment_data.outseg)
            }

            outsegs = [
                k for k in list(upsegs.keys()) if k > 0
            ]  # exclude 0, which is the outlet designator

            # for each outseg key, for each upseg, check for more upsegs,
            # append until headwaters has been reached
            for outseg in outsegs:

                up = True
                upsegslist = upsegs[outseg]
                while up:
                    added_upsegs = []
                    for us in upsegslist:
                        if us in outsegs:
                            added_upsegs += upsegs[us]
                    if len(added_upsegs) == 0:
                        up = False
                        break
                    else:
                        upsegslist = added_upsegs
                        upsegs[outseg] += added_upsegs

            # the above algorithm is recursive, so lower order streams
            # get duplicated many times use a set to get unique upsegs
            all_upsegs[per] = {u: list(set(upsegs[u])) for u in outsegs}
        return all_upsegs

    def get_variable_by_stress_period(self, varname):

        dtype = []
        all_data = np.zeros((self.nss, self.nper), dtype=float)
        for per in range(self.nper):
            inds = self.segment_data[per].nseg - 1
            all_data[inds, per] = self.segment_data[per][varname]
            dtype.append(("{}{}".format(varname, per), float))
        isvar = all_data.sum(axis=1) != 0
        ra = np.core.records.fromarrays(
            all_data[isvar].transpose().copy(), dtype=dtype
        )
        segs = self.segment_data[0].nseg[isvar]
        isseg = np.array(
            [True if s in segs else False for s in self.reach_data.iseg]
        )
        isinlet = isseg & (self.reach_data.ireach == 1)
        rd = np.array(self.reach_data[isinlet])[
            ["k", "i", "j", "iseg", "ireach"]
        ]
        ra = recfunctions.merge_arrays([rd, ra], flatten=True, usemask=False)
        return ra.view(np.recarray)

    def repair_outsegs(self):
        isasegment = np.in1d(
            self.segment_data[0].outseg, self.segment_data[0].nseg
        )
        isasegment = isasegment | (self.segment_data[0].outseg < 0)
        self.segment_data[0]["outseg"][~isasegment] = 0.0
        self._graph = None

    def renumber_segments(self):
        """
        Renumber segments so that segment numbering is continuous and always
        increases in the downstream direction. This may speed convergence of
        the NWT solver in some situations.

        Returns
        -------
        r : dictionary mapping old segment numbers to new
        """

        nseg = sorted(list(self.graph.keys()))
        outseg = [self.graph[k] for k in nseg]

        # explicitly fix any gaps in the numbering
        # (i.e. from removing segments)
        nseg2 = np.arange(1, len(nseg) + 1)
        # intermediate mapping that
        r1 = dict(zip(nseg, nseg2))
        r1[0] = 0
        outseg2 = np.array([r1[s] for s in outseg])

        # function re-assigning upseg numbers consecutively at one level
        # relative to outlet(s).  Counts down from the number of segments
        def reassign_upsegs(r, nexts, upsegs):
            nextupsegs = []
            for u in upsegs:
                r[u] = nexts if u > 0 else u  # handle lakes
                nexts -= 1
                nextupsegs += list(nseg2[outseg2 == u])
            return r, nexts, nextupsegs

        ns = len(nseg)

        # start at outlets with nss;
        # renumber upsegs consecutively at each level
        # until all headwaters have been reached
        nexts = ns
        r2 = {0: 0}
        nextupsegs = nseg2[outseg2 == 0]
        for _ in range(ns):
            r2, nexts, nextupsegs = reassign_upsegs(r2, nexts, nextupsegs)
            if len(nextupsegs) == 0:
                break
        # map original segment numbers to new numbers
        r = {k: r2.get(v, v) for k, v in r1.items()}

        # renumber segments in all stress period data
        for per in self.segment_data.keys():
            self.segment_data[per]["nseg"] = [
                r.get(s, s) for s in self.segment_data[per].nseg
            ]
            self.segment_data[per]["outseg"] = [
                r.get(s, s) for s in self.segment_data[per].outseg
            ]
            self.segment_data[per].sort(order="nseg")
            nseg = self.segment_data[per].nseg
            outseg = self.segment_data[per].outseg
            inds = (outseg > 0) & (nseg > outseg)
            assert not np.any(inds)
            assert (
                len(self.segment_data[per]["nseg"])
                == self.segment_data[per]["nseg"].max()
            )
        self._graph = None  # reset routing dict

        # renumber segments in reach_data
        self.reach_data["iseg"] = [r.get(s, s) for s in self.reach_data.iseg]
        self.reach_data.sort(order=["iseg", "ireach"])
        self.reach_data["reachID"] = np.arange(1, len(self.reach_data) + 1)
        self.set_outreaches()  # reset the outreaches to ensure continuity

        # renumber segments in other datasets
        def renumber_channel_data(d):
            if d is not None:
                d2 = {}
                for k, v in d.items():
                    d2[k] = {}
                    for s, vv in v.items():
                        d2[k][r[s]] = vv
            else:
                d2 = None
            return d2

        self.channel_geometry_data = renumber_channel_data(
            self.channel_geometry_data
        )
        self.channel_flow_data = renumber_channel_data(self.channel_flow_data)
        return r

    def plot_path(self, start_seg=None, end_seg=0, plot_segment_lines=True):
        """
        Plot a profile of streambed elevation and model top
        along a path of segments.

        Parameters
        ----------
        start_seg : int
            Number of first segment in path.
        end_seg : int
            Number of last segment in path (defaults to 0/outlet).
        plot_segment_lines : bool
            Controls plotting of segment end locations along profile.
            (default True)

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot object
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use ModflowSfr2.plot_path()"
            )
        if not pd:
            err_msg = "ModflowSfr2.plot_path: pandas not available"
            raise ImportError(err_msg)

        df = self.df
        m = self.parent
        mfunits = m.modelgrid.units

        to_miles = {"feet": 1 / 5280.0, "meters": 1 / (0.3048 * 5280.0)}

        # slice the path
        path = np.array(self.paths[start_seg])
        endidx = np.where(path == end_seg)[0]
        endidx = endidx if len(endidx) > 0 else None
        path = path[: np.squeeze(endidx)]
        path = [s for s in path if s > 0]  # skip lakes for now

        # get the values
        groups = df.groupby("iseg")
        tmp = pd.concat([groups.get_group(s) for s in path])
        tops = m.dis.top.array[tmp.i, tmp.j]
        dist = np.cumsum(tmp.rchlen.values) * to_miles.get(mfunits, 1.0)

        # segment starts
        starts = dist[np.where(tmp.ireach.values == 1)[0]]

        ax = plt.subplots(figsize=(11, 8.5))[-1]
        ax.plot(dist, tops, label="Model top")
        ax.plot(dist, tmp.strtop, label="Streambed top")
        ax.set_xlabel("Distance along path, in miles")
        ax.set_ylabel("Elevation, in {}".format(mfunits))
        ymin, ymax = ax.get_ylim()
        plt.autoscale(False)

        if plot_segment_lines:  # plot segment ends as vertical lines
            ax.vlines(
                x=starts,
                ymin=ymin,
                ymax=ymax,
                lw=0.1,
                alpha=0.1,
                label="Gray lines indicate\nsegment ends.",
            )
        ax.legend()

        # plot selected segment numbers along path
        stride = np.floor(len(dist) / 10)
        stride = 1 if stride < 1 else stride
        inds = np.arange(0, len(dist), stride, dtype=int)
        plot_segnumbers = tmp.iseg.values[inds]
        xlocs = dist[inds]
        pad = 0.04 * (ymax - ymin)
        for x, sn in zip(xlocs, plot_segnumbers):
            ax.text(x, ymin + pad, str(sn), va="top")
        ax.text(
            xlocs[0],
            ymin + pad * 1.2,
            "Segment numbers:",
            va="bottom",
            fontweight="bold",
        )
        ax.text(dist[-1], ymin + pad, str(end_seg), ha="center", va="top")
        return ax

    def _get_headwaters(self, per=0):
        """
        List all segments that are not outsegs (that do not have any
        segments upstream).

        Parameters
        ----------
        per : int
            Stress period for which to list headwater segments (default 0)

        Returns
        -------
        headwaters : np.ndarray (1-D)
            One dimensional array listing all headwater segments.
        """
        upsegs = [
            self.segment_data[per]
            .nseg[self.segment_data[per].outseg == s]
            .tolist()
            for s in self.segment_data[0].nseg
        ]
        return self.segment_data[per].nseg[
            np.array([i for i, u in enumerate(upsegs) if len(u) == 0])
        ]

    def _interpolate_to_reaches(self, segvar1, segvar2, per=0):
        """
        Interpolate values in datasets 6b and 6c to each reach in
        stream segment

        Parameters
        ----------
        segvar1 : str
            Column/variable name in segment_data array for representing start
            of segment (e.g. hcond1 for hydraulic conductivity)
            For segments with icalc=2 (specified channel geometry); if width1
            is given, the eighth distance point (XCPT8) from dataset 6d will
            be used as the stream width.
            For icalc=3, an arbitrary width of 5 is assigned.
            For icalc=4, the mean value for width given in item 6e is used.
        segvar2 : str
            Column/variable name in segment_data array for representing start
            of segment (e.g. hcond2 for hydraulic conductivity)
        per : int
            Stress period with segment data to interpolate

        Returns
        -------
        reach_values : 1D array
            One dimensional array of interpolated values of same length as
            reach_data array. For example, hcond1 and hcond2 could be entered
            as inputs to get values for the strhc1 (hydraulic conductivity)
            column in reach_data.

        """
        reach_data = self.reach_data
        segment_data = self.segment_data[per]
        segment_data.sort(order="nseg")
        reach_data.sort(order=["iseg", "ireach"])
        reach_values = []
        for seg in segment_data.nseg:
            reaches = reach_data[reach_data.iseg == seg]
            dist = np.cumsum(reaches.rchlen) - 0.5 * reaches.rchlen
            icalc = segment_data.icalc[segment_data.nseg == seg]
            # get width from channel cross section length
            if "width" in segvar1 and icalc == 2:
                channel_geometry_data = self.channel_geometry_data[per]
                reach_values += list(
                    np.ones(len(reaches)) * channel_geometry_data[seg][0][-1]
                )
            # assign arbitrary width since width is based on flow
            elif "width" in segvar1 and icalc == 3:
                reach_values += list(np.ones(len(reaches)) * 5)
            # assume width to be mean from streamflow width/flow table
            elif "width" in segvar1 and icalc == 4:
                channel_flow_data = self.channel_flow_data[per]
                reach_values += list(
                    np.ones(len(reaches)) * np.mean(channel_flow_data[seg][2])
                )
            else:
                fp = [
                    segment_data[segment_data["nseg"] == seg][segvar1][0],
                    segment_data[segment_data["nseg"] == seg][segvar2][0],
                ]
                xp = [dist[0], dist[-1]]
                reach_values += np.interp(dist, xp, fp).tolist()
        return np.array(reach_values)

    def _write_1c(self, f_sfr):

        # NSTRM NSS NSFRPAR NPARSEG CONST DLEAK ipakcb  ISTCB2
        # [ISFROPT] [NSTRAIL] [ISUZN] [NSFRSETS] [IRTFLG] [NUMTIM] [WEIGHT] [FLWTOL]
        f_sfr.write(
            "{:.0f} {:.0f} {:.0f} {:.0f} {:.8f} {:.8f} {:.0f} {:.0f} ".format(
                self.nstrm,
                self.nss,
                self.nsfrpar,
                self.nparseg,
                self.const,
                self.dleak,
                self.ipakcb,
                self.istcb2,
            )
        )
        if self.reachinput:
            self.nstrm = abs(
                self.nstrm
            )  # see explanation for dataset 1c in online guide
            f_sfr.write("{:.0f} ".format(self.isfropt))
            if self.isfropt > 1:
                f_sfr.write(
                    "{:.0f} {:.0f} {:.0f} ".format(
                        self.nstrail, self.isuzn, self.nsfrsets
                    )
                )
        if self.nstrm < 0:
            f_sfr.write("{:.0f} ".format(self.isfropt))
            if self.isfropt > 1:
                f_sfr.write(
                    "{:.0f} {:.0f} {:.0f} ".format(
                        self.nstrail, self.isuzn, self.nsfrsets
                    )
                )
        if self.nstrm < 0 or self.transroute:
            f_sfr.write("{:.0f} ".format(self.irtflg))
            if self.irtflg > 0:
                f_sfr.write(
                    "{:.0f} {:.8f} {:.8f} ".format(
                        self.numtim, self.weight, self.flwtol
                    )
                )
        f_sfr.write("\n")

    def _write_reach_data(self, f_sfr):

        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(
            self.reach_data, np.recarray
        ), "MfList.__tofile() data arg not a recarray"

        # decide which columns to write
        # columns = self._get_item2_names()
        columns = _get_item2_names(
            self.nstrm,
            self.reachinput,
            self.isfropt,
            structured=self.parent.structured,
        )

        # Add one to the kij indices
        # names = self.reach_data.dtype.names
        # lnames = []
        # [lnames.append(name.lower()) for name in names]
        # --make copy of data for multiple calls
        d = np.array(self.reach_data)
        for idx in ["k", "i", "j", "node"]:
            if idx in columns:
                d[idx] += 1
        d = d[columns]  # data columns sorted
        formats = _fmt_string(d) + "\n"
        for rec in d:
            f_sfr.write(formats.format(*rec))

    def _write_segment_data(self, i, j, f_sfr):
        cols = [
            "nseg",
            "icalc",
            "outseg",
            "iupseg",
            "iprior",
            "nstrpts",
            "flow",
            "runoff",
            "etsw",
            "pptsw",
            "roughch",
            "roughbk",
            "cdpth",
            "fdpth",
            "awdth",
            "bwdth",
        ]
        seg_dat = np.array(self.segment_data[i])[cols][j]
        fmts = _fmt_string_list(seg_dat)

        (
            nseg,
            icalc,
            outseg,
            iupseg,
            iprior,
            nstrpts,
            flow,
            runoff,
            etsw,
            pptsw,
            roughch,
            roughbk,
            cdpth,
            fdpth,
            awdth,
            bwdth,
        ) = [0 if v == self.default_value else v for v in seg_dat]

        f_sfr.write(
            " ".join(fmts[0:4]).format(nseg, icalc, outseg, iupseg) + " "
        )

        if iupseg > 0:
            f_sfr.write(fmts[4].format(iprior) + " ")
        if icalc == 4:
            f_sfr.write(fmts[5].format(nstrpts) + " ")

        f_sfr.write(
            " ".join(fmts[6:10]).format(flow, runoff, etsw, pptsw) + " "
        )

        if icalc in [1, 2]:
            f_sfr.write(fmts[10].format(roughch) + " ")
        if icalc == 2:
            f_sfr.write(fmts[11].format(roughbk) + " ")

        if icalc == 3:
            f_sfr.write(
                " ".join(fmts[12:16]).format(cdpth, fdpth, awdth, bwdth) + " "
            )
        f_sfr.write("\n")

        self._write_6bc(
            i,
            j,
            f_sfr,
            cols=[
                "hcond1",
                "thickm1",
                "elevup",
                "width1",
                "depth1",
                "thts1",
                "thti1",
                "eps1",
                "uhc1",
            ],
        )
        self._write_6bc(
            i,
            j,
            f_sfr,
            cols=[
                "hcond2",
                "thickm2",
                "elevdn",
                "width2",
                "depth2",
                "thts2",
                "thti2",
                "eps2",
                "uhc2",
            ],
        )

    def _write_6bc(self, i, j, f_sfr, cols=()):
        cols = list(cols)
        icalc = self.segment_data[i][j][1]
        seg_dat = np.array(self.segment_data[i])[cols][j]
        fmts = _fmt_string_list(seg_dat)
        hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc = [
            0 if v == self.default_value else v for v in seg_dat
        ]

        if self.isfropt in [0, 4, 5] and icalc <= 0:
            f_sfr.write(
                " ".join(fmts[0:5]).format(
                    hcond, thickm, elevupdn, width, depth
                )
                + " "
            )

        elif self.isfropt in [0, 4, 5] and icalc == 1:
            f_sfr.write(fmts[0].format(hcond) + " ")

            if i == 0:
                f_sfr.write(
                    " ".join(fmts[1:4]).format(thickm, elevupdn, width) + " "
                )
                if self.isfropt in [4, 5]:
                    f_sfr.write(
                        " ".join(fmts[5:8]).format(thts, thti, eps) + " "
                    )

                if self.isfropt == 5:
                    f_sfr.write(fmts[8].format(uhc) + " ")

            elif i > 0 and self.isfropt == 0:
                f_sfr.write(
                    " ".join(fmts[1:4]).format(thickm, elevupdn, width) + " "
                )

        elif self.isfropt in [0, 4, 5] and icalc >= 2:
            f_sfr.write(fmts[0].format(hcond) + " ")

            if self.isfropt in [4, 5] and i > 0 and icalc == 2:
                pass
            else:
                f_sfr.write(" ".join(fmts[1:3]).format(thickm, elevupdn) + " ")

                if self.isfropt in [4, 5] and icalc == 2 and i == 0:
                    f_sfr.write(
                        " ".join(fmts[3:6]).format(thts, thti, eps) + " "
                    )

                    if self.isfropt == 5:
                        f_sfr.write(fmts[8].format(uhc) + " ")
                else:
                    pass
        elif self.isfropt == 1 and icalc <= 1:
            f_sfr.write(fmts[3].format(width) + " ")
            if icalc <= 0:
                f_sfr.write(fmts[4].format(depth) + " ")
        elif self.isfropt in [2, 3]:
            if icalc <= 0:
                f_sfr.write(fmts[3].format(width) + " ")
                f_sfr.write(fmts[4].format(depth) + " ")
            elif icalc == 1:
                if i > 0:
                    return
                else:
                    f_sfr.write(fmts[3].format(width) + " ")
            else:
                return

        else:
            return
        f_sfr.write("\n")

    def write_file(self, filename=None):
        """
        Write the package file.

        Returns
        -------
        None

        """

        # tabfiles = False
        # tabfiles_dict = {}
        # transroute = False
        # reachinput = False
        if filename is not None:
            self.fn_path = filename

        f_sfr = open(self.fn_path, "w")

        # Item 0 -- header
        f_sfr.write("{0}\n".format(self.heading))

        # Item 1
        if (
            isinstance(self.options, OptionBlock)
            and self.parent.version == "mfnwt"
        ):
            self.options.update_from_package(self)
            self.options.write_options(f_sfr)
        elif isinstance(self.options, OptionBlock):
            self.options.update_from_package(self)
            self.options.block = False
            self.options.write_options(f_sfr)
        else:
            pass

        self._write_1c(f_sfr)

        # item 2
        self._write_reach_data(f_sfr)

        # items 3 and 4 are skipped (parameters not supported)

        for i in range(0, self.nper):

            # item 5
            itmp = self.dataset_5[i][0]
            f_sfr.write(" ".join(map(str, self.dataset_5[i])) + "\n")
            if itmp > 0:

                # Item 6
                for j in range(itmp):

                    # write datasets 6a, 6b and 6c
                    self._write_segment_data(i, j, f_sfr)

                    icalc = self.segment_data[i].icalc[j]
                    nseg = self.segment_data[i].nseg[j]
                    if icalc == 2:
                        # or isfropt <= 1:
                        if (
                            i == 0
                            or self.nstrm > 0
                            and not self.reachinput
                            or self.isfropt <= 1
                        ):
                            for k in range(2):
                                for d in self.channel_geometry_data[i][nseg][
                                    k
                                ]:
                                    f_sfr.write("{:.2f} ".format(d))
                                f_sfr.write("\n")

                    if icalc == 4:
                        # nstrpts = self.segment_data[i][j][5]
                        for k in range(3):
                            for d in self.channel_flow_data[i][nseg][k]:
                                f_sfr.write("{:.2f} ".format(d))
                            f_sfr.write("\n")
            if self.tabfiles and i == 0:
                for j in sorted(self.tabfiles_dict.keys()):
                    f_sfr.write(
                        "{:.0f} {:.0f} {:.0f}\n".format(
                            j,
                            self.tabfiles_dict[j]["numval"],
                            self.tabfiles_dict[j]["inuit"],
                        )
                    )
            else:
                continue
        f_sfr.close()

    def export(self, f, **kwargs):
        if isinstance(f, str) and f.lower().endswith(".shp"):
            from flopy.utils.geometry import Polygon
            from flopy.export.shapefile_utils import recarray2shp

            geoms = []
            for ix, i in enumerate(self.reach_data.i):
                verts = self.parent.modelgrid.get_cell_vertices(
                    i, self.reach_data.j[ix]
                )
                geoms.append(Polygon(verts))
            recarray2shp(self.reach_data, geoms, shpname=f, **kwargs)
        else:
            from flopy import export

            return export.utils.package_export(f, self, **kwargs)

    def export_linkages(self, f, **kwargs):
        """
        Export linework shapefile showing all routing connections between
        SFR reaches. A length field containing the distance between connected
        reaches can be used to filter for the longest connections in a GIS.

        """
        from flopy.utils.geometry import LineString
        from flopy.export.shapefile_utils import recarray2shp

        rd = self.reach_data.copy()
        m = self.parent
        rd.sort(order=["reachID"])

        # get the cell centers for each reach
        mg = m.modelgrid
        x0 = mg.xcellcenters[rd.i, rd.j]
        y0 = mg.ycellcenters[rd.i, rd.j]
        loc = dict(zip(rd.reachID, zip(x0, y0)))

        # make lines of the reach connections between cell centers
        geoms = []
        lengths = []
        for r in rd.reachID:
            x0, y0 = loc[r]
            outreach = rd.outreach[r - 1]
            if outreach == 0:
                x1, y1 = x0, y0
            else:
                x1, y1 = loc[outreach]
            geoms.append(LineString([(x0, y0), (x1, y1)]))
            lengths.append(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        lengths = np.array(lengths)

        # append connection lengths for filtering in GIS
        rd = recfunctions.append_fields(
            rd,
            names=["length"],
            data=[lengths],
            usemask=False,
            asrecarray=True,
        )
        recarray2shp(rd, geoms, f, **kwargs)

    def export_outlets(self, f, **kwargs):
        """
        Export point shapefile showing locations where streamflow is leaving
        the model (outset=0).

        """
        from flopy.utils.geometry import Point
        from flopy.export.shapefile_utils import recarray2shp

        rd = self.reach_data
        if np.min(rd.outreach) == np.max(rd.outreach):
            self.set_outreaches()
        rd = self.reach_data[self.reach_data.outreach == 0].copy()
        m = self.parent
        rd.sort(order=["iseg", "ireach"])

        # get the cell centers for each reach
        mg = m.modelgrid
        x0 = mg.xcellcenters[rd.i, rd.j]
        y0 = mg.ycellcenters[rd.i, rd.j]
        geoms = [Point(x, y) for x, y in zip(x0, y0)]
        recarray2shp(rd, geoms, f, **kwargs)

    def export_transient_variable(self, f, varname, **kwargs):
        """
        Export point shapefile showing locations with a given segment_data
        variable applied. For example, segments where streamflow is entering
        or leaving the upstream end of a stream segment (FLOW) or where RUNOFF
        is applied. Cell centroids of the first reach of segments with non-zero
        terms of varname are exported; values of varname are exported by stress
        period in the attribute fields (e.g. flow0, flow1, flow2... for FLOW
        in stress periods 0, 1, 2...

        Parameters
        ----------
        f : str, filename
        varname : str
            Variable in SFR Package dataset 6a (see SFR package documentation)

        """
        from flopy.utils.geometry import Point
        from flopy.export.shapefile_utils import recarray2shp

        rd = self.reach_data
        if np.min(rd.outreach) == np.max(rd.outreach):
            self.set_outreaches()
        ra = self.get_variable_by_stress_period(varname.lower())

        # get the cell centers for each reach
        m = self.parent
        mg = m.modelgrid
        x0 = mg.xcellcenters[ra.i, ra.j]
        y0 = mg.ycellcenters[ra.i, ra.j]
        geoms = [Point(x, y) for x, y in zip(x0, y0)]
        recarray2shp(ra, geoms, f, **kwargs)

    @staticmethod
    def _ftype():
        return "SFR"

    @staticmethod
    def _defaultunit():
        return 17


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
        self.sfr = copy.copy(sfrpackage)
        self.mg = self.sfr.parent.modelgrid
        self.reach_data = sfrpackage.reach_data
        self.segment_data = sfrpackage.segment_data
        self.verbose = verbose
        self.level = level
        self.passed = []
        self.warnings = []
        self.errors = []
        self.txt = "\n{} ERRORS:\n".format(self.sfr.name[0])
        self.summary_array = None

    def _boolean_compare(
        self,
        array,
        col1,
        col2,
        level0txt="{} violations encountered.",
        level1txt="Violations:",
        sort_ascending=True,
        print_delimiter=" ",
    ):
        """
        Compare two columns in a record array. For each row,
        tests if value in col1 is greater than col2. If any values
        in col1 are > col2, subsets array to only include rows where
        col1 is greater. Creates another column with differences
        (col1-col2), and prints the array sorted by the differences
        column (diff).

        Parameters
        ----------
        array : record array
            Array with columns to compare.
        col1 : string
            Column name in array.
        col2 : string
            Column name in array.
        sort_ascending : T/F; default True
            If True, printed array will be sorted by differences in
            ascending order.
        print_delimiter : str
            Delimiter for printed array.

        Returns
        -------
        txt : str
            Error messages and printed array (if .level attribute of
            checker is set to 1). Returns an empty string if no
            values in col1 are greater than col2.

        Notes
        -----
        info about appending to record arrays (views vs. copies and upcoming
        changes to numpy):
        http://stackoverflow.com/questions/22865877/how-do-i-write-to-multiple-fields-of-a-structured-array
        """
        txt = ""
        array = array.view(np.recarray).copy()
        if isinstance(col1, np.ndarray):
            array = recfunctions.append_fields(
                array, names="tmp1", data=col1, asrecarray=True
            )
            col1 = "tmp1"
        if isinstance(col2, np.ndarray):
            array = recfunctions.append_fields(
                array, names="tmp2", data=col2, asrecarray=True
            )
            col2 = "tmp2"
        if isinstance(col1, tuple):
            array = recfunctions.append_fields(
                array, names=col1[0], data=col1[1], asrecarray=True
            )
            col1 = col1[0]
        if isinstance(col2, tuple):
            array = recfunctions.append_fields(
                array, names=col2[0], data=col2[1], asrecarray=True
            )
            col2 = col2[0]

        failed = array[col1] > array[col2]
        if np.any(failed):
            failed_info = np.array(array)[failed]
            txt += level0txt.format(len(failed_info)) + "\n"
            if self.level == 1:
                diff = failed_info[col2] - failed_info[col1]
                cols = [
                    c
                    for c in failed_info.dtype.names
                    if failed_info[c].sum() != 0
                    and c != "diff"
                    and "tmp" not in c
                ]
                failed_info = recfunctions.append_fields(
                    failed_info[cols].copy(),
                    names="diff",
                    data=diff,
                    usemask=False,
                    asrecarray=False,
                )
                failed_info.sort(order="diff", axis=0)
                if not sort_ascending:
                    failed_info = failed_info[::-1]
                txt += level1txt + "\n"
                txt += _print_rec_array(failed_info, delimiter=print_delimiter)
            txt += "\n"
        return txt

    def _txt_footer(
        self, headertxt, txt, testname, passed=False, warning=True
    ):
        if len(txt) == 0 or passed:
            txt += "passed."
            self.passed.append(testname)
        elif warning:
            self.warnings.append(testname)
        else:
            self.errors.append(testname)
        if self.verbose:
            print(txt + "\n")
        self.txt += headertxt + txt + "\n"

    def for_nans(self):
        """
        Check for nans in reach or segment data

        """
        headertxt = "Checking for nan values...\n"
        txt = ""
        passed = False
        isnan = np.any(np.isnan(np.array(self.reach_data.tolist())), axis=1)
        nanreaches = self.reach_data[isnan]
        if np.any(isnan):
            txt += "Found {} reachs with nans:\n".format(len(nanreaches))
            if self.level == 1:
                txt += _print_rec_array(nanreaches, delimiter=" ")
        for per, sd in self.segment_data.items():
            isnan = np.any(np.isnan(np.array(sd.tolist())), axis=1)
            nansd = sd[isnan]
            if np.any(isnan):
                txt += "Per {}: found {} segments with nans:\n".format(
                    per, len(nanreaches)
                )
                if self.level == 1:
                    txt += _print_rec_array(nansd, delimiter=" ")
        if len(txt) == 0:
            passed = True
        self._txt_footer(headertxt, txt, "nan values", passed)

    def run_all(self):
        return self.sfr.check()

    def numbering(self):
        """
        Checks for continuity in segment and reach numbering
        """

        headertxt = (
            "Checking for continuity in segment and reach numbering...\n"
        )
        if self.verbose:
            print(headertxt.strip())
        txt = ""
        passed = False

        sd = self.segment_data[0]
        # check segment numbering
        txt += _check_numbers(
            self.sfr.nss, sd["nseg"], level=self.level, datatype="segment"
        )

        # check reach numbering
        for segment in np.arange(1, self.sfr.nss + 1):
            reaches = self.reach_data.ireach[self.reach_data.iseg == segment]
            t = _check_numbers(
                len(reaches), reaches, level=self.level, datatype="reach"
            )
            if len(t) > 0:
                txt += "Segment {} has {}".format(segment, t)
        if txt == "":
            passed = True
        self._txt_footer(
            headertxt,
            txt,
            "continuity in segment and reach numbering",
            passed,
            warning=False,
        )

        headertxt = "Checking for increasing segment numbers in downstream direction...\n"
        txt = ""
        passed = False
        if self.verbose:
            print(headertxt.strip())
        # for per, segment_data in self.segment_data.items():

        inds = (sd.outseg < sd.nseg) & (sd.outseg > 0)

        if len(txt) == 0 and np.any(inds):
            decreases = np.array(sd[inds])[["nseg", "outseg"]]
            txt += "Found {} segment numbers decreasing in the downstream direction.\n".format(
                len(decreases)
            )
            txt += "MODFLOW will run but convergence may be slowed:\n"
            if self.level == 1:
                txt += "nseg outseg\n"
                t = ""
                for nseg, outseg in decreases:
                    t += "{} {}\n".format(nseg, outseg)
                txt += t  # '\n'.join(textwrap.wrap(t, width=10))
        if len(t) == 0:
            passed = True
        self._txt_footer(headertxt, txt, "segment numbering order", passed)

    def routing(self):
        """
        Checks for breaks in routing and does comprehensive check for
        circular routing

        """
        headertxt = "Checking for circular routing...\n"
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        # txt += self.sfr.get_outlets(level=self.level, verbose=False)  # will print twice if verbose=True
        # simpler check method using paths from routing graph
        circular_segs = [k for k, v in self.sfr.paths.items() if v is None]
        if len(circular_segs) > 0:
            txt += "{0} instances where an outlet was not found after {1} consecutive segments!\n".format(
                len(circular_segs), self.sfr.nss
            )
            if self.level == 1:
                txt += " ".join(map(str, circular_segs)) + "\n"
            else:
                f = os.path.join(
                    self.sfr.parent._model_ws, "circular_routing.chk.csv"
                )
                np.savetxt(
                    f, circular_segs, fmt="%d", delimiter=",", header=txt
                )
                txt += "See {} for details.".format(f)
            if self.verbose:
                print(txt)
        self._txt_footer(headertxt, txt, "circular routing", warning=False)

        # check reach connections for proximity
        if self.mg is not None:
            rd = self.sfr.reach_data.copy()
            rd.sort(order=["reachID"])
            xcentergrid = self.mg.xcellcenters
            ycentergrid = self.mg.ycellcenters

            x0 = xcentergrid[rd.i, rd.j]
            y0 = ycentergrid[rd.i, rd.j]
            loc = dict(zip(rd.reachID, zip(x0, y0)))

            # compute distances between node centers of connected reaches
            headertxt = "Checking reach connections for proximity...\n"
            txt = ""
            if self.verbose:
                print(headertxt.strip())
            dist = []
            for r in rd.reachID:
                x0, y0 = loc[r]
                outreach = rd.outreach[r - 1]
                if outreach == 0:
                    dist.append(0)
                else:
                    x1, y1 = loc[outreach]
                    dist.append(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            dist = np.array(dist)

            # compute max width of reach nodes (hypotenuse for rectangular nodes)
            delr = self.mg.delr
            delc = self.mg.delc

            dx = delr[rd.j]
            dy = delc[rd.i]
            hyp = np.sqrt(dx ** 2 + dy ** 2)

            # breaks are when the connection distance is greater than
            # max node with * a tolerance
            # 1.25 * hyp is greater than distance of two diagonally adjacent nodes
            # where one is 1.5x larger than the other
            breaks = np.where(dist > hyp * 1.25)
            breaks_reach_data = rd[breaks]
            segments_with_breaks = set(breaks_reach_data.iseg)
            if len(breaks) > 0:
                txt += (
                    "{0} segments ".format(len(segments_with_breaks))
                    + "with non-adjacent reaches found.\n"
                )
                if self.level == 1:
                    txt += "At segments:\n"
                    txt += " ".join(map(str, segments_with_breaks)) + "\n"
                else:
                    fpath = os.path.join(
                        self.sfr.parent._model_ws,
                        "reach_connection_gaps.chk.csv",
                    )
                    with open(fpath, "w") as fp:
                        fp.write(",".join(rd.dtype.names) + "\n")
                        np.savetxt(fp, rd, "%s", ",")
                    txt += "See {} for details.".format(fpath)
                if self.verbose:
                    print(txt)
            self._txt_footer(
                headertxt, txt, "reach connections", warning=False
            )
        else:
            txt += (
                "No DIS package or SpatialReference object; cannot "
                "check reach proximities."
            )
            self._txt_footer(headertxt, txt, "")

    def overlapping_conductance(self, tol=1e-6):
        """
        Checks for multiple SFR reaches in one cell; and whether more than
        one reach has Cond > 0

        """
        headertxt = (
            "Checking for model cells with multiple non-zero "
            "SFR conductances...\n"
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        # make nreach vectors of each conductance parameter
        reach_data = np.array(self.reach_data)
        # if no dis file was supplied, can't compute node numbers
        # make nodes based on unique row, col pairs
        # if np.diff(reach_data.node).max() == 0:
        # always use unique rc, since flopy assigns nodes by k, i, j
        uniquerc = {}
        for i, (r, c) in enumerate(reach_data[["i", "j"]]):
            if (r, c) not in uniquerc:
                uniquerc[(r, c)] = i + 1
        reach_data["node"] = [
            uniquerc[(r, c)] for r, c in reach_data[["i", "j"]]
        ]

        K = reach_data["strhc1"]
        if K.max() == 0:
            K = self.sfr._interpolate_to_reaches("hcond1", "hcond2")
        b = reach_data["strthick"]
        if b.max() == 0:
            b = self.sfr._interpolate_to_reaches("thickm1", "thickm2")
        L = reach_data["rchlen"]
        w = self.sfr._interpolate_to_reaches("width1", "width2")

        # Calculate SFR conductance for each reach
        binv = np.zeros(b.shape, dtype=b.dtype)
        idx = b > 0.0
        binv[idx] = 1.0 / b[idx]
        Cond = K * w * L * binv

        shared_cells = _get_duplicates(reach_data["node"])

        nodes_with_multiple_conductance = set()
        for node in shared_cells:

            # select the collocated reaches for this cell
            conductances = Cond[reach_data["node"] == node].copy()
            conductances.sort()

            # list nodes with multiple non-zero SFR reach conductances
            if conductances[-1] != 0.0 and (
                conductances[0] / conductances[-1] > tol
            ):
                nodes_with_multiple_conductance.update({node})

        if len(nodes_with_multiple_conductance) > 0:
            txt += (
                "{} model cells with multiple non-zero SFR conductances found.\n"
                "This may lead to circular routing between collocated reaches.\n".format(
                    len(nodes_with_multiple_conductance)
                )
            )
            if self.level == 1:
                txt += "Nodes with overlapping conductances:\n"

                reach_data["strthick"] = b
                reach_data["strhc1"] = K

                cols = [
                    c
                    for c in reach_data.dtype.names
                    if c
                    in [
                        "k",
                        "i",
                        "j",
                        "iseg",
                        "ireach",
                        "rchlen",
                        "strthick",
                        "strhc1",
                        "width",
                        "conductance",
                    ]
                ]

                reach_data = recfunctions.append_fields(
                    reach_data,
                    names=["width", "conductance"],
                    data=[w, Cond],
                    usemask=False,
                    asrecarray=False,
                )
                has_multiple = np.array(
                    [
                        True if n in nodes_with_multiple_conductance else False
                        for n in reach_data["node"]
                    ]
                )
                reach_data = reach_data[has_multiple]
                reach_data = reach_data[cols]
                txt += _print_rec_array(reach_data, delimiter="\t")

        self._txt_footer(headertxt, txt, "overlapping conductance")

    def elevations(self, min_strtop=-10, max_strtop=15000):
        """
        Checks streambed elevations for downstream rises and inconsistencies
        with model grid

        """
        headertxt = (
            "Checking for streambed tops of less "
            "than {}...\n".format(min_strtop)
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        passed = False
        if self.sfr.isfropt in [1, 2, 3]:
            if np.diff(self.reach_data.strtop).max() == 0:
                txt += "isfropt setting of 1,2 or 3 requires strtop information!\n"
            else:
                is_less = self.reach_data.strtop < min_strtop
                if np.any(is_less):
                    below_minimum = self.reach_data[is_less]
                    txt += "{} instances of streambed top below minimum found.\n".format(
                        len(below_minimum)
                    )
                    if self.level == 1:
                        txt += "Reaches with low strtop:\n"
                        txt += _print_rec_array(below_minimum, delimiter="\t")
                if len(txt) == 0:
                    passed = True
        else:
            txt += "strtop not specified for isfropt={}\n".format(
                self.sfr.isfropt
            )
            passed = True
        self._txt_footer(headertxt, txt, "minimum streambed top", passed)

        headertxt = (
            "Checking for streambed tops of "
            "greater than {}...\n".format(max_strtop)
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        passed = False
        if self.sfr.isfropt in [1, 2, 3]:
            if np.diff(self.reach_data.strtop).max() == 0:
                txt += (
                    "isfropt setting of 1,2 or 3 "
                    "requires strtop information!\n"
                )
            else:
                is_greater = self.reach_data.strtop > max_strtop
                if np.any(is_greater):
                    above_max = self.reach_data[is_greater]
                    txt += (
                        "{} instances ".format(len(above_max))
                        + "of streambed top above the maximum found.\n"
                    )
                    if self.level == 1:
                        txt += "Reaches with high strtop:\n"
                        txt += _print_rec_array(above_max, delimiter="\t")
                if len(txt) == 0:
                    passed = True
        else:
            txt += "strtop not specified for isfropt={}\n".format(
                self.sfr.isfropt
            )
            passed = True
        self._txt_footer(headertxt, txt, "maximum streambed top", passed)

        headertxt = (
            "Checking segment_data for "
            "downstream rises in streambed elevation...\n"
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        # decide whether to check elevup and elevdn from items 6b/c
        # (see online guide to SFR input; Data Set 6b description)
        passed = False
        if self.sfr.isfropt in [0, 4, 5]:
            pers = sorted(self.segment_data.keys())
            for per in pers:
                segment_data = self.segment_data[per][
                    self.segment_data[per].elevup > -999999
                ]

                # enforce consecutive increasing segment numbers (for indexing)
                segment_data.sort(order="nseg")
                t = _check_numbers(
                    len(segment_data),
                    segment_data.nseg,
                    level=1,
                    datatype="Segment",
                )
                if len(t) > 0:
                    txt += (
                        "Elevation check requires "
                        "consecutive segment numbering."
                    )
                    self._txt_footer(headertxt, txt, "")
                    return

                # first check for segments where elevdn > elevup
                d_elev = segment_data.elevdn - segment_data.elevup
                segment_data = recfunctions.append_fields(
                    segment_data, names="d_elev", data=d_elev, asrecarray=True
                )
                txt += self._boolean_compare(
                    np.array(segment_data)[
                        ["nseg", "outseg", "elevup", "elevdn", "d_elev"]
                    ],
                    col1="d_elev",
                    col2=np.zeros(len(segment_data)),
                    level0txt="Stress Period {}: ".format(per + 1)
                    + "{} segments encountered with elevdn > elevup.",
                    level1txt="Backwards segments:",
                )

                # next check for rises between segments
                non_outlets = segment_data.outseg > 0
                non_outlets_seg_data = segment_data[
                    non_outlets
                ]  # lake outsegs are < 0
                outseg_elevup = np.array(
                    [
                        segment_data.elevup[o - 1]
                        for o in segment_data.outseg
                        if o > 0
                    ]
                )
                d_elev2 = outseg_elevup - segment_data.elevdn[non_outlets]
                non_outlets_seg_data = recfunctions.append_fields(
                    non_outlets_seg_data,
                    names=["outseg_elevup", "d_elev2"],
                    data=[outseg_elevup, d_elev2],
                    usemask=False,
                    asrecarray=False,
                )

                txt += self._boolean_compare(
                    non_outlets_seg_data[
                        [
                            "nseg",
                            "outseg",
                            "elevdn",
                            "outseg_elevup",
                            "d_elev2",
                        ]
                    ],
                    col1="d_elev2",
                    col2=np.zeros(len(non_outlets_seg_data)),
                    level0txt="Stress Period {}: ".format(per + 1)
                    + "{} segments encountered with segments encountered "
                    "with outseg elevup > elevdn.",
                    level1txt="Backwards segment connections:",
                )

            if len(txt) == 0:
                passed = True
        else:
            txt += (
                "Segment elevup and elevdn not specified for nstrm={} "
                "and isfropt={}\n".format(self.sfr.nstrm, self.sfr.isfropt)
            )
            passed = True
        self._txt_footer(headertxt, txt, "segment elevations", passed)

        headertxt = (
            "Checking reach_data for "
            "downstream rises in streambed elevation...\n"
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())
        passed = False
        if (
            self.sfr.nstrm < 0
            or self.sfr.reachinput
            and self.sfr.isfropt in [1, 2, 3]
        ):  # see SFR input instructions

            # compute outreaches if they aren't there already
            if np.diff(self.sfr.reach_data.outreach).max() == 0:
                self.sfr.set_outreaches()

            # compute changes in elevation
            rd = self.reach_data.copy()
            elev = dict(zip(rd.reachID, rd.strtop))
            dnelev = {
                rid: elev[rd.outreach[i]] if rd.outreach[i] != 0 else -9999
                for i, rid in enumerate(rd.reachID)
            }
            strtopdn = np.array([dnelev[r] for r in rd.reachID])
            diffs = np.array(
                [
                    (dnelev[i] - elev[i]) if dnelev[i] != -9999 else -0.001
                    for i in rd.reachID
                ]
            )

            reach_data = (
                self.sfr.reach_data
            )  # inconsistent with other checks that work with
            # reach_data attribute of check class. Want to have get_outreaches as a method of sfr class
            # (for other uses). Not sure if other check methods should also copy reach_data directly from
            # SFR package instance for consistency.

            # use outreach values to get downstream elevations
            # non_outlets = reach_data[reach_data.outreach != 0]
            # outreach_elevdn = np.array([reach_data.strtop[o - 1] for o in reach_data.outreach])
            # d_strtop = outreach_elevdn[reach_data.outreach != 0] - non_outlets.strtop
            rd = recfunctions.append_fields(
                rd,
                names=["strtopdn", "d_strtop"],
                data=[strtopdn, diffs],
                usemask=False,
                asrecarray=False,
            )

            txt += self._boolean_compare(
                rd[
                    [
                        "k",
                        "i",
                        "j",
                        "iseg",
                        "ireach",
                        "strtop",
                        "strtopdn",
                        "d_strtop",
                        "reachID",
                    ]
                ],
                col1="d_strtop",
                col2=np.zeros(len(rd)),
                level0txt="{} reaches encountered with strtop < strtop of downstream reach.",
                level1txt="Elevation rises:",
            )
            if len(txt) == 0:
                passed = True
        else:
            txt += "Reach strtop not specified for nstrm={}, reachinput={} and isfropt={}\n".format(
                self.sfr.nstrm, self.sfr.reachinput, self.sfr.isfropt
            )
            passed = True
        self._txt_footer(headertxt, txt, "reach elevations", passed)

        headertxt = "Checking reach_data for inconsistencies between streambed elevations and the model grid...\n"
        if self.verbose:
            print(headertxt.strip())
        txt = ""
        if self.sfr.parent.dis is None:
            txt += "No DIS file supplied; cannot check SFR elevations against model grid."
            self._txt_footer(headertxt, txt, "")
            return
        passed = False
        warning = True
        if (
            self.sfr.nstrm < 0
            or self.sfr.reachinput
            and self.sfr.isfropt in [1, 2, 3]
        ):  # see SFR input instructions
            reach_data = np.array(self.reach_data)
            i, j, k = reach_data["i"], reach_data["j"], reach_data["k"]

            # check streambed bottoms in relation to respective cell bottoms
            bots = self.sfr.parent.dis.botm.array[k, i, j]
            streambed_bots = reach_data["strtop"] - reach_data["strthick"]
            reach_data = recfunctions.append_fields(
                reach_data,
                names=["layerbot", "strbot"],
                data=[bots, streambed_bots],
                usemask=False,
                asrecarray=False,
            )

            txt += self._boolean_compare(
                reach_data[
                    [
                        "k",
                        "i",
                        "j",
                        "iseg",
                        "ireach",
                        "strtop",
                        "strthick",
                        "strbot",
                        "layerbot",
                        "reachID",
                    ]
                ],
                col1="layerbot",
                col2="strbot",
                level0txt="{} reaches encountered with streambed bottom below layer bottom.",
                level1txt="Layer bottom violations:",
            )
            if len(txt) > 0:
                warning = (
                    False  # this constitutes an error (MODFLOW won't run)
                )
            # check streambed elevations in relation to model top
            tops = self.sfr.parent.dis.top.array[i, j]
            reach_data = recfunctions.append_fields(
                reach_data,
                names="modeltop",
                data=tops,
                usemask=False,
                asrecarray=False,
            )

            txt += self._boolean_compare(
                reach_data[
                    [
                        "k",
                        "i",
                        "j",
                        "iseg",
                        "ireach",
                        "strtop",
                        "modeltop",
                        "strhc1",
                        "reachID",
                    ]
                ],
                col1="strtop",
                col2="modeltop",
                level0txt="{} reaches encountered with streambed above model top.",
                level1txt="Model top violations:",
            )

            if len(txt) == 0:
                passed = True
        else:
            txt += "Reach strtop, strthick not specified for nstrm={}, reachinput={} and isfropt={}\n".format(
                self.sfr.nstrm, self.sfr.reachinput, self.sfr.isfropt
            )
            passed = True
        self._txt_footer(
            headertxt,
            txt,
            "reach elevations vs. grid elevations",
            passed,
            warning=warning,
        )

        # In cases where segment end elevations/thicknesses are used,
        # do these need to be checked for consistency with layer bottoms?

        headertxt = (
            "Checking segment_data for inconsistencies "
            "between segment end elevations and the model grid...\n"
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())
        passed = False
        if self.sfr.isfropt in [0, 4, 5]:
            reach_data = self.reach_data
            pers = sorted(self.segment_data.keys())
            for per in pers:
                segment_data = self.segment_data[per][
                    self.segment_data[per].elevup > -999999
                ]

                # enforce consecutive increasing segment numbers (for indexing)
                segment_data.sort(order="nseg")
                t = _check_numbers(
                    len(segment_data),
                    segment_data.nseg,
                    level=1,
                    datatype="Segment",
                )
                if len(t) > 0:
                    raise Exception(
                        "Elevation check requires consecutive segment numbering."
                    )

            first_reaches = reach_data[reach_data.ireach == 1].copy()
            last_reaches = reach_data[
                np.append((np.diff(reach_data.iseg) == 1), True)
            ].copy()
            segment_ends = recfunctions.stack_arrays(
                [first_reaches, last_reaches], asrecarray=True, usemask=False
            )
            segment_ends["strtop"] = np.append(
                segment_data["elevup"], segment_data["elevdn"]
            )
            i, j = segment_ends.i, segment_ends.j
            tops = self.sfr.parent.dis.top.array[i, j]
            diff = tops - segment_ends.strtop
            segment_ends = recfunctions.append_fields(
                segment_ends,
                names=["modeltop", "diff"],
                data=[tops, diff],
                usemask=False,
                asrecarray=False,
            )

            txt += self._boolean_compare(
                segment_ends[
                    [
                        "k",
                        "i",
                        "j",
                        "iseg",
                        "strtop",
                        "modeltop",
                        "diff",
                        "reachID",
                    ]
                ].copy(),
                col1=np.zeros(len(segment_ends)),
                col2="diff",
                level0txt="{} reaches encountered with streambed above model top.",
                level1txt="Model top violations:",
            )

            if len(txt) == 0:
                passed = True
        else:
            txt += "Segment elevup and elevdn not specified for nstrm={} and isfropt={}\n".format(
                self.sfr.nstrm, self.sfr.isfropt
            )
            passed = True
        self._txt_footer(
            headertxt, txt, "segment elevations vs. model grid", passed
        )

    def slope(self, minimum_slope=1e-4, maximum_slope=1.0):
        """Checks that streambed slopes are greater than or equal to a specified minimum value.
        Low slope values can cause "backup" or unrealistic stream stages with icalc options
        where stage is computed.
        """
        headertxt = (
            "Checking for streambed slopes of less than {}...\n".format(
                minimum_slope
            )
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        passed = False
        if self.sfr.isfropt in [1, 2, 3]:
            if np.diff(self.reach_data.slope).max() == 0:
                txt += (
                    "isfropt setting of 1,2 or 3 requires slope information!\n"
                )
            else:
                is_less = self.reach_data.slope < minimum_slope
                if np.any(is_less):
                    below_minimum = self.reach_data[is_less]
                    txt += "{} instances of streambed slopes below minimum found.\n".format(
                        len(below_minimum)
                    )
                    if self.level == 1:
                        txt += "Reaches with low slopes:\n"
                        txt += _print_rec_array(below_minimum, delimiter="\t")
                if len(txt) == 0:
                    passed = True
        else:
            txt += "slope not specified for isfropt={}\n".format(
                self.sfr.isfropt
            )
            passed = True
        self._txt_footer(headertxt, txt, "minimum slope", passed)

        headertxt = (
            "Checking for streambed slopes of greater than {}...\n".format(
                maximum_slope
            )
        )
        txt = ""
        if self.verbose:
            print(headertxt.strip())

        passed = False
        if self.sfr.isfropt in [1, 2, 3]:
            if np.diff(self.reach_data.slope).max() == 0:
                txt += (
                    "isfropt setting of 1,2 or 3 requires slope information!\n"
                )
            else:
                is_greater = self.reach_data.slope > maximum_slope

                if np.any(is_greater):
                    above_max = self.reach_data[is_greater]
                    txt += "{} instances of streambed slopes above maximum found.\n".format(
                        len(above_max)
                    )
                    if self.level == 1:
                        txt += "Reaches with high slopes:\n"
                        txt += _print_rec_array(above_max, delimiter="\t")
                if len(txt) == 0:
                    passed = True
        else:
            txt += "slope not specified for isfropt={}\n".format(
                self.sfr.isfropt
            )
            passed = True
        self._txt_footer(headertxt, txt, "maximum slope", passed)


def _check_numbers(n, numbers, level=1, datatype="reach"):
    """
    Check that a sequence of numbers is consecutive
    (that the sequence is equal to the range from 1 to n+1, where n is
    the expected length of the sequence).

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
    txt = ""
    num_range = np.arange(1, n + 1)
    if not np.array_equal(num_range, numbers):
        txt += "Invalid {} numbering\n".format(datatype)
        if level == 1:
            # consistent dimension for boolean array
            non_consecutive = np.append(np.diff(numbers) != 1, False)
            gaps = num_range[non_consecutive] + 1
            if len(gaps) > 0:
                gapstr = " ".join(map(str, gaps))
                txt += "Gaps in numbering at positions {}\n".format(gapstr)
    return txt


def _isnumeric(s):
    try:
        float(s)
        return True
    except:
        return False


def _markitzero(recarray, inds):
    """
    Subtracts 1 from columns specified in inds argument, to convert from
    1 to 0-based indexing

    """
    lnames = [n.lower() for n in recarray.dtype.names]
    for idx in inds:
        if idx in lnames:
            recarray[idx] -= 1


def _pop_item(line):
    try:
        return float(line.pop(0))
    except:
        return 0.0


def _get_dataset(line, dataset):
    # interpret number supplied with decimal points as floats, rest as ints
    # this could be a bad idea (vs. explicitly formatting values for each dataset)
    for i, s in enumerate(line_parse(line)):
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
    """
    Returns duplicate values in an array, similar to pandas .duplicated()
    method
    http://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
    """
    s = np.sort(a, axis=None)
    equal_to_previous_item = np.append(
        s[1:] == s[:-1], False
    )  # maintain same dimension for boolean array
    return np.unique(s[equal_to_previous_item])


def _get_item2_names(nstrm, reachinput, isfropt, structured=False):
    """
    Determine which variables should be in item 2, based on model grid type,
    reachinput specification, and isfropt.

    Returns
    -------
    names : list of str
        List of names (same as variables in SFR Package input instructions) of
        columns to assign (upon load) or retain (upon write) in reach_data
        array.

    Notes
    -----
    Lowercase is used for all variable names.

    """
    names = []
    if structured:
        names += ["k", "i", "j"]
    else:
        names += ["node"]
    names += ["iseg", "ireach", "rchlen"]
    if nstrm < 0 or reachinput:
        if isfropt in [1, 2, 3]:
            names += ["strtop", "slope", "strthick", "strhc1"]
            if isfropt in [2, 3]:
                names += ["thts", "thti", "eps"]
                if isfropt == 3:
                    names += ["uhc"]
    return names


def _fmt_string_list(array, float_format=default_float_format):
    fmt_list = []
    for name in array.dtype.names:
        vtype = array.dtype[name].str[1].lower()
        if vtype == "v":
            continue
        if vtype == "i":
            fmt_list.append("{:d}")
        elif vtype == "f":
            fmt_list.append(float_format)
        elif vtype == "o":
            float_format = "{!s}"
        elif vtype == "s":
            raise ValueError(
                "'str' type found in dtype for {!r}. "
                "This gives unpredictable results when "
                "recarray to file - change to 'object' type".format(name)
            )
        else:
            raise ValueError(
                "unknown dtype for {!r}: {!r}".format(name, vtype)
            )
    return fmt_list


def _fmt_string(array, float_format=default_float_format):
    return " ".join(_fmt_string_list(array, float_format))


def _print_rec_array(
    array, cols=None, delimiter=" ", float_format=default_float_format
):
    """
    Print out a numpy record array to string, with column names.

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
    txt = ""
    if cols is not None:
        cols = [c for c in array.dtype.names if c in cols]
    else:
        cols = list(array.dtype.names)
    # drop columns with no data
    if np.shape(array)[0] > 1:
        cols = [c for c in cols if array[c].min() > -999999]
    # add _fmt_string call here
    array = np.array(array)[cols]
    fmts = _fmt_string_list(array, float_format=float_format)
    txt += delimiter.join(cols) + "\n"
    txt += "\n".join([delimiter.join(fmts).format(*r) for r in array.tolist()])
    return txt


def _parse_1c(line, reachinput, transroute):
    """
    Parse Data Set 1c for SFR2 package.
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
    # line = line.strip().split()
    line = line_parse(line)

    nstrm = int(line.pop(0))
    nss = int(line.pop(0))
    nsfrpar = int(line.pop(0))
    nparseg = int(line.pop(0))
    const = float(line.pop(0))
    dleak = float(line.pop(0))
    ipakcb = int(line.pop(0))
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
        if isfropt > 1:
            nstrail = int(line.pop(0))
            isuzn = int(line.pop(0))
            nsfrsets = int(line.pop(0))

    irtflg, numtim, weight, flwtol = na, na, na, na
    if nstrm < 0 or transroute:
        irtflg = int(_pop_item(line))
        if irtflg > 0:
            numtim = int(line.pop(0))
            weight = float(line.pop(0))
            flwtol = float(line.pop(0))

    # auxiliary variables (MODFLOW-LGR)
    option = [
        line[i]
        for i in np.arange(1, len(line))
        if "aux" in line[i - 1].lower()
    ]

    return (
        nstrm,
        nss,
        nsfrpar,
        nparseg,
        const,
        dleak,
        ipakcb,
        istcb2,
        isfropt,
        nstrail,
        isuzn,
        nsfrsets,
        irtflg,
        numtim,
        weight,
        flwtol,
        option,
    )


def _parse_6a(line, option):
    """
    Parse Data Set 6a for SFR2 package.
    See http://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?sfr.htm for more info

    Parameters
    ----------
    line : str
        line read from SFR package input file

    Returns
    -------
        a list of length 13 containing all variables for Data Set 6a
    """
    # line = line.strip().split()
    line = line_parse(line)

    xyz = []
    # handle any aux variables at end of line
    for s in line:
        if s.lower() in option:
            xyz.append(s.lower())

    na = 0
    nseg = int(_pop_item(line))
    icalc = int(_pop_item(line))
    outseg = int(_pop_item(line))
    iupseg = int(_pop_item(line))
    iprior = na
    nstrpts = na

    if iupseg > 0:
        iprior = int(_pop_item(line))
    if icalc == 4:
        nstrpts = int(_pop_item(line))

    flow = _pop_item(line)
    runoff = _pop_item(line)
    etsw = _pop_item(line)
    pptsw = _pop_item(line)
    roughch = na
    roughbk = na

    if icalc in [1, 2]:
        roughch = _pop_item(line)
    if icalc == 2:
        roughbk = _pop_item(line)

    cdpth, fdpth, awdth, bwdth = na, na, na, na
    if icalc == 3:
        cdpth, fdpth, awdth, bwdth = map(float, line)
    return (
        nseg,
        icalc,
        outseg,
        iupseg,
        iprior,
        nstrpts,
        flow,
        runoff,
        etsw,
        pptsw,
        roughch,
        roughbk,
        cdpth,
        fdpth,
        awdth,
        bwdth,
        xyz,
    )


def _parse_6bc(line, icalc, nstrm, isfropt, reachinput, per=0):
    """
    Parse Data Set 6b for SFR2 package.
    See http://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?sfr.htm for more info

    Parameters
    ----------
    line : str
        line read from SFR package input file

    Returns
    -------
        a list of length 9 containing all variables for Data Set 6b

    """
    nvalues = sum([_isnumeric(s) for s in line_parse(line)])
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
        if isfropt in [4, 5] and per > 0:
            pass
        else:
            thickm = line.pop(0)
            elevupdn = line.pop(0)
            # depth is not read if icalc == 1; see table in online guide
            width = line.pop(0)
            thts = _pop_item(line)
            thti = _pop_item(line)
            eps = _pop_item(line)
        if isfropt == 5 and per == 0:
            uhc = line.pop(0)
    elif isfropt in [0, 4, 5] and icalc >= 2:
        hcond = line.pop(0)
        if isfropt in [4, 5] and per > 0 and icalc == 2:
            pass
        else:
            thickm = line.pop(0)
            elevupdn = line.pop(0)
            if isfropt in [4, 5] and per == 0:
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
    elif isfropt in [2, 3]:
        if icalc <= 0:
            width = line.pop(0)
            depth = line.pop(0)

        elif icalc == 1:
            if per > 0:
                pass
            else:
                width = line.pop(0)

        else:
            pass
    else:
        pass
    return hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc


def find_path(graph, start, end=0):
    """Get a path through the routing network,
    from a segment to an outlet.

    Parameters
    ----------
    graph : dict
        Dictionary of seg : outseg numbers
    start : int
        Starting segment
    end : int
        Ending segment (default 0)

    Returns
    -------
    path : list
        List of segment numbers along routing path.
    """
    graph = graph.copy()
    return _find_path(graph, start, end=end)


def _find_path(graph, start, end=0, path=None):
    """Like find_path, but doesn't copy the routing
    dictionary (graph) so that the recursion works.
    """
    if path is None:
        path = list()
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    if not isinstance(graph[start], list):
        graph[start] = [graph[start]]
    for node in graph[start]:
        if node not in path:
            newpath = _find_path(graph, node, end, path)
            if newpath:
                return newpath
    return None
