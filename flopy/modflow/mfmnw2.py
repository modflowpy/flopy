import os
import sys
import warnings

import numpy as np
from .mfdis import get_layer
from ..utils import check
from ..utils.flopy_io import line_parse, pop_item, get_next_line
from ..utils import MfList
from ..utils.recarray_utils import create_empty_recarray

from ..pakbase import Package


class Mnw(object):
    """
    Multi-Node Well object class

    Parameters
    ----------
    wellid : str or int
        is the name of the well. This is a unique alphanumeric identification
        label for each well. The text string is limited to 20 alphanumeric
        characters. If the name of the well includes spaces, then enclose the
        name in quotes. Flopy converts wellid string to lower case.
    nnodes : int
        is the number of cells (nodes) associated with this well.
        NNODES normally is > 0, but for the case of a vertical borehole,
        setting NNODES < 0 will allow the user to specify the elevations of
        the tops and bottoms of well screens or open intervals (rather than
        grid layer numbers), and the absolute value of NNODES equals the
        number of open intervals (or well screens) to be specified in dataset
        2d. If this option is used, then the model will compute the layers in
        which the open intervals occur, the lengths of the open intervals,
        and the relative vertical position of the open interval within a model
        layer (for example, see figure 14 and related discussion).
    losstype : str
        is a character flag to determine the user-specified model for well loss
        (equation 2). Available options (that is, place one of the following
        approved words in this field) are:
        NONE    there are no well corrections and the head in the well is
                assumed to equal the head in the cell. This option (hWELL = hn)
                is only valid for a single-node well (NNODES = 1). (This is
                equivalent to using the original WEL Package of MODFLOW,
                but specifying the single-node well within the MNW2 Package
                enables the use of constraints.)
        THIEM   this option allows for only the cell-to-well correction at the
                well based on the Thiem (1906) equation; head in the well is
                determined from equation 2 as (hWELL = hn + AQn), and the model
                computes A on the basis of the user-specified well radius (Rw)
                and previously defined values of cell transmissivity and grid
                spacing. Coefficients B and C in equation 2 are automatically
                set = 0.0. User must define Rw in dataset 2c or 2d.
        SKIN    this option allows for formation damage or skin corrections at
                the well. hWELL = hn + AQn + BQn (from equation 2), where A is
                determined by the model from the value of Rw, and B is
                determined by the model from Rskin and Kskin. User must define
                Rw, Rskin, and Kskin in dataset 2c or 2d.
        GENERAL head loss is defined with coefficients A, B, and C and power
                exponent P (hWELL = hn + AQn + BQn + CQnP). A is determined by
                the model from the value of Rw. User must define Rw, B, C, and
                P in dataset 2c or 2d. A value of P = 2.0 is suggested if no
                other data are available (the model allows 1.0 <= P <= 3.5).
                Entering a value of C = 0 will result in a "linear" model in
                which the value of B is entered directly (rather than entering
                properties of the skin, as with the SKIN option).
        SPECIFYcwc the user specifies an effective conductance value
                (equivalent to the combined effects of the A, B, and C
                well-loss coefficients expressed in equation 15) between the
                well and the cell representing the aquifer, CWC. User must
                define CWC in dataset 2c or 2d. If there are multiple screens
                within the grid cell or if partial penetration corrections are
                to be made, then the effective value of CWC for the node may
                be further adjusted automatically by MNW2.
    pumploc : int
        is an integer flag pertaining to the location along the borehole of
        the pump intake (if any). If PUMPLOC = 0, then either there is no pump
        or the intake location (or discharge point for an injection well) is
        assumed to occur above the first active node associated with the multi-
        node well (that is, the node closest to the land surface or to the
        wellhead). If PUMPLOC > 0, then the cell in which the intake (or
        outflow) is located will be specified in dataset 2e as a LAY-ROW-COL
        grid location. For a vertical well only, specifying PUMPLOC < 0, will
        enable the option to define the vertical position of the pump intake
        (or outflow) as an elevation in dataset 2e (for the given spatial grid
        location [ROW-COL] defined for this well in 2d).
    qlimit : int
        is an integer flag that indicates whether the water level (head) in
        the well will be used to constrain the pumping rate. If Qlimit = 0,
        then there are no constraints for this well. If Qlimit > 0, then
        pumpage will be limited (constrained) by the water level in the well,
        and relevant parameters are constant in time and defined below in
        dataset 2f. If Qlimit < 0, then pumpage will be limited (constrained)
        by the water level in the well, and relevant parameters can vary with
        time and are defined for every stress period in dataset 4b.
    ppflag : int
        is an integer flag that determines whether the calculated head in the
        well will be corrected for the effect of partial penetration of the
        well screen in the cell. If PPFLAG = 0, then the head in the well will
        not be adjusted for the effects of partial penetration. If PPFLAG > 0,
        then the head in the well will be adjusted for the effects of partial
        penetration if the section of well containing the well screen is
        vertical (as indicated by identical row-column locations in the grid).
        If NNODES < 0 (that is, the open intervals of the well are defined by
        top and bottom elevations), then the model will automatically calculate
        the fraction of penetration for each node and the relative vertical
        position of the well screen. If NNODES > 0, then the fraction of
        penetration for each node must be defined in dataset 2d (see below)
        and the well screen will be assumed to be centered vertically within
        the thickness of the cell (except if the well is located in the
        uppermost model layer that is under unconfined conditions, in which
        case the bottom of the well screen will be assumed to be aligned with
        the bottom boundary of the cell and the assumed length of well screen
        will be based on the initial head in that cell).
    pumpcap : int
        is an integer flag and value that determines whether the discharge of
        a pumping (withdrawal) well (Q < 0.0) will be adjusted for changes in
        the lift (or total dynamic head) with time. If PUMPCAP = 0, then the
        discharge from the well will not be adjusted on the basis of changes
        in lift. If PUMPCAP > 0 for a withdrawal well, then the discharge from
        the well will be adjusted on the basis of the lift, as calculated from
        the most recent water level in the well. In this case, data describing
        the head-capacity relation for the pump must be listed in datasets 2g
        and 2h, and the use of that relation can be switched on or off for
        each stress period using a flag in dataset 4a. The number of entries
        (lines) in dataset 2h corresponds to the value of PUMPCAP. If PUMPCAP
        does not equal 0, it must be set to an integer value of between 1 and
        25, inclusive.
    rw : float
        radius of the well (losstype == 'THIEM', 'SKIN', or 'GENERAL')
    rskin : float
        radius to the outer limit of the skin (losstype == 'SKIN')
    kskin : float
        hydraulic conductivity of the skin
    B : float
        coefficient of the well-loss eqn. (eqn. 2 in MNW2 documentation)
        (losstype == 'GENERAL')
    C : float
        coefficient of the well-loss eqn. (eqn. 2 in MNW2 documentation)
        (losstype == 'GENERAL')
    P : float
        coefficient of the well-loss eqn. (eqn. 2 in MNW2 documentation)
        (losstype == 'GENERAL')
    cwc : float
        cell-to-well conductance.
        (losstype == 'SPECIFYcwc')
    pp : float
        fraction of partial penetration for the cell. Only specify if
        PFLAG > 0 and NNODES > 0.
    k : int
        layer index of well (zero-based)
    i : int
        row index of well (zero-based)
    j : int
        column index of well (zero-based)
    ztop : float
        top elevation of open intervals of vertical well.
    zbotm : float
        bottom elevation of open intervals of vertical well.
    node_data : numpy record array
        table containing MNW data by node. A blank node_data template can be
        created via the ModflowMnw2.get_empty_mnw_data() static method.

        Note: Variables in dataset 2d (e.g. rw) can be entered as a single
        value for the entire well (above), or in node_data (or dataset 2d) by
        node. Variables not in dataset 2d (such as pumplay) can be included
        in node data for convenience (to allow construction of MNW2 package
        from a table), but are only written to MNW2 as a single variable.
        When writing non-dataset 2d variables to MNW2 input, the first value
        for the well will be used.

        Other variables (e.g. hlim) can be entered here as
        constant for all stress periods, or by stress period below in stress_period_data.
        See MNW2 input instructions for more details.

        Columns are:
            k : int
                layer index of well (zero-based)
            i : int
                row index of well (zero-based)
            j : int
                column index of well (zero-based)
            ztop : float
                top elevation of open intervals of vertical well.
            zbotm : float
                bottom elevation of open intervals of vertical well.
            wellid : str
            losstype : str
            pumploc : int
            qlimit : int
            ppflag : int
            pumpcap : int
            rw : float
            rskin : float
            kskin : float
            B : float
            C : float
            P : float
            cwc : float
            pp : float
            pumplay : int
            pumprow : int
            pumpcol : int
            zpump : float
            hlim : float
            qcut : int
            qfrcmn : float
            qfrcmx : float
            hlift : float
            liftq0 : float
            liftqmax : float
            hwtol : float
            liftn : float
            qn : float

    stress_period_data : numpy record array
        table containing MNW pumping data for all stress periods (dataset 4 in
        the MNW2 input instructions). A blank stress_period_data template can
        be created via the Mnw.get_empty_stress_period_data() static method.
        Columns are:
            per : int
                stress period
            qdes : float
                is the actual (or maximum desired, if constraints are to be
                applied) volumetric pumping rate (negative for withdrawal or
                positive for injection) at the well (L3/T). Qdes should be
                set to 0 for nonpumping wells. If constraints are applied,
                then the calculated volumetric withdrawal or injection rate
                may be adjusted to range from 0 to Qdes and is not allowed
                to switch directions between withdrawal and injection
                conditions during any stress period. When PUMPCAP > 0, in the
                first stress period in which Qdes is specified with a negative
                value, Qdes represents the maximum operating discharge for the
                pump; in subsequent stress periods, any different negative
                values of Qdes are ignored, although values are subject to
                adjustment for CapMult. If Qdes >= 0.0, then pump-capacity
                adjustments are not applied.
            capmult : int
                is a flag and multiplier for implementing head-capacity
                relations during a given stress period. Only specify if
                PUMPCAP > 0 for this well. If CapMult <= 0, then
                head-capacity relations are ignored for this stress period.
                If CapMult = 1.0, then head-capacity relations defined
                in datasets 2g and 2h are used. If CapMult equals any other
                positive value (for example, 0.6 or 1.1), then head-capacity
                relations are used but adjusted and shifted by multiplying
                the discharge value indicated by the head-capacity curve by
                the value of CapMult.
            cprime : float
                is the concentration in the injected fluid. Only specify if
                Qdes > 0 and GWT process is active.
            hlim : float
            qcut : int
            qfrcmn : float
            qfrcmx : float
        Note: If auxiliary variables are also being used, additional columns
        for these must be included.
    pumplay : int
    pumprow : int
    pumpcol : int
        PUMPLAY, PUMPROW, and PUMPCOL are the layer, row, and column numbers,
        respectively, of the cell (node) in this multi-node well where the
        pump intake (or outflow) is located. The location defined in dataset
        2e should correspond with one of the nodes listed in 2d for this
        multi-node well. These variables are only read if PUMPLOC > 0 in 2b.
    zpump : float
        is the elevation of the pump intake (or discharge pipe location for an
        injection well). Zpump is read only if PUMPLOC < 0; in this case,
        the model assumes that the borehole is vertical and will compute the
        layer of the grid in which the pump intake is located.
    hlim : float
        is the limiting water level (head) in the well, which is a minimum for
        discharging wells and a maximum for injection wells. For example, in a
        discharging well, when hWELL falls below hlim, the flow from the well
        is constrained.
    qcut : int
        is an integer flag that indicates how pumping limits Qfrcmn and
        Qfrcmx will be specified. If pumping limits are to be specified as a
        rate (L3/T), then set QCUT > 0; if pumping limits are to be specified
        as a fraction of the specified pumping rate (Qdes), then set QCUT < 0.
        If there is not a minimum pumping rate below which the pump becomes
        inactive, then set QCUT = 0.
    qfrcmn : float
        is the minimum pumping rate or fraction of original pumping rate
        (a choice that depends on QCUT) that a well must exceed to remain
        active during a stress period. The absolute value of Qfrcmn must be
        less than the absolute value of Qfrcmx (defined next). Only specify
        if QCUT != 0.
    qfrcmx : float
        is the minimum pumping rate or fraction of original pumping rate that
        must be exceeded to reactivate a well that had been shut off based on
        Qfrcmn during a stress period. The absolute value of Qfrcmx must be
        greater than the absolute value of Qfrcmn. Only specify if QCUT != 0.
    hlift : float
        is the reference head (or elevation) corresponding to the discharge
        point for the well. This is typically at or above the land surface,
        and can be increased to account for additional head loss due to
        friction in pipes.
    liftq0 : float
        is the value of lift (total dynamic head) that exceeds the capacity of
        the pump. If the calculated lift equals or exceeds this value, then
        the pump is shut off and discharge from the well ceases.
    liftqmax : float
        is the value of lift (total dynamic head) corresponding to the maximum
        pumping (discharge) rate for the pump. If the calculated lift is less
        than or equal to LIFTqmax, then the pump will operate at its design
        capacity, assumed to equal the user-specified value of Qdes
        (in dataset 4a). LIFTqmax will be associated with the value of Qdes in
        the first stress period in which Qdes for the well is less than 0.0.
    hwtol : float
        is a minimum absolute value of change in the computed water level in
        the well allowed between successive iterations; if the value of hWELL
        changes from one iteration to the next by a value smaller than this
        tolerance, then the value of discharge computed from the head capacity
        curves will be locked for the remainder of that time step. It is
        recommended that HWtol be set equal to a value approximately one or
        two orders of magnitude larger than the value of HCLOSE, but if the
        solution fails to converge, then this may have to be adjusted.
    liftn : float
        is a value of lift (total dynamic head) that corresponds to a known
        value of discharge (Qn) for the given pump, specified as the second
        value in this line.
    qn : float
        is the value of discharge corresponding to the height of lift
        (total dynamic head) specified previously on this line. Sign
        (positive or negative) is ignored.
    mnwpackage : ModflowMnw2 instance
        package that mnw is attached to

    Returns
    -------
    None

    """

    by_node_variables = [
        "k",
        "i",
        "j",
        "ztop",
        "zbotm",
        "rw",
        "rskin",
        "kskin",
        "B",
        "C",
        "P",
        "cwc",
        "pp",
    ]

    def __init__(
        self,
        wellid,
        nnodes=1,
        nper=1,
        losstype="skin",
        pumploc=0,
        qlimit=0,
        ppflag=0,
        pumpcap=0,
        rw=1,
        rskin=2,
        kskin=10,
        B=None,
        C=0,
        P=2.0,
        cwc=None,
        pp=1,
        k=0,
        i=0,
        j=0,
        ztop=0,
        zbotm=0,
        node_data=None,
        stress_period_data=None,
        pumplay=0,
        pumprow=0,
        pumpcol=0,
        zpump=None,
        hlim=None,
        qcut=None,
        qfrcmn=None,
        qfrcmx=None,
        hlift=None,
        liftq0=None,
        liftqmax=None,
        hwtol=None,
        liftn=None,
        qn=None,
        mnwpackage=None,
    ):
        """
        Class constructor
        """

        self.nper = nper
        self.mnwpackage = mnwpackage  # associated ModflowMnw2 instance
        self.aux = None if mnwpackage is None else mnwpackage.aux

        # dataset 2a
        if isinstance(wellid, str):
            wellid = wellid.lower()
        self.wellid = wellid
        self.nnodes = nnodes
        # dataset 2b
        self.losstype = losstype.lower()
        self.pumploc = pumploc
        self.qlimit = qlimit
        self.ppflag = ppflag
        self.pumpcap = pumpcap
        # dataset 2c (can be entered by node)
        self.rw = rw
        self.rskin = rskin
        self.kskin = kskin
        self.B = B
        self.C = C
        self.P = P
        self.cwc = cwc
        self.pp = pp
        # dataset 2d (entered by node)
        # indices should be lists (for iteration over nodes)
        self.k = k
        self.i = i
        self.j = j
        self.ztop = ztop
        self.zbotm = zbotm
        for v in self.by_node_variables:
            if not isinstance(self.__dict__[v], list):
                self.__dict__[v] = [self.__dict__[v]]
        # dataset 2e
        self.pumplay = pumplay
        self.pumprow = pumprow
        self.pumpcol = pumpcol
        self.zpump = zpump
        # dataset 2f
        self.hlim = hlim
        self.qcut = qcut
        self.qfrcmn = qfrcmn
        self.qfrcmx = qfrcmx
        # dataset 2g
        self.hlift = hlift
        self.liftq0 = liftq0
        self.liftqmax = liftqmax
        self.hwtol = hwtol
        # dataset 2h
        self.liftn = liftn
        self.qn = qn

        # dataset 4

        # accept stress period data (pumping rates) from structured array
        # does this need to be Mflist?
        self.stress_period_data = self.get_empty_stress_period_data(nper)
        if stress_period_data is not None:
            for n in stress_period_data.dtype.names:
                self.stress_period_data[n] = stress_period_data[n]

        # accept node data from structured array
        self.node_data = ModflowMnw2.get_empty_node_data(
            np.abs(nnodes), aux_names=self.aux
        )
        if node_data is not None:
            for n in node_data.dtype.names:
                self.node_data[n] = node_data[n]
                # convert strings to lower case
                if isinstance(n, str):
                    for idx, v in enumerate(self.node_data[n]):
                        self.node_data[n][idx] = self.node_data[n][idx]

        # build recarray of node data from MNW2 input file
        if node_data is None:
            self.make_node_data()
        else:
            self._set_attributes_from_node_data()

        for n in ["k", "i", "j"]:
            if len(self.__dict__[n]) > 0:
                # need to set for each period
                self.stress_period_data[n] = [self.__dict__[n][0]]

    def make_node_data(self):
        """
        Make the node data array from variables entered individually.

        Returns
        -------
        None

        """
        nnodes = self.nnodes
        node_data = ModflowMnw2.get_empty_node_data(
            np.abs(nnodes), aux_names=self.aux
        )

        names = Mnw.get_item2_names(self)
        for n in names:
            node_data[n] = self.__dict__[n]
        self.node_data = node_data

    @staticmethod
    def get_empty_stress_period_data(
        nper=0, aux_names=None, structured=True, default_value=0
    ):
        """
        Get an empty stress_period_data recarray that corresponds to dtype

        Parameters
        ----------
        nper : int

        aux_names
        structured
        default_value

        Returns
        -------
        ra : np.recarray
            Recarray

        """
        #
        dtype = Mnw.get_default_spd_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(nper, dtype, default_value=default_value)

    @staticmethod
    def get_default_spd_dtype(structured=True):
        """
        Get the default stress period data dtype

        Parameters
        ----------
        structured : bool
            Boolean that defines if a structured (True) or unstructured (False)
            dtype will be created (default is True). Not implemented for
            unstructured.

        Returns
        -------
        dtype : np.dtype

        """
        if structured:
            return np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("per", int),
                    ("qdes", np.float32),
                    ("capmult", int),
                    ("cprime", np.float32),
                    ("hlim", np.float32),
                    ("qcut", int),
                    ("qfrcmn", np.float32),
                    ("qfrcmx", np.float32),
                ]
            )
        else:
            msg = (
                "Mnw2: get_default_spd_dtype not implemented for "
                + "unstructured grids"
            )
            raise NotImplementedError(msg)

    @staticmethod
    def get_item2_names(mnw2obj=None, node_data=None):
        """
        Get names for unknown things...

        Parameters
        ----------
        mnw2obj : Mnw object
            Mnw object (default is None)
        node_data : unknown
            Unknown what is in this parameter (default is None)

        Returns
        -------
        names : list
        List of dtype names.

        """

        if node_data is not None:
            nnodes = Mnw.get_nnodes(node_data)
            losstype = node_data.losstype[0].lower()
            ppflag = node_data.ppflag[0]
            pumploc = node_data.pumploc[0]
            qlimit = node_data.qlimit[0]
            pumpcap = node_data.pumpcap[0]
            qcut = node_data.qcut[0]
        # get names based on mnw2obj attribute values
        else:
            nnodes = mnw2obj.nnodes
            losstype = mnw2obj.losstype.lower()
            ppflag = mnw2obj.ppflag
            pumploc = mnw2obj.pumploc
            qlimit = mnw2obj.qlimit
            pumpcap = mnw2obj.pumpcap
            qcut = mnw2obj.qcut

        names = ["i", "j"]
        if nnodes > 0:
            names += ["k"]
        if nnodes < 0:
            names += ["ztop", "zbotm"]
        names += [
            "wellid",
            "losstype",
            "pumploc",
            "qlimit",
            "ppflag",
            "pumpcap",
        ]
        if losstype.lower() == "thiem":
            names += ["rw"]
        elif losstype.lower() == "skin":
            names += ["rw", "rskin", "kskin"]
        elif losstype.lower() == "general":
            names += ["rw", "B", "C", "P"]
        elif losstype.lower() == "specifycwc":
            names += ["cwc"]
        if ppflag > 0 and nnodes > 0:
            names += ["pp"]
        if pumploc != 0:
            if pumploc > 0:
                names += ["pumplay", "pumprow", "pumpcol"]
            if pumploc < 0:
                names += ["zpump"]
        if qlimit > 0:
            names += ["hlim", "qcut"]
            if qcut != 0:
                names += ["qfrcmn", "qfrcmx"]
        if pumpcap > 0:
            names += ["hlift", "liftq0", "liftqmax", "hwtol"]
            names += ["liftn", "qn"]
        return names

    @staticmethod
    def get_nnodes(node_data):
        """
        Get the number of MNW2 nodes.

        Parameters
        ----------
        node_data : list
            List of nodes???

        Returns
        -------
        nnodes : int
            Number of MNW2 nodes

        """
        nnodes = len(node_data)
        # check if ztop and zbotm were entered,
        # flip nnodes for format 2
        if np.sum(node_data.ztop - node_data.zbotm) > 0:
            nnodes *= -1
        return nnodes

    @staticmethod
    def sort_node_data(node_data):
        # sort by layer (layer input option)
        if np.any(np.diff(node_data["k"]) < 0):
            node_data.sort(order=["k"])

        # reverse sort by ztop if it's specified and not sorted correctly
        if np.any(np.diff(node_data["ztop"]) > 0):
            node_data = np.sort(node_data, order=["ztop"])[::-1]
        return node_data

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check mnw object for common errors.

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
        chk : flopy.utils.check object

        """
        chk = self._get_check(f, verbose, level, checktype)
        if self.losstype.lower() not in [
            "none",
            "thiem",
            "skin",
            "general",
            "sepecifycwc",
        ]:
            chk._add_to_summary(
                type="Error",
                k=self.k,
                i=self.i,
                j=self.j,
                value=self.losstype,
                desc="Invalid losstype.",
            )

        chk.summarize()
        return chk

    def _get_check(self, f, verbose, level, checktype):
        if checktype is not None:
            return checktype(self, f=f, verbose=verbose, level=level)
        else:
            return check(self, f=f, verbose=verbose, level=level)

    def _set_attributes_from_node_data(self):
        """
        Populates the Mnw object attributes with values from node_data table.
        """
        names = Mnw.get_item2_names(node_data=self.node_data)
        for n in names:
            # assign by node variables as lists if they are being included
            if (
                n in self.by_node_variables
            ):  # and len(np.unique(self.node_data[n])) > 1:
                self.__dict__[n] = list(self.node_data[n])
            else:
                self.__dict__[n] = self.node_data[n][0]

    def _write_2(self, f_mnw, float_format=" {:15.7E}", indent=12):
        """
        Write out dataset 2 for MNW.

        Parameters
        ----------
        f_mnw : package file handle
            file handle for MNW2 input file
        float_format : str
            python format statement for floats (default is ' {:15.7E}').
        indent : int
            number of spaces to indent line (default is 12).

        Returns
        -------
        None

        """
        # enforce sorting of node data
        self.node_data = Mnw.sort_node_data(self.node_data)

        # update object attributes with values from node_data
        self._set_attributes_from_node_data()

        indent = " " * indent
        # dataset 2a
        fmt = "{} {:.0f}\n"
        f_mnw.write(fmt.format(self.wellid, self.nnodes))
        # dataset 2b
        fmt = indent + "{} {:.0f} {:.0f} {:.0f} {:.0f}\n"
        f_mnw.write(
            fmt.format(
                self.losstype,
                self.pumploc,
                self.qlimit,
                self.ppflag,
                self.pumpcap,
            )
        )

        # dataset 2c
        def _assign_by_node_var(var):
            """Assign negative number if variable is entered by node."""
            if len(np.unique(var)) > 1:
                return -1
            return var[0]

        if self.losstype.lower() != "none":
            if self.losstype.lower() != "specifycwc":
                fmt = indent + float_format + " "
                f_mnw.write(fmt.format(_assign_by_node_var(self.rw)))
                if self.losstype.lower() == "skin":
                    fmt = "{0} {0}".format(float_format)
                    f_mnw.write(
                        fmt.format(
                            _assign_by_node_var(self.rskin),
                            _assign_by_node_var(self.kskin),
                        )
                    )
                elif self.losstype.lower() == "general":
                    fmt = "{0} {0} {0}".format(float_format)
                    f_mnw.write(
                        fmt.format(
                            _assign_by_node_var(self.B),
                            _assign_by_node_var(self.C),
                            _assign_by_node_var(self.P),
                        )
                    )
            else:
                fmt = indent + float_format
                f_mnw.write(fmt.format(_assign_by_node_var(self.cwc)))
            f_mnw.write("\n")
        # dataset 2d
        if self.nnodes > 0:

            def _getloc(n):
                """Output for dataset 2d1."""
                return indent + "{:.0f} {:.0f} {:.0f}".format(
                    self.k[n] + 1, self.i[n] + 1, self.j[n] + 1
                )

        elif self.nnodes < 0:

            def _getloc(n):
                """Output for dataset 2d2."""
                fmt = (
                    indent + "{0} {0} ".format(float_format) + "{:.0f} {:.0f}"
                )
                return fmt.format(
                    self.node_data.ztop[n],
                    self.node_data.zbotm[n],
                    self.node_data.i[n] + 1,
                    self.node_data.j[n] + 1,
                )

        for n in range(np.abs(self.nnodes)):
            f_mnw.write(_getloc(n))
            for var in ["rw", "rskin", "kskin", "B", "C", "P", "cwc", "pp"]:
                val = self.__dict__[var]
                if val is None:
                    continue
                # only write variables by node if they are unique lists > length 1
                if len(np.unique(val)) > 1:
                    # if isinstance(val, list) or val < 0:
                    fmt = " " + float_format
                    f_mnw.write(fmt.format(self.node_data[var][n]))
            f_mnw.write("\n")
        # dataset 2e
        if self.pumploc != 0:
            if self.pumploc > 0:
                f_mnw.write(
                    indent
                    + "{:.0f} {:.0f} {:.0f}\n".format(
                        self.pumplay, self.pumprow, self.pumpcol
                    )
                )
            elif self.pumploc < 0:
                fmt = indent + "{}\n".format(float_format)
                f_mnw.write(fmt.format(self.zpump))
        # dataset 2f
        if self.qlimit > 0:
            fmt = indent + "{} ".format(float_format) + "{:.0f}"
            f_mnw.write(fmt.format(self.hlim, self.qcut))
            if self.qcut != 0:
                fmt = " {0} {0}".format(float_format)
                f_mnw.write(fmt.format(self.qfrcmn, self.qfrcmx))
            f_mnw.write("\n")
        # dataset 2g
        if self.pumpcap > 0:
            fmt = indent + "{0} {0} {0} {0}\n".format(float_format)
            f_mnw.write(
                fmt.format(self.hlift, self.liftq0, self.liftqmax, self.hwtol)
            )
        # dataset 2h
        if self.pumpcap > 0:
            fmt = indent + "{0} {0}\n".format(float_format)
            f_mnw.write(fmt.format(self.liftn, self.qn))


class ModflowMnw2(Package):
    """
    Multi-Node Well 2 Package Class

    Parameters
    ----------
    model : model object
        The model object (of type :class:'flopy.modflow.mf.Modflow') to which
        this package will be added.
    mnwmax : int
        The absolute value of MNWMAX is the maximum number of multi-node wells
        (MNW) to be simulated. If MNWMAX is a negative number, NODTOT is read.
    nodtot : int
        Maximum number of nodes.
        The code automatically estimates the maximum number of nodes (NODTOT)
        as required for allocation of arrays. However, if a large number of
        horizontal wells are being simulated, or possibly for other reasons,
        this default estimate proves to be inadequate, a new input option has
        been added to allow the user to directly specify a value for NODTOT.
        If this is a desired option, then it can be implemented by specifying
        a negative value for "MNWMAX"--the first value listed in Record 1
        (Line 1) of the MNW2 input data file. If this is done, then the code
        will assume that the very next value on that line will be the desired
        value of "NODTOT". The model will then reset "MNWMAX" to its absolute
        value. The value of "ipakcb" will become the third value on that
        line, etc.
    ipakcb : int
        is a flag and a unit number:
            if ipakcb > 0, then it is the unit number to which MNW cell-by-cell
            flow terms will be recorded whenever cell-by-cell budget data are
            written to a file (as determined by the outputcontrol options of
            MODFLOW).
            if ipakcb = 0, then MNW cell-by-cell flow terms will not be printed
                or recorded.
            if ipakcb < 0, then well injection or withdrawal rates and water
                levels in the well and its multiple cells will be printed in
                the main MODFLOW listing (output) file whenever cell-by-cell
                budget data are written to a file (as determined by the output
                control options of MODFLOW).
    mnwprnt : integer
        Flag controlling the level of detail of information about multi-node
        wells to be written to the main MODFLOW listing (output) file.
        If MNWPRNT = 0, then only basic well information will be printed in
        the main MODFLOW output file; increasing the value of MNWPRNT yields
        more information, up to a maximum level of detail corresponding
        with MNWPRNT = 2. (default is 0)
    aux : list of strings
        (listed as "OPTION" in MNW2 input instructions)
        is an optional list of character values in the style of "AUXILIARY abc"
        or "AUX abc" where "abc" is the name of an auxiliary parameter to be
        read for each multi-node well as part of dataset 4a. Up to 20
        parameters can be specified, each of which must be preceded by
        "AUXILIARY" or "AUX." These parameters will not be used by the MNW2
        Package, but they will be available for use by other packages.
        (default is None)
    node_data : numpy record array
        master table describing multi-node wells in package. Same format as
        node_data tables for each Mnw object. See Mnw class documentation for
        more information.
    mnw : list or dict of Mnw objects
        Can be supplied instead of node_data and stress_period_data tables
        (in which case the tables are constructed from the Mnw objects).
        Otherwise the a dict of Mnw objects (keyed by wellid) is constructed
        from the tables.
    stress_period_data : dict of numpy record arrays
        master dictionary of record arrays (keyed by stress period) containing
        transient input for multi-node wells. Format is the same as stress
        period data for individual Mnw objects, except the 'per' column is
        replaced by 'wellid' (containing wellid for each MNW). See Mnw class
        documentation for more information.
    itmp : list of ints
        is an integer value for reusing or reading multi-node well data; it
        can change each stress period. ITMP must be >= 0 for the first stress
        period of a simulation.
        if ITMP > 0, then ITMP is the total number of active multi-node wells
            simulated during the stress period, and only wells listed in
            dataset 4a will be active during the stress period. Characteristics
            of each well are defined in datasets 2 and 4.
        if ITMP = 0, then no multi-node wells are active for the stress period
            and the following dataset is skipped.
        if ITMP < 0, then the same number of wells and well information will
            be reused from the previous stress period and dataset 4 is skipped.
    extension : string
        Filename extension (default is 'mnw2')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.
    gwt : boolean
        Flag indicating whether GW transport process is active

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

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> mnw2 = flopy.modflow.ModflowMnw2(ml, ...)

    """

    def __init__(
        self,
        model,
        mnwmax=0,
        nodtot=None,
        ipakcb=0,
        mnwprnt=0,
        aux=[],
        node_data=None,
        mnw=None,
        stress_period_data=None,
        itmp=[],
        extension="mnw2",
        unitnumber=None,
        filenames=None,
        gwt=False,
    ):
        """
        Package constructor
        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowMnw2._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                ipakcb, fname=fname, package=ModflowMnw2._ftype()
            )
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowMnw2._ftype()]
        units = [unitnumber]
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

        self.url = "mnw2.htm"
        self.nper = self.parent.nrow_ncol_nlay_nper[-1]
        self.nper = (
            1 if self.nper == 0 else self.nper
        )  # otherwise iterations from 0, nper won't run
        self.structured = self.parent.structured

        # Dataset 0
        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        # Dataset 1
        # maximum number of multi-node wells to be simulated
        self.mnwmax = int(mnwmax)
        self.nodtot = nodtot  # user-specified maximum number of nodes
        self.ipakcb = ipakcb
        self.mnwprnt = int(mnwprnt)  # -verbosity flag
        self.aux = aux  # -list of optional auxiliary parameters

        # Datasets 2-4 are contained in node_data and stress_period_data tables
        # and/or in Mnw objects
        self.node_data = self.get_empty_node_data(0, aux_names=aux)

        if node_data is not None:
            self.node_data = self.get_empty_node_data(
                len(node_data), aux_names=aux
            )
            names = [
                n
                for n in node_data.dtype.names
                if n in self.node_data.dtype.names
            ]
            for n in names:
                self.node_data[n] = node_data[
                    n
                ]  # recarray of Mnw properties by node
            self.nodtot = len(self.node_data)
            self._sort_node_data()
            # self.node_data.sort(order=['wellid', 'k'])

            # Python 3.5.0 produces a segmentation fault when trying to sort BR MNW wells
            # self.node_data.sort(order='wellid', axis=0)
        self.mnw = mnw  # dict or list of Mnw objects

        self.stress_period_data = MfList(
            self,
            {
                0: self.get_empty_stress_period_data(
                    0, aux_names=aux, structured=self.structured
                )
            },
            dtype=self.get_default_spd_dtype(structured=self.structured),
        )
        if stress_period_data is not None:
            for per, data in stress_period_data.items():
                spd = ModflowMnw2.get_empty_stress_period_data(
                    len(data), aux_names=aux
                )
                names = [n for n in data.dtype.names if n in spd.dtype.names]
                for n in names:
                    spd[n] = data[n]
                spd.sort(order="wellid")
                self.stress_period_data[per] = spd

        self.itmp = itmp
        self.gwt = gwt

        if mnw is None:
            self.make_mnw_objects()
        elif node_data is None and mnw is not None:
            if isinstance(mnw, list):
                self.mnw = {mnwobj.wellid: mnwobj for mnwobj in mnw}
            elif isinstance(mnw, Mnw):
                self.mnw = {mnw.wellid: mnw}
            self.make_node_data(self.mnw)
            self.make_stress_period_data(self.mnw)

        if stress_period_data is not None:
            if (
                "k"
                not in stress_period_data[
                    list(stress_period_data.keys())[0]
                ].dtype.names
            ):
                self._add_kij_to_stress_period_data()

        self.parent.add_package(self)

    def _add_kij_to_stress_period_data(self):
        for per in self.stress_period_data.data.keys():
            for d in ["k", "i", "j"]:
                self.stress_period_data[per][d] = [
                    self.mnw[wellid].__dict__[d][0]
                    for wellid in self.stress_period_data[per].wellid
                ]

    def _sort_node_data(self):

        node_data = self.node_data
        node_data_list = []
        wells = sorted(np.unique(node_data["wellid"]).tolist())
        for wellid in wells:
            nd = node_data[node_data["wellid"] == wellid]
            nd = Mnw.sort_node_data(nd)
            node_data_list.append(nd)
        node_data = np.concatenate(node_data_list, axis=0)
        self.node_data = node_data.view(np.recarray)

    @staticmethod
    def get_empty_node_data(
        maxnodes=0, aux_names=None, structured=True, default_value=0
    ):
        """
        get an empty recarray that corresponds to dtype

        Parameters
        ----------
        maxnodes : int
            Total number of nodes to be simulated (default is 0)
        aux_names : list
            List of aux name strings (default is None)
        structured : bool
            Boolean indicating if a structured (True) or unstructured (False)
            model (default is True).
        default_value : float
            Default value for float variables (default is 0).

        Returns
        -------
        r : np.recarray
            Recarray of default dtype of shape maxnode
        """
        dtype = ModflowMnw2.get_default_node_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(
            maxnodes, dtype, default_value=default_value
        )

    @staticmethod
    def get_default_node_dtype(structured=True):
        """
        Get default dtype for node data

        Parameters
        ----------
        structured : bool
            Boolean indicating if a structured (True) or unstructured (False)
            model (default is True).

        Returns
        -------
        dtype : np.dtype
            node data dtype

        """
        if structured:
            return np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("ztop", np.float32),
                    ("zbotm", np.float32),
                    ("wellid", object),
                    ("losstype", object),
                    ("pumploc", int),
                    ("qlimit", int),
                    ("ppflag", int),
                    ("pumpcap", int),
                    ("rw", np.float32),
                    ("rskin", np.float32),
                    ("kskin", np.float32),
                    ("B", np.float32),
                    ("C", np.float32),
                    ("P", np.float32),
                    ("cwc", np.float32),
                    ("pp", np.float32),
                    ("pumplay", int),
                    ("pumprow", int),
                    ("pumpcol", int),
                    ("zpump", np.float32),
                    ("hlim", np.float32),
                    ("qcut", int),
                    ("qfrcmn", np.float32),
                    ("qfrcmx", np.float32),
                    ("hlift", np.float32),
                    ("liftq0", np.float32),
                    ("liftqmax", np.float32),
                    ("hwtol", np.float32),
                    ("liftn", np.float32),
                    ("qn", np.float32),
                ]
            )
        else:
            msg = "get_default_node_dtype: unstructured model not supported"
            raise NotImplementedError(msg)

    @staticmethod
    def get_empty_stress_period_data(
        itmp=0, aux_names=None, structured=True, default_value=0
    ):
        """
        Get an empty stress period data recarray

        Parameters
        ----------
        itmp : int
            Number of entries in this stress period (default is 0).
        aux_names : list
            List of aux names (default is None).
        structured : bool
            Boolean indicating if a structured (True) or unstructured (False)
            model (default is True).
        default_value : float
            Default value for float variables (default is 0).

        Returns
        -------
        r : np.recarray
            Recarray of default dtype of shape itmp

        """
        dtype = ModflowMnw2.get_default_spd_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(itmp, dtype, default_value=default_value)

    @staticmethod
    def get_default_spd_dtype(structured=True):
        """
        Get default dtype for stress period data

        Parameters
        ----------
        structured : bool
            Boolean indicating if a structured (True) or unstructured (False)
            model (default is True).

        Returns
        -------
        dtype : np.dtype
            node data dtype

        """
        if structured:
            return np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("wellid", object),
                    ("qdes", np.float32),
                    ("capmult", int),
                    ("cprime", np.float32),
                    ("hlim", np.float32),
                    ("qcut", int),
                    ("qfrcmn", np.float32),
                    ("qfrcmx", np.float32),
                ]
            )
        else:
            msg = "get_default_spd_dtype: unstructured model not supported"
            raise NotImplementedError(msg)

    @classmethod
    def load(cls, f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):
        """

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.ModflowMnw2`) to
            which this package will be added.
        nper : int
            Number of periods
        gwt : bool
        nsol : int
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.


        Returns
        -------

        """

        if model.verbose:
            sys.stdout.write("loading mnw2 package file...\n")

        structured = model.structured
        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
            nper = (
                1 if nper == 0 else nper
            )  # otherwise iterations from 0, nper won't run

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 (header)
        while True:
            line = get_next_line(f)
            if line[0] != "#":
                break
        # dataset 1
        mnwmax, nodtot, ipakcb, mnwprint, option = _parse_1(line)
        # dataset 2
        node_data = ModflowMnw2.get_empty_node_data(0)
        mnw = {}
        for i in range(mnwmax):
            # create a Mnw object by parsing dataset 2
            mnwobj = _parse_2(f)
            # populate stress period data table for each well object
            # this is filled below under dataset 4
            mnwobj.stress_period_data = Mnw.get_empty_stress_period_data(
                nper, aux_names=option
            )
            mnw[mnwobj.wellid] = mnwobj
            # master table with all node data
            node_data = np.append(node_data, mnwobj.node_data).view(
                np.recarray
            )

        stress_period_data = (
            {}
        )  # stress period data table for package (flopy convention)
        itmp = []
        for per in range(0, nper):
            # dataset 3
            itmp_per = int(line_parse(get_next_line(f))[0])
            # dataset4
            # dict might be better here to only load submitted values
            if itmp_per > 0:
                current_4 = ModflowMnw2.get_empty_stress_period_data(
                    itmp_per, aux_names=option
                )
                for i in range(itmp_per):
                    wellid, qdes, capmult, cprime, xyz = _parse_4a(
                        get_next_line(f), mnw, gwt=gwt
                    )
                    hlim, qcut, qfrcmn, qfrcmx = 0, 0, 0, 0
                    if mnw[wellid].qlimit < 0:
                        hlim, qcut, qfrcmn, qfrcmx = _parse_4b(
                            get_next_line(f)
                        )
                    # update package stress period data table
                    ndw = node_data[node_data.wellid == wellid]
                    kij = [ndw.k[0], ndw.i[0], ndw.j[0]]
                    current_4[i] = tuple(
                        kij
                        + [
                            wellid,
                            qdes,
                            capmult,
                            cprime,
                            hlim,
                            qcut,
                            qfrcmn,
                            qfrcmx,
                        ]
                        + xyz
                    )
                    # update well stress period data table
                    mnw[wellid].stress_period_data[per] = tuple(
                        kij
                        + [per]
                        + [qdes, capmult, cprime, hlim, qcut, qfrcmn, qfrcmx]
                        + xyz
                    )
                stress_period_data[per] = current_4
            elif itmp_per == 0:  # no active mnws this stress period
                pass
            else:
                # copy pumping rates from previous stress period
                mnw[wellid].stress_period_data[per] = mnw[
                    wellid
                ].stress_period_data[per - 1]
            itmp.append(itmp_per)

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowMnw2._ftype():
                    unitnumber = key
                    filenames[0] = os.path.basename(value.filename)

                if ipakcb > 0:
                    if key == ipakcb:
                        filenames[1] = os.path.basename(value.filename)
                        model.add_pop_key_list(key)

        return cls(
            model,
            mnwmax=mnwmax,
            nodtot=nodtot,
            ipakcb=ipakcb,
            mnwprnt=mnwprint,
            aux=option,
            node_data=node_data,
            mnw=mnw,
            stress_period_data=stress_period_data,
            itmp=itmp,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check mnw2 package data for common errors.

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
        chk : check object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.mnw2.check()
        """
        chk = self._get_check(f, verbose, level, checktype)

        # itmp
        if self.itmp[0] < 0:
            chk._add_to_summary(
                type="Error",
                value=self.itmp[0],
                desc="Itmp must be >= 0 for first stress period.",
            )
        invalid_itmp = np.array(self.itmp) > self.mnwmax
        if np.any(invalid_itmp):
            for v in np.array(self.itmp)[invalid_itmp]:
                chk._add_to_summary(
                    type="Error",
                    value=v,
                    desc="Itmp value greater than MNWMAX",
                )

        chk.summarize()
        return chk

    def get_allnode_data(self):
        """
        Get a version of the node_data array that has all MNW2 nodes listed
        explicitly. For example, MNWs with open intervals encompassing
        multiple layers would have a row entry for each layer. Ztop and zbotm
        values indicate the top and bottom elevations of the node (these are
        the same as the layer top and bottom if the node fully penetrates
        that layer).

        Returns
        -------
        allnode_data : np.recarray
            Numpy record array of same form as node_data, except each row
            represents only one node.

        """
        from numpy.lib.recfunctions import stack_arrays

        nd = []
        for i in range(len(self.node_data)):
            r = self.node_data[i]
            if r["ztop"] - r["zbotm"] > 0:
                startK = get_layer(self.parent.dis, r["i"], r["j"], r["ztop"])
                endK = get_layer(self.parent.dis, r["i"], r["j"], r["zbotm"])
                if startK == endK:
                    r = r.copy()
                    r["k"] = startK
                    nd.append(r)
                else:
                    for k in np.arange(startK, endK + 1):
                        rk = r.copy()
                        rk["k"] = k
                        if k > startK:
                            loc = (k - 1, rk["i"], rk["j"])
                            rk["ztop"] = self.parent.dis.botm[loc]
                        if k < endK:
                            loc = (k, rk["i"], rk["j"])
                            rk["zbotm"] = self.parent.dis.botm[loc]
                        nd.append(rk)
            else:
                nd.append(r)
        return stack_arrays(nd, usemask=False).view(np.recarray)

    def make_mnw_objects(self):
        """
        Make a Mnw object

        Returns
        -------
        None

        """
        node_data = self.node_data
        stress_period_data = self.stress_period_data
        self.mnw = {}
        mnws = np.unique(node_data["wellid"])
        for wellid in mnws:
            nd = node_data[node_data.wellid == wellid]
            nnodes = Mnw.get_nnodes(nd)
            # if tops and bottoms are specified, flip nnodes
            # maxtop = np.max(nd.ztop)
            # minbot = np.min(nd.zbotm)
            # if maxtop - minbot > 0 and nnodes > 0:
            #    nnodes *= -1
            # reshape stress period data to well
            mnwspd = Mnw.get_empty_stress_period_data(
                self.nper, aux_names=self.aux
            )
            for per, itmp in enumerate(self.itmp):
                inds = stress_period_data[per].wellid == wellid
                if itmp > 0 and np.any(inds):
                    names = [
                        n
                        for n in stress_period_data[per][inds].dtype.names
                        if n in mnwspd.dtype.names
                    ]
                    mnwspd[per]["per"] = per
                    for n in names:
                        mnwspd[per][n] = stress_period_data[per][inds][n][0]
                elif itmp == 0:
                    continue
                elif itmp < 0:
                    mnwspd[per] = mnwspd[per - 1]

            self.mnw[wellid] = Mnw(
                wellid,
                nnodes=nnodes,
                nper=self.nper,
                node_data=nd,
                stress_period_data=mnwspd,
                mnwpackage=self,
            )

    def make_node_data(self, mnwobjs):
        """
        Make node_data recarray from Mnw objects

        Parameters
        ----------
        mnwobjs : Mnw object

        Returns
        -------
        None

        """
        if isinstance(mnwobjs, dict):
            mnwobjs = list(mnwobjs.values())
        elif isinstance(mnwobjs, Mnw):
            mnwobjs = [mnwobjs]

        mnwobj_node_data = []
        for mnwobj in mnwobjs:
            for rec in mnwobj.node_data:
                mnwobj_node_data.append(rec)
        node_data = ModflowMnw2.get_empty_node_data(len(mnwobj_node_data))

        for ix, node in enumerate(mnwobj_node_data):
            for jx, name in enumerate(node_data.dtype.names):
                node_data[name][ix] = node[jx]

        self.node_data = node_data

    def make_stress_period_data(self, mnwobjs):
        """
        Make stress_period_data recarray from Mnw objects

        Parameters
        ----------
        mnwobjs : Mnw object

        Returns
        -------
        None

        """
        if isinstance(mnwobjs, dict):
            mnwobjs = list(mnwobjs.values())
        elif isinstance(mnwobjs, Mnw):
            mnwobjs = [mnwobjs]
        stress_period_data = {}
        for per, itmp in enumerate(self.itmp):
            if itmp > 0:
                stress_period_data[
                    per
                ] = ModflowMnw2.get_empty_stress_period_data(
                    itmp, aux_names=self.aux
                )
                i = 0
                for mnw in mnwobjs:
                    if per in mnw.stress_period_data.per:
                        i += 1
                        if i > itmp:
                            raise ItmpError(itmp, i)
                        names = [
                            n
                            for n in mnw.stress_period_data.dtype.names
                            if n in stress_period_data[per].dtype.names
                        ]
                        stress_period_data[per]["wellid"][i - 1] = mnw.wellid
                        for n in names:
                            stress_period_data[per][n][
                                i - 1
                            ] = mnw.stress_period_data[n][per]
                stress_period_data[per].sort(order="wellid")
                if i < itmp:
                    raise ItmpError(itmp, i)
            elif itmp == 0:
                continue
            else:  # itmp < 0
                stress_period_data[per] = stress_period_data[per - 1]
        self.stress_period_data = MfList(
            self, stress_period_data, dtype=stress_period_data[0].dtype
        )

    def export(self, f, **kwargs):
        """
        Export MNW2 data

        Parameters
        ----------
        f : file
        kwargs

        Returns
        -------
        e : export object


        """
        # A better strategy would be to build a single 4-D MfList
        # (currently the stress period data array has everything in layer 0)
        self.node_data_MfList = MfList(
            self, self.get_allnode_data(), dtype=self.node_data.dtype
        )
        # make some modifications to ensure proper export
        # avoid duplicate entries for qfrc
        wellids = np.unique(self.node_data.wellid)
        todrop = ["hlim", "qcut", "qfrcmn", "qfrcmx"]
        # move duplicate fields from node_data to stress_period_data
        for wellid in wellids:
            wellnd = self.node_data.wellid == wellid
            if np.max(self.node_data.qlimit[wellnd]) > 0:
                for per in self.stress_period_data.data.keys():
                    for col in todrop:
                        inds = self.stress_period_data[per].wellid == wellid
                        self.stress_period_data[per][col][
                            inds
                        ] = self.node_data[wellnd][col]
        self.node_data_MfList = self.node_data_MfList.drop(todrop)
        """
        todrop = {'qfrcmx', 'qfrcmn'}
        names = list(set(self.stress_period_data.dtype.names).difference(todrop))
        dtype = np.dtype([(k, d) for k, d in self.stress_period_data.dtype.descr if k not in todrop])
        spd = {}
        for k, v in self.stress_period_data.data.items():
            newarr = np.array(np.zeros_like(self.stress_period_data[k][names]),
                              dtype=dtype).view(np.recarray)
            for n in dtype.names:
                newarr[n] = self.stress_period_data[k][n]
            spd[k] = newarr
        self.stress_period_data = MfList(self, spd, dtype=dtype)
        """

        return super(ModflowMnw2, self).export(f, **kwargs)

    def _write_1(self, f_mnw):
        """

        Parameters
        ----------
        f_mnw : file object
            File object for MNW2 input file


        Returns
        -------
        None

        """
        f_mnw.write("{:.0f} ".format(self.mnwmax))
        if self.mnwmax < 0:
            f_mnw.write("{:.0f} ".format(self.nodtot))
        f_mnw.write("{:.0f} {:.0f}".format(self.ipakcb, self.mnwprnt))
        if len(self.aux) > 0:
            for abc in self.aux:
                f_mnw.write(" aux {}".format(abc))
        f_mnw.write("\n")

    def write_file(
        self, filename=None, float_format=" {:15.7E}", use_tables=True
    ):
        """
        Write the package file.

        Parameters
        ----------
        filename : str
        float_format
        use_tables

        Returns
        -------
        None

        """

        if use_tables:
            # update mnw objects from node and stress_period_data tables
            self.make_mnw_objects()

        if filename is not None:
            self.fn_path = filename

        f_mnw = open(self.fn_path, "w")

        # dataset 0 (header)
        f_mnw.write("{0}\n".format(self.heading))

        # dataset 1
        self._write_1(f_mnw)

        # dataset 2
        # need a method that assigns attributes from table to objects!
        # call make_mnw_objects?? (table is definitive then)
        if use_tables:
            mnws = np.unique(
                self.node_data.wellid
            ).tolist()  # preserve any order
        else:
            mnws = self.mnw.values()
        for k in mnws:
            self.mnw[k]._write_2(f_mnw, float_format=float_format)

        # dataset 3
        for per in range(self.nper):
            f_mnw.write(
                "{:.0f}  Stress Period {:.0f}\n".format(
                    self.itmp[per], per + 1
                )
            )
            if self.itmp[per] > 0:

                for n in range(self.itmp[per]):
                    # dataset 4
                    wellid = self.stress_period_data[per].wellid[n]
                    qdes = self.stress_period_data[per].qdes[n]
                    fmt = "{} " + float_format
                    f_mnw.write(fmt.format(wellid, qdes))
                    if self.mnw[wellid].pumpcap > 0:
                        fmt = " " + float_format
                        f_mnw.write(
                            fmt.format(
                                *self.stress_period_data[per].capmult[n]
                            )
                        )
                    if qdes > 0 and self.gwt:
                        f_mnw.write(
                            fmt.format(*self.stress_period_data[per].cprime[n])
                        )
                    if len(self.aux) > 0:
                        for var in self.aux:
                            fmt = " " + float_format
                            f_mnw.write(
                                fmt.format(
                                    *self.stress_period_data[per][var][n]
                                )
                            )
                    f_mnw.write("\n")
                    if self.mnw[wellid].qlimit < 0:
                        hlim, qcut = self.stress_period_data[per][
                            ["hlim", "qcut"]
                        ][n]
                        fmt = float_format + " {:.0f}"
                        f_mnw.write(fmt.format(hlim, qcut))
                        if qcut != 0:
                            fmt = " {} {}".format(float_format)
                            f_mnw.write(
                                fmt.format(
                                    *self.stress_period_data[per][
                                        ["qfrcmn", "qfrcmx"]
                                    ][n]
                                )
                            )
                        f_mnw.write("\n")
        f_mnw.close()

    @staticmethod
    def _ftype():
        return "MNW2"

    @staticmethod
    def _defaultunit():
        return 34


def _parse_1(line):
    """

    Parameters
    ----------
    line

    Returns
    -------

    """
    line = line_parse(line)
    mnwmax = pop_item(line, int)
    nodtot = None
    if mnwmax < 0:
        nodtot = pop_item(line, int)
    ipakcb = pop_item(line, int)
    mnwprint = pop_item(line, int)
    option = []  # aux names
    if len(line) > 0:
        option += [
            line[i]
            for i in np.arange(1, len(line))
            if "aux" in line[i - 1].lower()
        ]
    return mnwmax, nodtot, ipakcb, mnwprint, option


def _parse_2(f):
    """

    Parameters
    ----------
    f

    Returns
    -------

    """
    # dataset 2a
    line = line_parse(get_next_line(f))
    if len(line) > 2:
        warnings.warn(
            "MNW2: {}\n".format(line)
            + "Extra items in Dataset 2a!"
            + "Check for WELLIDs with space "
            + "but not enclosed in quotes."
        )
    wellid = pop_item(line).lower()
    nnodes = pop_item(line, int)
    # dataset 2b
    line = line_parse(get_next_line(f))
    losstype = pop_item(line)
    pumploc = pop_item(line, int)
    qlimit = pop_item(line, int)
    ppflag = pop_item(line, int)
    pumpcap = pop_item(line, int)

    # dataset 2c
    names = [
        "ztop",
        "zbotm",
        "k",
        "i",
        "j",
        "rw",
        "rskin",
        "kskin",
        "B",
        "C",
        "P",
        "cwc",
        "pp",
    ]
    d2d = {n: [] for n in names}  # dataset 2d; dict of lists for each variable
    # set default values of 0 for all 2c items
    d2dw = dict(zip(["rw", "rskin", "kskin", "B", "C", "P", "cwc"], [0] * 7))
    if losstype.lower() != "none":
        # update d2dw items
        d2dw.update(
            _parse_2c(get_next_line(f), losstype)
        )  # dict of values for well
        for k, v in d2dw.items():
            if v > 0:
                d2d[k].append(v)
    # dataset 2d
    pp = 1  # partial penetration flag
    for i in range(np.abs(nnodes)):
        line = line_parse(get_next_line(f))
        if nnodes > 0:
            d2d["k"].append(pop_item(line, int) - 1)
            d2d["i"].append(pop_item(line, int) - 1)
            d2d["j"].append(pop_item(line, int) - 1)
        elif nnodes < 0:
            d2d["ztop"].append(pop_item(line, float))
            d2d["zbotm"].append(pop_item(line, float))
            d2d["i"].append(pop_item(line, int) - 1)
            d2d["j"].append(pop_item(line, int) - 1)
        d2di = _parse_2c(
            line,
            losstype,
            rw=d2dw["rw"],
            rskin=d2dw["rskin"],
            kskin=d2dw["kskin"],
            B=d2dw["B"],
            C=d2dw["C"],
            P=d2dw["P"],
            cwc=d2dw["cwc"],
        )
        # append only the returned items
        for k, v in d2di.items():
            d2d[k].append(v)
        if ppflag > 0 and nnodes > 0:
            d2d["pp"].append(pop_item(line, float))

    # dataset 2e
    pumplay = None
    pumprow = None
    pumpcol = None
    zpump = None
    if pumploc != 0:
        line = line_parse(get_next_line(f))
        if pumploc > 0:
            pumplay = pop_item(line, int)
            pumprow = pop_item(line, int)
            pumpcol = pop_item(line, int)
        else:
            zpump = pop_item(line, float)
    # dataset 2f
    hlim = None
    qcut = None
    qfrcmx = None
    qfrcmn = None
    if qlimit > 0:
        # Only specify dataset 2f if the value of Qlimit in dataset 2b is positive.
        # Do not enter fractions as percentages.
        line = line_parse(get_next_line(f))
        hlim = pop_item(line, float)
        qcut = pop_item(line, int)
        if qcut != 0:
            qfrcmn = pop_item(line, float)
            qfrcmx = pop_item(line, float)
    # dataset 2g
    hlift = None
    liftq0 = None
    liftqmax = None
    hwtol = None
    if pumpcap > 0:
        # The number of additional data points on the curve (and lines in dataset 2h)
        # must correspond to the value of PUMPCAP for this well (where PUMPCAP <= 25).
        line = line_parse(get_next_line(f))
        hlift = pop_item(line, float)
        liftq0 = pop_item(line, float)
        liftqmax = pop_item(line, float)
        hwtol = pop_item(line, float)
    # dataset 2h
    liftn = None
    qn = None
    if pumpcap > 0:
        # Enter data in order of decreasing lift
        # (that is, start with the point corresponding
        # to the highest value of total dynamic head) and increasing discharge.
        # The discharge value for the last data point in the sequence
        # must be less than the value of LIFTqmax.
        for i in range(len(pumpcap)):
            line = line_parse(get_next_line(f))
            liftn = pop_item(line, float)
            qn = pop_item(line, float)

    return Mnw(
        wellid,
        nnodes=nnodes,
        losstype=losstype,
        pumploc=pumploc,
        qlimit=qlimit,
        ppflag=ppflag,
        pumpcap=pumpcap,
        k=d2d["k"],
        i=d2d["i"],
        j=d2d["j"],
        ztop=d2d["ztop"],
        zbotm=d2d["zbotm"],
        rw=d2d["rw"],
        rskin=d2d["rskin"],
        kskin=d2d["kskin"],
        B=d2d["B"],
        C=d2d["C"],
        P=d2d["P"],
        cwc=d2d["cwc"],
        pp=d2d["pp"],
        pumplay=pumplay,
        pumprow=pumprow,
        pumpcol=pumpcol,
        zpump=zpump,
        hlim=hlim,
        qcut=qcut,
        qfrcmn=qfrcmn,
        qfrcmx=qfrcmx,
        hlift=hlift,
        liftq0=liftq0,
        liftqmax=liftqmax,
        hwtol=hwtol,
        liftn=liftn,
        qn=qn,
    )


def _parse_2c(
    line, losstype, rw=-1, rskin=-1, kskin=-1, B=-1, C=-1, P=-1, cwc=-1
):
    """

    Parameters
    ----------
    line
    losstype
    rw
    rskin
    kskin
    B
    C
    P
    cwc

    Returns
    -------

    """
    if not isinstance(line, list):
        line = line_parse(line)
    nd = {}  # dict of dataset 2c/2d items
    if losstype.lower() != "specifycwc":
        if rw < 0:
            nd["rw"] = pop_item(line, float)
        if losstype.lower() == "skin":
            if rskin < 0:
                nd["rskin"] = pop_item(line, float)
            if kskin < 0:
                nd["kskin"] = pop_item(line, float)
        elif losstype.lower() == "general":
            if B < 0:
                nd["B"] = pop_item(line, float)
            if C < 0:
                nd["C"] = pop_item(line, float)
            if P < 0:
                nd["P"] = pop_item(line, float)
    else:
        if cwc < 0:
            nd["cwc"] = pop_item(line, float)
    return nd


def _parse_4a(line, mnw, gwt=False):
    """

    Parameters
    ----------
    line
    mnw
    gwt

    Returns
    -------

    """
    capmult = 0
    cprime = 0
    line = line_parse(line)
    wellid = pop_item(line).lower()
    pumpcap = mnw[wellid].pumpcap
    qdes = pop_item(line, float)
    if pumpcap > 0:
        capmult = pop_item(line, int)
    if qdes > 0 and gwt:
        cprime = pop_item(line, float)
    xyz = line
    return wellid, qdes, capmult, cprime, xyz


def _parse_4b(line):
    """

    Parameters
    ----------
    line

    Returns
    -------

    """
    qfrcmn = 0
    qfrcmx = 0
    line = line_parse(line)
    hlim = pop_item(line, float)
    qcut = pop_item(line, int)
    if qcut != 0:
        qfrcmn = pop_item(line, float)
        qfrcmx = pop_item(line, float)
    return hlim, qcut, qfrcmn, qfrcmx


class ItmpError(Exception):
    def __init__(self, itmp, nactivewells):
        self.itmp = itmp
        self.nactivewells = nactivewells

    def __str__(self):
        s = (
            "\n\nItmp value of {} ".format(self.itmp)
            + "is positive but does not equal the number of active wells "
            + "specified ({}). ".format(self.nactivewells)
            + "See MNW2 package documentation for details."
        )
        return s
