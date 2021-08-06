"""
mfuzf1 module.  Contains the ModflowUzf1 class. Note that the user can access
the ModflowUzf1 class as `flopy.modflow.ModflowUzf1`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?uzf_unsaturated_zone_flow_pack.htm>`_.

"""

import sys
import numpy as np
from ..utils.flopy_io import pop_item, line_parse
from ..pakbase import Package
from ..utils import Util2d, Transient2d
from ..utils.optionblock import OptionBlock
from collections import OrderedDict
import warnings


class ModflowUzf1(Package):
    """
    MODFLOW Unsaturated Zone Flow 1 Boundary Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    nuztop : integer
        used to define which cell in a vertical column that recharge and
        discharge is simulated. (default is 1)

        1   Recharge to and discharge from only the top model layer. This
            option assumes land surface is defined as top of layer 1.
        2   Recharge to and discharge from the specified layer in variable
            IUZFBND. This option assumes land surface is defined as top of
            layer specified in IUZFBND.
        3   Recharge to and discharge from the highest active cell in each
            vertical column. Land surface is determined as top of layer
            specified in IUZFBND. A constant head node intercepts any recharge
            and prevents deeper percolation.

    iuzfopt : integer
        equal to 1 or 2. A value of 1 indicates that the vertical hydraulic
        conductivity will be specified within the UZF1 Package input file using
        array VKS. A value of 2 indicates that the vertical hydraulic
        conductivity will be specified within either the BCF or LPF Package
        input file. (default is 0)
    irunflg : integer
        specifies whether ground water that discharges to land surface will
        be routed to stream segments or lakes as specified in the IRUNBND
        array (IRUNFLG not equal to zero) or if ground-water discharge is
        removed from the model simulation and accounted for in the
        ground-water budget as a loss of water (IRUNFLG=0). The
        Streamflow-Routing (SFR2) and(or) the Lake (LAK3) Packages must be
        active if IRUNFLG is not zero. (default is 0)
    ietflg : integer
        specifies whether or not evapotranspiration (ET) will be simulated.
        ET will not be simulated if IETFLG is zero, otherwise it will be
        simulated. (default is 0)
    ipakcb : integer
        flag for writing ground-water recharge, ET, and ground-water
        discharge to land surface rates to a separate unformatted file using
        subroutine UBUDSV. If ipakcb>0, it is the unit number to which the
        cell-by-cell rates will be written when 'SAVE BUDGET' or a non-zero
        value for ICBCFL is specified in Output Control. If ipakcb less than
        or equal to 0, cell-by-cell rates will not be written to a file.
        (default is 57)
    iuzfcb2 : integer
        flag for writing ground-water recharge, ET, and ground-water
        discharge to land surface rates to a separate unformatted file using
        module UBDSV3. If IUZFCB2>0, it is the unit number to which
        cell-by-cell rates will be written when 'SAVE BUDGET' or a non-zero
        value for ICBCFL is specified in Output Control. If IUZFCB2 less than
        or equal to 0, cell-by-cell rates will not be written to file.
        (default is 0)
    ntrail2 : integer
        equal to the number of trailing waves used to define the
        water-content profile following a decrease in the infiltration rate.
        The number of trailing waves varies depending on the problem, but a
        range between 10 and 20 is usually adequate. More trailing waves may
        decrease mass-balance error and will increase computational
        requirements and memory usage. (default is 10)
    nsets : integer
        equal to the number of wave sets used to simulate multiple
        infiltration periods. The number of wave sets should be set to 20 for
        most problems involving time varying infiltration. The total number of
        waves allowed within an unsaturated zone cell is equal to
        NTRAIL2*NSETS2. An error will occur if the number of waves in a cell
        exceeds this value. (default is 20)
    surfdep : float
        The average height of undulations, D (Figure 1 in UZF documentation),
        in the land surface altitude. (default is 1.0)
    iuzfbnd : integer
        used to define the aerial extent of the active model in which recharge
        and discharge will be simulated. (default is 1)
    irunbnd : integer
        used to define the stream segments within the Streamflow-Routing
        (SFR2) Package or lake numbers in the Lake (LAK3) Package that
        overland runoff from excess infiltration and ground-water
        discharge to land surface will be added. A positive integer value
        identifies the stream segment and a negative integer value identifies
        the lake number. (default is 0)
    vks : float
        used to define the saturated vertical hydraulic conductivity of the
        unsaturated zone (LT-1). (default is 1.0E-6)
    eps : float
        values for each model cell used to define the Brooks-Corey epsilon of
        the unsaturated zone. Epsilon is used in the relation of water
        content to hydraulic conductivity (Brooks and Corey, 1966).
        (default is 3.5)
    thts : float
        used to define the saturated water content of the unsaturated zone in
        units of volume of water to total volume (L3L-3). (default is 0.35)
    thtr : float
        used to define the residual water content for each vertical column of
        cells in units of volume of water to total volume (L3L-3). THTR is
        the irreducible water content and the unsaturated water content
        cannot drain to water contents less than THTR. This variable is not
        included unless the key word SPECIFYTHTR is specified. (default is
        0.15)
    thti : float
        used to define the initial water content for each vertical column of
        cells in units of volume of water at start of simulation to total
        volume (L3L-3). THTI should not be specified for steady-state
        simulations. (default is 0.20)
    row_col_iftunit_iuzopt : list
        used to specify where information will be printed for each time step.
        row and col are zero-based. IUZOPT specifies what that information
        will be. (default is [])
        IUZOPT is

        1   Prints time, ground-water head, and thickness of unsaturated zone,
            and cumulative volumes of infiltration, recharge, storage, change
            in storage and ground-water discharge to land surface.
        2   Same as option 1 except rates of infiltration, recharge, change in
            storage, and ground-water discharge also are printed.
        3   Prints time, ground-water head, thickness of unsaturated zone,
            followed by a series of depths and water contents in the
            unsaturated zone.

    nwt_11_fmt : boolean
        flag indicating whether or not to utilize a newer (MODFLOW-NWT
        version 1.1 or later) format style, i.e., uzf1 optional variables
        appear line-by-line rather than in a specific order on a single
        line. True means that optional variables (e.g., SPECIFYTHTR,
        SPECIFYTHTI, NOSURFLEAK) appear on new lines. True also supports
        a number of newer optional variables (e.g., SPECIFYSURFK,
        REJECTSURFK, SEEPSURFK). False means that optional variables
        appear on one line.  (default is False)
    specifythtr : boolean
        key word for specifying optional input variable THTR (default is 0)
    specifythti : boolean
        key word for specifying optional input variable THTI. (default is 0)
    nosurfleak : boolean
        key word for inactivating calculation of surface leakage.
        (default is 0)
    specifysurfk : boolean
        (MODFLOW-NWT version 1.1 and MODFLOW-2005 1.12 or later)
        An optional character variable. When SPECIFYSURFK is specified,
        the variable SURFK is specified in Data Set 4b.
    rejectsurfk : boolean
        (MODFLOW-NWT version 1.1 and MODFLOW-2005 1.12 or later)
        An optional character variable. When REJECTSURFK is specified,
        SURFK instead of VKS is used for calculating rejected infiltration.
        REJECTSURFK only is included if SPECIFYSURFK is included.
    seepsurfk : boolean
        (MODFLOW-NWT version 1.1 and MODFLOW-2005 1.12 or later)
        An optional character variable. When SEEPSURFK is specified,
        SURFK instead of VKS is used for calculating surface leakage.
        SEEPSURFK only is included if SPECIFYSURFK is included.
    etsquare : float (smoothfact)
        (MODFLOW-NWT version 1.1 and MODFLOW-2005 1.12 or later)
        An optional character variable. When ETSQUARE is specified,
        groundwater ET is simulated using a constant potential ET rate,
        and is smoothed over a specified smoothing interval.
        This option is recommended only when using the NWT solver.

        etsquare is activated in flopy by specifying a real value
        for smoothfact (default is None).
        For example, if the interval factor (smoothfact)
        is specified as smoothfact=0.1 (recommended),
        then the smoothing interval will be calculated as:
        SMOOTHINT = 0.1*EXTDP and  is applied over the range for groundwater
        head (h):
        *   h < CELTOP-EXTDP, ET is zero;
        *   CELTOP-EXTDP < h < CELTOP-EXTDP+SMOOTHINT, ET is smoothed;
        CELTOP-EXTDP+SMOOTHINT < h, ET is equal to potential ET.
    uzgage : dict of lists or list of lists
        Dataset 8 in UZF Package documentation. Each entry in the dict
        is keyed by iftunit.
            Dict of lists: If iftunit is negative, the list is empty.
            If iftunit is positive, the list includes [IUZROW, IUZCOL, IUZOPT]
            List of lists:
            Lists follow the format described in the documentation:
            [[IUZROW, IUZCOL, IFTUNIT, IUZOPT]] or [[-IFTUNIT]]
    netflux : list of [Unitrech (int), Unitdis (int)]
        (MODFLOW-NWT version 1.1 and MODFLOW-2005 1.12 or later)
        An optional character variable. When NETFLUX is specified,
        the sum of recharge (L3/T) and the sum of discharge (L3/T) is written
        to separate unformatted files using module UBDSV3.

        netflux is activated in flopy by specifying a list for
        Unitrech and Unitdis (default is None).
        Unitrech and Unitdis are the unit numbers to which these values
        are written when “SAVE BUDGET” is specified in Output Control.
        Values written to Unitrech are the sum of recharge values
        for the UZF, SFR2, and LAK packages, and values written to Unitdis
        are the sum of discharge values for the UZF, SFR2, and LAK packages.
        Values are averaged over the period between output times.

        [NETFLUX unitrech unitdis]
    finf : float, 2-D array, or dict of {kper:value}
        where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
        Used to define the infiltration rates (LT-1) at land surface for each
        vertical column of cells. If FINF is specified as being greater than
        the vertical hydraulic conductivity then FINF is set equal to the
        vertical unsaturated hydraulic conductivity. Excess water is routed
        to streams or lakes when IRUNFLG is not zero, and if SFR2 or LAK3 is
        active. (default is 1.0E-8)
    pet : float, 2-D array, or dict of {kper:value}
        where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
        Used to define the ET demand rates (L1T-1) within the ET extinction
        depth interval for each vertical column of cells. (default is 5.0E-8)
    extdp : float, 2-D array, or dict of {kper:value}
        where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
        Used to define the ET extinction depths. The quantity of ET removed
        from a cell is limited by the volume of water stored in the
        unsaturated zone above the extinction depth. If ground water is
        within the ET extinction depth, then the rate removed is based
        on a linear decrease in the maximum rate at land surface and zero at
        the ET extinction depth. The linear decrease is the same method used
        in the Evapotranspiration Package (McDonald and Harbaugh, 1988, chap.
        10). (default is 15.0)
    extwc : float, 2-D array, or dict of {kper:value}
        where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
        Used to define the extinction water content below which ET cannot be
        removed from the unsaturated zone.  EXTWC must have a value between
        (THTS-Sy) and THTS, where Sy is the specific yield specified in
        either the LPF or BCF Package. (default is 0.1)
     air_entry : float, 2-D array, or dict of {kper: value}
        where kper is the zero-based stress period.
        Used to define the air entry pressure of the unsaturated zone in
        MODFLOW-NWT simulations
    hroot : float, 2-D array, or dict of {kper: value}
        where kper is the zero-based stress period.
        Used to define the pressure potential at roots in the unsaturated
        zone in MODFLOW-NWT simulations
    rootact : float, 2-D array, or dict of {kper: value}
        where kper is the zero-based stress period.
        Used to define the root activity in the unsaturated
        zone in MODFLOW-NWT simulations
    uzfbud_ext : list
        appears to be used for sequential naming of budget output files
        (default is [])
    extension : string
        Filename extension (default is 'uzf')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output, uzf output, and uzf
        observation names will be created using the model name and .cbc,
        uzfcb2.bin, and  .uzf#.out extensions (for example, modflowtest.cbc,
        and modflowtest.uzfcd2.bin), if ipakcbc, iuzfcb2, and len(uzgag) are
        numbers greater than zero. For uzf observations the file extension is
        created using the uzf observation file unit number (for example, for
        uzf observations written to unit 123 the file extension would be
        .uzf123.out). If a single string is passed the package name will be
        set to the string and other uzf output files will be set to the model
        name with the appropriate output file extensions. To define the names
        for all package files (input and output) the length of the list of
        strings should be 3 + len(uzgag). Default is None.
    surfk : float
        An optional array of positive real values used to define the hydraulic
        conductivity (LT-1). SURFK is used for calculating the rejected
        infiltration and/or surface leakage. IF SURFK is set greater than
        VKS then it is set equal to VKS. Only used if SEEPSURFK is True.

    Attributes
    ----------
    nuzgag : integer (deprecated - counter is set based on length of uzgage)
        equal to the number of cells (one per vertical column) that will be
        specified for printing detailed information on the unsaturated zone
        water budget and water content. A gage also may be used to print
        the budget summed over all model cells.  (default is None)

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> uzf = flopy.modflow.ModflowUzf1(ml, ...)

    """

    _options = OrderedDict(
        [
            ("specifythtr", OptionBlock.simple_flag),
            ("specifythti", OptionBlock.simple_flag),
            ("nosurfleak", OptionBlock.simple_flag),
            ("specifysurfk", OptionBlock.simple_flag),
            ("rejectsurfk", OptionBlock.simple_flag),
            ("seepsurfk", OptionBlock.simple_flag),
            ("capillaryuzet", OptionBlock.simple_flag),
            (
                "etsquare",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 1,
                    OptionBlock.vars: {"smoothfact": OptionBlock.simple_float},
                },
            ),
            (
                "netflux",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 2,
                    OptionBlock.vars: OrderedDict(
                        [
                            ("unitrech", OptionBlock.simple_int),
                            ("unitdis", OptionBlock.simple_int),
                        ]
                    ),
                },
            ),
            ("savefinf", OptionBlock.simple_flag),
        ]
    )

    def __init__(
        self,
        model,
        nuztop=1,
        iuzfopt=0,
        irunflg=0,
        ietflg=0,
        ipakcb=None,
        iuzfcb2=None,
        ntrail2=10,
        nsets=20,
        surfdep=1.0,
        iuzfbnd=1,
        irunbnd=0,
        vks=1.0e-6,
        eps=3.5,
        thts=0.35,
        thtr=0.15,
        thti=0.20,
        specifythtr=False,
        specifythti=False,
        nosurfleak=False,
        finf=1.0e-8,
        pet=5.0e-8,
        extdp=15.0,
        extwc=0.1,
        air_entry=0.0,
        hroot=0.0,
        rootact=0.0,
        nwt_11_fmt=False,
        specifysurfk=False,
        rejectsurfk=False,
        seepsurfk=False,
        etsquare=None,
        netflux=None,
        capillaryuzet=False,
        nuzgag=None,
        uzgag=None,
        extension="uzf",
        unitnumber=None,
        filenames=None,
        options=None,
        surfk=0.1,
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowUzf1._defaultunit()

        # set filenames
        nlen = 3
        if uzgag is not None:
            nlen += len(uzgag)
        if filenames is None:
            filenames = [None for x in range(nlen)]
        elif isinstance(filenames, str):
            filenames = [filenames] + [None for x in range(nlen)]
        elif isinstance(filenames, list):
            if len(filenames) < nlen:
                for idx in range(len(filenames), nlen + 1):
                    filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                abs(ipakcb), fname=fname, package=ModflowUzf1._ftype()
            )
        else:
            ipakcb = 0

        if iuzfcb2 is not None:
            fname = filenames[2]
            model.add_output_file(
                abs(iuzfcb2),
                fname=fname,
                extension="uzfcb2.bin",
                package=ModflowUzf1._ftype(),
            )
        else:
            iuzfcb2 = 0

        ipos = 3
        if uzgag is not None:
            # convert to dict
            if isinstance(uzgag, list):
                d = {}
                for l in uzgag:
                    if len(l) > 1:
                        d[l[2]] = [l[0], l[1], l[3]]
                    else:
                        d[-np.abs(l[0])] = []
                uzgag = d
            for key, value in uzgag.items():
                fname = filenames[ipos]
                iu = abs(key)
                uzgagext = "uzf{}.out".format(iu)
                model.add_output_file(
                    iu,
                    fname=fname,
                    binflag=False,
                    extension=uzgagext,
                    package=ModflowUzf1._ftype(),
                )
                ipos += 1
                # handle case where iftunit is listed in the values
                # (otherwise, iftunit will be written instead of iuzopt)
                if len(value) == 4:
                    uzgag[key] = value[:2] + value[-1:]
                elif len(value) == 1:
                    uzgag[-np.abs(key)] = []

        # Fill namefile items
        name = [ModflowUzf1._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        if (
            self.parent.get_package("RCH") != None
            or self.parent.get_package("EVT") != None
        ):
            print(
                "WARNING!\n The RCH and EVT packages should not be "
                "active when the UZF1 package is active!"
            )
        if self.parent.version == "mf2000":
            print(
                "WARNING!\nThe UZF1 package is only compatible "
                "with MODFLOW-2005 and MODFLOW-NWT!"
            )
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.url = "uzf_unsaturated_zone_flow_pack.htm"

        # Data Set 1a
        if nwt_11_fmt:
            warnings.warn(
                "nwt_11_fmt has been deprecated,"
                " and will be removed in the next release"
                " please provide a flopy.utils.OptionBlock object"
                " to the options argument",
                DeprecationWarning,
            )
        self.nwt_11_fmt = nwt_11_fmt
        self.specifythtr = bool(specifythtr)
        self.specifythti = bool(specifythti)
        self.nosurfleak = bool(nosurfleak)
        self.specifysurfk = bool(specifysurfk)
        self.rejectsurfk = bool(rejectsurfk)
        self.seepsurfk = bool(seepsurfk)
        self.capillaryuzet = bool(capillaryuzet)
        self.etsquare = False
        self.smoothfact = None
        if etsquare is not None:
            try:
                float(etsquare)
            except:
                print(
                    "etsquare must be specified by entering a real "
                    "number for smoothfact."
                )
            self.etsquare = True
            self.smoothfact = etsquare
        self.netflux = False
        self.unitrech = None
        self.unitdis = None
        if netflux is not None:
            e = "netflux must be a length=2 sequence of unitrech, unitdis"
            assert len(netflux) == 2, e
            self.netflux = True
            self.unitrech, self.unitdis = netflux

        if options is None:
            if (
                specifythti,
                specifythtr,
                nosurfleak,
                specifysurfk,
                rejectsurfk,
                seepsurfk,
                self.etsquare,
                self.netflux,
            ) != (False, False, False, False, False, False, False, False):
                options = OptionBlock("", ModflowUzf1, block=False)
                if "nwt" in self.parent.version:
                    options.block = True

        self.options = options

        # Data Set 1b
        # NUZTOP IUZFOPT IRUNFLG IETFLG ipakcb IUZFCB2 [NTRAIL2 NSETS2] NUZGAG SURFDEP
        self.nuztop = nuztop
        self.iuzfopt = iuzfopt
        # The Streamflow-Routing (SFR2) and(or) the Lake (LAK3) Packages
        # must be active if IRUNFLG is not zero.
        self.irunflg = irunflg
        self.ietflg = ietflg
        self.ipakcb = ipakcb
        self.iuzfcb2 = iuzfcb2
        if iuzfopt > 0:
            self.ntrail2 = ntrail2
            self.nsets = nsets
        self.surfdep = surfdep

        # Data Set 2
        # IUZFBND (NCOL, NROW) -- U2DINT
        self.iuzfbnd = Util2d(
            model, (nrow, ncol), np.int32, iuzfbnd, name="iuzfbnd"
        )

        # If IRUNFLG > 0: Read item 3
        # Data Set 3
        # [IRUNBND (NCOL, NROW)] -- U2DINT
        if irunflg > 0:
            self.irunbnd = Util2d(
                model, (nrow, ncol), np.int32, irunbnd, name="irunbnd"
            )

        # IF the absolute value of IUZFOPT = 1: Read item 4.
        # Data Set 4
        # [VKS (NCOL, NROW)] -- U2DREL
        if abs(iuzfopt) in [0, 1]:
            self.vks = Util2d(model, (nrow, ncol), np.float32, vks, name="vks")

        if seepsurfk or specifysurfk:
            self.surfk = Util2d(
                model, (nrow, ncol), np.float32, surfk, name="surfk"
            )

        if iuzfopt > 0:
            # Data Set 5
            # EPS (NCOL, NROW) -- U2DREL
            self.eps = Util2d(model, (nrow, ncol), np.float32, eps, name="eps")
            # Data Set 6a
            # THTS (NCOL, NROW) -- U2DREL
            self.thts = Util2d(
                model, (nrow, ncol), np.float32, thts, name="thts"
            )
            # Data Set 6b
            # THTS (NCOL, NROW) -- U2DREL
            if self.specifythtr > 0:
                self.thtr = Util2d(
                    model, (nrow, ncol), np.float32, thtr, name="thtr"
                )
            # Data Set 7
            # [THTI (NCOL, NROW)] -- U2DREL
            self.thti = Util2d(
                model, (nrow, ncol), np.float32, thti, name="thti"
            )

        # Data Set 8
        # {IFTUNIT: [IUZROW, IUZCOL, IUZOPT]}
        self._uzgag = uzgag

        # Dataset 9, 11, 13 and 15 will be written automatically in the
        # write_file function
        # Data Set 10
        # [FINF (NCOL, NROW)] – U2DREL

        self.finf = Transient2d(
            model, (nrow, ncol), np.float32, finf, name="finf"
        )
        if ietflg > 0:
            self.pet = Transient2d(
                model, (nrow, ncol), np.float32, pet, name="pet"
            )
            self.extdp = Transient2d(
                model, (nrow, ncol), np.float32, extdp, name="extdp"
            )
            self.extwc = Transient2d(
                model, (nrow, ncol), np.float32, extwc, name="extwc"
            )

        if capillaryuzet and "nwt" in model.version:
            self.air_entry = Transient2d(
                model, (nrow, ncol), np.float32, air_entry, name="air_entry"
            )
            self.hroot = Transient2d(
                model, (nrow, ncol), np.float32, hroot, name="hroot"
            )
            self.rootact = Transient2d(
                model, (nrow, ncol), np.float32, rootact, name="rootact"
            )

        self.parent.add_package(self)

    def __setattr__(self, key, value):
        if key == "uzgag":
            print(
                "Uzgag must be set by the constructor modifying this "
                "attribute requires creating a new ModflowUzf1 instance"
            )
        else:
            super().__setattr__(key, value)

    @property
    def nuzgag(self):
        """Number of uzf gages

        Returns
        -------
        nuzgag : int
            Number of uzf gages

        """
        if self.uzgag is None:
            return 0
        else:
            return len(self.uzgag)

    @property
    def uzgag(self):
        """Get the uzf gage data

        Returns
        -------
        uzgag : dict
            Dictionary containing uzf gage data for each gage

        """
        return self._uzgag

    def _2list(self, arg):
        # input as a 3D array
        if isinstance(arg, np.ndarray) and len(arg.shape) == 3:
            lst = [arg[per, :, :] for per in range(arg.shape[0])]
        # input is not a 3D array, and not a list
        # (could be numeric value or 2D array)
        elif not isinstance(arg, list):
            lst = [arg]
        # input was already a list
        else:
            lst = arg
        return lst

    def _ncells(self):
        """Maximum number of cells that have uzf (developed for
        MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of uzf cells

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        return nrow * ncol

    def _write_1a(self, f_uzf):

        # the nwt_11_fmt code is slated for removal (deprecated!)
        if not self.nwt_11_fmt:
            specify_temp = ""
            if self.specifythtr > 0:
                specify_temp += "SPECIFYTHTR "
            if self.specifythti > 0:
                specify_temp += "SPECIFYTHTI "
            if self.nosurfleak > 0:
                specify_temp += "NOSURFLEAK"
            if (self.specifythtr + self.specifythti + self.nosurfleak) > 0:
                f_uzf.write("{}\n".format(specify_temp))
            del specify_temp
        else:
            txt = "options\n"
            for var in [
                "specifythtr",
                "specifythti",
                "nosurfleak",
                "specifysurfk",
                "rejectsurfk",
                "seepsurfk",
            ]:
                value = self.__dict__[var]
                if int(value) > 0:
                    txt += "{}\n".format(var)
            if self.etsquare:
                txt += "etsquare {}\n".format(self.smoothfact)
            if self.netflux:
                txt += "netflux {} {}\n".format(self.unitrech, self.unitdis)
            txt += "end\n"
            f_uzf.write(txt)

    def write_file(self, f=None):
        """
        Write the package file.

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        if f is not None:
            if isinstance(f, str):
                f_uzf = open(f, "w")
            else:
                f_uzf = f
        else:
            f_uzf = open(self.fn_path, "w")
        f_uzf.write("{}\n".format(self.heading))

        # Dataset 1a
        if (
            isinstance(self.options, OptionBlock)
            and self.parent.version == "mfnwt"
        ):
            self.options.update_from_package(self)
            self.options.write_options(f_uzf)

        else:
            self._write_1a(f_uzf)

        # Dataset 1b
        if self.iuzfopt > 0:
            comment = " #NUZTOP IUZFOPT IRUNFLG IETFLG ipakcb IUZFCB2 NTRAIL NSETS NUZGAGES"
            f_uzf.write(
                "{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}{6:10d}{7:10d}{8:10d}{9:15.6E}{10:100s}\n".format(
                    self.nuztop,
                    self.iuzfopt,
                    self.irunflg,
                    self.ietflg,
                    self.ipakcb,
                    self.iuzfcb2,
                    self.ntrail2,
                    self.nsets,
                    self.nuzgag,
                    self.surfdep,
                    comment,
                )
            )
        else:
            comment = " #NUZTOP IUZFOPT IRUNFLG IETFLG ipakcb IUZFCB2 NUZGAGES"
            f_uzf.write(
                "{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}{6:10d}{7:15.6E}{8:100s}\n".format(
                    self.nuztop,
                    self.iuzfopt,
                    self.irunflg,
                    self.ietflg,
                    self.ipakcb,
                    self.iuzfcb2,
                    self.nuzgag,
                    self.surfdep,
                    comment,
                )
            )
        f_uzf.write(self.iuzfbnd.get_file_entry())
        if self.irunflg > 0:
            f_uzf.write(self.irunbnd.get_file_entry())
        # IF the absolute value of IUZFOPT = 1: Read item 4.
        # Data Set 4
        # [VKS (NCOL, NROW)] -- U2DREL
        if abs(self.iuzfopt) in [0, 1]:
            f_uzf.write(self.vks.get_file_entry())

        # Dataset 4b modflow 2005 v. 1.12 and modflow-nwt v. 1.1
        if self.seepsurfk or self.specifysurfk:
            f_uzf.write(self.surfk.get_file_entry())

        if self.iuzfopt > 0:
            # Data Set 5
            # EPS (NCOL, NROW) -- U2DREL
            f_uzf.write(self.eps.get_file_entry())
            # Data Set 6a
            # THTS (NCOL, NROW) -- U2DREL
            f_uzf.write(self.thts.get_file_entry())
            # Data Set 6b
            # THTR (NCOL, NROW) -- U2DREL
            if self.specifythtr > 0.0:
                f_uzf.write(self.thtr.get_file_entry())
            # Data Set 7
            # [THTI (NCOL, NROW)] -- U2DREL
            if (
                not self.parent.get_package("DIS").steady[0]
                or self.specifythti > 0.0
            ):
                f_uzf.write(self.thti.get_file_entry())
        # If NUZGAG>0: Item 8 is repeated NUZGAG times
        # Data Set 8
        # [IUZROW] [IUZCOL] IFTUNIT [IUZOPT]
        if self.nuzgag > 0:
            for iftunit, values in self.uzgag.items():
                if iftunit > 0:
                    values[0] += 1
                    values[1] += 1
                    comment = " #IUZROW IUZCOL IFTUNIT IUZOPT"
                    values.insert(2, iftunit)
                    for v in values:
                        f_uzf.write("{:10d}".format(v))
                    f_uzf.write("{}\n".format(comment))
                else:
                    comment = " #IFTUNIT"
                    f_uzf.write("{:10d}".format(iftunit))
                    f_uzf.write("{}\n".format(comment))

        def write_transient(name):
            invar, var = self.__dict__[name].get_kper_entry(n)

            comment = " #{} for stress period ".format(name) + str(n + 1)
            f_uzf.write("{0:10d}{1:20s}\n".format(invar, comment))
            if invar >= 0:
                f_uzf.write(var)

        for n in range(nper):
            write_transient("finf")
            if self.ietflg > 0:
                write_transient("pet")
                write_transient("extdp")
                if self.iuzfopt > 0:
                    write_transient("extwc")
            if self.capillaryuzet and "nwt" in self.parent.version:
                write_transient("air_entry")
                write_transient("hroot")
                write_transient("rootact")

        f_uzf.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None, check=False):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        uzf : ModflowUZF1 object
            ModflowUZF1 object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> uzf = flopy.modflow.ModflowUZF1.load('test.uzf', m)

        """
        if model.verbose:
            sys.stdout.write("loading uzf package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        # determine problem dimensions
        nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # dataset 1a
        specifythtr = False
        specifythti = False
        nosurfleak = False
        specifysurfk = False
        capillaryuzet = False
        etsquare = None
        netflux = None
        rejectsurfk = False
        seepsurfk = False
        options = None
        if model.version == "mfnwt" and "options" in line.lower():
            options = OptionBlock.load_options(f, ModflowUzf1)
            line = f.readline()

        else:
            query = (
                "specifythtr",
                "specifythti",
                "nosurfleak",
                "specifysurfk",
                "rejectsurfk",
                "seepsurfk",
                "etsquare",
                "netflux",
                "savefinf",
            )
            for i in query:
                if i in line.lower():
                    options = OptionBlock(
                        line.lower().strip(), ModflowUzf1, block=False
                    )
                    line = f.readline()
                    break

        if options is not None:
            specifythtr = options.specifythtr
            specifythti = options.specifythti
            nosurfleak = options.nosurfleak
            rejectsurfk = options.rejectsurfk
            seepsurfk = options.seepsurfk
            specifysurfk = options.specifysurfk
            capillaryuzet = options.capillaryuzet

            if options.etsquare:
                etsquare = options.smoothfact
            if options.netflux:
                netflux = [options.unitrech, options.unitdis]

        # dataset 1b
        (
            nuztop,
            iuzfopt,
            irunflg,
            ietflg,
            ipakcb,
            iuzfcb2,
            ntrail2,
            nsets2,
            nuzgag,
            surfdep,
        ) = _parse1(line)

        arrays = {
            "finf": {},
            # datasets 10, 12, 14, 16 are lists of util2d arrays
            "pet": {},
            "extdp": {},
            "extwc": {},
            "air_entry": {},
            "hroot": {},
            "rootact": {},
        }

        def load_util2d(name, dtype, per=None):
            print("   loading {} array...".format(name))
            if per is not None:
                arrays[name][per] = Util2d.load(
                    f, model, (nrow, ncol), dtype, name, ext_unit_dict
                )
            else:
                arrays[name] = Util2d.load(
                    f, model, (nrow, ncol), dtype, name, ext_unit_dict
                )

        # dataset 2
        load_util2d("iuzfbnd", np.int32)

        # dataset 3
        if irunflg > 0:
            load_util2d("irunbnd", np.int32)

        # dataset 4
        if iuzfopt in [0, 1]:
            load_util2d("vks", np.float32)

        # dataset 4b
        if seepsurfk or specifysurfk:
            load_util2d("surfk", np.float32)

        if iuzfopt > 0:
            # dataset 5
            load_util2d("eps", np.float32)

            # dataset 6
            load_util2d("thts", np.float32)

            if specifythtr:
                # dataset 6b (residual water content)
                load_util2d("thtr", np.float32)

            if specifythti or np.all(~model.dis.steady.array):
                # dataset 7 (initial water content;
                #            only read if not steady-state)
                load_util2d("thti", np.float32)

        # dataset 8
        uzgag = {}
        if nuzgag > 0:
            for i in range(nuzgag):
                iuzrow, iuzcol, iftunit, iuzopt = _parse8(f.readline())
                tmp = [iuzrow, iuzcol] if iftunit > 0 else []
                tmp.append(iftunit)
                if iuzopt > 0:
                    tmp.append(iuzopt)
                uzgag[iftunit] = tmp

        # dataset 9
        for per in range(nper):
            print("stress period {}:".format(per + 1))
            line = line_parse(f.readline())
            nuzf1 = pop_item(line, int)

            # dataset 10
            if nuzf1 >= 0:
                load_util2d("finf", np.float32, per=per)

            if ietflg > 0:
                # dataset 11
                line = line_parse(f.readline())
                nuzf2 = pop_item(line, int)
                if nuzf2 >= 0:
                    # dataset 12
                    load_util2d("pet", np.float32, per=per)
                # dataset 13
                line = line_parse(f.readline())
                nuzf3 = pop_item(line, int)
                if nuzf3 >= 0:
                    # dataset 14
                    load_util2d("extdp", np.float32, per=per)
                # dataset 15
                line = line_parse(f.readline())
                nuzf4 = pop_item(line, int)
                if nuzf4 >= 0:
                    # dataset 16
                    load_util2d("extwc", np.float32, per=per)

                if capillaryuzet:
                    # dataset 17
                    line = line_parse(f.readline())
                    nuzf5 = pop_item(line, int)
                    if nuzf5 > 0:
                        # dataset 18
                        load_util2d("air_entry", np.float32, per=per)

                    # dataset 19
                    line = line_parse(f.readline())
                    nuzf6 = pop_item(line, int)
                    if nuzf6 > 0:
                        # dataset 20
                        load_util2d("hroot", np.float32, per=per)

                    # dataset21
                    line = line_parse(f.readline())
                    nuzf7 = pop_item(line, int)
                    if nuzf7 > 0:
                        # dataset 22
                        load_util2d("rootact", np.float32, per=per)

        # close the file
        f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None for x in range(3 + nuzgag)]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowUzf1._ftype()
            )
            if abs(ipakcb) > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(ipakcb)
                )
                model.add_pop_key_list(ipakcb)
            if abs(iuzfcb2) > 0:
                iu, filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(iuzfcb2)
                )
                model.add_pop_key_list(abs(iuzfcb2))

            ipos = 3
            if nuzgag > 0:
                for key, value in uzgag.items():
                    iu, filenames[ipos] = model.get_ext_dict_attr(
                        ext_unit_dict, unit=abs(key)
                    )
                    model.add_pop_key_list(abs(iu))
                    ipos += 1

        # create uzf object
        return cls(
            model,
            nuztop=nuztop,
            iuzfopt=iuzfopt,
            irunflg=irunflg,
            ietflg=ietflg,
            ipakcb=ipakcb,
            iuzfcb2=iuzfcb2,
            ntrail2=ntrail2,
            nsets=nsets2,
            surfdep=surfdep,
            uzgag=uzgag,
            specifythtr=specifythtr,
            specifythti=specifythti,
            nosurfleak=nosurfleak,
            etsquare=etsquare,
            netflux=netflux,
            seepsurfk=seepsurfk,
            specifysurfk=specifysurfk,
            rejectsurfk=rejectsurfk,
            capillaryuzet=capillaryuzet,
            unitnumber=unitnumber,
            filenames=filenames,
            options=options,
            **arrays
        )

    @staticmethod
    def _ftype():
        return "UZF"

    @staticmethod
    def _defaultunit():
        return 19


def _parse1a(line):
    line = line_parse(line)
    line = [s.lower() if isinstance(s, str) else s for s in line]
    specifythtr = True if "specifythtr" in line else False
    specifythti = True if "specifythti" in line else False
    nosurfleak = True if "nosurfleak" in line else False
    return specifythtr, specifythti, nosurfleak


def _parse1(line):
    ntrail2 = None
    nsets2 = None
    line = line_parse(line)
    nuztop = pop_item(line, int)
    iuzfopt = pop_item(line, int)
    irunflg = pop_item(line, int)
    ietflag = pop_item(line, int)
    ipakcb = pop_item(line, int)
    iuzfcb2 = pop_item(line, int)
    if iuzfopt > 0:
        ntrail2 = pop_item(line, int)
        nsets2 = pop_item(line, int)
    nuzgag = pop_item(line, int)
    surfdep = pop_item(line, float)
    return (
        nuztop,
        iuzfopt,
        irunflg,
        ietflag,
        ipakcb,
        iuzfcb2,
        ntrail2,
        nsets2,
        nuzgag,
        surfdep,
    )


def _parse8(line):
    iuzrow = None
    iuzcol = None
    iuzopt = 0
    line = line_parse(line)
    if (len(line) > 1 and not int(line[0]) < 0) or (
        len(line) > 1 and line[1].isdigit()
    ):
        iuzrow = pop_item(line, int) - 1
        iuzcol = pop_item(line, int) - 1
        iftunit = pop_item(line, int)
        iuzopt = pop_item(line, int)
    else:
        iftunit = pop_item(line, int)
    return iuzrow, iuzcol, iftunit, iuzopt
