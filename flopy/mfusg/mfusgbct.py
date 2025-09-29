"""
Mfusgbct module.

Contains the MfUsgBct class. Note that the user can
access the MfUsgBct class as `flopy.mfusg.MfUsgBct`.
"""

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d, read1d
from ..utils.utils_def import (
    get_open_file_object,
    get_util2d_shape_for_layer,
    type_from_iterable,
)
from .mfusg import MfUsg


class MfUsgBct(Package):
    """Block Centered Transport (BCT) Package Class for MODFLOW-USG Transport.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    itrnsp : int (0,1,2,3,4,5,-1), (default is 1)
        transport simulation flag
    ipakcb : int (0,1,-1), (default is 0)
        a flag and a unit number >0 .
    mcomp : int (default is 1)
        number of mobile component species simulated
    icbndflg : int (default is 1)
        a flag that determines if active domain for transport the same as  for flow
    itvd : int (default is 3)
        0  : upstream weighted scheme is used for simulating the advective term
        >0 : number of TVD correction iterations used for simulating the advective term
    iadsorb : int (default is 0)
        adsorption flag (0: no adsorption, 1: linear isotherm, 2: Freundlich isotherm,
        3: Langmuir isotherm)
    ict : int (default is 0)
        transport solution scheme (0:water phase concentration, 1:total concentration)
    cinact : float (default is -1.0)
        concentration value that will be output at inactive nodes
    ciclose : float (default is 1e-6)
        concentration tolerance for convergence of the matrix solver
    idisp : int (default is 1)
        flag of dispersion (0: no dispersion, 1: isotropic, 2: anisotropic)
    ixdisp : int (default is 0)
        flag of cross-dispersion (0: no cross-dispersion, 1: cross-dispersion)
    diffnc : float (default is 0.57024)
        molecular diffusion coefficient
    izod : int (default is 0)
        flag of zero-order decay (0: no zero-order decay, 1: in water
        2: on soil, 3: on water and soil,  4: on air-water interface)
    ifod : int (default is 0)
        flag of first-order decay (0: no first-order decay, 1: in water
        2: on soil, 3: on water and soil,  4: on air-water interface)
    ifmbc : int (default is 0)
        flag of flux mass balance errors (0: not considered, 1: computed and reported)
    iheat : int (default is 0)
        flag of energy balance equation (0: not considered, 1: computed)
    imcomp : int (default is 0)
        number of immobile component species simulated
    idispcln : int (default is 0)
        index connection between GWF and CLN cells (0: finite difference approximation
        1: Thiem solution, 2: Thiem solution with local borehole thermal resistance)
    nseqitr : int (default is 0)
        an index or count for performing sequential iterations (0/1: no iterations
        >1: number of iterations of the transport and reaction modules)
    icbund : int or array of ints (nlay, nrow, ncol)
        is the cell-by-cell flag for the transport simulation
    prsity : float or array of floats (nlay, nrow, ncol), default is 0.15
        is the porosity of the porous medium
    bulkd : float or array of floats (nlay, nrow, ncol), default is 157.0
        is the bulk density of the porous medium
    anglex : float or array of floats (njag)
        is the angle (in radians) between the horizontal x-axis and the outward
        normal to the face between a node and its connecting nodes. The angle
        varies between zero and 6.283185 (two pi being 360 degrees).
    dl : float or array of floats (nlay, nrow, ncol), default is 1.0
        longitudinal dispersivity
    dt : float or array of floats (nlay, nrow, ncol), default is 0.1
        transverse dispersivity
    dlx : float or array of floats (nlay-1, nrow, ncol), default is 1.0
        x-direction longitudinal dispersivity
    dly : float or array of floats (nlay, nrow, ncol), default is 1.0
        y-direction longitudinal dispersivity
    dlz : float or array of floats (nlay, nrow, ncol), default is 0.1
        z-direction longitudinal dispersivity
    dtxy : float or array of floats (nlay, nrow, ncol), default is 0.1
        xy-direction transverse dispersivity
    dtyz : float or array of floats (nlay, nrow, ncol), default is 0.1
        yz-direction transverse dispersivity
    dtxz : float or array of floats (nlay, nrow, ncol), default is 0.1
        xz-direction transverse dispersivity
    adsorb : float or array of floats (nlay, nrow, ncol), default is none
        adsorption coefficient of a contaminant species
    flich : float or array of floats (nlay, nrow, ncol), default is none
        Freundlich adsorption isotherm exponent
    zodrw : float or array of floats (nlay, nrow, ncol), default is none
        zero-order decay coefficient in water
    zodrs : float or array of floats (nlay, nrow, ncol),default is none
        zero-order decay coefficient on soil
    zodraw : float or array of floats (nlay, nrow, ncol), default is none
        zero-order decay coefficient on air-water interface
    fodrw : float or array of floats (nlay, nrow, ncol), default is none
        first-order decay coefficient in water
    fodrs : float or array of floats (nlay, nrow, ncol), default is none
        first-order decay coefficient on soil
    fodraw : float or array of floats (nlay, nrow, ncol), default is none
        first-order decay coefficient on air-water interface
    conc : float or array of floats (nlay, nrow, ncol), default is 0.0
        initial concentration of each contaminant species
    extension : string
        mbegwunf  : flow imbalance information
        mbegwunt  : transport mass imbalance information
        mbeclnunf : flow imbalance information for CLN domain
        mbeclnunt : transport mass imbalance information for the CLN domain
        (default is ['bct','mbegwunf','mbegwunt','mbeclnunf','mbeclnunt']).
    unitnumber : int
        File unit number and the output files.
        (default is [59, 0, 0, 0, 0, 0, 0] ).
    filenames : str or list of str
        Filenames to use for the package and the output files.
    add_package : bool, default is True
        Flag to add the initialised package object to the parent model object.

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.mfusg.MfUsg(exe_name='USGs_1.exe')
    >>> disu = flopy.mfusg.MfUsgDisU(model=ml, nlay=1, nodes=1,
                 iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
    >>> bct = flopy.mfusg.MfUsgBct(ml)"""

    def __init__(
        self,
        model,
        itrnsp=1,
        ipakcb=0,
        mcomp=1,
        icbndflg=1,
        itvd=1,
        iadsorb=0,
        ict=0,
        cinact=-999.0,
        ciclose=1.0e-6,
        idisp=1,
        ixdisp=0,
        diffnc=0.0,
        izod=0,
        ifod=0,
        ifmbc=0,
        iheat=0,
        imcomp=0,
        idispcln=0,
        nseqitr=0,
        icbund=1,
        prsity=0.15,
        bulkd=1.0,
        anglex=0.0,
        dl=1.0,
        dt=0.1,
        dlx=1.0,
        dly=1.0,
        dlz=0.1,
        dtxy=0.1,
        dtyz=0.1,
        dtxz=0.1,
        conc=0.0,
        extension=["bct", "cbt", "mbegwf", "mbegwt", "mbeclnf", "mbeclnt"],
        unitnumber=None,
        filenames=None,
        add_package=True,
        **kwargs,
    ):
        """Constructs the MfUsgBct object."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if unitnumber is None:
            unitnumber = MfUsgBct._defaultunit()
        elif isinstance(unitnumber, list):
            if len(unitnumber) < 6:
                for idx in range(len(unitnumber), 6):
                    unitnumber.append(0)

        # set filenames
        filenames = self._prepare_filenames(filenames, 6)

        # cbc output file
        self.set_cbc_output_file(ipakcb, model, filenames[1])

        super().__init__(
            model,
            extension,
            self._ftype(),
            unitnumber,
            filenames,
        )

        self._generate_heading()
        self.itrnsp = itrnsp
        self.ipakcb = ipakcb
        self.mcomp = mcomp
        self.icbndflg = icbndflg
        self.itvd = itvd
        self.iadsorb = iadsorb
        self.ict = ict
        self.cinact = cinact
        self.ciclose = ciclose
        self.idisp = idisp
        self.ixdisp = ixdisp
        self.diffnc = diffnc
        self.izod = izod
        self.ifod = ifod
        self.ifmbc = ifmbc
        self.iheat = iheat
        self.imcomp = imcomp
        self.idispcln = idispcln
        self.nseqitr = nseqitr

        model.itrnsp = itrnsp
        model.mcomp = mcomp
        model.iheat = self.iheat

        structured = self.parent.structured
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        shape_3d = (nlay, nrow, ncol)

        # Options
        opts = []
        self.timeweight = None
        if "timeweight" in kwargs:
            self.timeweight = float(kwargs.pop("timeweight"))
            opts.append(f" TIMEWEIGHT {self.timeweight} ")

        self.chaindecay = None
        if kwargs.get("chaindecay"):
            self.chaindecay = 1
            opts.append(" CHAINDECAY ")

        self.only_satadsorb = None
        if kwargs.get("only_satadsorb"):
            self.only_satadsorb = 1
            opts.append(" ONLY_SATADSORB ")

        self.spatialreact = None
        if kwargs.get("spatialreact"):
            self.spatialreact = 1
            opts.append(" SPATIALREACT ")

        if self.chaindecay:
            if "nparent" in kwargs and len(kwargs["nparent"]) == mcomp:
                self.nparent = kwargs["nparent"]
            jparent = kwargs["jparent"]
            stotio = kwargs["stotio"]
            self.jparent = [0] * mcomp
            self.stotio = [0] * mcomp
            # sptlrct = kwargs["sptlrct"]
            self.sptlrct = [0] * mcomp
            for icomp in range(mcomp):
                if self.nparent[icomp] > 0:
                    self.jparent[icomp] = Util2d(
                        model, (mcomp,), np.int32, jparent[icomp], name="jparent"
                    )
                    self.stotio[icomp] = Util2d(
                        model, (mcomp,), np.float32, stotio[icomp], name="stotio"
                    )
                    # if self.spatialreact :
                    # self.sptlrct[icomp] = Util2d(
                    #     model, (mcomp,), np.float32, sptlrct[icomp], name="sptlrct"
                    # )

        self.solubility = None
        if kwargs.get("solubility"):
            self.sollim = Util2d(
                model, (mcomp,), np.float32, kwargs["sollim"], name="sollim"
            )
            self.solslope = Util2d(
                model, (mcomp,), np.float32, kwargs["solslope"], name="solslope"
            )
            self.solubility = 1
            opts.append(" SOLUBILITY ")

        self.aw_adsorb = None
        if kwargs.get("aw_adsorb"):
            self.iarea_fn = kwargs["iarea_fn"]
            self.ikawi_fn = kwargs["ikawi_fn"]
            self.aw_adsorb = 1
            opts.append(f" A-W_ADSORB {self.iarea_fn} {self.ikawi_fn}")

        if kwargs.get("imasswr"):
            opts.append(f" WRITE_GWMASS {kwargs['imasswr']} ")

        if "crootname" in kwargs:
            opts.append(f" MULTIFILE {kwargs['crootname']} ")

        self.options = " ".join(opts)

        # Assign input parameter array values
        if self.iheat:
            self.htcondw = kwargs["htcondw"]
            self.rhow = kwargs["rhow"]
            self.htcapw = kwargs["htcapw"]
            self.htcaps = Util3d(
                model, shape_3d, np.float32, kwargs["htcaps"], name="htcaps"
            )
            self.htconds = Util3d(
                model, shape_3d, np.float32, kwargs["htconds"], name="htconds"
            )
            self.heat = Util3d(model, shape_3d, np.float32, kwargs["heat"], name="heat")

        if self.icbndflg == 0:
            self.icbund = Util3d(model, shape_3d, np.int32, icbund, name="icbund")

        self.prsity = Util3d(model, shape_3d, np.float32, prsity, name="prsity")

        if self.iadsorb or self.iheat:
            self.bulkd = Util3d(model, shape_3d, np.float32, bulkd, name="bulkd")

        if not structured and self.idisp:
            njag = self.parent.get_package("DISU").njag
            self.anglex = Util2d(model, (njag,), np.float32, anglex, name="anglex")

        if self.idisp == 1:
            self.dl = Util3d(model, shape_3d, np.float32, dl, name="dl")
            self.dt = Util3d(model, shape_3d, np.float32, dt, name="dt")

        if self.idisp == 2:
            self.dlx = Util3d(model, shape_3d, np.float32, dlx, name="dlx")
            self.dly = Util3d(model, shape_3d, np.float32, dly, name="dly")
            self.dlz = Util3d(model, shape_3d, np.float32, dlz, name="dlz")
            self.dtxy = Util3d(model, shape_3d, np.float32, dtxy, name="dtxy")
            self.dtyz = Util3d(model, shape_3d, np.float32, dtyz, name="dtyz")
            self.dtxz = Util3d(model, shape_3d, np.float32, dtxz, name="dtxz")

        if self.iadsorb:
            adsorb = kwargs["adsorb"]
            self.adsorb = self.mcomp_Util3d(
                model, shape_3d, np.float32, adsorb, "adsorb", mcomp
            )

        if self.iadsorb == 2 or self.iadsorb == 3:
            flich = kwargs["flich"]
            self.flich = self.mcomp_Util3d(
                model, shape_3d, np.float32, flich, "flich", mcomp
            )

        if self.izod in {1, 3, 4}:
            zodrw = kwargs["zodrw"]
            self.zodrw = self.mcomp_Util3d(
                model, shape_3d, np.float32, zodrw, "zodrw", mcomp
            )

        if self.iadsorb and self.izod in {2, 3, 4}:
            zodrs = kwargs["zodrs"]
            self.zodrs = self.mcomp_Util3d(
                model, shape_3d, np.float32, zodrs, "zodrs", mcomp
            )

        if self.aw_adsorb and self.izod == 4:
            zodraw = kwargs["zodraw"]
            self.zodraw = self.mcomp_Util3d(
                model, shape_3d, np.float32, zodraw, "zodraw", mcomp
            )

        if self.ifod in {1, 3, 4}:
            fodrw = kwargs["fodrw"]
            self.fodrw = self.mcomp_Util3d(
                model, shape_3d, np.float32, fodrw, "fodrw", mcomp
            )

        if self.iadsorb and self.ifod in {2, 3, 4}:
            fodrs = kwargs["fodrs"]
            self.fodrs = self.mcomp_Util3d(
                model, shape_3d, np.float32, fodrs, "fodrs", mcomp
            )

        if self.aw_adsorb and self.ifod == 4:
            fodraw = kwargs["fodraw"]
            self.fodraw = self.mcomp_Util3d(
                model, shape_3d, np.float32, fodraw, "fodraw", mcomp
            )

        if self.aw_adsorb and self.iarea_fn == 1:
            self.awamax = Util3d(
                model, shape_3d, np.float32, kwargs["awamax"], name="awamax"
            )

        if self.aw_adsorb and self.iarea_fn == 2:
            self.grain_dia = Util3d(
                model, shape_3d, np.float32, kwargs["grain_dia"], name="grain_dia"
            )

        if self.aw_adsorb and self.iarea_fn in {1, 2, 3}:
            alangaw = kwargs["alangaw"]
            self.alangaw = self.mcomp_Util3d(
                model, shape_3d, np.float32, alangaw, "alangaw", mcomp
            )
            blangaw = kwargs["blangaw"]
            self.blangaw = self.mcomp_Util3d(
                model, shape_3d, np.float32, blangaw, "blangaw", mcomp
            )

        self.conc = self.mcomp_Util3d(model, shape_3d, np.float32, conc, "conc", mcomp)

        if self.imcomp > 0:
            imconc = kwargs["imconc"]
            self.imconc = self.mcomp_Util3d(
                model, shape_3d, np.float32, imconc, "imconc", mcomp
            )

        if add_package:
            self.parent.add_package(self)

    @staticmethod
    def mcomp_Util3d(model, shape, dtype, value, name, mcomp):
        if isinstance(value, (int, float)):
            value = [value] * mcomp
        mcomp3D = [0] * mcomp
        for icomp in range(mcomp):
            mcomp3D[icomp] = Util3d(
                model,
                shape,
                dtype,
                value[icomp],
                f"{name} of comp {icomp + 1}",
            )
        return mcomp3D

    def write_file(self, f=None):
        """
        Write the BCT package file.

        Parameters
        ----------
        f : open file object.
            Default is None, which will result in MfUsg.fn_path being
            opened for writing.

        """
        # Open file for writing
        if f is None:
            f_obj = open(self.fn_path, "w")

        # Item 1a: ITRNSP ipakcb MCOMP ICBNDFLG ITVD IADSORB ICT CINACT CICLOSE
        # IDISP IXDISP DIFFNC IZOD IFOD IFMBC IHEAT IMCOMP IDISPCLN NSEQITR
        f_obj.write(f"{self.heading}\n")

        print(self.parent.free_format_input)

        if self.parent.free_format_input:
            f_obj.write(
                f" {self.itrnsp:3d} {self.ipakcb:3d} {self.mcomp:3d}"
                f" {self.icbndflg:3d} "
                f"{self.itvd:3d} {self.iadsorb:3d} {self.ict:3d} {self.cinact:14.6e} "
                f"{self.ciclose:14.6e} {self.idisp:3d} {self.ixdisp:3d}"
                f" {self.diffnc:14.6e} "
                f"{self.izod:3d} {self.ifod:3d} {self.ifmbc:3d} {self.iheat:3d} "
                f"{self.imcomp:3d} {self.idispcln:3d} {self.nseqitr:3d} "
            )
        else:
            f_obj.write(
                f" {self.itrnsp:9d} {self.ipakcb:9d}"
                f" {self.mcomp:9d} {self.icbndflg:9d} "
                f"{self.itvd:9d} {self.iadsorb:9d} {self.ict:9d} {self.cinact:9.2e} "
                f"{self.ciclose:9.2e} {self.idisp:9d}"
                f" {self.ixdisp:9d} {self.diffnc:9.2e} "
                f"{self.izod:9d} {self.ifod:9d} {self.ifmbc:9d} {self.iheat:9d} "
                f"{self.imcomp:9d} {self.idispcln:9d} {self.nseqitr:9d} "
            )

        f_obj.write(self.options + "\n")

        # Item 1b.
        if self.ifmbc:
            f_obj.write(
                f" {self.unit_number[1]:9d} {self.unit_number[2]:9d}"
                f" {self.unit_number[3]:9d} {self.unit_number[4]:9d}\n"
            )

        # Item 1c: IHEAT == 1
        if self.iheat == 1:
            f_obj.write(f" {self.htcondw:9.2e} {self.rhow:9.2e} {self.htcapw:9.2e}\n")

        # Not implemented - item 1d aw_adsorb used and iarea_fn = 5 or ikawi_fn = 4
        # "nazones": int
        # "natabrows": int
        # Not implemented - item 1e aw_adsorb used and iarea_fn = 5 or ikawi_fn = 4
        # "iawizonmap": int
        # Not implemented - item 1f aw_adsorb used and iarea_fn = 3
        # "rog_sigma": int
        # Not implemented - item 1g aw_adsorb used and ikawi_fn = 3
        # "sigma_rt": int
        # Not implemented - item 1h aw_adsorb used and iarea_fn = 5
        # "awi_area_tab": int

        # Item 2: ICBUND
        if self.icbndflg == 0:
            f_obj.write(self.icbund.get_file_entry())

        # Item 3: PRSITY
        f_obj.write(self.prsity.get_file_entry())

        # Item 4: BULKD
        if self.iadsorb or self.iheat:
            f_obj.write(self.bulkd.get_file_entry())

        # Item 5: ANGLEX
        structured = self.parent.structured
        if (not structured) and self.idisp:
            f_obj.write(self.anglex.get_file_entry())

        # Item 6 & 7: DL & DT
        if self.idisp == 1:
            f_obj.write(self.dl.get_file_entry())
            f_obj.write(self.dt.get_file_entry())

        # Item 8 - 13: DLX, DLY, DLZ, DTXY, DTYZ, DTXZ
        if self.idisp == 2:
            f_obj.write(self.dlx.get_file_entry())
            f_obj.write(self.dly.get_file_entry())
            f_obj.write(self.dlz.get_file_entry())
            f_obj.write(self.dtxy.get_file_entry())
            f_obj.write(self.dtyz.get_file_entry())
            f_obj.write(self.dtxz.get_file_entry())

        if self.iheat == 1:
            f_obj.write(self.htcaps.get_file_entry())
            f_obj.write(self.htconds.get_file_entry())

        if self.aw_adsorb and self.iarea_fn == 1:
            f_obj.write(self.awamax.get_file_entry())
        if self.aw_adsorb and self.iarea_fn == 2:
            f_obj.write(self.grain_dia.get_file_entry())
        for icomp in range(self.mcomp):
            if self.chaindecay:
                f_obj.write(f"{self.nparent[icomp]:9d}\n")
                if self.nparent[icomp] > 0:
                    f_obj.write(self.jparent[icomp].get_file_entry())
                    f_obj.write(self.stotio[icomp].get_file_entry())
                    if self.spatialreact:
                        f_obj.write(self.sptlrct[icomp].get_file_entry())

            # Item A6-A7: ALANGAW, BLANGAW
            if self.aw_adsorb and self.iarea_fn in {1, 2, 3}:
                f_obj.write(self.alangaw[icomp].get_file_entry())
                f_obj.write(self.blangaw[icomp].get_file_entry())

            # Item 14 - 19: ADSORB, FLICH, ZODRW, ZODRS, ZODRAW, FODRW, FODRS, FODRAW
            if self.iadsorb:
                f_obj.write(self.adsorb[icomp].get_file_entry())
            if self.iadsorb == 2 or self.iadsorb == 3:
                f_obj.write(self.flich[icomp].get_file_entry())
            if self.izod == 1 or self.izod == 3 or self.izod == 4:
                f_obj.write(self.zodrw[icomp].get_file_entry())
            if self.iadsorb and (self.izod == 2 or self.izod == 3 or self.izod == 4):
                f_obj.write(self.zodrs[icomp].get_file_entry())
            if self.aw_adsorb and self.izod == 4:
                f_obj.write(self.zodraw[icomp].get_file_entry())
            if self.ifod == 1 or self.ifod == 3 or self.ifod == 4:
                f_obj.write(self.fodrw[icomp].get_file_entry())
            if self.iadsorb and (self.ifod == 2 or self.ifod == 3 or self.ifod == 4):
                f_obj.write(self.fodrs[icomp].get_file_entry())
            if self.aw_adsorb and self.ifod == 4:
                f_obj.write(self.fodraw[icomp].get_file_entry())

            # Item 20: CONC
            f_obj.write(self.conc[icomp].get_file_entry())

        if self.iheat == 1:
            f_obj.write(self.heat.get_file_entry())

        if self.imcomp > 0:
            for icomp in range(self.imcomp):
                f_obj.write(self.imconc[icomp].get_file_entry())

        # close the file
        f_obj.close()

    # Not implemented yet
    def check(
        self,
        f=None,
        verbose=True,
        level=1,
        checktype=None,
    ):
        """
        Check package data for common errors.

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

        Notes
        -----
        Unstructured models not checked for extreme recharge transmissivity
        ratios.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.rch.check()

        """
        chk = self._get_check(f, verbose, level, checktype)
        if self.parent.bas6 is not None:
            active = self.parent.bas6.ibound.array.sum(axis=0) != 0
        else:
            active = np.ones(self.rech.array[0][0].shape, dtype=bool)

        chk.summarize()
        return chk

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
        bct : MfUsgBct object

        Examples
        --------

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg()
        >>> dis = flopy.modflow.ModflowDis.load('Test1.dis', ml)
        >>> bct = flopy.mfusg.MfUsgBct.load('Test1.BTN', ml)

        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # determine problem dimensions
        nlay = model.nlay
        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")
            njag = dis.njag

        if model.verbose:
            print("loading bct package file...")

        f_obj = get_open_file_object(f, "r")

        # item 0
        line = f_obj.readline().upper()
        while line.startswith("#"):
            line = f_obj.readline().upper()

        t = line.split()
        kwargs = {}

        # item 1a
        vars = {
            "itrnsp": int,
            "ibctcb": int,
            "mcomp": int,
            "icbndflg": int,
            "itvd": int,
            "iadsorb": int,
            "ict": int,
            "cinact": float,
            "ciclose": float,
            "idisp": int,
            "ixdisp": int,
            "diffnc": float,
        }

        for i, (v, c) in enumerate(vars.items()):
            kwargs[v] = c(t[i].strip())
            print(f"{v}={kwargs[v]}")

        ibctcb = kwargs["ibctcb"]

        vars = {
            "izod": int,
            "ifod": int,
            "ifmbc": int,
            "iheat": int,
            "imcomp": int,
            "idispcln": int,
            "nseqitr": int,
        }

        for i, (v, c) in enumerate(vars.items()):
            kwargs[v] = type_from_iterable(t, index=12 + i, _type=c, default_val=0)
            print(f"{v}={kwargs[v]}")

        # item 1a - options
        if "TIMEWEIGHT" in t:
            idx = t.index("TIMEWEIGHT")
            kwargs["timeweight"] = float(t[idx + 1])

        kwargs["chaindecay"] = "CHAINDECAY" in t
        kwargs["only_satadsorb"] = "ONLY_SATADSORB" in t
        kwargs["spatialreact"] = "SPATIALREACT" in t
        kwargs["solubility"] = "SOLUBILITY" in t
        kwargs["chaindecay"] = "CHAINDECAY" in t
        kwargs["chaindecay"] = "CHAINDECAY" in t

        if "A-W_ADSORB" in t:
            idx = t.index("A-W_ADSORB")
            kwargs["aw_adsorb"] = 1
            kwargs["iarea_fn"] = int(t[idx + 1])
            kwargs["ikawi_fn"] = int(t[idx + 2])
        else:
            kwargs["aw_adsorb"] = None

        if "WRITE_GWMASS" in t:
            idx = t.index("WRITE_GWMASS")
            kwargs["imasswr"] = int(t[idx + 1])

        if "MULTIFILE" in t:
            idx = t.index("MULTIFILE")
            kwargs["crootname"] = str(t[idx + 1])

        # item 1b ifmbc == 1
        mbegwunf, mbegwunt, mbeclnunf, mbeclnunt = 0, 0, 0, 0
        if kwargs["ifmbc"] == 1:
            line = f_obj.readline().upper()
            t = line.split()
            mbegwunf, mbegwunt, mbeclnunf, mbeclnunt = (
                int(t[0]),
                int(t[1]),
                int(t[2]),
                int(t[3]),
            )

        # item 1c iheat == 1
        if kwargs["iheat"]:
            vars = {
                "htcondw": float,
                "rhow": float,
                "htcapw": float,
            }
            line = f_obj.readline().upper()
            t = line.split()
            for i, (v, c) in enumerate(vars.items()):
                kwargs[v] = c(t[i].strip())

        # Not implemented - item 1d aw_adsorb used and iarea_fn = 5 or ikawi_fn = 4
        # "nazones": int
        # "natabrows": int
        # Not implemented - item 1e aw_adsorb used and iarea_fn = 5 or ikawi_fn = 4
        # "iawizonmap": int
        # Not implemented - item 1f aw_adsorb used and iarea_fn = 3
        # "rog_sigma": int
        # Not implemented - item 1g aw_adsorb used and ikawi_fn = 3
        # "sigma_rt": int
        # Not implemented - item 1h aw_adsorb used and iarea_fn = 5
        # "awi_area_tab": int

        # item 2 ICBUND
        if kwargs["icbndflg"] == 0:
            kwargs["icbund"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "icbund", ext_unit_dict
            )

        # item 3 PRSITY
        kwargs["prsity"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "prsity", ext_unit_dict
        )

        # item 4 BULKD
        if kwargs["iadsorb"] or kwargs["iheat"]:
            kwargs["bulkd"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "bulkd", ext_unit_dict
            )

        mcomp = kwargs["mcomp"]
        # item 5 ANGLEX
        if (not model.structured) and kwargs["idisp"]:
            if model.verbose:
                print("   loading ANGLEX...")
            kwargs["anglex"] = Util2d.load(
                f_obj, model, (njag,), np.float32, "anglex", ext_unit_dict
            )

        # item 6 & 7 DL & DT
        if kwargs["idisp"] == 1:
            kwargs["dl"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dl", ext_unit_dict
            )
            kwargs["dt"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dt", ext_unit_dict
            )

        # item 8 - 13 DLX, DLY, DLZ, DTXY, DTYZ, DTXZ
        if kwargs["idisp"] == 2:
            kwargs["dlx"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dlx", ext_unit_dict
            )
            kwargs["dly"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dly", ext_unit_dict
            )
            kwargs["dlz"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dlz", ext_unit_dict
            )
            kwargs["dtxy"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dtxy", ext_unit_dict
            )
            kwargs["dtyz"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dtyz", ext_unit_dict
            )
            kwargs["dtxz"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dtxz", ext_unit_dict
            )

        # item H1 AND H2
        if kwargs["iheat"] == 1:
            kwargs["htcaps"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "htcondw", ext_unit_dict
            )
            kwargs["htconds"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "rhow", ext_unit_dict
            )

        # Read item A1  - A5 if AW_ADSORB option is on
        if kwargs["aw_adsorb"]:
            if kwargs["iarea_fn"] == 1:
                kwargs["awamax"] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "awamax", ext_unit_dict
                )
            if kwargs["iarea_fn"] == 2:
                kwargs["grain_dia"] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "grain_dia", ext_unit_dict
                )
            # Not implemented if AW_ADSORB option is on and iarea_fn = 4
            # if  kwargs["iarea_fn"] == 4:
            #     kwargs["awarea_x2"] = cls._load_prop_arrays(
            #         f_obj, model, nlay, np.float32, "awarea_x2", ext_unit_dict
            #     )
            #     kwargs["awarea_x1"] = cls._load_prop_arrays(
            #         f_obj, model, nlay, np.float32, "awarea_x1", ext_unit_dict
            #     )
            #     kwargs["awarea_x0"] = cls._load_prop_arrays(
            #         f_obj, model, nlay, np.float32, "awarea_x0", ext_unit_dict
            #     )

        adsorb = [0] * mcomp
        flich = [0] * mcomp
        zodrw = [0] * mcomp
        zodrs = [0] * mcomp
        zodraw = [0] * mcomp
        fodrw = [0] * mcomp
        fodrs = [0] * mcomp
        fodraw = [0] * mcomp
        conc = [0] * mcomp

        nparent = np.zeros(mcomp, dtype=np.int32)
        jparent = [0] * mcomp
        stotio = [0] * mcomp
        sptlrct = [0] * mcomp

        sollim = [0] * mcomp
        solslope = [0] * mcomp

        alangaw = [0] * mcomp
        blangaw = [0] * mcomp

        for icomp in range(mcomp):
            # item S1  - S2 if SOLUBILITY option is on  -- not tested
            if kwargs["solubility"] == 1:
                if model.verbose:
                    print("   loading SOLUBILITY ...")
                sollim[icomp] = read1d(f_obj, sollim)
                solslope[icomp] = read1d(f_obj, solslope)

            # item C1  - C3 if CHAINDECAY option is on
            if kwargs["chaindecay"] == 1:
                if model.verbose:
                    print("   loading CHAINDECAY...")
                t = f_obj.readline().split()
                # print(t)
                nparent[icomp] = int(t[0].strip())
                if nparent[icomp] > 0:
                    jparent[icomp] = Util2d.load(
                        f_obj, model, (mcomp,), np.int32, "jparent", ext_unit_dict
                    )
                    stotio[icomp] = Util2d.load(
                        f_obj, model, (mcomp,), np.float32, "stotio", ext_unit_dict
                    )
                    # item C4 if SPATIALREACT option is on - Not tested
                    if kwargs["spatialreact"] == 1:
                        if model.verbose:
                            print("   loading SPATIALREACT...")
                        sptlrct[icomp] = Util3d.load(
                            f_obj, model, (nlay,), np.float32, "sptlrct", ext_unit_dict
                        )

            # item A6  - A8 if AW_ADSORB option is on
            if kwargs["aw_adsorb"] == 1 and kwargs["iarea_fn"] in {1, 2, 3}:
                alangaw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "alangaw", ext_unit_dict
                )
                blangaw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "blangaw", ext_unit_dict
                )

            # item A8 not implemented if AW_ADSORB option is on and iarea_fn = 4

            # item 14 - 20 ADSORB, FLICH, ZODRW, ZODRS, ZODRAW, FODRW, FODRS, FODRAW
            if kwargs["iadsorb"]:
                adsorb[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "adsorb", ext_unit_dict
                )
            if kwargs["iadsorb"] == 2 or kwargs["iadsorb"] == 3:
                flich[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "flich", ext_unit_dict
                )
            if kwargs["izod"] == 1 or kwargs["izod"] == 3 or kwargs["izod"] == 4:
                zodrw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "zodrw", ext_unit_dict
                )
            if kwargs["iadsorb"] and (
                kwargs["izod"] == 2 or kwargs["izod"] == 3 or kwargs["izod"] == 4
            ):
                zodrs[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "zodrs", ext_unit_dict
                )
            if kwargs["aw_adsorb"] and (kwargs["izod"] == 4):
                zodraw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "zodraw", ext_unit_dict
                )
            if kwargs["ifod"] == 1 or kwargs["ifod"] == 3 or kwargs["ifod"] == 4:
                fodrw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "fodrw", ext_unit_dict
                )
            if kwargs["iadsorb"] and (
                kwargs["ifod"] == 2 or kwargs["ifod"] == 3 or kwargs["ifod"] == 4
            ):
                fodrs[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "fodrs", ext_unit_dict
                )
            if kwargs["aw_adsorb"] and (kwargs["ifod"] == 4):
                fodraw[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "fodraw", ext_unit_dict
                )

            # item 20 CONC
            conc[icomp] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "conc", ext_unit_dict
            )

        if kwargs["iheat"] == 1:
            kwargs["heat"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "heat", ext_unit_dict
            )

        if kwargs["imcomp"] > 0:
            imconc = [0] * kwargs["imcomp"]
            for icomp in range(kwargs["imcomp"]):
                imconc[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "imconc", ext_unit_dict
                )
            kwargs["imconc"] = imconc

        kwargs["nparent"] = nparent
        kwargs["jparent"] = jparent
        kwargs["stotio"] = stotio
        kwargs["sptlrct"] = sptlrct
        kwargs["sollim"] = sollim
        kwargs["solslope"] = solslope
        kwargs["alangaw"] = alangaw
        kwargs["blangaw"] = blangaw

        kwargs["adsorb"] = adsorb
        kwargs["flich"] = flich
        kwargs["zodrw"] = zodrw
        kwargs["zodrs"] = zodrs
        kwargs["zodraw"] = zodraw
        kwargs["fodrw"] = fodrw
        kwargs["fodrs"] = fodrs
        kwargs["fodraw"] = fodraw
        kwargs["conc"] = conc

        f_obj.close()

        # set package unit number
        # reset unit numbers

        unitnumber = [None] * 6
        filenames = [None] * 6
        if ext_unit_dict is not None:
            unitnumber[0], filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )
            file_unit_items = [ibctcb, mbegwunf, mbegwunt, mbeclnunf, mbeclnunt]
            for idx, item in enumerate(file_unit_items):
                unitnumber[idx + 1] = item
                _, filenames[idx + 1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(item)
                )
                model.add_pop_key_list(abs(item))

        bct = cls(model, unitnumber=unitnumber, filenames=filenames, **kwargs)

        if check:
            bct.check(
                f=f"{bct.name[0]}.chk",
                verbose=bct.parent.verbose,
                level=0,
            )

        return bct

    @staticmethod
    def _load_prop_arrays(f_obj, model, nlay, dtype, name, ext_unit_dict):
        if model.verbose:
            print(f"   loading {name} ...")
        prop_array = [0] * nlay
        for layer in range(nlay):
            util2d_shape = get_util2d_shape_for_layer(model, layer=layer)
            prop_array[layer] = Util2d.load(
                f_obj, model, util2d_shape, dtype, name, ext_unit_dict
            )
        return prop_array

    @staticmethod
    def _ftype():
        return "BCT"

    @staticmethod
    def _defaultunit():
        return [150, 0, 0, 0, 0, 0]
