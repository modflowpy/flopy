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
        itvd=3,
        iadsorb=0,
        ict=0,
        cinact=-1.0,
        ciclose=1e-6,
        idisp=1,
        ixdisp=0,
        diffnc=0.57024,
        izod=0,
        ifod=0,
        ifmbc=0,
        iheat=0,
        imcomp=0,
        idispcln=0,
        nseqitr=0,
        timeweight=None,
        chaindecay=False,
        nparent=0,
        jparent=0,
        stotio=0,
        sptlrct=0,
        only_satadsorb=False,
        spatialreact=False,
        solubility=False,
        aw_adsorb=False,
        iarea_fn=None,
        ikawi_fn=None,
        imasswr=None,
        crootname=None,
        mbegwunf=0,
        mbegwunt=0,
        mbeclnunf=0,
        mbeclnunt=0,
        htcondw=0,
        rhow=0,
        htcapw=0,
        htcaps=0,
        htconds=0,
        heat=0.0,
        icbund=1,
        prsity=0.15,
        bulkd=157.0,
        anglex=None,
        dl=1.0,
        dt=0.1,
        dlx=1.0,
        dly=1.0,
        dlz=0.1,
        dtxy=0.1,
        dtyz=0.1,
        dtxz=0.1,
        sollim=0.0,
        solslope=0.0,
        alangaw=0.0,
        blangaw=0.0,
        adsorb=0,
        flich=0,
        zodrw=0,
        zodrs=0,
        zodraw=0,
        fodrw=0,
        fodrs=0,
        fodraw=0,
        conc=0.0,
        imconc=0.0,
        extension=["bct", "cbt", "mbegwf", "mbegwt", "mbeclnf", "mbeclnt"],
        unitnumber=None,
        filenames=None,
        add_package=True,
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

        model.mcomp = mcomp
        model.iheat = iheat

        self.timeweight = timeweight
        self.chaindecay = chaindecay
        self.only_satadsorb = only_satadsorb
        self.spatialreact = spatialreact
        self.solubility = solubility
        self.aw_adsorb = aw_adsorb
        self.iarea_fn = iarea_fn
        self.ikawi_fn = ikawi_fn
        self.imasswr = imasswr
        self.crootname = crootname

        self.mbegwunf = mbegwunf
        self.mbegwunt = mbegwunt
        self.mbeclnunf = mbeclnunf
        self.mbeclnunt = mbeclnunt

        self.htcondw = htcondw
        self.rhow = rhow
        self.htcapw = htcapw

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        if self.icbndflg == 0:
            if icbund is None:
                icbund = 1
            self.icbund = Util3d(
                model, (nlay, nrow, ncol), np.int32, icbund, name="icbund"
            )

        self.prsity = Util3d(
            model, (nlay, nrow, ncol), np.float32, prsity, name="prsity"
        )

        if self.iadsorb or self.iheat:
            if bulkd is None:
                bulkd = 157.0
            self.bulkd = Util3d(
                model, (nlay, nrow, ncol), np.float32, bulkd, name="bulkd"
            )

        if anglex is not None:
            njag = self.parent.get_package("DISU").njag
            self.anglex = Util2d(model, (njag,), np.float32, anglex, name="anglex")

        if self.idisp == 1:
            self.dl = Util3d(model, (nlay, nrow, ncol), np.float32, dl, name="dl")
            self.dt = Util3d(model, (nlay, nrow, ncol), np.float32, dt, name="dt")

        if self.idisp == 2:
            self.dlx = Util3d(model, (nlay, nrow, ncol), np.float32, dlx, name="dlx")
            self.dly = Util3d(model, (nlay, nrow, ncol), np.float32, dly, name="dly")
            self.dlz = Util3d(model, (nlay, nrow, ncol), np.float32, dlz, name="dlz")
            self.dtxy = Util3d(model, (nlay, nrow, ncol), np.float32, dtxy, name="dtxy")
            self.dtyz = Util3d(model, (nlay, nrow, ncol), np.float32, dtyz, name="dtyz")
            self.dtxz = Util3d(model, (nlay, nrow, ncol), np.float32, dtxz, name="dtxz")
        if self.iheat == 1:
            self.htcaps = Util3d(
                model, (nlay, nrow, ncol), np.float32, htcaps, name="htcaps"
            )
            self.htconds = Util3d(
                model, (nlay, nrow, ncol), np.float32, htconds, name="htconds"
            )
        if self.solubility == 1:
            self.sollim = Util2d(model, (mcomp,), np.float32, sollim, name="sollim")
            self.solslope = Util2d(
                model, (mcomp,), np.float32, solslope, name="solslope"
            )

        if isinstance(adsorb, (int, float)):
            adsorb = [adsorb] * mcomp
        if isinstance(flich, (int, float)):
            flich = [flich] * mcomp
        if isinstance(zodrw, (int, float)):
            zodrw = [zodrw] * mcomp
        if isinstance(zodrs, (int, float)):
            zodrs = [zodrs] * mcomp
        if isinstance(zodraw, (int, float)):
            zodraw = [zodraw] * mcomp
        if isinstance(fodrw, (int, float)):
            fodrw = [fodrw] * mcomp
        if isinstance(fodrs, (int, float)):
            fodrs = [fodrs] * mcomp
        if isinstance(fodraw, (int, float)):
            fodraw = [fodraw] * mcomp
        if isinstance(conc, (int, float)):
            conc = [conc] * mcomp

        self.adsorb = [0] * mcomp
        self.flich = [0] * mcomp
        self.zodrw = [0] * mcomp
        self.zodrs = [0] * mcomp
        self.zodraw = [0] * mcomp
        self.fodrw = [0] * mcomp
        self.fodrs = [0] * mcomp
        self.fodraw = [0] * mcomp
        self.conc = [0] * mcomp

        self.nparent = nparent
        self.jparent = [0] * mcomp
        self.stotio = [0] * mcomp
        self.sptlrct = [0] * mcomp

        if isinstance(alangaw, (int, float)):
            alangaw = [alangaw] * mcomp
        if isinstance(blangaw, (int, float)):
            blangaw = [blangaw] * mcomp
        self.alangaw = [0] * mcomp
        self.blangaw = [0] * mcomp

        for icomp in range(mcomp):
            if self.chaindecay and self.nparent[icomp] > 0:
                self.jparent[icomp] = Util2d(
                    model, (mcomp,), np.float32, jparent[icomp], name="jparent"
                )
                self.stotio[icomp] = Util2d(
                    model, (mcomp,), np.float32, stotio[icomp], name="stotio"
                )
                # Not tested
                if self.spatialreact:
                    self.sptlrct[icomp] = Util3d(
                        model,
                        (nlay, nrow, ncol),
                        np.float32,
                        sptlrct[icomp],
                        name="sptlrct",
                    )

            if self.aw_adsorb and self.iarea_fn in (1, 2, 3):
                self.alangaw[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    alangaw[icomp],
                    name="alangaw species {icomp+1}",
                )
                self.blangaw[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    blangaw[icomp],
                    name="blangaw species {icomp+1}",
                )

            self.adsorb[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                adsorb[icomp],
                name=f"adsorb species {icomp+1}",
            )
            self.flich[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                flich[icomp],
                name=f"flich species {icomp+1}",
            )
            self.zodrw[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                zodrw[icomp],
                name=f"zodrw species {icomp+1}",
            )
            self.zodrs[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                zodrs[icomp],
                name=f"zodrs species {icomp+1}",
            )
            self.zodraw[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                zodraw[icomp],
                name=f"zodraw species {icomp+1}",
            )
            self.fodrw[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                fodrw[icomp],
                name=f"fodrw species {icomp+1}",
            )
            self.fodrs[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                fodrs[icomp],
                name=f"fodrs species {icomp+1}",
            )
            self.fodraw[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                fodraw[icomp],
                name=f"fodraw species {icomp+1}",
            )
            self.conc[icomp] = Util3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                conc[icomp],
                name=f"conc species {icomp+1}",
            )

        if self.iheat == 1:
            self.heat = Util3d(model, (nlay, nrow, ncol), np.float32, heat, name="heat")

        if self.imcomp > 0:
            if isinstance(imconc, (int, float)):
                imconc = [imconc] * mcomp
            self.imconc = [0] * mcomp
            for icomp in range(self.imcomp):
                self.imconc[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    imconc[icomp],
                    name=f"imconc species {icomp+1}",
                )

        if add_package:
            self.parent.add_package(self)

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

        # Item 1: ITRNSP ipakcb MCOMP ICBNDFLG ITVD IADSORB ICT CINACT CICLOSE
        # IDISP IXDISP DIFFNC IZOD IFOD IFMBC IHEAT IMCOMP IDISPCLN NSEQITR
        f_obj.write(f"{self.heading}\n")
        f_obj.write(
            f" {self.itrnsp:9d} {self.ipakcb:9d} {self.mcomp:9d} {self.icbndflg:9d} "
            f"{self.itvd:9d} {self.iadsorb:9d} {self.ict:9d} {self.cinact:9.2f} "
            f"{self.ciclose:9.2e} {self.idisp:9d} {self.ixdisp:9d} {self.diffnc:9.2f} "
            f"{self.izod:9d} {self.ifod:9d} {self.ifmbc:9d} {self.iheat:9d} "
            f"{self.imcomp:9d} {self.idispcln:9d} {self.nseqitr:9d}"
        )

        # Options: TIMEWEIGHT CHAINDECAY ONLY_SATADSORB SPATIALREACT SOLUBILITY
        # A-W_ADSORB WRITE_GWMASS MULTIFILE
        if self.chaindecay:
            f_obj.write(" CHAINDECAY")
        if self.only_satadsorb:
            f_obj.write(" ONLY_SATADSORB")
        if self.spatialreact:
            f_obj.write(" SPATIALREACT")
        if self.solubility:
            f_obj.write(" SOLUBILITY")
        if self.timeweight is not None:
            f_obj.write(f" TIMEWEIGHT {self.timeweight:9.2f}")
        if self.aw_adsorb:
            f_obj.write(f" A-W_ADSORB {self.iarea_fn:9d} {self.ikawi_fn:9d}")
        if self.imasswr is not None:
            f_obj.write(f" WRITE_GWMASS {self.imasswr:9d}")
        if self.crootname is not None:
            f_obj.write(f" MULTIFILE {self.crootname}")

        f_obj.write("\n")

        # Item 1b: IFMBC == 1
        if self.ifmbc == 1:
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

        for icomp in range(self.mcomp):
            if self.chaindecay:
                f_obj.write(f"{self.nparent[icomp]:9d}\n")
                if self.nparent[icomp] > 0:
                    f_obj.write(self.jparent[icomp].get_file_entry())
                    f_obj.write(self.stotio[icomp].get_file_entry())
                    if self.spatialreact:
                        f_obj.write(self.sptlrct[icomp].get_file_entry())

            # Item A6-A7: ALANGAW, BLANGAW
            if self.aw_adsorb and self.iarea_fn in (1, 2, 3):
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

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
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
            "ipakcb": int,
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
        else:
            kwargs["timeweight"] = None

        if "CHAINDECAY" in t:
            kwargs["chaindecay"] = 1
        else:
            kwargs["chaindecay"] = 0

        if "ONLY_SATADSORB" in t:
            kwargs["only_satadsorb"] = 1
        else:
            kwargs["only_satadsorb"] = 0

        if "SPATIALREACT" in t:
            kwargs["spatialreact"] = 1
        else:
            kwargs["spatialreact"] = 0

        if "SOLUBILITY" in t:
            kwargs["solubility"] = 1
        else:
            kwargs["solubility"] = 0

        if "A-W_ADSORB" in t:
            idx = t.index("A-W_ADSORB")
            kwargs["aw_adsorb"] = 1
            kwargs["iarea_fn"] = int(t[idx + 1])
            kwargs["ikawi_fn"] = int(t[idx + 2])
        else:
            kwargs["aw_adsorb"] = 0
            kwargs["iarea_fn"] = None
            kwargs["ikawi_fn"] = None

        if "WRITE_GWMASS" in t:
            idx = t.index("WRITE_GWMASS")
            kwargs["imasswr"] = int(t[idx + 1])
        else:
            kwargs["imasswr"] = None

        if "MULTIFILE" in t:
            idx = t.index("MULTIFILE")
            kwargs["crootname"] = str(t[idx + 1])
        else:
            kwargs["crootname"] = None

        # item 1b ifmbc == 1
        if kwargs["ifmbc"] == 1:
            vars = {
                "mbegwunf": int,
                "mbegwunt": int,
                "mbeclnunf": int,
                "mbeclnunt": int,
            }
            line = f_obj.readline().upper()
            t = line.split()
            for i, (v, c) in enumerate(vars.items()):
                kwargs[v] = c(t[i].strip())
        else:
            kwargs["mbegwunf"] = 0
            kwargs["mbegwunt"] = 0
            kwargs["mbeclnunf"] = 0
            kwargs["mbeclnunt"] = 0

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

        # Not implemented item A1  - A5 if AW_ADSORB option is on

        mcomp = kwargs["mcomp"]

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

            # item A6  - A7 if AW_ADSORB option is on
            if kwargs["aw_adsorb"] == 1 and kwargs["iarea_fn"] in (1, 2, 3):
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
        unitnumber = cls._defaultunit()
        filenames = [None] * 6
        unitnumber[0], filenames[0] = model.get_ext_dict_attr(
            ext_unit_dict, filetype=cls._ftype()
        )
        if kwargs["ipakcb"] > 0:
            unitnumber[1], filenames[1] = model.get_ext_dict_attr(
                ext_unit_dict, unit=kwargs["ipakcb"]
            )
        if kwargs["mbegwunf"] > 0:
            unitnumber[2], filenames[2] = model.get_ext_dict_attr(
                ext_unit_dict, unit=kwargs["mbegwunf"]
            )
        if kwargs["mbegwunt"] > 0:
            unitnumber[3], filenames[3] = model.get_ext_dict_attr(
                ext_unit_dict, unit=kwargs["mbegwunt"]
            )
        if kwargs["mbeclnunf"] > 0:
            unitnumber[4], filenames[4] = model.get_ext_dict_attr(
                ext_unit_dict, unit=kwargs["mbeclnunf"]
            )
        if kwargs["mbeclnunt"] > 0:
            unitnumber[5], filenames[5] = model.get_ext_dict_attr(
                ext_unit_dict, unit=kwargs["mbeclnunt"]
            )

        return cls(model, unitnumber=unitnumber, filenames=filenames, **kwargs)

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
