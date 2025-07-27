"""
Mfusgdpt module.

Contains the MfUsgDpt class. Note that the user can
access the MfUsgDpt class as `flopy.mfusg.MfUsgDpt`.
"""

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d
from ..utils.utils_def import (
    get_open_file_object,
    get_unitnumber_from_ext_unit_dict,
    get_util2d_shape_for_layer,
)
from .mfusg import MfUsg


class MfUsgDpt(Package):
    """Dual Porosity Transport (dpt) Package Class for MODFLOW-USG Transport.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    ipakcb : int (0,1,-1), (default is 0)
        a flag and a unit number >0 for cell-by-cell mass flux terms.
    idptcon : int (0,1), (default is 0)
        a flag and a unit number >0 for immobile domain concentrations
    icbndimflg : int (0,1), (default is 1)
        a flag active domain for the immobile (matrix) domain the same as
        that for the mobile (fracture) domain
    iadsorbim : int (0,1,2,3), (default is 0)
        a flag for adsorption in the immobile domain(0: no adsorption,
        1: linear isotherm, 2: Freundlich isotherm, 3: Langmuir isotherm)
    idispim : int (0,1), (default is 0)
        a flag for dispersion in the immobile domain (0: no dispersion,
        1: dispersion)
    izodim : int (0,1), (default is 0)
        a flag for zero-order decay in the immobile domain (0: no zero-order decay,
        1: in water, 2: on soil, 3: on water and soil,4: on air-water interface)
    ifodim : int (0,1), (default is 0)
        a flag for first-order decay in the immobile domain (0: no first-order decay
        1: in water, 2: on soil, 3: on water and soil,4: on air-water interface)
    frahk : bool, (default is False)
        a flag for fractional hydraulic conductivity in the immobile domain
    mobilesat : bool, (default is False)
        immobile domain saturation equal to initial mobile domain saturation
    inputsat : bool, (default is False)
        a flag of immobile domain saturation input
    icbundim : int or array of ints (nlay, nrow, ncol)
        is  cell-by-cell flag for transport simulation in immobile domain
    phif : float or array of floats (nlay, nrow, ncol)
        fraction of the total space that is occupied by the mobile domain
    prsityim : float or array of floats (nlay, nrow, ncol)
        effective transport porosity in the immobile domain
    bulkdim : float or array of floats (nlay, nrow, ncol)
        bulk density in the immobile domain
    dlim : float or array of floats (nlay, nrow, ncol)
        longitudinal dispersivity coefficient between mobile and immobile domains
    ddtr : float or array of floats (nlay, nrow, ncol)
        mass transfer coefficient between mobile and immobile domains
    sim : float or array of floats (nlay, nrow, ncol)
        saturation in the immobile domain
    htcapsim : float or array of floats (nlay, nrow, ncol)
        heat capacity in the immobile domain
    htcondsim : float or array of floats (nlay, nrow, ncol)
        heat conductivity in the immobile domain
    adsorbim : float or array of floats (nlay, nrow, ncol)
        adsorption coefficient in the immobile domain
    flichim : float or array of floats (nlay, nrow, ncol)
        Freundlich coefficient in the immobile domain
    zodrwim : float or array of floats (nlay, nrow, ncol)
        zero-order decay rate in water in the immobile domain
    zodrsim : float or array of floats (nlay, nrow, ncol)
        zero-order decay rate on soil in the immobile domain
    fodrwim : float or array of floats (nlay, nrow, ncol)
        first-order decay rate in water in the immobile domain
    fodrsim : float or array of floats (nlay, nrow, ncol)
        first-order decay rate on soil in the immobile domain
    concim : float or array of floats (nlay, nrow, ncol)
        initial concentration in the immobile domain
    extension : string,  (default is 'dpt').
    unitnumber : int, default is 58.
        File unit number.
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
    >>> ml = flopy.mfusg.MfUsg()
    >>> disu = flopy.mfusg.MfUsgDisU(model=ml, nlay=1, nodes=1,
                 iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
    >>> dpt = flopy.mfusg.MfUsgdpt(ml)"""

    def __init__(
        self,
        model,
        ipakcb=0,
        idptcon=0,
        icbndimflg=1,
        iadsorbim=0,
        idispim=0,
        izodim=0,
        ifodim=0,
        frahk=False,
        mobilesat=False,
        inputsat=False,
        icbundim=1,
        phif=0.4,
        prsityim=0.4,
        bulkdim=1.6,
        dlim=0.5,
        ddtr=0.5,
        sim=0.5,
        htcapsim=0.0,
        htcondsim=0.0,
        adsorbim=0.0,
        flichim=0.0,
        zodrwim=0.0,
        zodrsim=0.0,
        fodrwim=0.0,
        fodrsim=0.0,
        concim=0.0,
        extension="dpt",
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        """Constructs the MfUsgdpt object."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # set default unit number of one is not specified
        if unitnumber is None:
            self.unitnumber = self._defaultunit()

        super().__init__(
            model,
            extension,
            self._ftype(),
            unitnumber,
            self._prepare_filenames(filenames),
        )

        self._generate_heading()
        self.ipakcb = ipakcb
        self.idptcon = idptcon
        self.icbndimflg = icbndimflg
        self.iadsorbim = iadsorbim
        self.idispim = idispim
        self.izodim = izodim
        self.ifodim = ifodim
        # options
        self.frahk = frahk
        self.mobilesat = mobilesat
        self.inputsat = inputsat

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        if self.icbndimflg == 0:
            self.icbundim = Util3d(
                model, (nlay, nrow, ncol), np.int32, icbundim, name="icbundim"
            )

        self.phif = Util3d(model, (nlay, nrow, ncol), np.float32, phif, name="phif")

        self.prsityim = Util3d(
            model, (nlay, nrow, ncol), np.float32, prsityim, name="prsityim"
        )

        if self.iadsorbim:
            self.bulkdim = Util3d(
                model, (nlay, nrow, ncol), np.float32, bulkdim, name="bulkdim"
            )

        self.dlim = Util3d(model, (nlay, nrow, ncol), np.float32, dlim, name="dlim")

        self.ddtr = Util3d(model, (nlay, nrow, ncol), np.float32, ddtr, name="ddtr")

        if self.inputsat:
            self.sim = Util3d(model, (nlay, nrow, ncol), np.float32, sim, name="sim")

        if model.iheat:
            self.htcapsim = Util3d(
                model, (nlay, nrow, ncol), np.float32, htcapsim, name="htcapsim"
            )

            self.htcondsim = Util3d(
                model, (nlay, nrow, ncol), np.float32, htcondsim, name="htcondsim"
            )

        mcomp = model.mcomp

        if isinstance(adsorbim, (int, float)):
            adsorbim = [adsorbim] * mcomp
        if isinstance(flichim, (int, float)):
            flichim = [flichim] * mcomp
        if isinstance(zodrwim, (int, float)):
            zodrwim = [zodrwim] * mcomp
        if isinstance(zodrsim, (int, float)):
            zodrsim = [zodrsim] * mcomp
        if isinstance(fodrwim, (int, float)):
            fodrwim = [fodrwim] * mcomp
        if isinstance(fodrsim, (int, float)):
            fodrsim = [fodrsim] * mcomp

        if isinstance(concim, (int, float)):
            concim = [concim] * mcomp

        self.adsorbim = [0] * mcomp
        self.flichim = [0] * mcomp
        self.zodrwim = [0] * mcomp
        self.zodrsim = [0] * mcomp
        self.fodrwim = [0] * mcomp
        self.fodrsim = [0] * mcomp
        self.concim = [0] * mcomp

        for icomp in range(mcomp):
            if self.iadsorbim:
                self.adsorbim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    adsorbim[icomp],
                    name="adsorbim",
                )

            if self.iadsorbim == 2 or self.iadsorbim == 3:
                self.flichim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    flichim[icomp],
                    name="flichim",
                )

            if self.izodim == 1 or self.izodim == 3 or self.izodim == 4:
                self.zodrwim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    zodrwim[icomp],
                    name="zodrwim",
                )

            if self.iadsorbim and (
                self.izodim == 2 or self.izodim == 3 or self.izodim == 4
            ):
                self.zodrsim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    zodrsim[icomp],
                    name="zodrsim",
                )

            if self.ifodim == 1 or self.ifodim == 3 or self.ifodim == 4:
                self.fodrwim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    fodrwim[icomp],
                    name="fodrwim",
                )

            if self.iadsorbim and (
                self.ifodim == 2 or self.ifodim == 3 or self.ifodim == 4
            ):
                self.fodrsim[icomp] = Util3d(
                    model,
                    (nlay, nrow, ncol),
                    np.float32,
                    fodrsim[icomp],
                    name="fodrsim",
                )

            self.concim[icomp] = Util3d(
                model, (nlay, nrow, ncol), np.float32, concim[icomp], name="concim"
            )

        if add_package:
            self.parent.add_package(self)

    def write_file(self, f=None):
        """
        Write the dpt package file.

        Parameters
        ----------
        f : open file object.
            Default is None, which will result in MfUsg.fn_path being
            opened for writing.

        Examples
        --------
        """
        # Open file for writing
        if f is None:
            f_obj = open(self.fn_path, "w")

        #        f_obj.write(f"{self.heading}\n")

        # Item 0: IPAKCB, IDPTCON
        f_obj.write(
            f" {self.ipakcb:9d} {self.idptcon:9d} {self.icbndimflg:9d}"
            f" {self.iadsorbim:9d} {self.idispim:9d} {self.izodim:9d} {self.ifodim:9d}"
        )

        # Options
        if self.frahk:
            f_obj.write(" FRAHK")

        if self.mobilesat:
            f_obj.write(" MOBILESAT")

        if self.inputsat:
            f_obj.write(" INPUTSAT")

        f_obj.write("\n")

        # Item 1: ICBUNDIM
        if self.icbndimflg == 0:
            f_obj.write(self.icbundim.get_file_entry())

        # Item 2: PHIF
        f_obj.write(self.phif.get_file_entry())

        # Item 3: PRSITYIM
        f_obj.write(self.prsityim.get_file_entry())

        # Item 4: BULKDIM
        if self.iadsorbim:
            f_obj.write(self.bulkdim.get_file_entry())

        # Item 5: DLIM
        if self.parent.idpf:
            f_obj.write(self.dlim.get_file_entry())

        # Item 6: DDTR
        f_obj.write(self.ddtr.get_file_entry())

        # Item 7: SIM
        if self.inputsat:
            f_obj.write(self.sim.get_file_entry())

        # Item 8: HTCAPSIM, HTCONDSIM
        if self.parent.iheat == 1:
            f_obj.write(self.htcapsim.get_file_entry())
            f_obj.write(self.htcondsim.get_file_entry())

        # Item 9: ADSORBIM, FLICHIM, ZODRWIM, ZODRSIM, FODRWIM, FODRSIM, CONCIM
        mcomp = self.parent.mcomp
        for icomp in range(mcomp):
            if self.iadsorbim:
                f_obj.write(self.adsorbim[icomp].get_file_entry())

            if self.iadsorbim == 2 or self.iadsorbim == 3:
                f_obj.write(self.flichim[icomp].get_file_entry())

            if self.izodim == 1 or self.izodim == 3 or self.izodim == 4:
                f_obj.write(self.zodrwim[icomp].get_file_entry())

            if self.iadsorbim and (
                self.izodim == 2 or self.izodim == 3 or self.izodim == 4
            ):
                f_obj.write(self.zodrsim[icomp].get_file_entry())

            if self.ifodim == 1 or self.ifodim == 3 or self.ifodim == 4:
                f_obj.write(self.fodrwim[icomp].get_file_entry())

            if self.iadsorbim and (
                self.ifodim == 2 or self.ifodim == 3 or self.ifodim == 4
            ):
                f_obj.write(self.fodrsim[icomp].get_file_entry())

            f_obj.write(self.concim[icomp].get_file_entry())

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
        dpt : MfUsgdpt object

        Examples
        --------

        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg()
        >>> dis = flopy.mfusg.MfUsgDisU.load('SeqDegEg.dis', ml)
        >>> dpt = flopy.mfusg.MfUsgdpt.load('SeqDegEg.dpt', ml)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading dpt package file...")

        f_obj = get_open_file_object(f, "r")

        # determine problem dimensions
        nlay = model.nlay

        # item 0
        line = f_obj.readline().upper()
        while line.startswith("#"):
            line = f_obj.readline().upper()

        t = line.split()
        kwargs = {}

        # item 1a
        vars = {
            "ipakcb": int,
            "idptcon": int,
            "icbndimflg": int,
            "iadsorbim": int,
            "idispim": int,
            "izodim": int,
            "ifodim": int,
        }

        for i, (v, c) in enumerate(vars.items()):
            kwargs[v] = c(t[i].strip())
            # print(f"{v}={kwargs[v]}")

        # item 1a - options
        if "frahk" in t:
            kwargs["frahk"] = 1
        else:
            kwargs["frahk"] = 0

        if "mobilesat" in t:
            kwargs["mobilesat"] = 1
        else:
            kwargs["mobilesat"] = 0

        if "inputsat" in t:
            kwargs["inputsat"] = 1
        else:
            kwargs["inputsat"] = 0

        # item 1b
        if kwargs["icbndimflg"] == 0:
            kwargs["icbundim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.int32, "icbundim", ext_unit_dict
            )

        # item 2
        kwargs["phif"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "phif", ext_unit_dict
        )

        # item 3
        kwargs["prsityim"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "prsityim", ext_unit_dict
        )

        # item 4
        if kwargs["iadsorbim"]:
            kwargs["bulkdim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "bulkdim", ext_unit_dict
            )

        # item 5
        if model.idpf:
            kwargs["dlim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "dlim", ext_unit_dict
            )

        # item 6
        kwargs["ddtr"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "ddtr", ext_unit_dict
        )

        # item 7
        if kwargs["inputsat"]:
            kwargs["sim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "sim", ext_unit_dict
            )

        # item 8
        if model.iheat == 1:
            kwargs["htcapsim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "htcapsim", ext_unit_dict
            )

            kwargs["htcondsim"] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "htcondsim", ext_unit_dict
            )

        # item 9
        mcomp = model.mcomp
        adsorbim = [0] * mcomp
        flichim = [0] * mcomp
        zodrwim = [0] * mcomp
        zodrsim = [0] * mcomp
        fodrwim = [0] * mcomp
        fodrsim = [0] * mcomp
        concim = [0] * mcomp

        for icomp in range(mcomp):
            if kwargs["iadsorbim"]:
                adsorbim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "adsorbim", ext_unit_dict
                )

            if kwargs["iadsorbim"] == 2 or kwargs["iadsorbim"] == 3:
                flichim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "flichim", ext_unit_dict
                )

            if kwargs["izodim"] == 1 or kwargs["izodim"] == 3 or kwargs["izodim"] == 4:
                zodrwim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "zodrwim", ext_unit_dict
                )

            if kwargs["iadsorbim"] and (
                kwargs["izodim"] == 2 or kwargs["izodim"] == 3 or kwargs["izodim"] == 4
            ):
                zodrsim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "zodrsim", ext_unit_dict
                )

            if kwargs["ifodim"] == 1 or kwargs["ifodim"] == 3 or kwargs["ifodim"] == 4:
                fodrwim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "fodrwim", ext_unit_dict
                )

            if kwargs["iadsorbim"] and (
                kwargs["ifodim"] == 2 or kwargs["ifodim"] == 3 or kwargs["ifodim"] == 4
            ):
                fodrsim[icomp] = cls._load_prop_arrays(
                    f_obj, model, nlay, np.float32, "fodrsim", ext_unit_dict
                )

            concim[icomp] = cls._load_prop_arrays(
                f_obj, model, nlay, np.float32, "concim", ext_unit_dict
            )

        kwargs["adsorbim"] = adsorbim
        kwargs["flichim"] = flichim
        kwargs["zodrwim"] = zodrwim
        kwargs["zodrsim"] = zodrsim
        kwargs["fodrwim"] = fodrwim
        kwargs["fodrsim"] = fodrsim
        kwargs["concim"] = concim

        f_obj.close()
        # set package unit number
        unitnumber, filenames = get_unitnumber_from_ext_unit_dict(
            model, cls, ext_unit_dict, kwargs["ipakcb"]
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
        return "DPT"

    @staticmethod
    def _defaultunit():
        return 158
