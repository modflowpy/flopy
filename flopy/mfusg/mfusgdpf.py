"""
Mfusgdpf module.

Contains the MfUsgDpf class. Note that the user can
access the MfUsgDpf class as `flopy.mfusg.MfUsgDpf`.
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


class MfUsgDpf(Package):
    """Dual Porosity Flow (dpf) Package Class for MODFLOW-USG Transport.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    ipakcb : int (0,1,-1), (default is 0)
        a flag and a unit number >0 for cell-by-cell mass flux terms.
    idpfhd : int, (default is 0)
        a flag and a unit number >0 for immobile domain heads.
    idpfdd : int, (default is 0)
        a flag and a unit number >0 for immobile domain drawdown.
    iuzontabim : int or array of ints, (default is 0)
        soil type index for all groundwater flow nodes in the immobile domain
    iboundim : int, (default is 0)
        boundary variable for the immobile domain (<0 constant head, =0 no flow)
    hnewim : float or array of floats
        initial (starting) head in the immobile domain
    phif : float or array of floats
        porosity of the mobile domain.
    ddftr : float or array of floats
        dual domain flow transfer rate
    sc1im : float or array of floats
        specific storage of the immobile domain
    sc2im : float or array of floats
        specific yield or porosity of the immobile domain
    alphaim : float or array of floats
        van Genuchten alpha coefficient of the immobile domain
    betaim : float or array of floats
        van Genuchten beta coefficient of the immobile domain
    srim : float or array of floats
        van Genuchten sr coefficient of the immobile domain
    brookim : float or array of floats
        Brooks-Corey exponent for the relative permeability of the immobile domain
    bpim : float or array of floats
        Bubble point or air entry pressure head of the immobile domain
    extension : string,  (default is 'dpf').
    unitnumber : int, default is 57.
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
    >>> dpf = flopy.mfusg.MfUsgdpf(ml)"""

    def __init__(
        self,
        model,
        ipakcb=0,
        idpfhd=0,
        idpfdd=0,
        # iuzontabim=0,
        iboundim=0,
        hnewim=0.0,
        phif=0.0,
        ddftr=0.0,
        sc1im=0.0,
        sc2im=0.0,
        # alphaim=0.0,
        # betaim=0.0,
        # srim=0.0,
        # brookim=0.0,
        # bpim=0.0,
        extension="dpf",
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        """Constructs the MfUsgdpf object."""
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
        self.idpfhd = idpfhd
        self.idpfdd = idpfdd

        model.idpf = 0

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        # if self.parent.tabrich:
        #     self.iuzontabim = Util3d(
        #         model, (nlay, nrow, ncol), np.int32, iuzontabim, name="iuzontabim"
        #     )

        self.iboundim = Util3d(
            model, (nlay, nrow, ncol), np.float32, iboundim, name="iboundim"
        )

        self.hnewim = Util3d(
            model, (nlay, nrow, ncol), np.float32, hnewim, name="hnewim"
        )

        self.phif = Util3d(model, (nlay, nrow, ncol), np.float32, phif, name="phif")

        self.ddftr = Util3d(model, (nlay, nrow, ncol), np.float32, ddftr, name="ddftr")

        self.sc1im = Util3d(model, (nlay, nrow, ncol), np.float32, sc1im, name="sc1im")

        self.sc2im = Util3d(model, (nlay, nrow, ncol), np.float32, sc2im, name="sc2im")

        # self.alphaim = Util3d(
        #     model, (nlay, nrow, ncol), np.float32, alphaim, name="alphaim"
        # )

        # self.betaim = Util3d(
        #     model, (nlay, nrow, ncol), np.float32, betaim, name="betaim"
        # )

        # self.srim = Util3d(
        #     model, (nlay, nrow, ncol), np.float32, srim, name="srim"
        # )

        # self.brookim = Util3d(
        #     model, (nlay, nrow, ncol), np.float32, brookim, name="brookim"
        # )

        # self.bpim = Util3d(
        #     model, (nlay, nrow, ncol), np.float32, bpim, name="bpim"
        # )

        if add_package:
            self.parent.add_package(self)

    def write_file(self, f=None):
        """
        Write the dpf package file.

        Parameters
        ----------
        f : open file object.
            Default is None, which will result in MfUsg.fn_path being
            opened for writing.

        """
        # Open file for writing
        if f is None:
            f_obj = open(self.fn_path, "w")

        #        f_obj.write(f"{self.heading}\n")

        # Item 0: IPAKCB, IdpfCON
        f_obj.write(f" {self.ipakcb:9d} {self.idpfhd:9d} {self.idpfdd:9d} \n")

        f_obj.write(self.iboundim.get_file_entry())
        f_obj.write(self.hnewim.get_file_entry())
        f_obj.write(self.phif.get_file_entry())
        f_obj.write(self.ddftr.get_file_entry())
        f_obj.write(self.sc1im.get_file_entry())
        f_obj.write(self.sc2im.get_file_entry())

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
        dpf : MfUsgdpf object

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg()
        >>> dis = flopy.modflow.ModflowDis.load('Test1.dis', ml)
        >>> dpf = flopy.mfusg.MfUsgdpf.load('Test1.BTN', ml)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading dpf package file...")

        nlay = model.nlay

        f_obj = get_open_file_object(f, "r")

        # item 0
        line = f_obj.readline().upper()
        while line.startswith("#"):
            line = f_obj.readline().upper()

        t = line.split()
        kwargs = {}

        # item 1a
        vars = {
            "ipakcb": int,
            "idpfhd": int,
            "idpfdd": int,
        }

        for i, (v, c) in enumerate(vars.items()):
            kwargs[v] = c(t[i].strip())
            # print(f"{v}={kwargs[v]}\n")

        # item 1b
        # if self.parent.tabrich:
        # kwargs["iuzontabim"] = cls._load_prop_arrays(
        #     f_obj, model, nlay, np.int32, "iuzontabim", ext_unit_dict
        # )

        kwargs["iboundim"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.int32, "iboundim", ext_unit_dict
        )

        kwargs["hnewim"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "hnewim", ext_unit_dict
        )

        kwargs["phif"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "phif", ext_unit_dict
        )

        kwargs["ddftr"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "ddftr", ext_unit_dict
        )

        kwargs["sc1im"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "sc1im", ext_unit_dict
        )

        kwargs["sc2im"] = cls._load_prop_arrays(
            f_obj, model, nlay, np.float32, "sc2im", ext_unit_dict
        )

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
        return "DPF"

    @staticmethod
    def _defaultunit():
        return 157
