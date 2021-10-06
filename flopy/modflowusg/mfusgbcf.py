"""
mfusgbcf module.

Contains the ModflowUsgBcf class. Note that the user can
access the ModflowUsgBcf class as `flopy.modflowusg.ModflowUsgBcf`.
"""
import warnings
import numpy as np

from ..modflow import ModflowBcf
from .mfusg import ModflowUsg
from ..utils import Util2d, Util3d
from ..utils.flopy_io import line_parse
from ..utils.utils_def import (
    get_util2d_shape_for_layer,
    get_unitnumber_from_ext_unit_dict,
)


class ModflowUsgBcf(ModflowBcf):
    """
    Block Centered Flow (BCF) Package Class for MODFLOW-USG.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 53)
    intercellt : int
        Intercell transmissivities, harmonic mean (0), arithmetic mean (1),
        logarithmic mean (2), combination (3). (default is 0)
    laycon : int
        Layer type, confined (0), unconfined (1), constant T, variable S (2),
        variable T, variable S (default is 3)
    trpy : float or array of floats (nlay)
        horizontal anisotropy ratio (default is 1.0)
    hdry : float
        head assigned when cell is dry - used as indicator(default is -1E+30)
    iwdflg : int
        flag to indicate if wetting is inactive (0) or not (non zero)
        (default is 0)
    wetfct : float
        factor used when cell is converted from dry to wet (default is 0.1)
    iwetit : int
        iteration interval in wetting/drying algorithm (default is 1)
    ihdwet : int
        flag to indicate how initial head is computed for cells that become
        wet (default is 0)
    ikvflag : int
        flag indicating if vertical hydraulic conductivity is input
        instead of leakance between two layers.
    ikcflag : int
        flag indicating if hydraulic conductivity or transmissivity
        information is input for each of the nodes or whether this information
        is directly input for the nodal connections. The easiest input format
        is to provide the hydraulic conductivity or transmissivity values to
        the cells using a zero value for IKCFLAG.
    anglex : float or array of floats (njag)
        is the angle (in radians) between the horizontal x-axis and the outward
        normal to the face between a node and its connecting nodes. The angle
        varies between zero and 6.283185 (two pi being 360 degrees).
    tran : float or array of floats (nlay, nrow, ncol), optional
        transmissivity (only read if laycon is 0 or 2) (default is 1.0)
    hy : float or array of floats (nlay, nrow, ncol)
        hydraulic conductivity (only read if laycon is 1 or 3)
        (default is 1.0)
    vcont : float or array of floats (nlay-1, nrow, ncol)
        vertical leakance between layers (default is 1.0)
    kv : float or array of floats (nlay, nrow, ncol)
        is the vertical hydraulic conductivity of the cell and the leakance is
        computed for each vertical connection.
    sf1 : float or array of floats (nlay, nrow, ncol)
        specific storage (confined) or storage coefficient (unconfined),
        read when there is at least one transient stress period.
        (default is 1e-5)
    sf2 : float or array of floats (nlay, nrow, ncol)
        specific yield, only read when laycon is 2 or 3 and there is at least
        one transient stress period (default is 0.15)
    wetdry : float
        a combination of the wetting threshold and a flag to indicate which
        neighboring cells can cause a cell to become wet (default is -0.01)
    ksat : float or array of floats (njag)
        inter-block saturated hydraulic conductivity or transmissivity
        (if IKCFLAG = 1) or the inter-block conductance (if IKCFLAG = - 1)
        of the connection between nodes n and m.
    extension : string
        Filename extension (default is 'bcf')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output name will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.
    add_package : bool
        Flag to add the initialised package object to the parent model object.
        Default is True.

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflowusg.ModflowUsg()
    >>> disu = flopy.modflowusg.ModflowUsgDisU(model=ml, nlay=1, nodes=1,
                 iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
    >>> bcf = flopy.modflowusg.ModflowUsgBcf(ml)"""

    def __init__(
        self,
        model,
        **kwargs,
    ):
        """
        Constructs the ModflowUsgBcf object.

        Overrides the parent ModflowBcf object.
        """
        msg = (
            "Model object must be of type flopy.modflowusg.ModflowUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, ModflowUsg), msg

        valid_args_defaults = {
            "ipakcb": None,
            "intercellt": 0,
            "laycon": 3,
            "trpy": 1.0,
            "hdry": -1e30,
            "iwdflg": 0,
            "wetfct": 0.1,
            "iwetit": 1,
            "ihdwet": 0,
            "ikvflag": 0,
            "ikcflag": 0,
            "tran": 1.0,
            "hy": 1.0,
            "vcont": 1.0,
            "kv": 1.0,
            "anglex": 0.0,
            "ksat": 1.0,
            "sf1": 1e-5,
            "sf2": 0.15,
            "wetdry": -0.01,
            "extension": "bcf",
            "unitnumber": None,
            "filenames": None,
            "add_package": True,
        }

        for arg, default_value in valid_args_defaults.items():
            setattr(self, arg, kwargs.pop(arg, default_value))

        super().__init__(
            model,
            ipakcb=self.ipakcb,
            intercellt=self.intercellt,
            laycon=self.laycon,
            trpy=self.trpy,
            hdry=self.hdry,
            iwdflg=self.iwdflg,
            wetfct=self.wetfct,
            iwetit=self.iwetit,
            ihdwet=self.ihdwet,
            tran=self.tran,
            hy=self.hy,
            vcont=self.vcont,
            sf1=self.sf1,
            sf2=self.sf2,
            wetdry=self.wetdry,
            extension=self.extension,
            unitnumber=self.unitnumber,
            filenames=self.filenames,
            add_package=False,
        )

        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")
        structured = self.parent.structured

        nrow, ncol, nlay, _ = self.parent.nrow_ncol_nlay_nper

        if not structured:
            njag = dis.njag
            self.anglex = Util2d(
                model,
                (njag,),
                np.float32,
                self.anglex,
                "anglex",
                locat=self.unit_number[0],
            )

        # item 1
        self.kv = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            self.kv,
            "Vertical Hydraulic Conductivity",
            locat=self.unit_number[0],
        )
        if not structured:
            self.ksat = Util3d(
                model,
                (njag,),
                np.float32,
                self.ksat,
                "ksat",
                locat=self.unit_number[0],
            )

        if self.add_package:
            self.parent.add_package(self)

        return

    def __getattr__(self, name):
        """
        Get class attributes to avoid pylint E1101 false positives

        Will only get called for undefined attributes, given dynamic variable definitions.
        """
        warnings.warn(
            f"No member '{name}' contained in {type(self).__name__})"
        )
        return ""

    def write_file(self, f=None):
        """
        Write the BCF package file.

        Parameters
        ----------
        f : open file object.
            Default is None, which will result in ModflowUsg.fn_path being
            opened for writing.
        """
        # get model information
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")

        # Open file for writing
        if f is None:
            f = open(self.fn_path, "w")

        # Item 1: ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET, IKVFLAG, IKCFLAG
        f.write(
            (
                f" {self.ipakcb:9d} {self.hdry:9.3G} {self.iwdflg:9d}"
                f" {self.wetfct:9.3G} {self.iwetit:9d} {self.ihdwet:9d}"
                f" {self.ikvflag:9d} {self.ikcflag:9d}\n"
            )
        )

        # LAYCON array
        for k in range(nlay):
            if self.intercellt[k] > 0:
                f.write(f"{self.intercellt[k]:1d} {self.laycon[k]:1d} ")
            else:
                f.write(f"0{self.laycon[k]:1d} ")
        f.write("\n")

        # TRPY, <ANGLEX>
        f.write(self.trpy.get_file_entry())
        transient = not dis.steady.all()
        structured = self.parent.structured
        anis = any([t != 1 for t in self.trpy])
        if (not structured) and anis:
            f.write(self.anglex.get_file_entry())

        # <SF1>, <TRAN>, <HY>, <VCONT>, <KV>, <SF2>, <WETDRY>
        for k in range(nlay):
            if transient:
                f.write(self.sf1[k].get_file_entry())

            if self.ikcflag == 0:
                self._write_hy_tran_vcont_kv(f, k)

            if (transient) and (
                (self.laycon[k] == 2)
                or (self.laycon[k] == 3)
                or (self.laycon[k] == 4)
            ):
                f.write(self.sf2[k].get_file_entry())

            if (self.iwdflg != 0) and (
                (self.laycon[k] == 1) or (self.laycon[k] == 3)
            ):
                f.write(self.wetdry[k].get_file_entry())

        # <KSAT> (if ikcflag==1)
        if abs(self.ikcflag == 1):
            f.write(self.ksat.get_file_entry())

        f.close()

    def _write_hy_tran_vcont_kv(self, f, k):
        """
        writes hy/tran and vcont/kv file entries

        Parameters
        ----------
        f : open file object.
        k : model layer index (base 0)
        """
        if (self.laycon[k] == 0) or (self.laycon[k] == 2):
            f.write(self.tran[k].get_file_entry())
        else:
            f.write(self.hy[k].get_file_entry())

        if (self.ikvflag == 0) and k < (self.parent.nlay - 1):
            f.write(self.vcont[k].get_file_entry())
        elif (self.ikvflag == 1) and (self.parent.nlay > 1):
            f.write(self.kv[k].get_file_entry())

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
        bcf : ModflowUsgBcf object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflowusg.ModflowUsg()
        >>> disu = flopy.modflowusg.ModflowUsgDisU(model=m, nlay=1, nodes=1, iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
        >>> bcf = flopy.modflowusg.ModflowUsgBcf.load('test.bcf', m)
        """
        msg = (
            "Model object must be of type flopy.modflowusg.ModflowUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, ModflowUsg), msg

        if model.verbose:
            print("loading bcf package file...")

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
        nr, nc, nlay, _ = model.get_nrow_ncol_nlay_nper()
        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")
            njag = dis.njag

        # Item 1: ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET - line already read above
        if model.verbose:
            print("   loading ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET...")
        text_list = line_parse(line)
        ipakcb, hdry, iwdflg, wetfct, iwetit, ihdwet = (
            int(text_list[0]),
            float(text_list[1]),
            int(text_list[2]),
            float(text_list[3]),
            int(text_list[4]),
            int(text_list[5]),
        )

        try:
            ikvflag = int(text_list[6])
        except IndexError:
            ikvflag = 0
        except ValueError:
            ikvflag = 0

        try:
            ikcflag = int(text_list[7])
        except IndexError:
            ikcflag = 0
        except ValueError:
            ikvflag = 0

        # LAYCON array
        laycon, intercellt = cls._load_laycon(f, model)

        # TRPY array
        if model.verbose:
            print("   loading TRPY...")
        trpy = Util2d.load(
            f, model, (nlay,), np.float32, "trpy", ext_unit_dict
        )

        # property data for each layer based on options
        transient = not dis.steady.all()
        anis = any([t != 1 for t in trpy])
        if (not model.structured) and anis:
            if model.verbose:
                print("loading ANGLEX...")
            anglex = Util2d.load(
                f, model, (njag,), np.float32, "anglex", ext_unit_dict
            )
        else:
            anglex = 0

        # hy, kv, storage
        (sf1, tran, hy, vcont, sf2, wetdry, kv) = cls._load_layer_arrays(
            f,
            model,
            nlay,
            ext_unit_dict,
            transient,
            laycon,
            ikvflag,
            ikcflag,
            iwdflg,
        )

        # Ksat  mfusg
        if (not model.structured) and abs(ikcflag == 1):
            if model.verbose:
                print("   loading ksat (njag)...")
            ksat = Util2d.load(
                f, model, (njag,), np.float32, "ksat", ext_unit_dict
            )
        else:
            ksat = 0

        if openfile:
            f.close()

        # set package unit number
        unitnumber, filenames = get_unitnumber_from_ext_unit_dict(
            model, cls, ext_unit_dict, ipakcb
        )

        # create instance of bcf object
        bcf = cls(
            model,
            ipakcb=ipakcb,
            intercellt=intercellt,
            laycon=laycon,
            trpy=trpy,
            hdry=hdry,
            iwdflg=iwdflg,
            wetfct=wetfct,
            iwetit=iwetit,
            ihdwet=ihdwet,
            ikvflag=ikvflag,
            ikcflag=ikcflag,
            tran=tran,
            hy=hy,
            vcont=vcont,
            kv=kv,
            anglex=anglex,
            ksat=ksat,
            sf1=sf1,
            sf2=sf2,
            wetdry=wetdry,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        # return bcf object
        return bcf

    @staticmethod
    def _load_laycon(f, model):
        """
        Loads laycon and intercellt file entries.

        Parameters
        ----------
        f : open file object.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.

        Returns
        -------
        laycon : Numpy array of laycon values
        intercellt : Numpy array of intercellt values
        """
        nlay = model.nlay
        ifrefm = model.get_ifrefm()

        if model.verbose:
            print("   loading LAYCON...")

        line = f.readline()

        if ifrefm:
            laycons = []
            line_split = line.strip().split()
            for item in line_split:
                laycons.append(item)
            # read the rest of the laycon values
            if len(laycons) < nlay:
                while True:
                    line = f.readline()
                    line_split = line.strip().split()
                    for item in line_split:
                        laycons.append(item)
                    if len(laycons) == nlay:
                        break
        else:
            laycons = []
            istart = 0
            for k in range(nlay):
                lcode = line[istart : istart + 2]
                if lcode.strip() == "":
                    # hit end of line before expected end of data
                    # read next line
                    line = f.readline()
                    istart = 0
                    lcode = line[istart : istart + 2]
                lcode = lcode.replace(" ", "0")
                laycons.append(lcode)
                istart += 2

        intercellt = np.zeros(nlay, dtype=np.int32)
        laycon = np.zeros(nlay, dtype=np.int32)
        for k in range(nlay):
            if len(laycons[k]) > 1:
                intercellt[k] = int(laycons[k][0])
                laycon[k] = int(laycons[k][1])
            else:
                laycon[k] = int(laycons[k])

        return laycon, intercellt

    @classmethod
    def _load_layer_arrays(
        cls,
        f,
        model,
        nlay,
        ext_unit_dict,
        transient,
        laycon,
        ikvflag,
        ikcflag,
        iwdflg,
    ):
        """
        Loads LPF layer property arrays.

        Parameters
        ----------
        f : open file object.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        nlay : int. Number of model layers
        laycon : disu laycon array
        ext_unit_dict : external unit dictionary
        transient : bool. Flag for transient vs steady state models
        ikvflag : int
            flag indicating if vertical hydraulic conductivity is input
            instead of leakance between two layers.
        ikcflag : int
            flag indicating if hydraulic conductivity or transmissivity
            information is input for each of the nodes or whether this information
            is directly input for the nodal connections.
        iwdflg : int
            flag to indicate if wetting is inactive (0) or not (non zero)

        Returns
        -------
        sf1, tran, hy, vcont, sf2, wetdry, kv : layer property arrays
        """
        sf1 = [0] * nlay
        tran = [0] * nlay
        hy = [0] * nlay
        if nlay > 1:
            vcont = [0] * (nlay - 1)
        else:
            vcont = [0] * nlay
        sf2 = [0] * nlay
        wetdry = [0] * nlay
        kv = [0] * nlay  # mfusg

        for k in range(nlay):

            util2d_shape = get_util2d_shape_for_layer(model, layer=k)

            # sf1
            if transient:
                if model.verbose:
                    print(f"   loading sf1 layer {k + 1:3d}...")
                sf1[k] = Util2d.load(
                    f, model, util2d_shape, np.float32, "sf1", ext_unit_dict
                )

            # hy/tran, and kv/vcont
            if ikcflag == 0:
                hy[k], tran[k], kv[k], vcont_k = cls._load_hy_tran_kv_vcont(
                    f, model, (k, laycon[k]), ext_unit_dict, ikvflag
                )
                if k < nlay - 1:
                    vcont[k] = vcont_k

            # sf2
            if transient and (
                (laycon[k] == 2) or (laycon[k] == 3) or (laycon[k] == 4)
            ):
                if model.verbose:
                    print(f"   loading sf2 layer {k + 1:3d}...")
                sf2[k] = Util2d.load(
                    f, model, util2d_shape, np.float32, "sf2", ext_unit_dict
                )

            # wetdry
            if (iwdflg != 0) and ((laycon[k] == 1) or (laycon[k] == 3)):
                if model.verbose:
                    print(f"   loading sf2 layer {k + 1:3d}...")
                wetdry[k] = Util2d.load(
                    f, model, util2d_shape, np.float32, "wetdry", ext_unit_dict
                )

        return sf1, tran, hy, vcont, sf2, wetdry, kv

    @staticmethod
    def _load_hy_tran_kv_vcont(f, model, laycon_k, ext_unit_dict, ikvflag):
        """
        Loads hy/tran and kv/vcont file entries.

        Parameters
        ----------
        f : open file object.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        laycon_k : tuple of two ints: (k, laycon for layer k)
        ext_unit_dict : external unit dictionary
        ikvflag : int
            flag indicating if vertical hydraulic conductivity is input
            instead of leakance between two layers.

        Returns
        -------
        _hy : Numpy array of hk values (or 0)
        _tran : Numpy array of tran values (or 0)
        _kv : Numpy array of kv values (or 0)
        _vcont : Numpy array of vcont values (or 0)
        """
        k = laycon_k[0]
        laycon_k = laycon_k[1]
        util2d_shape = get_util2d_shape_for_layer(model, layer=k)

        # hy or tran
        _tran = 0
        _hy = 0
        if (laycon_k == 0) or (laycon_k == 2):
            if model.verbose:
                print(f"   loading tran layer {k + 1:3d}...")
            _tran = Util2d.load(
                f,
                model,
                util2d_shape,
                np.float32,
                "tran",
                ext_unit_dict,
            )
        else:
            if model.verbose:
                print(f"   loading hy layer {k + 1:3d}...")
            _hy = Util2d.load(
                f, model, util2d_shape, np.float32, "hy", ext_unit_dict
            )

        # kv or vcont
        _kv = 0
        _vcont = 0
        if k < (model.nlay - 1):
            if model.verbose:
                print(f"   loading vcont layer {k + 1:3d}...")
            _vcont = Util2d.load(
                f,
                model,
                util2d_shape,
                np.float32,
                "vcont",
                ext_unit_dict,
            )
        elif (ikvflag == 1) and (model.nlay > 1):
            if model.verbose:
                print(f"   loading kv layer {k + 1:3d}...")
            _kv = Util2d.load(
                f, model, util2d_shape, np.float32, "kv", ext_unit_dict
            )

        return _hy, _tran, _kv, _vcont
