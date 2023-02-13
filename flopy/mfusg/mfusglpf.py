"""
Mfusglpf module.

Contains the MfUsgLpf class. Note that the user can access
the MfUsgLpf class as `flopy.mfusg.MfUsgLpf`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/lpf.html>`_.
"""
import numpy as np

from ..modflow.mflpf import ModflowLpf
from ..modflow.mfpar import ModflowPar as mfpar
from ..utils import Util2d, read1d
from ..utils.flopy_io import line_parse
from ..utils.utils_def import (
    get_open_file_object,
    get_unitnumber_from_ext_unit_dict,
    get_util2d_shape_for_layer,
)
from .mfusg import MfUsg


class MfUsgLpf(ModflowLpf):
    """MODFLOW Layer Property Flow Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflowusg.mfusg.MfUsg`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0)
    hdry : float
        Is the head that is assigned to cells that are converted to dry during
        a simulation. Although this value plays no role in the model
        calculations, it is useful as an indicator when looking at the
        resulting heads that are output from the model. HDRY is thus similar
        to HNOFLO in the Basic Package, which is the value assigned to cells
        that are no-flow cells at the start of a model simulation.
        (default is -1.e30).
    laytyp : int or array of ints (nlay)
        Layer type, contains a flag for each layer that specifies the layer
        type.
        0 confined
        >0 convertible
        <0 convertible unless the THICKSTRT option is in effect.
        (default is 0).
    layavg : int or array of ints (nlay)
        Layer average
        0 is harmonic mean
        1 is logarithmic mean
        2 is arithmetic mean of saturated thickness and logarithmic mean of
        of hydraulic conductivity
        (default is 0).
    chani : float or array of floats (nlay)
        contains a value for each layer that is a flag or the horizontal
        anisotropy. If CHANI is less than or equal to 0, then variable HANI
        defines horizontal anisotropy. If CHANI is greater than 0, then CHANI
        is the horizontal anisotropy for the entire layer, and HANI is not
        read. If any HANI parameters are used, CHANI for all layers must be
        less than or equal to 0. Use as many records as needed to enter a
        value of CHANI for each layer. The horizontal anisotropy is the ratio
        of the hydraulic conductivity along columns (the Y direction) to the
        hydraulic conductivity along rows (the X direction).
        (default is 1).
    layvka : int or array of ints (nlay)
        a flag for each layer that indicates whether variable VKA is vertical
        hydraulic conductivity or the ratio of horizontal to vertical
        hydraulic conductivity.
        0: VKA is vertical hydraulic conductivity
        not 0: VKA is the ratio of horizontal to vertical hydraulic conductivity
        (default is 0).
    laywet : int or array of ints (nlay)
        contains a flag for each layer that indicates if wetting is active.
        0 wetting is inactive
        not 0 wetting is active
        (default is 0).
    wetfct : float
        is a factor that is included in the calculation of the head that is
        initially established at a cell when it is converted from dry to wet.
        (default is 0.1).
    iwetit : int
        is the iteration interval for attempting to wet cells. Wetting is
        attempted every IWETIT iteration. If using the PCG solver
        (Hill, 1990), this applies to outer iterations, not inner iterations.
        If IWETIT  less than or equal to 0, it is changed to 1.
        (default is 1).
    ihdwet : int
        is a flag that determines which equation is used to define the
        initial head at cells that become wet.
        (default is 0)
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
    hk : float or array of floats (nlay, nrow, ncol)
        is the hydraulic conductivity along rows. HK is multiplied by
        horizontal anisotropy (see CHANI and HANI) to obtain hydraulic
        conductivity along columns.
        (default is 1.0).
    hani : float or array of floats (nlay, nrow, ncol)
        is the ratio of hydraulic conductivity along columns to hydraulic
        conductivity along rows, where HK of item 10 specifies the hydraulic
        conductivity along rows. Thus, the hydraulic conductivity along
        columns is the product of the values in HK and HANI.
        (default is 1.0).
    vka : float or array of floats (nlay, nrow, ncol)
        is either vertical hydraulic conductivity or the ratio of horizontal
        to vertical hydraulic conductivity depending on the value of LAYVKA.
        (default is 1.0).
    ss : float or array of floats (nlay, nrow, ncol)
        is specific storage unless the STORAGECOEFFICIENT option is used.
        When STORAGECOEFFICIENT is used, Ss is confined storage coefficient.
        (default is 1.e-5).
    sy : float or array of floats (nlay, nrow, ncol)
        is specific yield.
        (default is 0.15).
    vkcb : float or array of floats (nlay, nrow, ncol)
        is the vertical hydraulic conductivity of a Quasi-three-dimensional
        confining bed below a layer. (default is 0.0).  Note that if an array
        is passed for vkcb it must be of size (nlay, nrow, ncol) even though
        the information for the bottom layer is not needed.
    wetdry : float or array of floats (nlay, nrow, ncol)
        is a combination of the wetting threshold and a flag to indicate
        which neighboring cells can cause a cell to become wet.
        (default is -0.01).
    ksat : float or array of floats (njag)
        inter-block saturated hydraulic conductivity or transmissivity
        (if IKCFLAG = 1) or the inter-block conductance (if IKCFLAG = - 1)
        of the connection between nodes n and m.
    storagecoefficient : boolean
        indicates that variable Ss and SS parameters are read as storage
        coefficient rather than specific storage.
        (default is False).
    constantcv : boolean
         indicates that vertical conductance for an unconfined cell is
         computed from the cell thickness rather than the saturated thickness.
         The CONSTANTCV option automatically invokes the NOCVCORRECTION
         option. (default is False).
    thickstrt : boolean
        indicates that layers having a negative LAYTYP are confined, and their
        cell thickness for conductance calculations will be computed as
        STRT-BOT rather than TOP-BOT. (default is False).
    nocvcorrection : boolean
        indicates that vertical conductance is not corrected when the vertical
        flow correction is applied. (default is False).
    novfc : boolean
         turns off the vertical flow correction under dewatered conditions.
         This option turns off the vertical flow calculation described on p.
         5-8 of USGS Techniques and Methods Report 6-A16 and the vertical
         conductance correction described on p. 5-18 of that report.
         (default is False).
    extension : string
        Filename extension (default is 'lpf')
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
    >>> m = flopy.mfusg.MfUsg()
    >>> disu = flopy.mfusg.MfUsgDisU(
        model=m, nlay=1, nodes=1, iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
    >>> lpf = flopy.mfusg.MfUsgLpf(m)
    """

    def __init__(
        self,
        model,
        laytyp=0,
        layavg=0,
        chani=1.0,
        layvka=0,
        laywet=0,
        ipakcb=None,
        hdry=-1e30,
        iwdflg=0,
        wetfct=0.1,
        iwetit=1,
        ihdwet=0,
        ikcflag=0,
        anglex=0,
        hk=1.0,
        hani=1.0,
        vka=1.0,
        ss=1e-5,
        sy=0.15,
        vkcb=0.0,
        wetdry=-0.01,
        ksat=1.0,
        storagecoefficient=False,
        constantcv=False,
        thickstrt=False,
        nocvcorrection=False,
        novfc=False,
        extension="lpf",
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        """Constructs the MfUsgBcf object.

        Overrides the parent ModflowBcf object."""

        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        super().__init__(
            model,
            laytyp=laytyp,
            layavg=layavg,
            chani=chani,
            layvka=layvka,
            laywet=laywet,
            ipakcb=ipakcb,
            hdry=hdry,
            iwdflg=iwdflg,
            wetfct=wetfct,
            iwetit=iwetit,
            ihdwet=ihdwet,
            hk=hk,
            hani=hani,
            vka=vka,
            ss=ss,
            sy=sy,
            vkcb=vkcb,
            wetdry=wetdry,
            storagecoefficient=storagecoefficient,
            constantcv=constantcv,
            thickstrt=thickstrt,
            nocvcorrection=nocvcorrection,
            novfc=novfc,
            extension=extension,
            unitnumber=unitnumber,
            filenames=self._prepare_filenames(filenames),
            add_package=False,
        )

        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")
        structured = self.parent.structured

        self.ikcflag = ikcflag
        if structured:
            self.ikcflag = 0
        self.options = " "
        if storagecoefficient:
            self.options = self.options + "STORAGECOEFFICIENT "
        if constantcv:
            self.options = self.options + "CONSTANTCV "
        if thickstrt:
            self.options = self.options + "THICKSTRT "
        if nocvcorrection:
            self.options = self.options + "NOCVCORRECTION "
        if novfc:
            self.options = self.options + "NOVFC "

        if not structured:
            njag = dis.njag
            self.anglex = Util2d(
                model,
                (njag,),
                np.float32,
                anglex,
                "anglex",
                locat=self.unit_number[0],
            )

        if not structured:
            njag = dis.njag
            self.ksat = Util2d(
                model,
                (njag,),
                np.float32,
                ksat,
                "ksat",
                locat=self.unit_number[0],
            )

        if add_package:
            self.parent.add_package(self)

    def write_file(self, check=True, f=None):
        """
        Write the package file.

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        # allows turning off package checks when writing files at model level
        if check:
            self.check(
                f=f"{self.name[0]}.chk",
                verbose=self.parent.verbose,
                level=1,
            )

        # get model information
        nlay = self.parent.nlay
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")

        # Open file for writing
        if f is None:
            f_obj = open(self.fn_path, "w")

        # Item 0: text
        f_obj.write(f"{self.heading}\n")

        # Item 1: IBCFCB, HDRY, NPLPF, <IKCFLAG>, OPTIONS
        if self.parent.version == "mfusg" and not self.parent.structured:
            f_obj.write(
                f" {self.ipakcb:9d} {self.hdry:9.5G} {self.nplpf:9d}"
                f" {self.ikcflag:9d} {self.options:s}\n"
            )
        else:
            f_obj.write(
                f" {self.ipakcb:9d} {self.hdry:9.5G} {self.nplpf:9d} {self.options}\n"
            )
        # LAYTYP array
        f_obj.write(self.laytyp.string)
        # LAYAVG array
        f_obj.write(self.layavg.string)
        # CHANI array
        f_obj.write(self.chani.string)
        # LAYVKA array
        f_obj.write(self.layvka.string)
        # LAYWET array
        f_obj.write(self.laywet.string)
        # Item 7: WETFCT, IWETIT, IHDWET
        iwetdry = self.laywet.sum()
        if iwetdry > 0:
            f_obj.write(
                f"{self.wetfct:10f}{self.iwetit:10d}{self.ihdwet:10d}\n"
            )

        transient = not dis.steady.all()
        structured = self.parent.structured
        anis = any(ch != 1 for ch in self.chani)
        if (not structured) and anis:
            f_obj.write(self.anglex.get_file_entry())

        for layer in range(nlay):
            if self.ikcflag == 0:  # mfusg
                f_obj.write(self.hk[layer].get_file_entry())
                if self.chani[layer] <= 0.0:
                    f_obj.write(self.hani[layer].get_file_entry())
                f_obj.write(self.vka[layer].get_file_entry())

            if transient:
                f_obj.write(self.ss[layer].get_file_entry())
                if self.laytyp[layer] != 0:
                    f_obj.write(self.sy[layer].get_file_entry())
                if self.ikcflag == 0 and dis.laycbd[layer] > 0:
                    f_obj.write(self.vkcb[layer].get_file_entry())
                if self.laywet[layer] != 0 and self.laytyp[layer] != 0:
                    f_obj.write(self.wetdry[layer].get_file_entry())

        if abs(self.ikcflag == 1):
            f_obj.write(self.ksat.get_file_entry())

        f_obj.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None, check=True):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mfusg.MfUsg`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        lpf : MfUsgLpf object
            MfUsgLpf object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.mfusg.MfUsg()
        >>> disu = flopy.mfusg.MfUsgDisU(
            model=m, nlay=1, nodes=1, iac=[1], njag=1,ja=np.array([0]), fahl=[1.0], cl12=[1.0])
        >>> lpf = flopy.mfusg.MfUsgLpf.load('test.lpf', m)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading lpf package file...")

        f_obj = get_open_file_object(f, "r")

        # dataset 0 -- header
        while True:
            line = f_obj.readline()
            if line[0] != "#":
                break

        # determine problem dimensions
        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")
            njag = dis.njag

        (
            ipakcb,
            hdry,
            nplpf,
            ikcflag,
            storagecoefficient,
            constantcv,
            thickstrt,
            nocvcorrection,
            novfc,
        ) = cls._load_item1(line, model)

        (
            laytyp,
            layavg,
            chani,
            layvka,
            laywet,
            wetfct,
            iwetit,
            ihdwet,
            iwetdry,
        ) = cls._load_items_2_to_7(f_obj, model)

        # ANGLEX for unstructured grid with anisotropy
        anis = any(ch != 1 for ch in chani)
        anglex = 0
        if (not model.structured) and anis:
            if model.verbose:
                print("mfusg:   loading ANGLEX...")
            anglex = Util2d.load(
                f_obj, model, (njag,), np.float32, "anglex", ext_unit_dict
            )

        # load layer properties
        (hk, hani, vka, ss, sy, vkcb, wetdry) = cls._load_layer_properties(
            cls,
            f_obj,
            model,
            dis,
            ikcflag,
            layvka,
            chani,
            laytyp,
            laywet,
            nplpf,
            ext_unit_dict,
        )

        # Ksat  mfusg
        ksat = 1.0
        if abs(ikcflag) == 1:
            if model.verbose:
                print("   loading ksat...")
            ksat = Util2d.load(
                f_obj, model, (njag,), np.float32, "ksat", ext_unit_dict
            )

        f_obj.close()

        # set package unit number and io file names
        unitnumber, filenames = get_unitnumber_from_ext_unit_dict(
            model, cls, ext_unit_dict, ipakcb
        )

        # create instance of lpf class
        lpf = cls(
            model,
            ipakcb=ipakcb,
            laytyp=laytyp,
            layavg=layavg,
            chani=chani,
            layvka=layvka,
            laywet=laywet,
            hdry=hdry,
            iwdflg=iwetdry,
            wetfct=wetfct,
            iwetit=iwetit,
            ihdwet=ihdwet,
            ikcflag=ikcflag,
            anglex=anglex,
            hk=hk,
            hani=hani,
            vka=vka,
            ss=ss,
            sy=sy,
            vkcb=vkcb,
            wetdry=wetdry,
            ksat=ksat,
            storagecoefficient=storagecoefficient,
            constantcv=constantcv,
            thickstrt=thickstrt,
            novfc=novfc,
            nocvcorrection=nocvcorrection,
            unitnumber=unitnumber,
            filenames=filenames,
        )
        if check:
            lpf.check(
                f=f"{lpf.name[0]}.chk",
                verbose=lpf.parent.verbose,
                level=0,
            )
        return lpf

    @staticmethod
    def _load_item1(line, model):
        """Loads LPF item 1 and options."""
        # Item 1: IBCFCB, HDRY, NPLPF - line already read above
        if model.verbose:
            print("   loading IBCFCB, HDRY, NPLPF...")
        text_list = line_parse(line)
        ipakcb, hdry, nplpf = (
            int(text_list[0]),
            float(text_list[1]),
            int(text_list[2]),
        )
        ikcflag = 0
        if not model.structured:
            ikcflag = int(text_list[3])
        storagecoefficient = "STORAGECOEFFICIENT" in [
            item.upper() for item in text_list
        ]
        constantcv = "CONSTANTCV" in [item.upper() for item in text_list]
        thickstrt = "THICKSTRT" in [item.upper() for item in text_list]
        nocvcorrection = "NOCVCORRECTION" in [
            item.upper() for item in text_list
        ]
        novfc = "NOVFC" in [item.upper() for item in text_list]

        return (
            ipakcb,
            hdry,
            nplpf,
            ikcflag,
            storagecoefficient,
            constantcv,
            thickstrt,
            nocvcorrection,
            novfc,
        )

    @staticmethod
    def _load_items_2_to_7(f_obj, model):
        """Loads LPF items 2 through 7."""
        nlay = model.nlay

        # LAYTYP array
        if model.verbose:
            print("   loading LAYTYP...")
        laytyp = np.empty((nlay), dtype=np.int32)
        laytyp = read1d(f_obj, laytyp)

        # LAYAVG array
        if model.verbose:
            print("   loading LAYAVG...")
        layavg = np.empty((nlay), dtype=np.int32)
        layavg = read1d(f_obj, layavg)

        # CHANI array
        if model.verbose:
            print("   loading CHANI...")
        chani = np.empty((nlay), dtype=np.float32)
        chani = read1d(f_obj, chani)

        # LAYVKA array
        if model.verbose:
            print("   loading LAYVKA...")
        layvka = np.empty((nlay,), dtype=np.int32)
        layvka = read1d(f_obj, layvka)

        # LAYWET array
        if model.verbose:
            print("   loading LAYWET...")
        laywet = np.empty((nlay), dtype=np.int32)
        laywet = read1d(f_obj, laywet)

        # Item 7: WETFCT, IWETIT, IHDWET
        wetfct, iwetit, ihdwet = None, None, None
        iwetdry = laywet.sum()
        if iwetdry > 0:
            if model.verbose:
                print("   loading WETFCT, IWETIT, IHDWET...")
            line = f_obj.readline()
            text_list = line.strip().split()
            wetfct, iwetit, ihdwet = (
                float(text_list[0]),
                int(text_list[1]),
                int(text_list[2]),
            )

        return (
            laytyp,
            layavg,
            chani,
            layvka,
            laywet,
            wetfct,
            iwetit,
            ihdwet,
            iwetdry,
        )

    @staticmethod
    def _load_hy_tran_kv_vcont(
        f_obj, model, layer_vars, ext_unit_dict, par_types_parm_dict
    ):
        """
        Loads hy/tran and kv/vcont file entries.

        Parameters
        ----------
        f_obj : open file object.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        layer_vars : dict. Key/value pairs must be:
            "layer": layer number k (base 0),
            "layvka": layvka for layer k,
            "chani": chani for layer k
        ext_unit_dict : external unit dictionary
        par_types_parm_dict : tuple of (par_types, parm_dict)

        Returns
        -------
        hk_k : Numpy array of hk values for layer k (or 0)
        hani_k : Numpy array of hani values for layer k (or 0)
        vka_k : Numpy array of vka values for layer k (or 0)
        """
        par_types, parm_dict = par_types_parm_dict
        layer = layer_vars["layer"]
        layvka_k = layer_vars["layvka"]
        chani_k = layer_vars["chani"]
        util2d_shape = get_util2d_shape_for_layer(model, layer=layer)

        # hk
        if model.verbose:
            print(f"   loading hk layer {layer + 1:3d}...")
        if "hk" not in par_types:
            hk_k = Util2d.load(
                f_obj, model, util2d_shape, np.float32, "hk", ext_unit_dict
            )
        else:
            f_obj.readline()
            hk_k = mfpar.parameter_fill(
                model, util2d_shape, "hk", parm_dict, findlayer=layer
            )

        # hani
        hani_k = 1.0
        if chani_k <= 0.0:
            if model.verbose:
                print(f"   loading hani layer {layer + 1:3d}...")
            if "hani" not in par_types:
                hani_k = Util2d.load(
                    f_obj,
                    model,
                    util2d_shape,
                    np.float32,
                    "hani",
                    ext_unit_dict,
                )
            else:
                f_obj.readline()
                hani_k = mfpar.parameter_fill(
                    model, util2d_shape, "hani", parm_dict, findlayer=layer
                )

        # vka
        if model.verbose:
            print(f"   loading vka layer {layer + 1:3d}...")
        key = "vk"
        if layvka_k != 0:
            key = "vani"
        if "vk" not in par_types and "vani" not in par_types:
            vka_k = Util2d.load(
                f_obj, model, util2d_shape, np.float32, key, ext_unit_dict
            )
        else:
            f_obj.readline()
            key = "vk"
            if "vani" in par_types:
                key = "vani"
            vka_k = mfpar.parameter_fill(
                model, util2d_shape, key, parm_dict, findlayer=layer
            )

        return hk_k, hani_k, vka_k

    def _load_layer_properties(
        self,
        f_obj,
        model,
        dis,
        ikcflag,
        layvka,
        chani,
        laytyp,
        laywet,
        nplpf,
        ext_unit_dict,
    ):
        """Loads layer properties."""
        # parameters data
        par_types = []
        parm_dict = {}
        if nplpf > 0:
            par_types, parm_dict = mfpar.load(f_obj, nplpf, model.verbose)
            # print parm_dict

        # non-parameter data
        transient = not dis.steady.all()
        nlay = model.nlay
        hk = [0] * nlay
        hani = [0] * nlay
        vka = [0] * nlay
        ss = [0] * nlay
        sy = [0] * nlay
        vkcb = [0] * nlay
        wetdry = [0] * nlay

        # load by layer
        for layer in range(nlay):
            util2d_shape = get_util2d_shape_for_layer(model, layer=layer)

            if ikcflag == 0:
                (
                    hk[layer],
                    hani[layer],
                    vka[layer],
                ) = self._load_hy_tran_kv_vcont(
                    f_obj,
                    model,
                    {
                        "layer": layer,
                        "layvka": layvka[layer],
                        "chani": chani[layer],
                    },
                    ext_unit_dict,
                    (par_types, parm_dict),
                )

            # storage properties
            if transient:
                ss[layer], sy[layer] = self._load_storage(
                    f_obj,
                    model,
                    {"layer": layer, "laytyp": laytyp[layer]},
                    ext_unit_dict,
                    (par_types, parm_dict),
                )

            # vkcb
            if ikcflag == 0 and dis.laycbd[layer] != 0:
                if model.verbose:
                    print(f"   loading vkcb layer {layer + 1:3d}...")
                if "vkcb" not in par_types:
                    vkcb[layer] = Util2d.load(
                        f_obj,
                        model,
                        util2d_shape,
                        np.float32,
                        "vkcb",
                        ext_unit_dict,
                    )
                else:
                    _ = f_obj.readline()
                    vkcb[layer] = mfpar.parameter_fill(
                        model, util2d_shape, "vkcb", parm_dict, findlayer=layer
                    )

            # wetdry
            if laywet[layer] != 0 and not (laytyp[layer] not in [0, 4]):
                if model.verbose:
                    print(f"   loading wetdry layer {layer + 1:3d}...")
                wetdry[layer] = Util2d.load(
                    f_obj,
                    model,
                    util2d_shape,
                    np.float32,
                    "wetdry",
                    ext_unit_dict,
                )

        return hk, hani, vka, ss, sy, vkcb, wetdry

    @staticmethod
    def _load_storage(
        f_obj, model, layer_vars, ext_unit_dict, par_types_parm_dict
    ):
        """
        Loads ss, sy file entries.

        Parameters
        ----------
        f_obj : open file object.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        layer_vars : dict. Key/value pairs must be:
            "layer": layer number k (base 0),
            "laytyp": laytyp for layer k
        ext_unit_dict : external unit dictionary
        par_types_parm_dict : tuple of (par_types, parm_dict)

        Returns
        -------
        ss_k : Numpy array of ss values for layer k (or 0)
        sy_k : Numpy array of sy values for layer k (or 0)
        """
        par_types, parm_dict = par_types_parm_dict
        layer = layer_vars["layer"]
        laytyp_k = layer_vars["laytyp"]
        util2d_shape = get_util2d_shape_for_layer(model, layer=layer)

        # ss
        if model.verbose:
            print(f"   loading ss layer {layer + 1:3d}...")
        if "ss" not in par_types:
            ss_k = Util2d.load(
                f_obj, model, util2d_shape, np.float32, "ss", ext_unit_dict
            )
        else:
            f_obj.readline()
            ss_k = mfpar.parameter_fill(
                model, util2d_shape, "ss", parm_dict, findlayer=layer
            )

        # sy
        sy_k = 0.1
        if laytyp_k != 0:
            if model.verbose:
                print(f"   loading sy layer {layer + 1:3d}...")
            if "sy" not in par_types:
                sy_k = Util2d.load(
                    f_obj,
                    model,
                    util2d_shape,
                    np.float32,
                    "sy",
                    ext_unit_dict,
                )
            else:
                f_obj.readline()
                sy_k = mfpar.parameter_fill(
                    model, util2d_shape, "sy", parm_dict, findlayer=layer
                )

        return ss_k, sy_k
