"""
mfupw module.  Contains the ModflowUpw class. Note that the user can access
the ModflowUpw class as `flopy.modflow.ModflowUpw`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/upw_upstream_weighting_package.htm>`_.

"""

import sys
import numpy as np
from .mfpar import ModflowPar as mfpar
from ..pakbase import Package
from ..utils import Util2d, Util3d, read1d
from ..utils.flopy_io import line_parse


class ModflowUpw(Package):
    """
    Upstream weighting package class


    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    hdry : float
        Is the head that is assigned to cells that are converted to dry during
        a simulation. Although this value plays no role in the model
        calculations, it is useful as an indicator when looking at the
        resulting heads that are output from the model. HDRY is thus similar
        to HNOFLO in the Basic Package, which is the value assigned to cells
        that are no-flow cells at the start of a model simulation. (default
        is -1.e30).
    iphdry : int
        iphdry is a flag that indicates whether groundwater head will be set
        to hdry when the groundwater head is less than 0.0001 above the cell
        bottom (units defined by lenuni in the discretization package). If
        iphdry=0, then head will not be set to hdry. If iphdry>0, then head
        will be set to hdry. If the head solution from one simulation will be
        used as starting heads for a subsequent simulation, or if the
        Observation Process is used (Harbaugh and others, 2000), then hdry
        should not be printed to the output file for dry cells (that is, the
        upw package input variable should be set as iphdry=0). (default is 0)
    noparcheck : bool
        noparcheck turns off the checking that a value is defined for all
        cells when parameters are used to define layer data.
    laytyp : int or array of ints (nlay)
        Layer type (default is 0).
    layavg : int or array of ints (nlay)
        Layer average (default is 0).
        0 is harmonic mean
        1 is logarithmic mean
        2 is arithmetic mean of saturated thickness and logarithmic mean of
        of hydraulic conductivity
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
    layvka : int or array of ints (nlay)
        a flag for each layer that indicates whether variable VKA is vertical
        hydraulic conductivity or the ratio of horizontal to vertical
        hydraulic conductivity.
    laywet : int or array of ints (nlay)
        contains a flag for each layer that indicates if wetting is active.
        laywet should always be zero for the UPW Package because all cells
        initially active are wettable.
    hk : float or array of floats (nlay, nrow, ncol)
        is the hydraulic conductivity along rows. HK is multiplied by
        horizontal anisotropy (see CHANI and HANI) to obtain hydraulic
        conductivity along columns. (default is 1.0).
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
        is specific yield. (default is 0.15).
    vkcb : float or array of floats (nlay, nrow, ncol)
        is the vertical hydraulic conductivity of a Quasi-three-dimensional
        confining bed below a layer. (default is 0.0).
    extension : string
        Filename extension (default is 'upw')
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
    >>> m = flopy.modflow.Modflow()
    >>> lpf = flopy.modflow.ModflowLpf(m)

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
        iphdry=0,
        hk=1.0,
        hani=1.0,
        vka=1.0,
        ss=1e-5,
        sy=0.15,
        vkcb=0.0,
        noparcheck=False,
        extension="upw",
        unitnumber=None,
        filenames=None,
    ):

        if model.version != "mfnwt":
            err = (
                "Error: model version must be mfnwt to use "
                + "{} package".format(ModflowUpw._ftype())
            )
            raise Exception(err)

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowUpw._defaultunit()

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
                ipakcb, fname=fname, package=ModflowUpw._ftype()
            )
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowUpw._ftype()]
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

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.url = "upw_upstream_weighting_package.htm"

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # item 1
        self.ipakcb = ipakcb
        # Head in cells that are converted to dry during a simulation
        self.hdry = hdry
        # number of UPW parameters
        self.npupw = 0
        self.iphdry = iphdry
        self.laytyp = Util2d(model, (nlay,), np.int32, laytyp, name="laytyp")
        self.layavg = Util2d(model, (nlay,), np.int32, layavg, name="layavg")
        self.chani = Util2d(model, (nlay,), np.float32, chani, name="chani")
        self.layvka = Util2d(model, (nlay,), np.int32, layvka, name="vka")
        self.laywet = Util2d(model, (nlay,), np.int32, laywet, name="laywet")

        self.options = " "
        if noparcheck:
            self.options = self.options + "NOPARCHECK  "

        self.hk = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            hk,
            name="hk",
            locat=self.unit_number[0],
        )
        self.hani = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            hani,
            name="hani",
            locat=self.unit_number[0],
        )
        keys = []
        for k in range(nlay):
            key = "vka"
            if self.layvka[k] != 0:
                key = "vani"
            keys.append(key)
        self.vka = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            vka,
            name=keys,
            locat=self.unit_number[0],
        )
        self.ss = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            ss,
            name="ss",
            locat=self.unit_number[0],
        )
        self.sy = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            sy,
            name="sy",
            locat=self.unit_number[0],
        )
        self.vkcb = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            vkcb,
            name="vkcb",
            locat=self.unit_number[0],
        )
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
                f="{}.chk".format(self.name[0]),
                verbose=self.parent.verbose,
                level=1,
            )
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        if f is not None:
            f_upw = f
        else:
            f_upw = open(self.fn_path, "w")
        # Item 0: text
        f_upw.write("{}\n".format(self.heading))
        # Item 1: IBCFCB, HDRY, NPLPF
        f_upw.write(
            "{0:10d}{1:10.3G}{2:10d}{3:10d}{4:s}\n".format(
                self.ipakcb, self.hdry, self.npupw, self.iphdry, self.options
            )
        )
        # LAYTYP array
        f_upw.write(self.laytyp.string)
        # LAYAVG array
        f_upw.write(self.layavg.string)
        # CHANI array
        f_upw.write(self.chani.string)
        # LAYVKA array
        f_upw.write(self.layvka.string)
        # LAYWET array
        f_upw.write(self.laywet.string)
        # Item 7: WETFCT, IWETIT, IHDWET
        iwetdry = self.laywet.sum()
        if iwetdry > 0:
            raise Exception("LAYWET should be 0 for UPW")
        transient = not self.parent.get_package("DIS").steady.all()
        for k in range(nlay):
            f_upw.write(self.hk[k].get_file_entry())
            if self.chani[k] < 1:
                f_upw.write(self.hani[k].get_file_entry())
            f_upw.write(self.vka[k].get_file_entry())
            if transient:
                f_upw.write(self.ss[k].get_file_entry())
                if self.laytyp[k] != 0:
                    f_upw.write(self.sy[k].get_file_entry())
            if self.parent.get_package("DIS").laycbd[k] > 0:
                f_upw.write(self.vkcb[k].get_file_entry())
            if self.laywet[k] != 0 and self.laytyp[k] != 0:
                f_upw.write(self.laywet[k].get_file_entry())
        f_upw.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None, check=True):
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        dis : ModflowUPW object
            ModflowLpf object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> upw = flopy.modflow.ModflowUpw.load('test.upw', m)

        """

        if model.verbose:
            sys.stdout.write("loading upw package file...\n")

        if model.version != "mfnwt":
            msg = (
                "Warning: model version was reset from "
                + "'{}' to 'mfnwt' in order to load a UPW file".format(
                    model.version
                )
            )
            print(msg)
            model.version = "mfnwt"

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
        # Item 1: IBCFCB, HDRY, NPLPF - line already read above
        if model.verbose:
            print("   loading ipakcb, HDRY, NPUPW, IPHDRY...")
        t = line_parse(line)
        ipakcb, hdry, npupw, iphdry = (
            int(t[0]),
            float(t[1]),
            int(t[2]),
            int(t[3]),
        )

        # options
        noparcheck = False
        if len(t) > 3:
            for k in range(3, len(t)):
                if "NOPARCHECK" in t[k].upper():
                    noparcheck = True

        # LAYTYP array
        if model.verbose:
            print("   loading LAYTYP...")
        laytyp = np.empty((nlay,), dtype=np.int32)
        laytyp = read1d(f, laytyp)

        # LAYAVG array
        if model.verbose:
            print("   loading LAYAVG...")
        layavg = np.empty((nlay,), dtype=np.int32)
        layavg = read1d(f, layavg)

        # CHANI array
        if model.verbose:
            print("   loading CHANI...")
        chani = np.empty((nlay,), dtype=np.float32)
        chani = read1d(f, chani)

        # LAYVKA array
        if model.verbose:
            print("   loading LAYVKA...")
        layvka = np.empty((nlay,), dtype=np.int32)
        layvka = read1d(f, layvka)

        # LAYWET array
        if model.verbose:
            print("   loading LAYWET...")
        laywet = np.empty((nlay,), dtype=np.int32)
        laywet = read1d(f, laywet)

        # check that LAYWET is 0 for all layers
        iwetdry = laywet.sum()
        if iwetdry > 0:
            raise Exception("LAYWET should be 0 for UPW")

        # get parameters
        par_types = []
        if npupw > 0:
            par_types, parm_dict = mfpar.load(f, npupw, model.verbose)

        # get arrays
        transient = not model.get_package("DIS").steady.all()
        hk = [0] * nlay
        hani = [0] * nlay
        vka = [0] * nlay
        ss = [0] * nlay
        sy = [0] * nlay
        vkcb = [0] * nlay
        # load by layer
        for k in range(nlay):

            # hk
            if model.verbose:
                print("   loading hk layer {0:3d}...".format(k + 1))
            if "hk" not in par_types:
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "hk", ext_unit_dict
                )
            else:
                line = f.readline()
                t = mfpar.parameter_fill(
                    model, (nrow, ncol), "hk", parm_dict, findlayer=k
                )
            hk[k] = t

            # hani
            if chani[k] < 1:
                if model.verbose:
                    print("   loading hani layer {0:3d}...".format(k + 1))
                if "hani" not in par_types:
                    t = Util2d.load(
                        f,
                        model,
                        (nrow, ncol),
                        np.float32,
                        "hani",
                        ext_unit_dict,
                    )
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(
                        model, (nrow, ncol), "hani", parm_dict, findlayer=k
                    )
                hani[k] = t

            # vka
            if model.verbose:
                print("   loading vka layer {0:3d}...".format(k + 1))
            key = "vk"
            if layvka[k] != 0:
                key = "vani"
            if "vk" not in par_types and "vani" not in par_types:
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, key, ext_unit_dict
                )
            else:
                line = f.readline()
                t = mfpar.parameter_fill(
                    model, (nrow, ncol), key, parm_dict, findlayer=k
                )
            vka[k] = t

            # storage properties
            if transient:

                # ss
                if model.verbose:
                    print("   loading ss layer {0:3d}...".format(k + 1))
                if "ss" not in par_types:
                    t = Util2d.load(
                        f, model, (nrow, ncol), np.float32, "ss", ext_unit_dict
                    )
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(
                        model, (nrow, ncol), "ss", parm_dict, findlayer=k
                    )
                ss[k] = t

                # sy
                if laytyp[k] != 0:
                    if model.verbose:
                        print("   loading sy layer {0:3d}...".format(k + 1))
                    if "sy" not in par_types:
                        t = Util2d.load(
                            f,
                            model,
                            (nrow, ncol),
                            np.float32,
                            "sy",
                            ext_unit_dict,
                        )
                    else:
                        line = f.readline()
                        t = mfpar.parameter_fill(
                            model, (nrow, ncol), "sy", parm_dict, findlayer=k
                        )
                    sy[k] = t

            # vkcb
            if model.get_package("DIS").laycbd[k] > 0:
                if model.verbose:
                    print("   loading vkcb layer {0:3d}...".format(k + 1))
                if "vkcb" not in par_types:
                    t = Util2d.load(
                        f,
                        model,
                        (nrow, ncol),
                        np.float32,
                        "vkcb",
                        ext_unit_dict,
                    )
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(
                        model, (nrow, ncol), "vkcb", parm_dict, findlayer=k
                    )
                vkcb[k] = t

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowUpw._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)

        # create upw object
        upw = cls(
            model,
            ipakcb=ipakcb,
            iphdry=iphdry,
            hdry=hdry,
            noparcheck=noparcheck,
            laytyp=laytyp,
            layavg=layavg,
            chani=chani,
            layvka=layvka,
            laywet=laywet,
            hk=hk,
            hani=hani,
            vka=vka,
            ss=ss,
            sy=sy,
            vkcb=vkcb,
            unitnumber=unitnumber,
            filenames=filenames,
        )
        if check:
            upw.check(
                f="{}.chk".format(upw.name[0]),
                verbose=upw.parent.verbose,
                level=0,
            )

        # return upw object
        return upw

    @staticmethod
    def _ftype():
        return "UPW"

    @staticmethod
    def _defaultunit():
        return 31
