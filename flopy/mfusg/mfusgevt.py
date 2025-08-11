"""
mfusgevt module.  Contains the MfUsgEvt class. Note that the user can access
the MfUsgEvt class as `flopy.mfusg.MfUsgEvt`.

"""

import numpy as np

from ..modflow.mfparbc import ModflowParBc as mfparbc
from ..pakbase import Package
from ..utils import Transient2d, Util2d
from ..utils.utils_def import (
    get_pak_vals_shape,
    type_from_iterable,
)


class MfUsgEvt(Package):
    """
    MODFLOW Evapotranspiration Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mfusg.MfUsgEvt`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    nevtop : int
        is the recharge option code.
        1: ET is calculated only for cells in the top grid layer
        2: ET to layer defined in ievt
        3: ET to highest active cell (default is 3).
    surf : float or filename or ndarray or dict keyed on kper (zero-based)
        is the ET surface elevation. (default is 0.0, which is used for all
        stress periods).
    evtr: float or filename or ndarray or dict keyed on kper (zero-based)
        is the maximum ET flux (default is 1e-3, which is used for all
        stress periods).
    exdp : float or filename or ndarray or dict keyed on kper (zero-based)
        is the ET extinction depth (default is 1.0, which is used for all
        stress periods).
    ievt : int or filename or ndarray or dict keyed on kper (zero-based)
        is the layer indicator variable (default is 1, which is used for all
        stress periods).
    etfactor : float of array (mcomp) (default is 1.0)
        fraction of mass of the component that leaves with water
        0 = chemical component left behind in groundwater
        1 = chemical component leaves with water
        between 0 and 1 = fraction of mass of the component leaves
    iznevt : float of array (mcomp) (default is 1.0)
        array of zonal indices for applying a PET time series to zones.
        This PET input is independent of the stress period input, which
        is ignored when the zonal time series are provided.
    extension : string
        Filename extension (default is 'evt')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, mfusgtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
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
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mfusg.MfUsg()
    >>> evt = flopy.mfusg.MfUsgEvt(m, nevtop=3, evtr=1.2e-4)

    """

    def __init__(
        self,
        model,
        nevtop=3,
        ipakcb=None,
        surf=0.0,
        evtr=1e-3,
        exdp=1.0,
        ievt=1,
        mxetzones=0,
        ietfactor=0,
        etfactor=0.0,
        inznevt=0,
        iznevt=0,
        extension="evt",
        unitnumber=None,
        filenames=None,
        external=True,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = MfUsgEvt._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # cbc output file
        self.set_cbc_output_file(ipakcb, model, filenames[1])

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self._generate_heading()
        self.url = "evt.html"
        self.nevtop = nevtop

        self.mxetzones = mxetzones
        self.ietfactor = ietfactor
        self.etfactor = etfactor

        self.external = external
        if self.external is False:
            load = True
        else:
            load = model.load

        surf_u2d_shape = get_pak_vals_shape(model, surf)
        evtr_u2d_shape = get_pak_vals_shape(model, evtr)
        exdp_u2d_shape = get_pak_vals_shape(model, exdp)
        ievt_u2d_shape = get_pak_vals_shape(model, ievt)

        self.surf = Transient2d(model, surf_u2d_shape, np.float32, surf, name="surf")
        self.evtr = Transient2d(model, evtr_u2d_shape, np.float32, evtr, name="evtr")
        self.exdp = Transient2d(model, exdp_u2d_shape, np.float32, exdp, name="exdp")
        self.ievt = Transient2d(model, ievt_u2d_shape, np.int32, ievt, name="ievt")

        # self.inznevt = [inznevt]*nper
        # self.iznevt = iznevt
        # iznevt_u2d_shape = get_pak_vals_shape(model, iznevt)
        # self.iznevt = Transient2d(model, iznevt_u2d_shape,
        # np.int32, iznevt, name="iznevt")

        self.np = 0
        self.parent.add_package(self)

    def _ncells(self):
        """Maximum number of cells that have evapotranspiration (developed for
        MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of evt cells

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        return nrow * ncol

    def write_file(self, f=None):
        """
        Write the package file.

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        if f is not None:
            f_evt = f
        else:
            f_evt = open(self.fn_path, "w")
        f_evt.write(f"{self.heading}\n")
        f_evt.write(f"{self.nevtop:10d}{self.ipakcb:10d}")

        if self.parent.itrnsp and self.ietfactor != 0:
            f_evt.write(f"{self.ietfactor:10d}")
        if self.mxetzones > 0:
            f_evt.write(f"ETS {self.mxetzones:10d}")
        f_evt.write("\n")

        if self.nevtop == 2:
            ievt = {}
            for kper, u2d in self.ievt.transient_2ds.items():
                ievt[kper] = u2d.array + 1
            ievt = Transient2d(
                self.parent,
                self.ievt.shape,
                self.ievt.dtype,
                ievt,
                self.ievt.name,
            )
            if not self.parent.structured:
                mxndevt = np.max(
                    [u2d.array.size for kper, u2d in self.ievt.transient_2ds.items()]
                )
                f_evt.write(f"{mxndevt:10d}\n")

        if self.parent.itrnsp and self.ietfactor == 1:
            mcomp = self.parent.mcomp
            for icomp in range(mcomp):
                f_evt.write(f"{self.etfactor[icomp]:10.2e}")
            f_evt.write("\n")

        for n in range(nper):
            insurf, surf = self.surf.get_kper_entry(n)
            inevtr, evtr = self.evtr.get_kper_entry(n)
            inexdp, exdp = self.exdp.get_kper_entry(n)
            inievt = -1
            if self.nevtop == 2:
                inievt, file_entry_ievt = ievt.get_kper_entry(n)
                if inievt >= 0 and not self.parent.structured:
                    inievt = self.ievt[n].array.size
            comment = f"Evapotranspiration dataset 5 for stress period {n + 1}"
            f_evt.write(f"{insurf:10d}{inevtr:10d}{inexdp:10d}{inievt:10d} ")
            # if self.inznevt[n] > 0:
            #     f_evt.write(f"INEVTZONES {self.inznevt[n]:10d}\n")
            f_evt.write(f"#{comment}\n")

            if insurf >= 0:
                f_evt.write(surf)
            if inevtr >= 0:
                f_evt.write(evtr)
            if inexdp >= 0:
                f_evt.write(exdp)
            if self.nevtop == 2 and inievt >= 0:
                f_evt.write(file_entry_ievt)

            # if self.inznevt[n] > 0:
            #     iznevt = self.iznevt.get_kper_entry(n)
            #     f_evt.write(iznevt)

        f_evt.close()

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mfusg.mf.MfUsg`) to
            which this package will be added.
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        evt : MfUsgEvt object
            MfUsgEvt object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.mfusg.MfUsg()
        >>> evt = flopy.mfusg.mfevt.load('test.evt', m)

        """
        if model.verbose:
            print("loading evt package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        npar = 0
        if "parameter" in line.lower():
            raw = line.strip().split()
            npar = int(raw[1])
            if npar > 0:
                if model.verbose:
                    print("  Parameters detected. Number of parameters = ", npar)
            line = f.readline()
        # Dataset 2
        t = line.strip().split()
        nevtop = int(t[0])
        ipakcb = int(t[1])
        ietfactor = type_from_iterable(t, 2)

        # Options
        mxetzones = 0
        if "ETS" in t:
            idx = t.index("ETS")
            mxetzones = float(t[idx + 1])

        # dataset 2b for mfusg
        if not model.structured and nevtop == 2:
            line = f.readline()
            t = line.strip().split()
            mxndevt = int(t[0])

        # dataset 2c for mfusg
        etfactor = None
        mcomp = model.mcomp
        if mcomp > 0:
            etfactor = np.zeros(model.mcomp)
            if ietfactor < 0:
                etfactor = np.ones(mcomp)
            if ietfactor > 0:
                if mcomp > 0:
                    line = f.readline()
                    t = line.strip().split()
                    for icomp in range(mcomp):
                        etfactor[icomp] = float(t[icomp])

        # Dataset 3 and 4 - parameters data
        pak_parms = None
        if npar > 0:
            pak_parms = mfparbc.loadarray(f, npar, model.verbose)

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
        else:
            nrow, ncol, nlay, _ = model.get_nrow_ncol_nlay_nper()

        u2d_shape = (nrow, ncol)

        # Read data for every stress period
        surf = {}
        evtr = {}
        exdp = {}
        ievt = {}
        # iznevt = {}
        current_surf = []
        current_evtr = []
        current_exdp = []
        current_ievt = []
        current_iznevt = []
        # inznevt = [0] * nper
        for iper in range(nper):
            line = f.readline()
            t = line.strip().split()
            insurf = int(t[0])
            inevtr = int(t[1])
            inexdp = int(t[2])

            if nevtop == 2:
                inievt = int(t[3])
                if (not model.structured) and (inievt >= 0):
                    u2d_shape = (1, inievt)
            elif not model.structured:
                u2d_shape = (1, ncol[0])

            # if "INEVTZONES" in t:
            #     idx = t.index("INEVTZONES")
            #     inznevt[iper] = int(t[idx + 1])

            if insurf >= 0:
                if model.verbose:
                    print(f"   loading surf stress period {iper + 1:3d}...")
                t = Util2d.load(f, model, u2d_shape, np.float32, "surf", ext_unit_dict)
                current_surf = t
            surf[iper] = current_surf

            if inevtr >= 0:
                if npar == 0:
                    if model.verbose:
                        print(f"   loading evtr stress period {iper + 1:3d}...")
                    t = Util2d.load(
                        f,
                        model,
                        u2d_shape,
                        np.float32,
                        "evtr",
                        ext_unit_dict,
                    )
                else:
                    parm_dict = {}
                    for ipar in range(inevtr):
                        line = f.readline()
                        t = line.strip().split()
                        c = t[0].lower()
                        if len(c) > 10:
                            c = c[0:10]
                        pname = c
                        try:
                            c = t[1].lower()
                            instance_dict = pak_parms.bc_parms[pname][1]
                            if c in instance_dict:
                                iname = c
                            else:
                                iname = "static"
                        except:
                            iname = "static"
                        parm_dict[pname] = iname
                    t = mfparbc.parameter_bcfill(model, u2d_shape, parm_dict, pak_parms)

                current_evtr = t
            evtr[iper] = current_evtr
            if inexdp >= 0:
                if model.verbose:
                    print(f"   loading exdp stress period {iper + 1:3d}...")
                t = Util2d.load(f, model, u2d_shape, np.float32, "exdp", ext_unit_dict)
                current_exdp = t
            exdp[iper] = current_exdp
            if nevtop == 2:
                if inievt >= 0:
                    if model.verbose:
                        print(f"   loading ievt stress period {iper + 1:3d}...")
                    t = Util2d.load(
                        f, model, u2d_shape, np.int32, "ievt", ext_unit_dict
                    )
                    current_ievt = Util2d(
                        model, u2d_shape, np.int32, t.array - 1, "ievt"
                    )
                ievt[iper] = current_ievt
            # if inznevt[iper] > 0:
            #     if model.verbose:
            #         print(f"   loading iznevt stress period {iper + 1:3d}...")
            #     current_iznevt = Util2d.load(f, model,
            #     (inznevt[iper],), np.int32, "iznevt", ext_unit_dict)
            # iznevt[iper] = current_iznevt

        if openfile:
            f.close()

        # create evt object
        args = {}
        args["ievt"] = ievt
        args["nevtop"] = nevtop
        args["evtr"] = evtr
        args["surf"] = surf
        args["exdp"] = exdp
        args["ipakcb"] = ipakcb

        args["mxetzones"] = mxetzones
        args["etfactor"] = etfactor
        # args["inznevt"] = inznevt
        # args["iznevt"] = iznevt

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )
            _, filenames[1] = model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)

        # return evt object
        return cls(model, unitnumber=unitnumber, filenames=filenames, **args)

    @staticmethod
    def _ftype():
        return "EVT"

    @staticmethod
    def _defaultunit():
        return 22
