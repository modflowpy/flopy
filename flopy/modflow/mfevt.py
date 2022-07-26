"""
mfghb module.  Contains the ModflowEvt class. Note that the user can access
the ModflowEvt class as `flopy.modflow.ModflowEvt`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/evt.html>`_.

"""
import numpy as np

from ..pakbase import Package
from ..utils import Transient2d, Util2d
from ..utils.utils_def import get_pak_vals_shape
from .mfparbc import ModflowParBc as mfparbc


class ModflowEvt(Package):
    """
    MODFLOW Evapotranspiration Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.ModflowEvt`) to which
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
    extension : string
        Filename extension (default is 'evt')
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
    >>> m = flopy.modflow.Modflow()
    >>> evt = flopy.modflow.ModflowEvt(m, nevtop=3, evtr=1.2e-4)

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
        extension="evt",
        unitnumber=None,
        filenames=None,
        external=True,
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowEvt._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            model.add_output_file(
                ipakcb, fname=filenames[1], package=self._ftype()
            )
        else:
            ipakcb = 0

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
        self.ipakcb = ipakcb
        self.external = external
        if self.external is False:
            load = True
        else:
            load = model.load

        surf_u2d_shape = get_pak_vals_shape(model, evtr)
        evtr_u2d_shape = get_pak_vals_shape(model, evtr)
        exdp_u2d_shape = get_pak_vals_shape(model, exdp)
        ievt_u2d_shape = get_pak_vals_shape(model, ievt)

        self.surf = Transient2d(
            model, surf_u2d_shape, np.float32, surf, name="surf"
        )
        self.evtr = Transient2d(
            model, evtr_u2d_shape, np.float32, evtr, name="evtr"
        )
        self.exdp = Transient2d(
            model, exdp_u2d_shape, np.float32, exdp, name="exdp"
        )
        self.ievt = Transient2d(
            model, ievt_u2d_shape, np.int32, ievt, name="ievt"
        )
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
        f_evt.write(f"{self.nevtop:10d}{self.ipakcb:10d}\n")

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
                    [
                        u2d.array.size
                        for kper, u2d in self.ievt.transient_2ds.items()
                    ]
                )
                f_evt.write(f"{mxndevt:10d}\n")

        for n in range(nper):
            insurf, surf = self.surf.get_kper_entry(n)
            inevtr, evtr = self.evtr.get_kper_entry(n)
            inexdp, exdp = self.exdp.get_kper_entry(n)
            inievt = 0
            if self.nevtop == 2:
                inievt, file_entry_ievt = ievt.get_kper_entry(n)
                if inievt >= 0 and not self.parent.structured:
                    inievt = self.ievt[n].array.size
            comment = f"Evapotranspiration dataset 5 for stress period {n + 1}"
            f_evt.write(
                f"{insurf:10d}{inevtr:10d}{inexdp:10d}{inievt:10d} # {comment}\n"
            )
            if insurf >= 0:
                f_evt.write(surf)
            if inevtr >= 0:
                f_evt.write(evtr)
            if inexdp >= 0:
                f_evt.write(exdp)
            if self.nevtop == 2 and inievt >= 0:
                f_evt.write(file_entry_ievt)
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
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
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
        evt : ModflowEvt object
            ModflowEvt object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> evt = flopy.modflow.mfevt.load('test.evt', m)

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
                    print(
                        "  Parameters detected. Number of parameters = ", npar
                    )
            line = f.readline()
        # Dataset 2
        t = line.strip().split()
        nevtop = int(t[0])
        ipakcb = int(t[1])

        # dataset 2b for mfusg
        if not model.structured and nevtop == 2:
            line = f.readline()
            t = line.strip().split()
            mxndevt = int(t[0])

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
        current_surf = []
        current_evtr = []
        current_exdp = []
        current_ievt = []
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

            if insurf >= 0:
                if model.verbose:
                    print(f"   loading surf stress period {iper + 1:3d}...")
                t = Util2d.load(
                    f, model, u2d_shape, np.float32, "surf", ext_unit_dict
                )
                current_surf = t
            surf[iper] = current_surf

            if inevtr >= 0:
                if npar == 0:
                    if model.verbose:
                        print(
                            f"   loading evtr stress period {iper + 1:3d}..."
                        )
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
                    t = mfparbc.parameter_bcfill(
                        model, u2d_shape, parm_dict, pak_parms
                    )

                current_evtr = t
            evtr[iper] = current_evtr
            if inexdp >= 0:
                if model.verbose:
                    print(f"   loading exdp stress period {iper + 1:3d}...")
                t = Util2d.load(
                    f, model, u2d_shape, np.float32, "exdp", ext_unit_dict
                )
                current_exdp = t
            exdp[iper] = current_exdp
            if nevtop == 2:
                if inievt >= 0:
                    if model.verbose:
                        print(
                            f"   loading ievt stress period {iper + 1:3d}..."
                        )
                    t = Util2d.load(
                        f, model, u2d_shape, np.int32, "ievt", ext_unit_dict
                    )
                    current_ievt = Util2d(
                        model, u2d_shape, np.int32, t.array - 1, "ievt"
                    )
                ievt[iper] = current_ievt

        if openfile:
            f.close()

        # create evt object
        args = {}
        if ievt:
            args["ievt"] = ievt
        if nevtop:
            args["nevtop"] = nevtop
        if evtr:
            args["evtr"] = evtr
        if surf:
            args["surf"] = surf
        if exdp:
            args["exdp"] = exdp
        args["ipakcb"] = ipakcb

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowEvt._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)

        # set args for unitnumber and filenames
        args["unitnumber"] = unitnumber
        args["filenames"] = filenames

        evt = cls(model, **args)

        # return evt object
        return evt

    @staticmethod
    def _ftype():
        return "EVT"

    @staticmethod
    def _defaultunit():
        return 22
