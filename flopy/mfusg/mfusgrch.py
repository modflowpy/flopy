"""
mfusgrch module.  Contains the MfUsgRch class. Note that the user can access
the MfUsgRch class as `flopy.mfusg.MfUsgRch`.

"""

import numpy as np

from ..modflow.mfparbc import ModflowParBc as mfparbc
from ..modflow.mfrch import ModflowRch
from ..utils import Transient2d, Transient3d, Util2d, read1d
from ..utils.flopy_io import line_parse
from ..utils.utils_def import (
    get_pak_vals_shape,
    get_unitnumber_from_ext_unit_dict,
)
from .mfusg import MfUsg


class MfUsgRch(ModflowRch):
    """
    MFUSG TRANSPORT Recharge Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int, optional
        Toggles whether cell-by-cell budget data should be saved. If None or zero,
        budget data will not be saved (default is None).
    nrchop : int
        is the recharge option code.
        1: Recharge to top grid layer only
        2: Recharge to layer defined in irch
        3: Recharge to highest active cell (default is 3).
    rech : float or filename or ndarray or dict keyed on kper (zero-based)
        Recharge flux (default is 1.e-3, which is used for all stress periods)
    irch : int or filename or ndarray or dict keyed on kper (zero-based)
        Layer (for an unstructured grid) or node (for an unstructured grid) to
        which recharge is applied in each vertical column (only used when
        nrchop=2). Default is 0, which is used for all stress periods.
    extension : string
        Filename extension (default is 'rch')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcb is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
        created using the model name and .cbc extension, if ipakcb is a
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
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    """

    def __init__(
        self,
        model,
        nrchop=3,
        ipakcb=None,
        rech=1e-3,
        irch=0,
        seepelev=0,
        mxrtzones=0,
        iconc=0,
        selev=None,
        iznrch=None,
        rchconc=None,
        unitnumber=None,
        filenames=None,
    ):
        """Package constructor."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # call base package constructor
        super().__init__(
            model,
            nrchop=nrchop,
            ipakcb=ipakcb,
            rech=rech,
            irch=irch,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        self.mxrtzones = mxrtzones
        self.seepelev = seepelev
        self.iconc = iconc

        self.selev = None
        if selev is not None:
            selev_u2d_shape = get_pak_vals_shape(model, selev)
            self.selev = Transient2d(
                model, selev_u2d_shape, np.float32, rech, name="rech_selev"
            )

        self.iznrch = None
        if iznrch is not None:
            iznrch_u2d_shape = get_pak_vals_shape(model, iznrch)
            self.iznrch = Transient2d(
                model, iznrch_u2d_shape, np.int32, rech, name="rech_izn"
            )

        self.rchconc = None
        if rchconc is not None:
            mcomp = model.mcomp
            if model.iheat > 0:
                mcomp = model.mcomp + 1
            self.irchconc = [1] * mcomp
            rchconc_u3d_shape = (mcomp,) + get_pak_vals_shape(model, rech)
            self.rchconc = Transient3d(
                model, rchconc_u3d_shape, np.float32, rchconc, name="rech_conc"
            )

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
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        if f is not None:
            f_rch = f
        else:
            f_rch = open(self.fn_path, "w")
        f_rch.write(f"{self.heading}\n")
        f_rch.write(f"{self.nrchop:10d}{self.ipakcb:10d}")
        if self.seepelev:
            f_rch.write(" SEEPELEV")
        if self.iconc:
            f_rch.write(" CONC")
        if self.mxrtzones:
            f_rch.write(f" RTS {self.mxrtzones:4.0d}")
        f_rch.write("\n")

        mcomp = self.parent.mcomp
        if self.parent.iheat > 0:
            mcomp = mcomp + 1

        if self.nrchop == 2:
            irch = {}
            for kper, u2d in self.irch.transient_2ds.items():
                irch[kper] = u2d.array + 1
            irch = Transient2d(
                self.parent,
                self.irch.shape,
                self.irch.dtype,
                irch,
                self.irch.name,
            )
            if not self.parent.structured:
                mxndrch = np.max(
                    [u2d.array.size for kper, u2d in self.irch.transient_2ds.items()]
                )
                f_rch.write(f"{mxndrch:10d}\n")

        if self.iconc:
            for icomp in range(mcomp):
                f_rch.write(f"{self.irchconc[icomp]:10.0f}")
            f_rch.write("\n")

        for kper in range(nper):
            inrech, file_entry_rech = self.rech.get_kper_entry(kper)
            if self.nrchop == 2:
                inirch, file_entry_irch = irch.get_kper_entry(kper)
                if not self.parent.structured:
                    inirch = self.rech[kper].array.size
            else:
                inirch = -1

            f_rch.write(f"{inrech:10d}{inirch:10d} ")

            if self.iznrch is not None:
                f_rch.write(" INRCHZONES 1")
            if self.selev is not None:
                f_rch.write(" INSELEV 1")
            if self.rchconc is not None:
                f_rch.write(" INCONC 1")
            f_rch.write("# Stress period {kper + 1}\n")

            if inrech >= 0:
                f_rch.write(file_entry_rech)
            if self.nrchop == 2:
                if inirch >= 0:
                    f_rch.write(file_entry_irch)

            if self.iznrch is not None:
                inrech, file_entry_iznrch = self.iznrch.get_kper_entry(kper)
                f_rch.write(file_entry_iznrch)
            if self.selev is not None:
                inrech, file_entry_selev = self.selev.get_kper_entry(kper)
                f_rch.write(file_entry_selev)
            if self.rchconc is not None:
                inrech, file_entry_rchconc = self.rchconc.get_kper_entry(kper)
                f_rch.write(file_entry_rchconc)
        f_rch.close()

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None, check=True):
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        rch : MfUsgRch object
            MfUsgRch object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> rch = flopy.modflow.MfUsgRch.load('test.rch', m)

        """
        if model.verbose:
            print("loading rch package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
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
                    print(f"   Parameters detected. Number of parameters = {npar}")
            line = f.readline()
        # dataset 2
        t = line_parse(line)
        nrchop = int(t[0])
        ipakcb = int(t[1])

        # item 2 - options
        seepelev = 0
        if "SEEPELEV" in t:
            seepelev = 1

        iconc = 0
        if "CONC" in line:
            iconc = 1

        mxrtzones = 0
        if "RTS" in t:
            idx = t.index("RTS")
            mxrtzones = int(t[idx + 1])

        # dataset 2b for mfusg
        if not model.structured and nrchop == 2:
            line = f.readline()
            t = line_parse(line)
            mxndrch = int(t[0])

        mcomp = model.mcomp
        if model.iheat > 0:
            mcomp = model.mcomp + 1

        irchconc = np.empty((mcomp), dtype=np.int32)
        if iconc:
            if model.verbose:
                print(f"   loading IRCHCONC[{mcomp}] ...")
            irchconc = read1d(f, irchconc)

        print(f"irchconc: {irchconc}")

        # dataset 3 and 4 - parameters data
        pak_parms = None
        if npar > 0:
            pak_parms = mfparbc.loadarray(f, npar, model.verbose)

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
        else:
            nrow, ncol, nlay, _ = model.get_nrow_ncol_nlay_nper()

        u2d_shape = (nrow, ncol)

        # read data for every stress period
        current_rech = []
        rech = {}
        irch = None
        if nrchop == 2:
            irch = {}
        current_irch = []

        iznrch = None
        if mxrtzones:
            iznrch = {}
        iniznrch = [0] * nper
        current_iznrch = []

        selev = None
        if seepelev:
            selev = {}
        inselev = [0] * nper
        current_selev = []

        rchconc = None
        if iconc:
            rchconc = {}
        inconc = [0] * nper
        current_rchconc = []

        for iper in range(nper):
            line = f.readline()
            t = line_parse(line)
            inrech = int(t[0])

            if nrchop == 2:
                inirch = int(t[1])
                if (not model.structured) and (inirch >= 0):
                    u2d_shape = (1, inirch)
            elif not model.structured:
                u2d_shape = (1, ncol[0])

            if "INSELEV" in t:
                idx = t.index("INSELEV")
                inselev[iper] = int(t[idx + 1])

            if "INRCHZONES" in t:
                idx = t.index("INIZNRCH")
                iniznrch[iper] = int(t[idx + 1])

            if "INCONC" in t:
                inconc[iper] = 1

            if inrech >= 0:
                if npar == 0:
                    if model.verbose:
                        print(f"   loading rech stress period {iper + 1:3d}...")
                    t = Util2d.load(
                        f,
                        model,
                        u2d_shape,
                        np.float32,
                        "rech",
                        ext_unit_dict,
                    )
                else:
                    parm_dict = {}
                    for ipar in range(inrech):
                        line = f.readline()
                        t = line.strip().split()
                        pname = t[0].lower()
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

                current_rech = t
            rech[iper] = current_rech

            if nrchop == 2:
                if inirch >= 0:
                    if model.verbose:
                        print(f"   loading irch stress period {iper + 1:3d}...")
                    t = Util2d.load(
                        f, model, u2d_shape, np.int32, "irch", ext_unit_dict
                    )
                    current_irch = Util2d(
                        model, u2d_shape, np.int32, t.array - 1, "irch"
                    )
                irch[iper] = current_irch

            # Item 9 for mfusg transport
            if mxrtzones:
                if iniznrch[iper]:
                    if model.verbose:
                        print(f"   loading iznrch stress period {iper + 1:3d}...")
                    current_iznrch = Util2d.load(
                        f, model, u2d_shape, np.int32, "iznrch", ext_unit_dict
                    )
                iznrch[iper] = current_iznrch

            # Item 10 for mfusg transport
            if seepelev:
                if inselev[iper]:
                    if model.verbose:
                        print(f"   loading selev stress period {iper + 1:3d}...")
                    current_selev = Util2d.load(
                        f, model, u2d_shape, np.float32, "selev", ext_unit_dict
                    )
                selev[iper] = current_selev

            # Item 11 for mfusg transport
            if iconc:
                if inconc[iper]:
                    if model.verbose:
                        print(f"   loading rch conc stress period {iper + 1:3d}...")
                    if model.iheat > 0:
                        mcomp = model.mcomp + 1

                    current_rchconc = [0] * mcomp
                    for icomp in range(mcomp):
                        if irchconc[icomp]:
                            if model.verbose:
                                print(
                                    f"   loading rch conc stress period {iper + 1:3d}"
                                    f" component {icomp + 1}..."
                                )
                            current_rchconc[icomp] = Util2d.load(
                                f,
                                model,
                                u2d_shape,
                                np.float32,
                                "rchconc",
                                ext_unit_dict,
                            )
                rchconc[iper] = current_rchconc

        if openfile:
            f.close()

        unitnumber, filenames = get_unitnumber_from_ext_unit_dict(
            model, cls, ext_unit_dict, ipakcb
        )

        # create recharge package instance
        rch = cls(
            model,
            nrchop=nrchop,
            ipakcb=ipakcb,
            rech=rech,
            irch=irch,
            seepelev=seepelev,
            mxrtzones=mxrtzones,
            iconc=iconc,
            selev=selev,
            iznrch=iznrch,
            rchconc=rchconc,
            unitnumber=unitnumber,
            filenames=filenames,
        )
        if check:
            rch.check(
                f=f"{rch.name[0]}.chk",
                verbose=rch.parent.verbose,
                level=0,
            )
        return rch
