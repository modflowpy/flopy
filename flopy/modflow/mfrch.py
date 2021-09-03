"""
mfrch module.  Contains the ModflowRch class. Note that the user can access
the ModflowRch class as `flopy.modflow.ModflowRch`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?rch.htm>`_.

"""
import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Transient2d
from ..modflow.mfparbc import ModflowParBc as mfparbc
from ..utils.flopy_io import line_parse
from ..utils.utils_def import get_pak_vals_shape


class ModflowRch(Package):
    """
    MODFLOW Recharge Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
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
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> #steady state
    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> rch = flopy.modflow.ModflowRch(m, nrchop=3, rech=1.2e-4)

    >>> #transient with time-varying recharge
    >>> import flopy
    >>> rech = {}
    >>> rech[0] = 1.2e-4 #stress period 1 to 4
    >>> rech[4] = 0.0 #stress period 5 and 6
    >>> rech[6] = 1.2e-3 #stress period 7 to the end
    >>> m = flopy.modflow.Modflow()
    >>> rch = flopy.modflow.ModflowRch(m, nrchop=3, rech=rech)

    """

    def __init__(
        self,
        model,
        nrchop=3,
        ipakcb=None,
        rech=1e-3,
        irch=0,
        extension="rch",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowRch._defaultunit()

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
                ipakcb, fname=fname, package=ModflowRch._ftype()
            )
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowRch._ftype()]
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

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self._generate_heading()
        self.url = "rch.htm"

        self.nrchop = nrchop
        self.ipakcb = ipakcb

        rech_u2d_shape = get_pak_vals_shape(model, rech)
        irch_u2d_shape = get_pak_vals_shape(model, irch)

        self.rech = Transient2d(
            model, rech_u2d_shape, np.float32, rech, name="rech_"
        )
        if self.nrchop == 2:
            self.irch = Transient2d(
                model, irch_u2d_shape, np.int32, irch, name="irch_"
            )
        else:
            self.irch = None
        self.np = 0
        self.parent.add_package(self)

    def check(
        self,
        f=None,
        verbose=True,
        level=1,
        RTmin=2e-8,
        RTmax=2e-4,
        checktype=None,
    ):
        """
        Check package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a sting is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.
        RTmin : float
            Minimum product of recharge and transmissivity. Default is 2e-8
        RTmax : float
            Maximum product of recharge and transmissivity. Default is 2e-4

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

        # check for unusually high or low values of mean R/T
        hk_package = {"UPW", "LPF"}.intersection(
            set(self.parent.get_package_list())
        )
        if len(hk_package) > 0 and self.parent.structured:
            pkg = list(hk_package)[0]

            # handle quasi-3D layers
            # (ugly, would be nice to put this else where in a general function)
            if self.parent.dis.laycbd.sum() != 0:
                thickness = np.empty(
                    (
                        self.parent.dis.nlay,
                        self.parent.dis.nrow,
                        self.parent.dis.ncol,
                    ),
                    dtype=float,
                )
                l = 0
                for i, cbd in enumerate(self.parent.dis.laycbd):
                    thickness[i, :, :] = self.parent.modelgrid.thick[l, :, :]
                    if cbd > 0:
                        l += 1
                    l += 1
                assert l == self.parent.modelgrid.thick.shape[0]
            else:
                thickness = self.parent.modelgrid.thick
            assert thickness.shape == self.parent.get_package(pkg).hk.shape
            Tmean = (
                (self.parent.get_package(pkg).hk.array * thickness)[:, active]
                .sum(axis=0)
                .mean()
            )

            # get mean value of recharge array for each stress period
            period_means = self.rech.array.mean(axis=(1, 2, 3))

            if Tmean != 0:
                R_T = period_means / Tmean
                lessthan = np.where(R_T < RTmin)[0]
                greaterthan = np.where(R_T > RTmax)[0]

                if len(lessthan) > 0:
                    txt = (
                        "\r    Mean R/T ratio < checker warning threshold of "
                        "{} for {} stress periods".format(RTmin, len(lessthan))
                    )
                    chk._add_to_summary(
                        type="Warning", value=R_T.min(), desc=txt
                    )
                    chk.remove_passed(
                        f"Mean R/T is between {RTmin} and {RTmax}"
                    )

                if len(greaterthan) > 0:
                    txt = (
                        "\r    Mean R/T ratio > checker warning "
                        "threshold of {} for "
                        "{} stress periods".format(RTmax, len(greaterthan))
                    )
                    chk._add_to_summary(
                        type="Warning", value=R_T.max(), desc=txt
                    )
                    chk.remove_passed(
                        f"Mean R/T is between {RTmin} and {RTmax}"
                    )
                elif len(lessthan) == 0 and len(greaterthan) == 0:
                    chk.append_passed(
                        f"Mean R/T is between {RTmin} and {RTmax}"
                    )

        # check for NRCHOP values != 3
        if self.nrchop != 3:
            txt = "\r    Variable NRCHOP set to value other than 3"
            chk._add_to_summary(type="Warning", value=self.nrchop, desc=txt)
            chk.remove_passed("Variable NRCHOP set to 3.")
        else:
            chk.append_passed("Variable NRCHOP set to 3.")
        chk.summarize()
        return chk

    def _ncells(self):
        """Maximum number of cells that have recharge (developed for
        MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of rch cells

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        return nrow * ncol

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
        f_rch.write(f"{self.nrchop:10d}{self.ipakcb:10d}\n")

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
                    [
                        u2d.array.size
                        for kper, u2d in self.irch.transient_2ds.items()
                    ]
                )
                f_rch.write(f"{mxndrch:10d}\n")

        for kper in range(nper):
            inrech, file_entry_rech = self.rech.get_kper_entry(kper)
            if self.nrchop == 2:
                inirch, file_entry_irch = irch.get_kper_entry(kper)
                if not self.parent.structured:
                    inirch = self.rech[kper].array.size
            else:
                inirch = -1
            f_rch.write(
                f"{inrech:10d}{inirch:10d} # Stress period {kper + 1}\n"
            )
            if inrech >= 0:
                f_rch.write(file_entry_rech)
            if self.nrchop == 2:
                if inirch >= 0:
                    f_rch.write(file_entry_irch)
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
        rch : ModflowRch object
            ModflowRch object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> rch = flopy.modflow.ModflowRch.load('test.rch', m)

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
                    print(
                        f"   Parameters detected. Number of parameters = {npar}"
                    )
            line = f.readline()
        # dataset 2
        t = line_parse(line)
        nrchop = int(t[0])
        ipakcb = int(t[1])

        # dataset 2b for mfusg
        if not model.structured and nrchop == 2:
            line = f.readline()
            t = line_parse(line)
            mxndrch = int(t[0])

        # dataset 3 and 4 - parameters data
        pak_parms = None
        if npar > 0:
            pak_parms = mfparbc.loadarray(f, npar, model.verbose)

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
        else:
            nrow, ncol, nlay, _ = model.get_nrow_ncol_nlay_nper()
        # read data for every stress period
        rech = {}
        irch = None
        if nrchop == 2:
            irch = {}
        current_rech = []
        current_irch = []
        for iper in range(nper):
            line = f.readline()
            t = line_parse(line)
            inrech = int(t[0])

            if nrchop == 2:
                inirch = int(t[1])
            elif not model.structured:
                # usg uses only layer 1 nodes for options 1 and 3. ncol is nodelay for mfusg models.
                inirch = ncol[0]

            if model.structured:
                u2d_shape = (nrow, ncol)
            else:
                u2d_shape = (1, inirch)

            if inrech >= 0:
                if npar == 0:
                    if model.verbose:
                        print(
                            f"   loading rech stress period {iper + 1:3d}..."
                        )
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
                    t = mfparbc.parameter_bcfill(
                        model, u2d_shape, parm_dict, pak_parms
                    )

                current_rech = t
            rech[iper] = current_rech
            if nrchop == 2:
                if inirch >= 0:
                    if model.verbose:
                        print(
                            f"   loading irch stress period {iper + 1:3d}..."
                        )
                    t = Util2d.load(
                        f, model, u2d_shape, np.int32, "irch", ext_unit_dict
                    )
                    current_irch = Util2d(
                        model, u2d_shape, np.int32, t.array - 1, "irch"
                    )
                irch[iper] = current_irch

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowRch._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)

        # create recharge package instance
        rch = cls(
            model,
            nrchop=nrchop,
            ipakcb=ipakcb,
            rech=rech,
            irch=irch,
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

    @staticmethod
    def _ftype():
        return "RCH"

    @staticmethod
    def _defaultunit():
        return 19
