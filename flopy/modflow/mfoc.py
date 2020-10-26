"""
mfoc module.  Contains the ModflowOc class. Note that the user can access
the ModflowOc class as `flopy.modflow.ModflowOc`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?oc.htm>`_.

"""
import os
import sys

from ..pakbase import Package


class ModflowOc(Package):
    """
    MODFLOW Output Control Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ihedfm : int
        is a code for the format in which heads will be printed.
        (default is 0).
    iddnfm : int
        is a code for the format in which drawdown will be printed.
        (default is 0).
    chedfm : string
        is a character value that specifies the format for saving heads.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CHEDFM, then heads are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.
        (default is None)
    cddnfm : string
        is a character value that specifies the format for saving drawdown.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CDDNFM, then drawdowns are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.
        (default is None)
    cboufm : string
        is a character value that specifies the format for saving ibound.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CBOUFM, then ibounds are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.
        (default is None)
    stress_period_data : dictionary of lists
        Dictionary key is a tuple with the zero-based period and step
        (IPEROC, ITSOC) for each print/save option list. If stress_period_data
        is None, then heads are saved for the last time step of each stress
        period. (default is None)

        The list can have any valid MODFLOW OC print/save option:
            PRINT HEAD
            PRINT DRAWDOWN
            PRINT BUDGET
            SAVE HEAD
            SAVE DRAWDOWN
            SAVE BUDGET
            SAVE IBOUND

            The lists can also include (1) DDREFERENCE in the list to reset
            drawdown reference to the period and step and (2) a list of layers
            for PRINT HEAD, SAVE HEAD, PRINT DRAWDOWN, SAVE DRAWDOWN, and
            SAVE IBOUND.

        stress_period_data = {(0,1):['save head']}) would save the head for
        the second timestep in the first stress period.

    compact : boolean
        Save results in compact budget form. (default is True).
    extension : list of strings
        (default is ['oc', 'hds', 'ddn', 'cbc', 'ibo']).
    unitnumber : list of ints
        (default is [14, 51, 52, 53, 0]).
    filenames : str or list of str
        Filenames to use for the package and the head, drawdown, budget (not
        used), and ibound output files. If filenames=None the package name
        will be created using the model name and package extension and the
        output file names will be created using the model name and extensions.
        If a single string is passed the package will be set to the string and
        output names will be created using the model name and head, drawdown,
        budget, and ibound extensions. To define the names for all package
        files (input and output) the length of the list of strings should be 5.
        Default is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The "words" method for specifying output control is the only option
    available.  Also, the "compact" budget should normally be used as it
    produces files that are typically much smaller.  The compact budget form is
    also a requirement for using the MODPATH particle tracking program.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> spd = {(0, 0): ['print head'],
    ...   (0, 1): [],
    ...   (0, 249): ['print head'],
    ...   (0, 250): [],
    ...   (0, 499): ['print head', 'save ibound'],
    ...   (0, 500): [],
    ...   (0, 749): ['print head', 'ddreference'],
    ...   (0, 750): [],
    ...   (0, 999): ['print head']}
    >>> oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, cboufm='(20i5)')

    """

    def __init__(
        self,
        model,
        ihedfm=0,
        iddnfm=0,
        chedfm=None,
        cddnfm=None,
        cboufm=None,
        compact=True,
        stress_period_data={(0, 0): ["save head"]},
        extension=["oc", "hds", "ddn", "cbc", "ibo"],
        unitnumber=None,
        filenames=None,
        label="LABEL",
        **kwargs
    ):

        """
        Package constructor.

        """
        if unitnumber is None:
            unitnumber = ModflowOc._defaultunit()
        elif isinstance(unitnumber, list):
            if len(unitnumber) < 5:
                for idx in range(len(unitnumber), 6):
                    unitnumber.append(0)
        self.label = label
        # set filenames
        if filenames is None:
            filenames = [None, None, None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 5:
                for idx in range(len(filenames), 5):
                    filenames.append(None)

        # support structured and unstructured dis
        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")

        if stress_period_data is None:
            stress_period_data = {
                (kper, dis.nstp.array[kper] - 1): ["save head"]
                for kper in range(dis.nper)
            }

        # process kwargs
        if "save_every" in kwargs:
            save_every = int(kwargs.pop("save_every"))
        else:
            save_every = None
        if save_every is not None:
            if "save_types" in kwargs:
                save_types = kwargs.pop("save_types")
                if isinstance(save_types, str):
                    save_types = [save_types]
            else:
                save_types = ["save head", "print budget"]
            if "save_start" in kwargs:
                save_start = int(kwargs.pop("save_start"))
            else:
                save_start = 1
            stress_period_data = {}
            for kper in range(dis.nper):
                icnt = save_start
                for kstp in range(dis.nstp[kper]):
                    if icnt == save_every:
                        stress_period_data[(kper, kstp)] = save_types
                        icnt = 0
                    else:
                        stress_period_data[(kper, kstp)] = []
                    icnt += 1

        # set output unit numbers based on oc settings
        self.savehead, self.saveddn, self.savebud, self.saveibnd = (
            False,
            False,
            False,
            False,
        )
        for key, value in stress_period_data.items():
            tlist = list(value)
            for t in tlist:
                if "save head" in t.lower():
                    self.savehead = True
                    if unitnumber[1] == 0:
                        unitnumber[1] = 51
                if "save drawdown" in t.lower():
                    self.saveddn = True
                    if unitnumber[2] == 0:
                        unitnumber[2] = 52
                if "save budget" in t.lower():
                    self.savebud = True
                    if unitnumber[3] == 0 and filenames is None:
                        unitnumber[3] = 53
                if "save ibound" in t.lower():
                    self.saveibnd = True
                    if unitnumber[4] == 0:
                        unitnumber[4] = 54

        # do not create head, ddn, or cbc output files if output is not
        # specified in the oc stress_period_data
        if not self.savehead:
            unitnumber[1] = 0
        if not self.saveddn:
            unitnumber[2] = 0
        if not self.savebud:
            unitnumber[3] = 0
        if not self.saveibnd:
            unitnumber[4] = 0

        self.iuhead = unitnumber[1]
        self.iuddn = unitnumber[2]
        self.iubud = unitnumber[3]
        self.iuibnd = unitnumber[4]

        # add output files
        # head file
        if self.savehead:
            iu = unitnumber[1]
            binflag = True
            if chedfm is not None:
                binflag = False
            fname = filenames[1]
            model.add_output_file(
                iu, fname=fname, extension=extension[1], binflag=binflag
            )
        # drawdown file
        if self.saveddn:
            iu = unitnumber[2]
            binflag = True
            if cddnfm is not None:
                binflag = False
            fname = filenames[2]
            model.add_output_file(
                iu, fname=fname, extension=extension[2], binflag=binflag
            )
        # budget file
        # Nothing is needed for the budget file

        # ibound file
        ibouun = unitnumber[4]
        if self.saveibnd:
            iu = unitnumber[4]
            binflag = True
            if cboufm is not None:
                binflag = False
            fname = filenames[4]
            model.add_output_file(
                iu, fname=fname, extension=extension[4], binflag=binflag
            )

        name = [ModflowOc._ftype()]
        extra = [""]
        extension = [extension[0]]
        unitnumber = unitnumber[0]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=unitnumber,
            extra=extra,
            filenames=fname,
        )

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )

        self.url = "oc.htm"
        self.ihedfm = ihedfm
        self.iddnfm = iddnfm
        self.chedfm = chedfm
        self.cddnfm = cddnfm

        self.ibouun = ibouun
        self.cboufm = cboufm

        self.compact = compact

        self.stress_period_data = stress_period_data

        self.parent.add_package(self)

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen.
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.oc.check()

        """
        chk = self._get_check(f, verbose, level, checktype)
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")
        if dis is None:
            chk._add_to_summary(
                "Error", package="OC", desc="DIS package not available"
            )
        else:
            # generate possible actions expected
            expected_actions = []
            for first in ["PRINT", "SAVE"]:
                for second in ["HEAD", "DRAWDOWN", "BUDGET", "IBOUND"]:
                    expected_actions.append([first, second])
            # remove exception
            del expected_actions[expected_actions.index(["PRINT", "IBOUND"])]
            keys = list(self.stress_period_data.keys())
            for kper in range(dis.nper):
                for kstp in range(dis.nstp[kper]):
                    kperkstp = (kper, kstp)
                    if kperkstp in keys:
                        del keys[keys.index(kperkstp)]
                        data = self.stress_period_data[kperkstp]
                        if not isinstance(data, list):
                            data = [data]
                        for action in data:
                            words = action.upper().split()
                            if len(words) < 2:
                                chk._add_to_summary(
                                    "Warning",
                                    package="OC",  # value=kperkstp,
                                    desc="action {!r} ignored; too few words".format(
                                        action
                                    ),
                                )
                            elif words[0:2] not in expected_actions:
                                chk._add_to_summary(
                                    "Warning",
                                    package="OC",  # value=kperkstp,
                                    desc="action {!r} ignored".format(action),
                                )
                            # TODO: check data list of layers for some actions
            for kperkstp in keys:
                # repeat as many times as remaining keys not used
                chk._add_to_summary(
                    "Warning",
                    package="OC",  # value=kperkstp,
                    desc="action(s) defined in OC stress_period_data ignored "
                    "as they are not part the stress periods defined by DIS",
                )
        chk.summarize()
        return chk

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f_oc = open(self.fn_path, "w")
        f_oc.write("{}\n".format(self.heading))

        # write options
        line = "HEAD PRINT FORMAT {0:3.0f}\n".format(self.ihedfm)
        f_oc.write(line)
        if self.chedfm is not None:
            line = "HEAD SAVE FORMAT {0:20s} {1}\n".format(
                self.chedfm, self.label
            )
            f_oc.write(line)
        if self.savehead:
            line = "HEAD SAVE UNIT {0:5.0f}\n".format(self.iuhead)
            f_oc.write(line)

        f_oc.write("DRAWDOWN PRINT FORMAT {0:3.0f}\n".format(self.iddnfm))
        if self.cddnfm is not None:
            line = "DRAWDOWN SAVE FORMAT {0:20s} {1}\n".format(
                self.cddnfm, self.label
            )
            f_oc.write(line)
        if self.saveddn:
            line = "DRAWDOWN SAVE UNIT {0:5.0f}\n".format(self.iuddn)
            f_oc.write(line)

        if self.saveibnd:
            if self.cboufm is not None:
                line = "IBOUND SAVE FORMAT {0:20s} {1}\n".format(
                    self.cboufm, self.label
                )
                f_oc.write(line)
            line = "IBOUND SAVE UNIT {0:5.0f}\n".format(self.iuibnd)
            f_oc.write(line)

        if self.compact:
            f_oc.write("COMPACT BUDGET AUX\n")

        # add a line separator between header and stress
        #  period data
        f_oc.write("\n")

        # write the transient sequence described by the data dict
        nr, nc, nl, nper = self.parent.get_nrow_ncol_nlay_nper()
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")
        nstp = dis.nstp

        keys = list(self.stress_period_data.keys())
        keys.sort()

        data = []
        ddnref = ""
        lines = ""
        for kper in range(nper):
            for kstp in range(nstp[kper]):
                kperkstp = (kper, kstp)
                if kperkstp in keys:
                    data = self.stress_period_data[kperkstp]
                    if not isinstance(data, list):
                        data = [data]
                    lines = ""
                    if len(data) > 0:
                        for item in data:
                            if "DDREFERENCE" in item.upper():
                                ddnref = item.lower()
                            else:
                                lines += "  {}\n".format(item)
                if len(lines) > 0:
                    f_oc.write(
                        "period {} step {} {}\n".format(
                            kper + 1, kstp + 1, ddnref
                        )
                    )
                    f_oc.write(lines)
                    f_oc.write("\n")
                    ddnref = ""
                    lines = ""

        # close oc file
        f_oc.close()

    def _set_singlebudgetunit(self, budgetunit):
        if budgetunit is None:
            budgetunit = self.parent.next_ext_unit()
        self.iubud = budgetunit

    def _set_budgetunit(self):
        iubud = []
        for i, pp in enumerate(self.parent.packagelist):
            if hasattr(pp, "ipakcb"):
                if pp.ipakcb > 0:
                    iubud.append(pp.ipakcb)
        if len(iubud) < 1:
            iubud = None
        elif len(iubud) == 1:
            iubud = iubud[0]
        self.iubud = iubud

    def get_budgetunit(self):
        """
        Get the budget file unit number(s).

        Parameters
        ----------
        None

        Returns
        -------
        iubud : integer ot list of integers
            Unit number or list of cell-by-cell budget output unit numbers.
            None is returned if ipakcb is less than one for all packages.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis(m)
        >>> bas = flopy.modflow.ModflowBas(m)
        >>> lpf = flopy.modflow.ModflowLpf(m, ipakcb=100)
        >>> wel_data = {0: [[0, 0, 0, -1000.]]}
        >>> wel = flopy.modflow.ModflowWel(m, ipakcb=101,
        ... stress_period_data=wel_data)
        >>> spd = {(0, 0): ['save head', 'save budget']}
        >>> oc = flopy.modflow.ModflowOc(m, stress_period_data=spd)
        >>> oc.get_budgetunit()
        [100, 101]

        """
        # set iubud by iterating through the packages
        self._set_budgetunit()
        return self.iubud

    def reset_budgetunit(self, budgetunit=None, fname=None):
        """
        Reset the cell-by-cell budget unit (ipakcb) for every package that
        can write cell-by-cell data when SAVE BUDGET is specified in the
        OC file to the specified budgetunit.

        Parameters
        ----------
        budgetunit : int, optional
            Unit number for cell-by-cell output data. If budgetunit is None
            then the next available external unit number is assigned. Default
            is None
        fname : string, optional
            Filename to use for cell-by-cell output file. If fname=None the
            cell-by-cell output file will be created using the model name and
            a '.cbc' file extension. Default is None.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis(m)
        >>> bas = flopy.modflow.ModflowBas(m)
        >>> lpf = flopy.modflow.ModflowLpf(m, ipakcb=100)
        >>> wel_data = {0: [[0, 0, 0, -1000.]]}
        >>> wel = flopy.modflow.ModflowWel(m, ipakcb=101,
        ... stress_period_data=wel_data)
        >>> spd = {(0, 0): ['save head', 'save budget']}
        >>> oc = flopy.modflow.ModflowOc(m, stress_period_data=spd)
        >>> oc.reset_budgetunit(budgetunit=1053, fname='test.cbc')

        """

        # remove existing output file
        for pp in self.parent.packagelist:
            if hasattr(pp, "ipakcb"):
                if pp.ipakcb > 0:
                    self.parent.remove_output(unit=pp.ipakcb)
                    pp.ipakcb = 0

        # set the unit number used for all cell-by-cell output
        self._set_singlebudgetunit(budgetunit)

        # add output file
        for pp in self.parent.packagelist:
            if hasattr(pp, "ipakcb"):
                pp.ipakcb = self.iubud
                self.parent.add_output_file(
                    pp.ipakcb, fname=fname, package=pp.name
                )

        return

    @staticmethod
    def get_ocoutput_units(f, ext_unit_dict=None):
        """
        Get head and drawdown units from a OC file.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        ihedun : integer
            Unit number of the head file.
        fhead : str
            File name of the head file. Is only defined if ext_unit_dict is
            passed and the unit number is a valid key.
            , headfilename, oc : ModflowOc object
            ModflowOc object.
        iddnun : integer
            Unit number of the drawdown file.
        fddn : str
            File name of the drawdown file. Is only defined if ext_unit_dict is
            passed and the unit number is a valid key.

        Examples
        --------

        >>> import flopy
        >>> ihds, hf, iddn, df = flopy.modflow.ModflowOc.get_ocoutput_units('test.oc')

        """

        # initialize
        ihedun = 0
        iddnun = 0
        fhead = None
        fddn = None

        numericformat = False

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # read header
        ipos = f.tell()
        while True:
            line = f.readline()
            if line[0] == "#":
                continue
            elif line[0] == []:
                continue
            else:
                lnlst = line.strip().split()
                try:
                    ihedun, iddnun = int(lnlst[2]), int(lnlst[3])
                    numericformat = True
                except:
                    f.seek(ipos)
                # exit so the remaining data can be read
                #  from the file based on numericformat
                break
        # read word formats
        if not numericformat:
            while True:
                line = f.readline()
                if len(line) < 1:
                    break
                lnlst = line.strip().split()
                if line[0] == "#":
                    continue

                # skip blank line in the OC file
                if len(lnlst) < 1:
                    continue

                # dataset 1 values
                elif (
                    "HEAD" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "UNIT" in lnlst[2].upper()
                ):
                    ihedun = int(lnlst[3])
                elif (
                    "DRAWDOWN" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "UNIT" in lnlst[2].upper()
                ):
                    iddnun = int(lnlst[3])
                # dataset 2
                elif "PERIOD" in lnlst[0].upper():
                    break
        #
        if ext_unit_dict is not None:
            if ihedun in ext_unit_dict:
                fhead = ext_unit_dict[ihedun]
            if iddnun in ext_unit_dict:
                fddn = ext_unit_dict[iddnun]

        if openfile:
            f.close()

        # return
        return ihedun, fhead, iddnun, fddn

    @classmethod
    def load(
        cls, f, model, nper=None, nstp=None, nlay=None, ext_unit_dict=None
    ):
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
        nstp : int or list of ints
            Integer of list of integers containing the number of time steps
            in each stress period. If nstp is None, then nstp will be obtained
            from the DIS or DISU packages attached to the model object. The
            length of nstp must be equal to nper. (default is None).
        nlay : int
            The number of model layers.  If nlay is None, then nnlay will be
            obtained from the model object. nlay only needs to be specified
            if an empty model object is passed in and the oc file being loaded
            is defined using numeric codes. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        oc : ModflowOc object
            ModflowOc object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> oc = flopy.modflow.ModflowOc.load('test.oc', m)

        """

        if model.verbose:
            sys.stdout.write("loading oc package file...\n")

        # set nper
        if nper is None or nlay is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        if nper == 0 or nlay == 0:
            msg = (
                "discretization package not defined for the model, "
                + "nper and nlay must be provided to the .load() method"
            )
            raise ValueError(msg)

        # set nstp
        if nstp is None:
            dis = model.get_package("DIS")
            if dis is None:
                dis = model.get_package("DISU")
            if dis is None:
                msg = (
                    "discretization package not defined for the model, "
                    + "a nstp list must be provided to the .load() method"
                )
                raise ValueError(msg)
            nstp = list(dis.nstp.array)
        else:
            if isinstance(nstp, (int, float)):
                nstp = [int(nstp)]

        # validate the size of nstp
        if len(nstp) != nper:
            msg = "nstp must be a list with {} entries, ".format(
                nper
            ) + "provided nstp list has {} entries.".format(len(nstp))
            raise IOError(msg)

        # initialize
        ihedfm = 0
        iddnfm = 0
        ihedun = 0
        iddnun = 0
        ibouun = 0
        compact = False
        chedfm = None
        cddnfm = None
        cboufm = None

        numericformat = False

        stress_period_data = {}

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        else:
            filename = os.path.basename(f.name)

        # read header
        ipos = f.tell()
        while True:
            line = f.readline()
            if line[0] == "#":
                continue
            elif line[0] == []:
                continue
            else:
                lnlst = line.strip().split()
                try:
                    ihedfm, iddnfm = int(lnlst[0]), int(lnlst[1])
                    ihedun, iddnun = int(lnlst[2]), int(lnlst[3])
                    numericformat = True
                except:
                    f.seek(ipos)
                # exit so the remaining data can be read
                #  from the file based on numericformat
                break
            # set pointer to current position in the OC file
            ipos = f.tell()

        # process each line
        lines = []
        if numericformat == True:
            for iperoc in range(nper):
                for itsoc in range(nstp[iperoc]):
                    line = f.readline()
                    lnlst = line.strip().split()
                    incode, ihddfl = int(lnlst[0]), int(lnlst[1])
                    ibudfl, icbcfl = int(lnlst[2]), int(lnlst[3])
                    # new print and save flags are needed if incode is not
                    #  less than 0.
                    if incode >= 0:
                        lines = []
                    # use print options from the last time step
                    else:
                        if len(lines) > 0:
                            stress_period_data[(iperoc, itsoc)] = list(lines)
                        continue
                    # set print and save budget flags
                    if ibudfl != 0:
                        lines.append("PRINT BUDGET")
                    if icbcfl != 0:
                        lines.append("SAVE BUDGET")
                    if incode == 0:
                        line = f.readline()
                        lnlst = line.strip().split()
                        hdpr, ddpr = int(lnlst[0]), int(lnlst[1])
                        hdsv, ddsv = int(lnlst[2]), int(lnlst[3])
                        if hdpr != 0:
                            lines.append("PRINT HEAD")
                        if ddpr != 0:
                            lines.append("PRINT DRAWDOWN")
                        if hdsv != 0:
                            lines.append("SAVE HEAD")
                        if ddsv != 0:
                            lines.append("SAVE DRAWDOWN")
                    elif incode > 0:
                        headprint = ""
                        headsave = ""
                        ddnprint = ""
                        ddnsave = ""
                        for k in range(nlay):
                            line = f.readline()
                            lnlst = line.strip().split()
                            hdpr, ddpr = int(lnlst[0]), int(lnlst[1])
                            hdsv, ddsv = int(lnlst[2]), int(lnlst[3])
                            if hdpr != 0:
                                headprint += " {}".format(k + 1)
                            if ddpr != 0:
                                ddnprint += " {}".format(k + 1)
                            if hdsv != 0:
                                headsave += " {}".format(k + 1)
                            if ddsv != 0:
                                ddnsave += " {}".format(k + 1)
                        if len(headprint) > 0:
                            lines.append("PRINT HEAD" + headprint)
                        if len(ddnprint) > 0:
                            lines.append("PRINT DRAWDOWN" + ddnprint)
                        if len(headsave) > 0:
                            lines.append("SAVE HEAD" + headsave)
                        if len(ddnsave) > 0:
                            lines.append("SAVE DRAWDOWN" + ddnsave)
                    stress_period_data[(iperoc, itsoc)] = list(lines)
        else:
            iperoc, itsoc = 0, 0
            while True:
                line = f.readline()
                if len(line) < 1:
                    break
                lnlst = line.strip().split()
                if line[0] == "#":
                    continue

                # added by JJS 12/12/14 to avoid error when there is a blank line in the OC file
                if lnlst == []:
                    continue
                # end add

                # dataset 1 values
                elif (
                    "HEAD" in lnlst[0].upper()
                    and "PRINT" in lnlst[1].upper()
                    and "FORMAT" in lnlst[2].upper()
                ):
                    ihedfm = int(lnlst[3])
                elif (
                    "HEAD" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "FORMAT" in lnlst[2].upper()
                ):
                    chedfm = lnlst[3]
                elif (
                    "HEAD" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "UNIT" in lnlst[2].upper()
                ):
                    ihedun = int(lnlst[3])
                elif (
                    "DRAWDOWN" in lnlst[0].upper()
                    and "PRINT" in lnlst[1].upper()
                    and "FORMAT" in lnlst[2].upper()
                ):
                    iddnfm = int(lnlst[3])
                elif (
                    "DRAWDOWN" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "FORMAT" in lnlst[2].upper()
                ):
                    cddnfm = lnlst[3]
                elif (
                    "DRAWDOWN" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "UNIT" in lnlst[2].upper()
                ):
                    iddnun = int(lnlst[3])
                elif (
                    "IBOUND" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "FORMAT" in lnlst[2].upper()
                ):
                    cboufm = lnlst[3]
                elif (
                    "IBOUND" in lnlst[0].upper()
                    and "SAVE" in lnlst[1].upper()
                    and "UNIT" in lnlst[2].upper()
                ):
                    ibouun = int(lnlst[3])
                elif "COMPACT" in lnlst[0].upper():
                    compact = True

                # dataset 2
                elif "PERIOD" in lnlst[0].upper():
                    if len(lines) > 0:
                        if iperoc > 0:
                            # create period step tuple
                            kperkstp = (iperoc - 1, itsoc - 1)
                            # save data
                            stress_period_data[kperkstp] = lines
                        # reset lines
                        lines = []
                    # turn off oc if required
                    if iperoc > 0:
                        if itsoc == nstp[iperoc - 1]:
                            iperoc1 = iperoc + 1
                            itsoc1 = 1
                        else:
                            iperoc1 = iperoc
                            itsoc1 = itsoc + 1
                    else:
                        iperoc1, itsoc1 = iperoc, itsoc
                        # update iperoc and itsoc
                    iperoc = int(lnlst[1])
                    itsoc = int(lnlst[3])
                    # do not used data that exceeds nper
                    if iperoc > nper:
                        break
                    # add a empty list if necessary
                    iempty = False
                    if iperoc != iperoc1:
                        iempty = True
                    else:
                        if itsoc != itsoc1:
                            iempty = True
                    if iempty == True:
                        kperkstp = (iperoc1 - 1, itsoc1 - 1)
                        stress_period_data[kperkstp] = []
                # dataset 3
                elif "PRINT" in lnlst[0].upper():
                    lines.append(
                        "{} {}".format(lnlst[0].lower(), lnlst[1].lower())
                    )
                elif "SAVE" in lnlst[0].upper():
                    lines.append(
                        "{} {}".format(lnlst[0].lower(), lnlst[1].lower())
                    )
                else:
                    print("Error encountered in OC import.")
                    print("Creating default OC package.")
                    return ModflowOc(model)

            # store the last record in word
            if len(lines) > 0:
                # create period step tuple
                kperkstp = (iperoc - 1, itsoc - 1)
                # save data
                stress_period_data[kperkstp] = lines
                # add a empty list if necessary
                iempty = False
                if iperoc != iperoc1:
                    iempty = True
                else:
                    if itsoc != itsoc1:
                        iempty = True
                if iempty == True:
                    kperkstp = (iperoc1 - 1, itsoc1 - 1)
                    stress_period_data[kperkstp] = []

        if openfile:
            f.close()

        # reset unit numbers
        unitnumber = [14, 0, 0, 0, 0]
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowOc._ftype():
                    unitnumber[0] = key
                    fname = os.path.basename(value.filename)
        else:
            fname = os.path.basename(filename)

        # initialize filenames list
        filenames = [fname, None, None, None, None]

        # fill remainder of filenames list
        if ihedun > 0:
            unitnumber[1] = ihedun
            try:
                filenames[1] = os.path.basename(ext_unit_dict[ihedun].filename)
            except:
                if model.verbose:
                    print("head file name will be generated by flopy")
        if iddnun > 0:
            unitnumber[2] = iddnun
            try:
                filenames[2] = os.path.basename(ext_unit_dict[iddnun].filename)
            except:
                if model.verbose:
                    print("drawdown file name will be generated by flopy")
        if ibouun > 0:
            unitnumber[4] = ibouun
            try:
                filenames[4] = os.path.basename(ext_unit_dict[ibouun].filename)
            except:
                if model.verbose:
                    print("ibound file name will be generated by flopy")
            if cboufm is None:
                cboufm = True

        # add unit numbers to pop_key_list
        for u in unitnumber:
            model.add_pop_key_list(u)

        return cls(
            model,
            ihedfm=ihedfm,
            iddnfm=iddnfm,
            chedfm=chedfm,
            cddnfm=cddnfm,
            cboufm=cboufm,
            compact=compact,
            stress_period_data=stress_period_data,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "OC"

    @staticmethod
    def _defaultunit():
        return [14, 0, 0, 0, 0]
