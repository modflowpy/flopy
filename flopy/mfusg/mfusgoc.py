"""
mfusgoc module.  Contains the MfusgOc class. Note that the user can access
the MfusgOc class as `flopy.mfusg.MfusgOc`.

"""

import os

from ..pakbase import Package
from ..utils.utils_def import type_from_iterable
from .mfusg import MfUsg


class MfUsgOc(Package):
    """
    MODFLOW USG Transport Output Control (OC) Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mfusg.MfUsg`) to which
        this package will be added.
    ihedfm : int (default is 0)
        is a code for the format in which heads will be printed.
    iddnfm : int (default is 0)
        is a code for the format in which drawdown will be printed.
    chedfm : string (default is None)
        is a character value that specifies the format for saving heads.
    cddnfm : string (default is None)
        is a character value that specifies the format for saving drawdown.
    cboufm : string (default is None)
        is a character value that specifies the format for saving ibound.
    stress_period_data : dictionary of lists
        Dictionary key is a tuple with the zero-based period and step
        (IPEROC, ITSOC) for each print/save option list. If stress_period_data
        is None, then heads are saved for the last time step of each stress
        period. (default is None)

        The list can have any valid MFUSG OC print/save option:
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
    >>> m = flopy.mfusg.MfUsg()
    >>> spd = {(0, 0): ['print head'],
    ...   (0, 1): [],
    ...   (0, 249): ['print head'],
    ...   (0, 250): [],
    ...   (0, 499): ['print head', 'save ibound'],
    ...   (0, 500): [],
    ...   (0, 749): ['print head', 'ddreference'],
    ...   (0, 750): [],
    ...   (0, 999): ['print head']}
    >>> oc = flopy.mfusg.MfusgOc(m, stress_period_data=spd, cboufm='(20i5)')

    """

    def __init__(
        self,
        model,
        ihedfm=0,
        iddnfm=0,
        ispcfm=0,
        chedfm=None,
        cddnfm=None,
        cspcfm=None,
        cboufm=None,
        compact=True,
        stress_period_data={(0, 0): ["save head"]},
        extension=["oc", "hds", "ddn", "cbc", "ibo", "con"],
        unitnumber=None,
        filenames=None,
        label="LABEL",
        **kwargs,
    ):
        """Constructs the MfusgOc object."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if unitnumber is None:
            unitnumber = MfUsgOc._defaultunit()
        elif isinstance(unitnumber, list):
            if len(unitnumber) < 6:
                for idx in range(len(unitnumber), 6):
                    unitnumber.append(0)
        self.label = label

        # set filenames
        filenames = self._prepare_filenames(filenames, 6)

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
                save_types = ["save head", "save budget"]
            if "save_conc" in kwargs:
                save_types.append("save conc")
            if "save_start" in kwargs:
                save_start = int(kwargs.pop("save_start"))
            else:
                save_start = 1
            stress_period_data = {}
            for kper in range(dis.nper):
                icnt = save_start
                for kstp in range(dis.nstp[kper]):
                    if icnt == save_every:
                        stress_period_data[kper, kstp] = save_types
                        icnt = 0
                    else:
                        stress_period_data[kper, kstp] = []
                    icnt += 1

        # adaptive time stepping
        self.atsa = 0
        self.nptimes = 0
        self.npsteps = 0
        self.timot = None

        # FASTFORWARD initial conditions of heads
        self.fastforward = False
        self.ispfast = 0
        self.itsfast = 0
        self.iugfast = 0
        self.iucfast = 0
        self.iudfast = 0

        # FASTFORWARDC initial conditions of concentration
        self.fastforwardc = False
        self.ispfastc = 0
        self.itsfastc = 0
        self.iugfastc = 0
        self.iucfastc = 0
        self.iudfastc = 0
        self.iumfastc = 0

        if "atsa" in kwargs:
            self.atsa = int(kwargs.pop("atsa"))
        if "nptimes" in kwargs:
            self.nptimes = int(kwargs.pop("nptimes"))
        if "npsteps" in kwargs:
            self.npsteps = int(kwargs.pop("npsteps"))
        if "timot" in kwargs:
            self.timot = kwargs.pop("timot")

        if "fastforward" in kwargs:
            self.fastforward = int(kwargs.pop("fastforward"))

        if self.fastforward:
            if "ispfast" in kwargs:
                self.ispfast = int(kwargs.pop("ispfast"))
            if "itsfast" in kwargs:
                self.itsfast = int(kwargs.pop("itsfast"))
            if "iugfast" in kwargs:
                self.iugfast = int(kwargs.pop("iugfast"))
            if "iucfast" in kwargs:
                self.iucfast = int(kwargs.pop("iucfast"))
            if "iudfast" in kwargs:
                self.iudfast = int(kwargs.pop("iudfast"))

        if "fastforwardc" in kwargs:
            self.fastforwardc = int(kwargs.pop("fastforwardc"))

        if self.fastforwardc:
            if "ispfastc" in kwargs:
                self.ispfastc = int(kwargs.pop("ispfastc"))
            if "itsfastc" in kwargs:
                self.itsfastc = int(kwargs.pop("itsfastc"))
            if "iugfastc" in kwargs:
                self.iugfastc = int(kwargs.pop("iugfastc"))
            if "iucfastc" in kwargs:
                self.iucfastc = int(kwargs.pop("iucfastc"))
            if "iudfastc" in kwargs:
                self.iudfastc = int(kwargs.pop("iudfastc"))
            if "iumfastc" in kwargs:
                self.iumfastc = int(kwargs.pop("iumfastc"))

        # set output unit numbers based on oc settings
        self.savehead = False
        self.saveddn = False
        self.savebud = False
        self.saveibnd = False
        self.savespc = False
        for key, value in stress_period_data.items():
            for t in list(value):
                tlwr = t.lower()
                if "save head" in tlwr:
                    self.savehead = True
                    if unitnumber[1] == 0:
                        unitnumber[1] = 51
                if "save drawdown" in tlwr:
                    self.saveddn = True
                    if unitnumber[2] == 0:
                        unitnumber[2] = 52
                if "save budget" in tlwr:
                    self.savebud = True
                    if unitnumber[3] == 0 and filenames is None:
                        unitnumber[3] = 53
                if "save ibound" in tlwr:
                    self.saveibnd = True
                    if unitnumber[4] == 0:
                        unitnumber[4] = 54
                if "save conc" in tlwr:
                    self.savespc = True
                    if unitnumber[5] == 0:
                        unitnumber[5] = 55

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
        if not self.savespc:
            unitnumber[5] = 0

        self.iuhead = unitnumber[1]
        self.iuddn = unitnumber[2]
        self.iubud = unitnumber[3]
        self.iuibnd = unitnumber[4]
        self.iuspc = unitnumber[5]

        # add output files
        # head file
        if self.savehead:
            model.add_output_file(
                unitnumber[1],
                fname=filenames[1],
                extension=extension[1],
                binflag=chedfm is None,
            )
        # drawdown file
        if self.saveddn:
            model.add_output_file(
                unitnumber[2],
                fname=filenames[2],
                extension=extension[2],
                binflag=cddnfm is None,
            )
        # budget file
        # Nothing is needed for the budget file

        # ibound file
        if self.saveibnd:
            model.add_output_file(
                unitnumber[4],
                fname=filenames[4],
                extension=extension[4],
                binflag=cboufm is None,
            )

        # concentration file
        if self.savespc:
            model.add_output_file(
                unitnumber[5],
                fname=filenames[5],
                extension=extension[5],
                binflag=cspcfm is None,
            )

        # call base package constructor
        super().__init__(
            model,
            extension=extension[0],
            name=self._ftype(),
            unit_number=unitnumber[0],
            filenames=filenames[0],
        )

        self._generate_heading()

        self.url = "oc.html"
        self.ihedfm = ihedfm
        self.iddnfm = iddnfm
        self.chedfm = chedfm
        self.cddnfm = cddnfm

        self.cboufm = cboufm

        self.ispcfm = ispcfm
        self.cspcfm = cspcfm

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
        >>> m = flopy.mfusg.MfUsg.load('model.nam')
        >>> m.oc.check()

        """
        chk = self._get_check(f, verbose, level, checktype)
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")
        if dis is None:
            chk._add_to_summary("Error", package="OC", desc="DIS package not available")
        else:
            # generate possible actions expected
            expected_actions = []
            for first in ["PRINT", "SAVE"]:
                for second in ["HEAD", "DRAWDOWN", "BUDGET", "IBOUND", "CONC"]:
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
                                    package="OC",
                                    desc=f"action {action!r} ignored; too few words",
                                )
                            elif words[0:2] not in expected_actions:
                                chk._add_to_summary(
                                    "Warning",
                                    package="OC",
                                    desc=f"action {action!r} ignored",
                                )
                            # TODO: check data list of layers for some actions
            for kperkstp in keys:
                # repeat as many times as remaining keys not used
                chk._add_to_summary(
                    "Warning",
                    package="OC",
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
        f_oc.write(f"{self.heading}\n")

        # write options
        if self.atsa:
            f_oc.write("ATSA ")
            if self.nptimes > 0:
                f_oc.write(f"NPTIMES {self.nptimes} ")
                for i in range(self.nptimes):
                    f_oc.write(f"{self.timot[i]} ")
            if self.npsteps > 0:
                f_oc.write(f"NPSTPS {self.npsteps} ")
            f_oc.write("\n")
        if self.fastforward:
            f_oc.write(
                f"FASTFORWARD {self.ispfast:3.0f} {self.itsfast:3.0f}"
                f" {self.iugfast:3.0f} {self.iucfast:3.0f} {self.iudfast:3.0f}\n"
            )
        if self.fastforwardc:
            f_oc.write(
                f"FASTFORWARDC {self.ispfastc:3.0f} {self.itsfastc:3.0f}"
                f" {self.iugfastc:3.0f} {self.iucfastc:3.0f} {self.iudfastc:3.0f}"
                f" {self.iumfastc:3.0f}\n"
            )

        if self.atsa and self.nptimes > 0:
            for i in range(self.nptimes):
                f_oc.write(f"{self.timot[i]} ")
            f_oc.write("\n")

        line = f"HEAD PRINT FORMAT {self.ihedfm:3.0f}\n"
        f_oc.write(line)
        if self.chedfm is not None:
            line = f"HEAD SAVE FORMAT {self.chedfm:20s} {self.label}\n"
            f_oc.write(line)
        if self.savehead:
            line = f"HEAD SAVE UNIT {self.iuhead:5.0f}\n"
            f_oc.write(line)

        f_oc.write(f"DRAWDOWN PRINT FORMAT {self.iddnfm:3.0f}\n")
        if self.cddnfm is not None:
            line = f"DRAWDOWN SAVE FORMAT {self.cddnfm:20s} {self.label}\n"
            f_oc.write(line)
        if self.saveddn:
            line = f"DRAWDOWN SAVE UNIT {self.iuddn:5.0f}\n"
            f_oc.write(line)

        if self.saveibnd:
            if self.cboufm is not None:
                line = f"IBOUND SAVE FORMAT {self.cboufm:20s} {self.label}\n"
                f_oc.write(line)
            line = f"IBOUND SAVE UNIT {self.iuibnd:5.0f}\n"
            f_oc.write(line)

        if self.savespc:
            f_oc.write(f"CONC PRINT FORMAT {self.ispcfm:3.0f}\n")
            if self.cspcfm is not None:
                line = f"CONC SAVE FORMAT {self.cspcfm:20s} {self.label}\n"
                f_oc.write(line)
            line = f"CONC SAVE UNIT {self.iuspc:5.0f}\n"
            f_oc.write(line)

        if self.compact:
            f_oc.write("COMPACT BUDGET AUX\n")

        # add a line separator between header and stress
        #  period data
        #        f_oc.write("\n")

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
                                lines += f"  {item}\n"
                if len(lines) > 0:
                    f_oc.write(f"period {kper + 1} step {kstp + 1} {ddnref}\n")
                    f_oc.write(lines)
                    #                    f_oc.write("\n")
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
        iubud : integer or list of integers
            Unit number or list of cell-by-cell budget output unit numbers.
            None is returned if ipakcb is less than one for all packages.
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
                self.parent.add_output_file(pp.ipakcb, fname=fname, package=pp.name)

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
            , headfilename, oc : MfUsgOc object
            MfUsgOc object.
        iddnun : integer
            Unit number of the drawdown file.
        fddn : str
            File name of the drawdown file. Is only defined if ext_unit_dict is
            passed and the unit number is a valid key.

        Examples
        --------

        >>> import flopy
        >>> ihds, hf, iddn, df = flopy.mfusg.MfUsgOc.get_ocoutput_units('test.oc')

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
    def load(cls, f, model, nper=None, nstp=None, nlay=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mfusg.MfUsg`) to
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
        oc : MfUsgOc object
            MfUsgOc object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.mfusg.Mfusg()
        >>> oc = flopy.mfusg.MfUsgOc.load('test.oc', m)

        """

        if model.verbose:
            print("loading oc package file...")

        # set nper
        if nper is None or nlay is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        if nper == 0 or nlay == 0:
            raise ValueError(
                "discretization package not defined for the model, "
                "nper and nlay must be provided to the .load() method"
            )

        # set nstp
        if nstp is None:
            dis = model.get_package("DIS")
            if dis is None:
                dis = model.get_package("DISU")
            if dis is None:
                raise ValueError(
                    "discretization package not defined for the model, "
                    "a nstp list must be provided to the .load() method"
                )
            nstp = list(dis.nstp.array)
        else:
            if isinstance(nstp, (int, float)):
                nstp = [int(nstp)]

        # validate the size of nstp
        if len(nstp) != nper:
            raise OSError(
                f"nstp must be a list with {nper} entries, "
                f"provided nstp list has {len(nstp)} entries."
            )

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

        # concentration of species
        ispcfm = 0
        ispcun = 0
        cspcfm = None

        stress_period_data = {}

        # read header
        kwargs = {}
        kwargs["atsa"] = 0
        kwargs["nptimes"] = 0
        kwargs["npsteps"] = 0
        kwargs["timot"] = None

        # FASTFORWARD initial conditions of heads
        # stress period number to fast-forward
        kwargs["ispfast"] = 0
        # time step number to fast-forward
        kwargs["itsfast"] = 0
        # unit number for groundwater head file
        kwargs["iugfast"] = 0
        # unit number for CLN domain head file
        kwargs["iucfast"] = 0
        # unit number for dual domain head file
        kwargs["iudfast"] = 0
        # FASTFORWARD initial conditions of concentration
        # stress period number to fast-forward concentration
        kwargs["ispfastc"] = 0
        # time step number to fast-forward concentration
        kwargs["itsfastc"] = 0
        # unit number for groundwater concentration file
        kwargs["iugfastc"] = 0
        # unit number for CLN domain concentration file
        kwargs["iucfastc"] = 0
        # unit number for dual domain concentration file
        kwargs["iudfastc"] = 0
        # unit number for matrix diffusion concentration file
        kwargs["iumfastc"] = 0

        # BOOTSTRAPPING initial conditions of heads
        # unit number for groundwater head file
        kwargs["iugboot"] = 0
        # unit number for CLN domain head file
        kwargs["iucboot"] = 0
        # unit number for dual domain head file
        kwargs["iudboot"] = 0

        numericformat = False

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        else:
            filename = os.path.basename(f.name)

        line = f.readline().upper()
        while line[0] == "#":
            line = f.readline().upper()

        lnlst = line.strip().split()
        try:
            ihedfm, iddnfm = int(lnlst[0]), int(lnlst[1])
            ihedun, iddnun = int(lnlst[2]), int(lnlst[3])
            numericformat = True
        except:
            f.seek(0)

        # read options
        ispcfm = type_from_iterable(lnlst, index=4)
        ispcun = type_from_iterable(lnlst, index=5)
        if "ATS" in lnlst or "ATSA" in lnlst:
            kwargs["atsa"] = 1
        if "NPTIMES" in lnlst:
            idx = lnlst.index("NPTIMES")
            kwargs["nptimes"] = int(lnlst[idx + 1])
        if "NPSTPS" in lnlst:
            idx = lnlst.index("NPSTPS")
            kwargs["npsteps"] = int(lnlst[idx + 1])
        if "FASTFORWARD" in lnlst:
            idx = lnlst.index("FASTFORWARD")
            kwargs["ispfast"] = int(lnlst[idx + 1])
            kwargs["itsfast"] = int(lnlst[idx + 2])
            kwargs["iugfast"] = int(lnlst[idx + 3])
            if model.icln:
                kwargs["iucfast"] = int(lnlst[idx + 4])
            if model.idpf:
                kwargs["iudfast"] = int(lnlst[idx + 5])
        if "FASTFORWARDC" in lnlst:
            idx = lnlst.index("FASTFORWARDC")
            kwargs["ispfastc"] = int(lnlst[idx + 1])
            kwargs["itsfastc"] = int(lnlst[idx + 2])
            kwargs["iugfastc"] = int(lnlst[idx + 3])
            if model.icln:
                kwargs["iucfastc"] = int(lnlst[idx + 4])
            if model.idpt:
                kwargs["iudfastc"] = int(lnlst[idx + 5])
            if model.imdt:
                kwargs["iumfastc"] = int(lnlst[idx + 5])
        if "BOOTSTRAPPING" in lnlst:
            idx = lnlst.index("BOOTSTRAPPING")
            kwargs["iugboot"] = int(lnlst[idx + 1])
            if model.icln:
                kwargs["iucboot"] = int(lnlst[idx + 2])
            if model.idpf:
                kwargs["iudboot"] = int(lnlst[idx + 3])

        # read numeric formats
        if numericformat:
            # read TIMOT values
            if kwargs["atsa"] and kwargs["nptimes"] > 0:
                line = f.readline()
                lnlst = line.strip().split()
                kwargs["timot"] = [0] * kwargs["nptimes"]
                for i in range(kwargs["nptimes"]):
                    kwargs["timot"][i] = int(lnlst[i])
            # process each line
            lines = []
            # Item 3 is read for each time step if adaptive time stepping is not used.
            for iperoc in range(nper):
                # adaptive time stepping is used, read item 3 for each stress period
                if kwargs["atsa"]:
                    nstp[iperoc] = 1
                for itsoc in range(nstp[iperoc]):
                    lines = []
                    if kwargs["atsa"]:
                        line = f.readline()
                        lnlst = line.strip().split()
                        lines.append("DELTAT {float(lnlst[0]):11.4e}")
                        lines.append("TMINAT {float(lnlst[1]):11.4e}")
                        lines.append("TMAXAT {float(lnlst[2]):11.4e}")
                        lines.append("TADJAT {float(lnlst[3]):11.4e}")
                        lines.append("TCUTAT {float(lnlst[4]):11.4e}")

                    line = f.readline()
                    lnlst = line.strip().split()
                    incode, ihddfl = int(lnlst[0]), int(lnlst[1])
                    ibudfl, icbcfl = int(lnlst[2]), int(lnlst[3])
                    ispcfl = type_from_iterable(lnlst, index=4)

                    # new print and save flags are needed if incode is not
                    #  less than 0.
                    if incode < 0:
                        if len(lines) > 0:
                            stress_period_data[iperoc, itsoc] = list(lines)
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
                        cnpr = type_from_iterable(lnlst, index=4)
                        cnsv = type_from_iterable(lnlst, index=5)
                        if hdpr != 0:
                            lines.append("PRINT HEAD")
                        if ddpr != 0:
                            lines.append("PRINT DRAWDOWN")
                        if cnpr != 0:
                            lines.append("PRINT CONC")
                        if hdsv != 0:
                            lines.append("SAVE HEAD")
                        if ddsv != 0:
                            lines.append("SAVE DRAWDOWN")
                        if cnsv != 0:
                            lines.append("SAVE CONC")
                    elif incode > 0:
                        headprint = ""
                        headsave = ""
                        ddnprint = ""
                        ddnsave = ""
                        spcprint = ""
                        spcsave = ""
                        for k in range(nlay):
                            line = f.readline()
                            lnlst = line.strip().split()
                            hdpr, ddpr = int(lnlst[0]), int(lnlst[1])
                            hdsv, ddsv = int(lnlst[2]), int(lnlst[3])
                            cnpr = type_from_iterable(lnlst, index=4)
                            cnsv = type_from_iterable(lnlst, index=5)
                            if hdpr != 0:
                                headprint += f" {k + 1}"
                            if ddpr != 0:
                                ddnprint += f" {k + 1}"
                            if hdsv != 0:
                                headsave += f" {k + 1}"
                            if ddsv != 0:
                                ddnsave += f" {k + 1}"
                            if cnpr != 0:
                                spcprint += f" {k + 1}"
                            if cnsv != 0:
                                spcsave += f" {k + 1}"
                        if len(headprint) > 0:
                            lines.append(f"PRINT HEAD{headprint}")
                        if len(ddnprint) > 0:
                            lines.append(f"PRINT DRAWDOWN{ddnprint}")
                        if len(headsave) > 0:
                            lines.append(f"SAVE HEAD{headsave}")
                        if len(ddnsave) > 0:
                            lines.append(f"SAVE DRAWDOWN{ddnsave}")
                        if len(spcprint) > 0:
                            lines.append(f"PRINT CONC{spcprint}")
                        if len(spcsave) > 0:
                            lines.append(f"SAVE CONC{spcsave}")
                    stress_period_data[iperoc, itsoc] = list(lines)

        # Output Control Using Words
        else:
            iperoc, itsoc = 0, 0
            lines = []
            while True:
                line = f.readline().upper()
                if not line:
                    if len(lines) > 0:
                        stress_period_data[iperoc, itsoc] = list(lines)
                    break
                lnlst = line.strip().split()
                if line[0] == "#":
                    continue

                # added by JJS 12/12/14 to avoid error when there is a blank line
                if lnlst == []:
                    continue
                # end add

                # dataset 1 values

                # TIMOT : output time values
                elif lnlst[0].isdigit():
                    if kwargs["atsa"] and kwargs["nptimes"] > 0:
                        kwargs["timot"] = [0] * kwargs["nptimes"]
                        for i in range(kwargs["nptimes"]):
                            kwargs["timot"][i] = int(lnlst[i])
                elif (
                    "HEAD" in lnlst[0] and "PRINT" in lnlst[1] and "FORMAT" in lnlst[2]
                ):
                    ihedfm = int(lnlst[3])

                elif "HEAD" in lnlst[0] and "SAVE" in lnlst[1] and "FORMAT" in lnlst[2]:
                    chedfm = lnlst[3]
                elif "HEAD" in lnlst[0] and "SAVE" in lnlst[1] and "UNIT" in lnlst[2]:
                    ihedun = int(lnlst[3])
                elif (
                    "DRAWDOWN" in lnlst[0]
                    and "PRINT" in lnlst[1]
                    and "FORMAT" in lnlst[2]
                ):
                    iddnfm = int(lnlst[3])
                elif (
                    "DRAWDOWN" in lnlst[0]
                    and "SAVE" in lnlst[1]
                    and "FORMAT" in lnlst[2]
                ):
                    cddnfm = lnlst[3]
                elif (
                    "DRAWDOWN" in lnlst[0] and "SAVE" in lnlst[1] and "UNIT" in lnlst[2]
                ):
                    iddnun = int(lnlst[3])
                elif (
                    "IBOUND" in lnlst[0] and "SAVE" in lnlst[1] and "FORMAT" in lnlst[2]
                ):
                    cboufm = lnlst[3]
                elif "IBOUND" in lnlst[0] and "SAVE" in lnlst[1] and "UNIT" in lnlst[2]:
                    ibouun = int(lnlst[3])
                elif (
                    "CONC" in lnlst[0] and "PRINT" in lnlst[1] and "FORMAT" in lnlst[2]
                ):
                    ispcfm = int(lnlst[3])
                elif "CONC" in lnlst[0] and "SAVE" in lnlst[1] and "FORMAT" in lnlst[2]:
                    cspcfm = lnlst[3]
                elif "CONC" in lnlst[0] and "SAVE" in lnlst[1] and "UNIT" in lnlst[2]:
                    ispcun = int(lnlst[3])
                elif "COMPACT" in lnlst[0]:
                    compact = True

                # dataset 2
                elif "PERIOD" in lnlst[0]:
                    iperoc1 = int(lnlst[1]) - 1
                    if "STEP" in lnlst:
                        itsoc1 = int(lnlst[3]) - 1
                    else:
                        itsoc1 = 0

                    if len(lines) > 0:
                        if iperoc1 > 0 or itsoc1 > 0:
                            # create period step tuple
                            kperkstp = (iperoc, itsoc)
                            # save data
                            stress_period_data[kperkstp] = lines
                        # reset lines
                        lines = []

                    # update iperoc and itsoc
                    iperoc, itsoc = iperoc1, itsoc1

                    # do not used data that exceeds nper
                    if iperoc > nper:
                        break

                # dataset 3
                elif "PRINT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {lnlst[1]}")
                elif "SAVE" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {lnlst[1]}")
                elif "DELTAT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "TMINAT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "TMAXAT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "TADJAT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "TCUTAT" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "HCLOSE" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "BTOL" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {float(lnlst[1]):11.4e}")
                elif "MXITER" in lnlst[0]:
                    lines.append(f"{lnlst[0]} {int(lnlst[1]):11d}")
                elif "BOOTSTRAP" in lnlst[0]:
                    lines.append(f"{lnlst[0]}")
                elif "NOBOOTSTRAP" in lnlst[0]:
                    lines.append(f"{lnlst[0]}")
                elif "BOOTSTRAPSCALE" in lnlst[0]:
                    lines.append(f"{lnlst[0]}")
                elif "NOBOOTSTRAPSCALE" in lnlst[0]:
                    lines.append(f"{lnlst[0]}")
                else:
                    continue

        if openfile:
            f.close()

        # reset unit numbers
        unitnumber = cls._defaultunit()
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == cls._ftype():
                    unitnumber[0] = key
                    fname = os.path.basename(value.filename)
        else:
            fname = os.path.basename(filename)

        # initialize filenames list
        filenames = [fname, None, None, None, None, None]

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
        if ispcun > 0:
            unitnumber[5] = ispcun
            try:
                filenames[5] = os.path.basename(ext_unit_dict[ispcun].filename)
            except:
                if model.verbose:
                    print("concentration file name will be generated by flopy")

        # add unit numbers to pop_key_list
        for u in unitnumber:
            model.add_pop_key_list(u)

        return cls(
            model,
            ihedfm=ihedfm,
            iddnfm=iddnfm,
            ispcfm=ispcfm,
            chedfm=chedfm,
            cddnfm=cddnfm,
            cboufm=cboufm,
            cspcfm=cspcfm,
            compact=compact,
            stress_period_data=stress_period_data,
            unitnumber=unitnumber,
            filenames=filenames,
            **kwargs,
        )

    @staticmethod
    def _ftype():
        return "OC"

    @staticmethod
    def _defaultunit():
        return [14, 0, 0, 0, 0, 0, 0]
