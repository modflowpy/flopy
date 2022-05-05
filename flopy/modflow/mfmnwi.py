from ..pakbase import Package
from ..utils.flopy_io import line_parse, pop_item


class ModflowMnwi(Package):
    """
    'Multi-Node Well Information Package Class'

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    wel1flag : integer
        Flag indicating output to be written for each MNW node at the end of
        each stress period
    qsumflag :integer
        Flag indicating output to be written for each multi-node well
    byndflag :integer
        Flag indicating output to be written for each MNW node
    mnwobs :integer
        Number of multi-node wells for which detailed flow, head, and solute
        data to be saved
    wellid_unit_qndflag_qhbflag_concflag : list of lists
        Containing wells and related information to be output
        (length : [MNWOBS][4or5])
    extension : string
        Filename extension (default is 'mnwi')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the output names will be created using
        the model name and output extensions. Default is None.

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
    >>> ml = flopy.modflow.Modflow()
    >>> ghb = flopy.modflow.ModflowMnwi(ml, ...)

    """

    def __init__(
        self,
        model,
        wel1flag=None,
        qsumflag=None,
        byndflag=None,
        mnwobs=1,
        wellid_unit_qndflag_qhbflag_concflag=None,
        extension="mnwi",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowMnwi._defaultunit()

        # determine the number of unique unit numbers in dataset 3
        unique_units = []
        if wellid_unit_qndflag_qhbflag_concflag is not None:
            for t in wellid_unit_qndflag_qhbflag_concflag:
                iu = int(t[1])
                if iu not in unique_units:
                    unique_units.append(iu)

        # set filenames
        nfn = 4 + len(unique_units)
        filenames = self._prepare_filenames(filenames, nfn)

        # update external file information with unit_pc output, if necessary
        if wel1flag is not None:
            model.add_output_file(
                wel1flag,
                fname=filenames[1],
                extension="wel1",
                binflag=False,
                package=self._ftype(),
            )
        else:
            wel1flag = 0

        # update external file information with unit_ts output, if necessary
        if qsumflag is not None:
            model.add_output_file(
                qsumflag,
                fname=filenames[2],
                extension="qsum",
                binflag=False,
                package=self._ftype(),
            )
        else:
            qsumflag = 0

        # update external file information with ipunit output, if necessary
        if byndflag is not None:
            model.add_output_file(
                byndflag,
                fname=filenames[3],
                extension="bynd",
                binflag=False,
                package=self._ftype(),
            )
        else:
            byndflag = 0

        for idx, iu in enumerate(unique_units, 4):
            model.add_output_file(
                iu,
                fname=filenames[idx],
                extension=f"{iu:04d}.mnwobs",
                binflag=False,
                package=self._ftype(),
            )

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        self.url = "mnwi.html"
        self._generate_heading()
        # integer flag indicating output to be written for each MNW node at
        # the end of each stress period
        self.wel1flag = wel1flag
        # integer flag indicating output to be written for each multi-node well
        self.qsumflag = qsumflag
        # integer flag indicating output to be written for each MNW node
        self.byndflag = byndflag
        # number of multi-node wells for which detailed flow, head, and solute
        # data to be saved
        self.mnwobs = mnwobs
        # list of lists containing wells and related information to be
        # output (length = [MNWOBS][4or5])
        self.wellid_unit_qndflag_qhbflag_concflag = (
            wellid_unit_qndflag_qhbflag_concflag
        )

        # -input format checks:
        assert (
            self.wel1flag >= 0
        ), "WEL1flag must be greater than or equal to zero."
        assert (
            self.qsumflag >= 0
        ), "QSUMflag must be greater than or equal to zero."
        assert (
            self.byndflag >= 0
        ), "BYNDflag must be greater than or equal to zero."

        if len(self.wellid_unit_qndflag_qhbflag_concflag) != self.mnwobs:
            print(
                "WARNING: number of listed well ids to be "
                "monitored does not match MNWOBS."
            )

        self.parent.add_package(self)

    @classmethod
    def load(cls, f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):

        if model.verbose:
            print("loading mnw2 package file...")

        structured = model.structured
        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
            # otherwise iterations from 0, nper won't run
            nper = 1 if nper == 0 else nper

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 1
        line = line_parse(next(f))
        wel1flag, qsumflag, byndflag = map(int, line)
        if wel1flag > 0:
            model.add_pop_key_list(wel1flag)
        if qsumflag > 0:
            model.add_pop_key_list(qsumflag)
        if byndflag > 0:
            model.add_pop_key_list(byndflag)

        # dataset 2
        unique_units = []
        mnwobs = pop_item(line_parse(next(f)), int)
        wellid_unit_qndflag_qhbflag_concflag = []
        if mnwobs > 0:
            for i in range(mnwobs):
                # dataset 3
                line = line_parse(next(f))
                wellid = pop_item(line, str)
                unit = pop_item(line, int)
                qndflag = pop_item(line, int)
                qbhflag = pop_item(line, int)
                tmp = [wellid, unit, qndflag, qbhflag]
                if gwt and len(line) > 0:
                    tmp.append(pop_item(line, int))
                wellid_unit_qndflag_qhbflag_concflag.append(tmp)
                if unit not in unique_units:
                    unique_units.append(unit)

        if openfile:
            f.close()

        for unit in unique_units:
            model.add_pop_key_list(unit)

        # determine specified unit number
        nfn = 4 + len(unique_units)
        unitnumber = None
        filenames = [None for x in range(nfn)]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowMnwi._ftype()
            )
            if wel1flag > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=wel1flag
                )
            if qsumflag > 0:
                iu, filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=qsumflag
                )
            if byndflag > 0:
                iu, filenames[3] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=byndflag
                )
            idx = 4
            for unit in unique_units:
                iu, filenames[idx] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=unit
                )
                idx += 1

        return cls(
            model,
            wel1flag=wel1flag,
            qsumflag=qsumflag,
            byndflag=byndflag,
            mnwobs=mnwobs,
            wellid_unit_qndflag_qhbflag_concflag=wellid_unit_qndflag_qhbflag_concflag,
            extension="mnwi",
            unitnumber=unitnumber,
            filenames=filenames,
        )

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check mnwi package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
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
        >>> m.mnwi.check()
        """
        chk = self._get_check(f, verbose, level, checktype)
        if "MNW2" not in self.parent.get_package_list():
            desc = "\r    MNWI package present without MNW2 package."
            chk._add_to_summary(type="Warning", value=0, desc=desc)

        chk.summarize()
        return chk

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """

        # -open file for writing
        f = open(self.fn_path, "w")

        # header not supported
        # # -write header
        # f.write('{}\n'.format(self.heading))

        # dataset 1 - WEL1flag QSUMflag SYNDflag
        line = f"{self.wel1flag:10d}"
        line += f"{self.qsumflag:10d}"
        line += f"{self.byndflag:10d}\n"
        f.write(line)

        # dataset 2 - MNWOBS
        f.write(f"{self.mnwobs:10d}\n")

        # dataset 3 -  WELLID UNIT QNDflag QBHflag {CONCflag}
        # (Repeat MNWOBS times)
        nitems = len(self.wellid_unit_qndflag_qhbflag_concflag[0])
        for i, t in enumerate(self.wellid_unit_qndflag_qhbflag_concflag):
            wellid = t[0]
            unit = t[1]
            qndflag = t[2]
            qhbflag = t[3]
            assert (
                qndflag >= 0
            ), "QNDflag must be greater than or equal to zero."
            assert (
                qhbflag >= 0
            ), "QHBflag must be greater than or equal to zero."
            line = f"{wellid:20s} "
            line += f"{unit:5d} "
            line += f"{qndflag:5d} "
            line += f"{qhbflag:5d} "
            if nitems == 5:
                concflag = t[4]
                assert (
                    0 <= concflag <= 3
                ), "CONCflag must be an integer between 0 and 3."
                assert isinstance(
                    concflag, int
                ), "CONCflag must be an integer between 0 and 3."
                line += f"{concflag:5d} "
            line += "\n"
            f.write(line)

        f.close()

    @staticmethod
    def _ftype():
        return "MNWI"

    @staticmethod
    def _defaultunit():
        return 58
