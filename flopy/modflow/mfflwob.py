import os
import sys
import numpy as np
from ..pakbase import Package
from ..utils import parsenamefile


class ModflowFlwob(Package):
    """
    Head-dependent flow boundary Observation package class. Minimal working
    example that will be refactored in a future version.

    Parameters
    ----------
    nqfb : int
        Number of cell groups for the head-dependent flow boundary
        observations
    nqcfb : int
        Greater than or equal to the total number of cells in all cell groups
    nqtfb : int
        Total number of head-dependent flow boundary observations for all cell
        groups
    iufbobsv : int
        unit number where output is saved
    tomultfb : float
        Time-offset multiplier for head-dependent flow boundary observations.
        The product of tomultfb and toffset must produce a time value in units
        consistent with other model input. tomultfb can be dimensionless or
        can be used to convert the units of toffset to the time unit used in
        the simulation.
    nqobfb : int list of length nqfb
        The number of times at which flows are observed for the group of cells
    nqclfb : int list of length nqfb
        Is a flag, and the absolute value of nqclfb is the number of cells in
        the group.  If nqclfb is less than zero, factor = 1.0 for all cells in
        the group.
    obsnam : string list of length nqtfb
        Observation name
    irefsp : int of length nqtfb
        The zero-based stress period to which the observation time is
        referenced.
        The reference point is the beginning of the specified stress period.
    toffset : float list of length nqtfb
        Is the time from the beginning of the stress period irefsp to the time
        of the observation.  toffset must be in units such that the product of
        toffset and tomultfb are consistent with other model input.  For
        steady state observations, specify irefsp as the steady state stress
        period and toffset less than or equal to perlen of the stress period.
        If perlen is zero, set toffset to zero.  If the observation falls
        within a time step, linearly interpolation is used between values at
        the beginning and end of the time step.
    flwobs : float list of length nqtfb
        Observed flow value from the head-dependent flow boundary into the
        aquifer (+) or the flow from the aquifer into the boundary (-)
    layer : int list of length(nqfb, nqclfb)
        The zero-based layer index for the cell included in the cell group.
    row : int list of length(nqfb, nqclfb)
        The zero-based row index for the cell included in the cell group.
    column : int list of length(nqfb, nqclfb)
        The zero-based column index of the cell included in the cell group.
    factor : float list of length(nqfb, nqclfb)
        Is the portion of the simulated gain or loss in the cell that is
        included in the total gain or loss for this cell group (fn of eq. 5).
    flowtype : string
        String that corresponds to the head-dependent flow boundary condition
        type (CHD, GHB, DRN, RIV)
    extension : list of string
        Filename extension. If extension is None, extension is set to
        ['chob','obc','gbob','obg','drob','obd', 'rvob','obr']
        (default is None).
    no_print : boolean
        When True or 1, a list of flow observations will not be
        written to the Listing File (default is False)
    options : list of strings
        Package options (default is None).
    unitnumber : list of int
        File unit number. If unitnumber is None, unitnumber is set to
        [40, 140, 41, 141, 42, 142, 43, 143] (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the flwob output name will be created using
        the model name and .out extension (for example,
        modflowtest.out), if iufbobsv is a number greater than zero.
        If a single string is passed the package will be set to the string
        and flwob output name will be created using the model name and .out
        extension, if iufbobsv is a number greater than zero. To define the
        names for all package files (input and output) the length of the list
        of strings should be 2. Default is None.


    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    This represents a minimal working example that will be refactored in a
    future version.

    """

    def __init__(
        self,
        model,
        nqfb=0,
        nqcfb=0,
        nqtfb=0,
        iufbobsv=0,
        tomultfb=1.0,
        nqobfb=None,
        nqclfb=None,
        obsnam=None,
        irefsp=None,
        toffset=None,
        flwobs=None,
        layer=None,
        row=None,
        column=None,
        factor=None,
        flowtype=None,
        extension=None,
        no_print=False,
        options=None,
        filenames=None,
        unitnumber=None,
    ):

        """
        Package constructor
        """
        if nqobfb is None:
            nqobfb = []
        if nqclfb is None:
            nqclfb = []
        if obsnam is None:
            obsnam = []
        if irefsp is None:
            irefsp = []
        if toffset is None:
            toffset = []
        if flwobs is None:
            flwobs = []
        if layer is None:
            layer = []
        if row is None:
            row = []
        if column is None:
            column = []
        if factor is None:
            factor = []
        if extension is None:
            extension = [
                "chob",
                "obc",
                "gbob",
                "obg",
                "drob",
                "obd",
                "rvob",
                "obr",
            ]
        pakunits = {"chob": 40, "gbob": 41, "drob": 42, "rvob": 43}
        outunits = {"chob": 140, "gbob": 141, "drob": 142, "rvob": 143}
        # if unitnumber is None:
        #     unitnumber = [40, 140, 41, 141, 42, 142, 43, 143]

        if flowtype.upper().strip() == "CHD":
            name = ["CHOB", "DATA"]
            extension = extension[0:2]
            # unitnumber = unitnumber[0:2]
            # iufbobsv = unitnumber[1]
            self._ftype = "CHOB"
            self.url = "chob.htm"
            self.heading = "# CHOB for MODFLOW, generated by Flopy."
        elif flowtype.upper().strip() == "GHB":
            name = ["GBOB", "DATA"]
            extension = extension[2:4]
            # unitnumber = unitnumber[2:4]
            # iufbobsv = unitnumber[1]
            self._ftype = "GBOB"
            self.url = "gbob.htm"
            self.heading = "# GBOB for MODFLOW, generated by Flopy."
        elif flowtype.upper().strip() == "DRN":
            name = ["DROB", "DATA"]
            extension = extension[4:6]
            # unitnumber = unitnumber[4:6]
            # iufbobsv = unitnumber[1]
            self._ftype = "DROB"
            self.url = "drob.htm"
            self.heading = "# DROB for MODFLOW, generated by Flopy."
        elif flowtype.upper().strip() == "RIV":
            name = ["RVOB", "DATA"]
            extension = extension[6:8]
            # unitnumber = unitnumber[6:8]
            # iufbobsv = unitnumber[1]
            self._ftype = "RVOB"
            self.url = "rvob.htm"
            self.heading = "# RVOB for MODFLOW, generated by Flopy."
        else:
            msg = "ModflowFlwob: flowtype must be CHD, GHB, DRN, or RIV"
            raise KeyError(msg)

        if unitnumber is None:
            unitnumber = [pakunits[name[0].lower()], outunits[name[0].lower()]]
        elif isinstance(unitnumber, int):
            unitnumber = [unitnumber]
        if len(unitnumber) == 1:
            if unitnumber[0] in outunits.keys():
                unitnumber = [pakunits[name[0].lower()], unitnumber[0]]
            else:
                unitnumber = [unitnumber[0], outunits[name[0].lower()]]
        iufbobsv = unitnumber[1]

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # call base package constructor
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=unitnumber,
            allowDuplicates=True,
            filenames=filenames,
        )

        self.nqfb = nqfb
        self.nqcfb = nqcfb
        self.nqtfb = nqtfb
        self.iufbobsv = iufbobsv
        self.tomultfb = tomultfb
        self.nqobfb = nqobfb
        self.nqclfb = nqclfb
        self.obsnam = obsnam
        self.irefsp = irefsp
        self.toffset = toffset
        self.flwobs = flwobs
        self.layer = layer
        self.row = row
        self.column = column
        self.factor = factor

        # -create empty arrays of the correct size
        self.layer = np.zeros(
            (self.nqfb, max(np.abs(self.nqclfb))), dtype="int32"
        )
        self.row = np.zeros(
            (self.nqfb, max(np.abs(self.nqclfb))), dtype="int32"
        )
        self.column = np.zeros(
            (self.nqfb, max(np.abs(self.nqclfb))), dtype="int32"
        )
        self.factor = np.zeros(
            (self.nqfb, max(np.abs(self.nqclfb))), dtype="float32"
        )
        self.nqobfb = np.zeros((self.nqfb), dtype="int32")
        self.nqclfb = np.zeros((self.nqfb), dtype="int32")
        self.irefsp = np.zeros((self.nqtfb), dtype="int32")
        self.toffset = np.zeros((self.nqtfb), dtype="float32")
        self.flwobs = np.zeros((self.nqtfb), dtype="float32")

        # -assign values to arrays

        self.nqobfb[:] = nqobfb
        self.nqclfb[:] = nqclfb
        self.obsnam[:] = obsnam
        self.irefsp[:] = irefsp
        self.toffset[:] = toffset
        self.flwobs[:] = flwobs
        for i in range(self.nqfb):
            self.layer[i, : len(layer[i])] = layer[i]
            self.row[i, : len(row[i])] = row[i]
            self.column[i, : len(column[i])] = column[i]
            self.factor[i, : len(factor[i])] = factor[i]

        # add more checks here

        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append("NOPRINT")
        self.options = options

        # add checks for input compliance (obsnam length, etc.)
        self.parent.add_package(self)

    def ftype(self):
        return self._ftype

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # open file for writing
        f_fbob = open(self.fn_path, "w")

        # write header
        f_fbob.write("{}\n".format(self.heading))

        # write sections 1 and 2 : NOTE- what about NOPRINT?
        line = "{:10d}".format(self.nqfb)
        line += "{:10d}".format(self.nqcfb)
        line += "{:10d}".format(self.nqtfb)
        line += "{:10d}".format(self.iufbobsv)
        if self.no_print or "NOPRINT" in self.options:
            line += "{: >10}".format("NOPRINT")
        line += "\n"
        f_fbob.write(line)
        f_fbob.write("{:10e}\n".format(self.tomultfb))

        # write sections 3-5 looping through observations groups
        c = 0
        for i in range(self.nqfb):
            #        while (i < self.nqfb):
            # write section 3
            f_fbob.write(
                "{:10d}{:10d}\n".format(self.nqobfb[i], self.nqclfb[i])
            )

            # Loop through observation times for the groups
            for j in range(self.nqobfb[i]):
                # write section 4
                line = "{:12}".format(self.obsnam[c])
                line += "{:8d}".format(self.irefsp[c] + 1)
                line += "{:16.10g}".format(self.toffset[c])
                line += " {:10.4g}\n".format(self.flwobs[c])
                f_fbob.write(line)
                c += 1  # index variable

                # write section 5 - NOTE- need to adjust factor for multiple
                # observations in the same cell
            for j in range(abs(self.nqclfb[i])):
                # set factor to 1.0 for all cells in group
                if self.nqclfb[i] < 0:
                    self.factor[i, :] = 1.0
                line = "{:10d}".format(self.layer[i, j] + 1)
                line += "{:10d}".format(self.row[i, j] + 1)
                line += "{:10d}".format(self.column[i, j] + 1)
                line += " ".format(self.factor[i, j])
                # note is 10f good enough here?
                line += "{:10f}\n".format(self.factor[i, j])
                f_fbob.write(line)

        f_fbob.close()

        #
        # swm: BEGIN hack for writing standard file
        sfname = self.fn_path
        sfname += "_ins"

        # write header
        f_ins = open(sfname, "w")
        f_ins.write("jif @\n")
        f_ins.write("StandardFile 0 1 {}\n".format(self.nqtfb))
        for i in range(0, self.nqtfb):
            f_ins.write("{}\n".format(self.obsnam[i]))

        f_ins.close()
        # swm: END hack for writing standard file

        return

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
        flwob : ModflowFlwob package object
            ModflowFlwob package object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> hobs = flopy.modflow.ModflowFlwob.load('test.drob', m)

        """

        if model.verbose:
            sys.stdout.write("loading flwob package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # read dataset 1 -- NQFB NQCFB NQTFB IUFBOBSV Options
        t = line.strip().split()
        nqfb = int(t[0])
        nqcfb = int(t[1])
        nqtfb = int(t[2])
        iufbobsv = int(t[3])
        options = []
        if len(t) > 4:
            options = t[4:]

        # read dataset 2 -- TOMULTFB
        line = f.readline()
        t = line.strip().split()
        tomultfb = float(t[0])

        nqobfb = np.zeros(nqfb, dtype=np.int32)
        nqclfb = np.zeros(nqfb, dtype=np.int32)
        obsnam = []
        irefsp = []
        toffset = []
        flwobs = []

        layer = []
        row = []
        column = []
        factor = []

        # read datasets 3, 4, and 5 for each of nqfb groups
        # of cells
        nobs = 0
        while True:

            # read dataset 3 -- NQOBFB NQCLFB
            line = f.readline()
            t = line.strip().split()
            nqobfb[nobs] = int(t[0])
            nqclfb[nobs] = int(t[1])

            # read dataset 4 -- OBSNAM IREFSP TOFFSET FLWOBS
            ntimes = 0
            while True:
                line = f.readline()
                t = line.strip().split()
                obsnam.append(t[0])
                irefsp.append(int(t[1]))
                toffset.append(float(t[2]))
                flwobs.append(float(t[3]))
                ntimes += 1
                if ntimes == nqobfb[nobs]:
                    break

            # read dataset 5 -- Layer Row Column Factor
            k = np.zeros(abs(nqclfb[nobs]), np.int32)
            i = np.zeros(abs(nqclfb[nobs]), np.int32)
            j = np.zeros(abs(nqclfb[nobs]), np.int32)
            fac = np.zeros(abs(nqclfb[nobs]), np.float32)

            ncells = 0
            while True:
                line = f.readline()
                t = line.strip().split()
                k[ncells] = int(t[0])
                i[ncells] = int(t[1])
                j[ncells] = int(t[2])
                fac[ncells] = float(t[3])

                ncells += 1
                if ncells == abs(nqclfb[nobs]):
                    layer.append(k)
                    row.append(i)
                    column.append(j)
                    factor.append(fac)
                    break

            nobs += 1
            if nobs == nqfb:
                break

        irefsp = np.array(irefsp) - 1
        layer = np.array(layer) - 1
        row = np.array(row) - 1
        column = np.array(column) - 1
        factor = np.array(factor)

        if openfile:
            f.close()

        # get ext_unit_dict if none passed
        if ext_unit_dict is None:
            namefile = os.path.join(model.model_ws, model.namefile)
            ext_unit_dict = parsenamefile(namefile, model.mfnam_packages)

        flowtype, ftype = _get_ftype_from_filename(f.name, ext_unit_dict)

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ftype.upper()
            )
            if iufbobsv > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iufbobsv
                )
                model.add_pop_key_list(iufbobsv)

        # create ModflowFlwob object instance
        flwob = cls(
            model,
            iufbobsv=iufbobsv,
            tomultfb=tomultfb,
            nqfb=nqfb,
            nqcfb=nqcfb,
            nqtfb=nqtfb,
            nqobfb=nqobfb,
            nqclfb=nqclfb,
            obsnam=obsnam,
            irefsp=irefsp,
            toffset=toffset,
            flwobs=flwobs,
            layer=layer,
            row=row,
            column=column,
            factor=factor,
            options=options,
            flowtype=flowtype,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        return flwob


def _get_ftype_from_filename(fn, ext_unit_dict=None):
    """
    Returns the boundary flowtype and filetype for a given ModflowFlwob
    package filename.

    Parameters
    ----------
    fn : str
        The filename to be parsed.
    ext_unit_dict : dictionary, optional
        If the arrays in the file are specified using EXTERNAL,
        or older style array control records, then `f` should be a file
        handle.  In this case ext_unit_dict is required, which can be
        constructed using the function
        :class:`flopy.utils.mfreadnam.parsenamefile`.

    Returns
    -------
    flowtype : str
        Corresponds to the type of the head-dependent boundary package for
        which observations are desired (e.g. "CHD", "GHB", "DRN", or "RIV").
    ftype : str
        Corresponds to the observation file type (e.g. "CHOB", "GBOB",
        "DROB", or "RVOB").
    """

    ftype = None

    # determine filetype from filename using ext_unit_dict
    if ext_unit_dict is not None:
        for key, value in ext_unit_dict.items():
            if value.filename == fn:
                ftype = value.filetype
                break

    # else, try to infer filetype from filename extension
    else:
        ext = fn.split(".")[-1].lower()
        if "ch" in ext.lower():
            ftype = "CHOB"
        elif "gb" in ext.lower():
            ftype = "GBOB"
        elif "dr" in ext.lower():
            ftype = "DROB"
        elif "rv" in ext.lower():
            ftype = "RVOB"

    msg = (
        "ModflowFlwob: filetype cannot be inferred "
        "from file name {}".format(fn)
    )
    if ftype is None:
        raise AssertionError(msg)

    flowtype_dict = {
        "CHOB": "CHD",
        "GOBO": "GHB",
        "DROB": "DRN",
        "RVOB": "RIV",
    }
    flowtype = flowtype_dict[ftype]

    return flowtype, ftype
