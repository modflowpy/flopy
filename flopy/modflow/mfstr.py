"""
mfstr module.  Contains the ModflowStr class. Note that the user can access
the ModflowStr class as `flopy.modflow.ModflowStr`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/str.htm>`_.

"""
import sys

import numpy as np
from ..utils import MfList
from ..pakbase import Package
from .mfparbc import ModflowParBc as mfparbc
from ..utils.recarray_utils import create_empty_recarray
from ..utils import read_fixed_var, write_fixed_var


class ModflowStr(Package):
    """
    MODFLOW Stream Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mxacts : int
        Maximum number of stream reaches that will be in use during any stress
        period. (default is 0)
    nss : int
        Number of stream segments. (default is 0)
    ntrib : int
        The number of stream tributaries that can connect to one segment. The
        program is currently dimensioned so that NTRIB cannot exceed 10.
        (default is 0)
    ndiv : int
        A flag, which when positive, specifies that diversions from segments
        are to be simulated. (default is 0)
    icalc : int
        A flag, which when positive, specifies that stream stages in reaches
        are to be calculated. (default is 0)
    const : float
        Constant value used in calculating stream stage in reaches whenever
        ICALC is greater than 0. This constant is 1.486 for flow units of
        cubic feet per second and 1.0 for units of cubic meters per second.
        The constant must be multiplied by 86,400 when using time units of
        days in the simulation. If ICALC is 0, const can be any real value.
        (default is 86400.)
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    istcb2 : int
        A flag that is used flag and a unit number for the option to store
        streamflow out of each reach in an unformatted (binary) file.
        If istcb2 is greater than zero streamflow data will be saved.
        (default is None).
    dtype : tuple, list, or numpy array of numpy dtypes
        is a tuple, list, or numpy array containing the dtype for
        datasets 6 and 8 and the dtype for datasets 9 and 10 data in
        stress_period_data and segment_data dictionaries.
        (default is None)
    irdflg : integer or dictionary
        is a integer or dictionary containing a integer flag, when positive
        suppresses printing of the stream input data for a stress period. If
        an integer is passed, all stress periods will use the same value.
        If a dictionary is passed, stress periods not in the dictionary will
        assigned a value of 1. Default is None which will assign a value of 1
        to all stress periods.
    iptflg : integer or dictionary
        is a integer or dictionary containing a integer flag, when positive
        suppresses printing of stream results for a stress period. If an
        integer is passed, all stress periods will use the same value.
        If a dictionary is passed, stress periods not in the dictionary will
        assigned a value of 1. Default is None which will assign a value of 1
        to all stress periods.
    stress_period_data : dictionary of reach data
        Each dictionary contains a list of str reach data for a stress period.

        Each stress period in the dictionary data contains data for
        datasets 6 and 8.

        The value for stress period data for a stress period can be an integer
        (-1 or 0), a list of lists, a numpy array, or a numpy recarray. If
        stress period data for a stress period contains an integer, a -1
        denotes data from the previous stress period will be reused and a 0
        indicates there are no str reaches for this stress period.

        Otherwise stress period data for a stress period should contain mxacts
        or fewer rows of data containing data for each reach. Reach data are
        specified through definition of layer (int), row (int), column (int),
        segment number (int), sequential reach number (int), flow entering a
        segment (float), stream stage (float), streambed hydraulic conductance
        (float), streambed bottom elevation (float), streambed top elevation
        (float), stream width (float), stream slope (float), roughness
        coefficient (float), and auxiliary variable data for auxiliary variables
        defined in options (float).

        If icalc=0 is specified, stream width, stream slope, and roughness
        coefficients, are not used and can be any value for each stress period.
        If data are specified for dataset 6 for a given stress period and
        icalc>0, then stream width, stream slope, and roughness coefficients
        should be appropriately set.

        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. For example, if mxacts=3 this gives the form of::

            stress_period_data =
            {0: [
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough]]
                ],
            1:  [
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough]]
                ], ...
            kper:
                [
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough],
                [lay, row, col, seg, reach, flow, stage, cond, sbot, stop, width, slope, rough]]
                ]
            }

    segment_data : dictionary of str segment data
        Each dictionary contains a list of segment str data for a stress period.

        Each stress period in the dictionary data contains data for
        datasets 9, and 10. Segment data for a stress period are ignored if
        a integer value is specified for stress period data.

        The value for segment data for a stress period can be an integer
        (-1 or 0), a list of lists, a numpy array, or a numpy recarray. If
        segment data for a stress period contains an integer, a -1 denotes
        data from the previous stress period will be reused and a 0 indicates
        there are no str segments for this stress period.

        Otherwise stress period data for a stress period should contain nss
        rows of data containing data for each segment. Segment data are
        specified through definition of itrib (int) data for up to 10
        tributaries and iupseg (int) data.

        If ntrib=0 is specified, itrib values are not used and can be any value
        for each stress period. If data are specified for dataset 6 for a given
        stress period and ntrib>0, then itrib data should be specified for
        columns 0:ntrib.

        If ndiv=0 is specified, iupseg values are not used and can be any value
        for each stress period. If data are specified for dataset 6 for a given
        stress period and ndiv>0, then iupseg data should be specified for the
        column in the dataset [10].

        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. For example, if nss=2 and ntrib>0 and/or ndiv>0 this gives the
        form of::

            segment_data =
            {0: [
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                ],
            1:  [
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                ], ...
            kper:
                [
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                [itrib1, itrib2, itrib3, itrib4, itrib5, itrib6, itrib7, itrib8, itrib9, itrib10, iupseg],
                ]
            }

    options : list of strings
        Package options. Auxiliary variables included as options should be
        constructed as options=['AUXILIARY IFACE', 'AUX xyx']. Either
        'AUXILIARY' or 'AUX' can be specified (case insensitive).
        (default is None).
    extension : string
        Filename extension (default is 'str')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output and str output name will be
        created using the model name and .cbc the .sfr.bin/.sfr.out extensions
        (for example, modflowtest.cbc, and modflowtest.str.bin), if ipakcbc and
        istcb2 are numbers greater than zero. If a single string is passed
        the package will be set to the string and cbc and sf routput names
        will be created using the model name and .cbc and .str.bin/.str.out
        extensions, if ipakcbc and istcb2 are numbers greater than zero. To
        define the names for all package files (input and output) the length
        of the list of strings should be 3. Default is None.

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
    >>> strd = {}
    >>> strd[0] = [[2, 3, 4, 15.6, 1050., -4]]  #this str boundary will be
    >>>                                         #applied to all stress periods
    >>> str = flopy.modflow.ModflowStr(m, stress_period_data=strd)

    """

    def __init__(
        self,
        model,
        mxacts=0,
        nss=0,
        ntrib=0,
        ndiv=0,
        icalc=0,
        const=86400.0,
        ipakcb=None,
        istcb2=None,
        dtype=None,
        stress_period_data=None,
        segment_data=None,
        irdflg=None,
        iptflg=None,
        extension="str",
        unitnumber=None,
        filenames=None,
        options=None,
        **kwargs
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowStr._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 3:
                for idx in range(len(filenames), 3):
                    filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                ipakcb, fname=fname, package=ModflowStr._ftype()
            )
        else:
            ipakcb = 0

        if istcb2 is not None:
            fname = filenames[2]
            model.add_output_file(
                istcb2, fname=fname, package=ModflowStr._ftype()
            )
        else:
            ipakcb = 0

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowStr._ftype()]
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
        self.url = "str.htm"
        self.mxacts = mxacts
        self.nss = nss
        self.icalc = icalc
        self.ntrib = ntrib
        self.ndiv = ndiv
        self.const = const
        self.ipakcb = ipakcb
        self.istcb2 = istcb2

        # issue exception if ntrib is greater than 10
        if ntrib > 10:
            raise Exception(
                "ModflowStr error: ntrib must be less that 10: "
                + "specified value = {}".format(ntrib)
            )

        if options is None:
            options = []
        self.options = options

        # parameters are not supported
        self.npstr = 0

        # dataset 5
        # check type of irdflg and iptflg
        msg = ""
        if irdflg is not None and not isinstance(irdflg, (int, dict)):
            msg = "irdflg"
        if iptflg is not None and not isinstance(iptflg, (int, dict)):
            if len(msg) > 0:
                msg += " and "
            msg += "iptflg"
        if len(msg) > 0:
            msg += " must be an integer or a dictionary"
            raise TypeError(msg)

        # process irdflg
        self.irdflg = {}
        for n in range(self.parent.nper):
            if irdflg is None:
                self.irdflg[n] = 1
            elif isinstance(irdflg, int):
                self.irdflg[n] = irdflg
            elif isinstance(irdflg, dict):
                if n in irdflg:
                    self.irdflg[n] = irdflg[n]
                else:
                    self.irdflg[n] = 1

        # process iptflg
        self.iptflg = {}
        for n in range(self.parent.nper):
            if iptflg is None:
                self.iptflg[n] = 1
            elif isinstance(iptflg, int):
                self.iptflg[n] = iptflg
            elif isinstance(iptflg, dict):
                if n in iptflg:
                    self.iptflg[n] = iptflg[n]
                else:
                    self.iptflg[n] = 1

        # determine dtype for dataset 6
        if dtype is not None:
            self.dtype = dtype[0]
            self.dtype2 = dtype[1]
        else:
            aux_names = []
            if len(options) > 0:
                aux_names = []
                it = 0
                while True:
                    if "aux" in options[it].lower():
                        t = options[it].split()
                        aux_names.append(t[-1].lower())
                    it += 1
                    if it >= len(options):
                        break
            if len(aux_names) < 1:
                aux_names = None
            d, d2 = self.get_empty(
                1, 1, aux_names=aux_names, structured=self.parent.structured
            )
            self.dtype = d.dtype
            self.dtype2 = d2.dtype

        # convert stress_period_data for datasets 6 and 8 to a recarray if
        # necessary
        if stress_period_data is not None:
            for key, d in stress_period_data.items():
                if isinstance(d, list):
                    d = np.array(d)
                if isinstance(d, np.recarray):
                    e = (
                        "ModflowStr error: recarray dtype: "
                        + str(d.dtype)
                        + " does not match "
                        + "self dtype: "
                        + str(self.dtype)
                    )
                    assert d.dtype == self.dtype, e
                elif isinstance(d, np.ndarray):
                    d = np.core.records.fromarrays(
                        d.transpose(), dtype=self.dtype
                    )
                elif isinstance(d, int):
                    if model.verbose:
                        if d < 0:
                            msg = (
                                3 * " "
                                + "reusing str data from previous stress period"
                            )
                            print(msg)
                        elif d == 0:
                            msg = (
                                3 * " "
                                + "no str data for stress "
                                + "period {}".format(key)
                            )
                            print(msg)
                else:
                    e = (
                        "ModflowStr error: unsupported data type: "
                        + str(type(d))
                        + " at kper "
                        + "{0:d}".format(key)
                    )
                    raise Exception(e)

        # add stress_period_data to package
        self.stress_period_data = MfList(self, stress_period_data)

        # convert segment_data for datasets 9 and 10 to a recarray if necessary
        if segment_data is not None:
            for key, d in segment_data.items():
                if isinstance(d, list):
                    d = np.array(d)
                if isinstance(d, np.recarray):
                    e = (
                        "ModflowStr error: recarray dtype: "
                        + str(d.dtype)
                        + " does not match "
                        + "self dtype: "
                        + str(self.dtype2)
                    )
                    assert d.dtype == self.dtype2, e
                elif isinstance(d, np.ndarray):
                    d = np.core.records.fromarrays(
                        d.transpose(), dtype=self.dtype2
                    )
                elif isinstance(d, int):
                    if model.verbose:
                        if d < 0:
                            msg = (
                                3 * " "
                                + "reusing str segment data "
                                + "from previous stress period"
                            )
                            print(msg)
                        elif d == 0:
                            msg = (
                                3 * " "
                                + "no str segment data for "
                                + "stress period {}".format(key)
                            )
                            print(msg)
                else:
                    e = (
                        "ModflowStr error: unsupported data type: "
                        + str(type(d))
                        + " at kper "
                        + "{0:d}".format(key)
                    )
                    raise Exception(e)

        # add segment_data to package
        self.segment_data = segment_data

        self.parent.add_package(self)
        return

    @staticmethod
    def get_empty(ncells=0, nss=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype, dtype2 = ModflowStr.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return (
            create_empty_recarray(ncells, dtype=dtype, default_value=-1.0e10),
            create_empty_recarray(nss, dtype=dtype2, default_value=0),
        )

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("segment", int),
                    ("reach", int),
                    ("flow", np.float32),
                    ("stage", np.float32),
                    ("cond", np.float32),
                    ("sbot", np.float32),
                    ("stop", np.float32),
                    ("width", np.float32),
                    ("slope", np.float32),
                    ("rough", np.float32),
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("node", int),
                    ("segment", int),
                    ("reach", int),
                    ("flow", np.float32),
                    ("stage", np.float32),
                    ("cond", np.float32),
                    ("sbot", np.float32),
                    ("stop", np.float32),
                    ("width", np.float32),
                    ("slope", np.float32),
                    ("rough", np.float32),
                ]
            )

        dtype2 = np.dtype(
            [
                ("itrib01", int),
                ("itrib02", int),
                ("itrib03", int),
                ("itrib04", int),
                ("itrib05", int),
                ("itrib06", int),
                ("itrib07", int),
                ("itrib08", int),
                ("itrib09", int),
                ("itrib10", int),
                ("iupseg", int),
            ]
        )
        return dtype, dtype2

    def _ncells(self):
        """Maximum number of cells that have streams (developed for
        MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of str cells

        """
        return self.mxacts

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # set free variable
        free = self.parent.free_format_input

        # open the str file
        f_str = open(self.fn_path, "w")

        # dataset 0
        f_str.write("{0}\n".format(self.heading))

        # dataset 1 - parameters not supported on write

        # dataset 2
        line = write_fixed_var(
            [
                self.mxacts,
                self.nss,
                self.ntrib,
                self.ndiv,
                self.icalc,
                self.const,
                self.ipakcb,
                self.istcb2,
            ],
            free=free,
        )
        for opt in self.options:
            line = line.rstrip()
            line += " " + str(opt) + "\n"
        f_str.write(line)

        # dataset 3  - parameters not supported on write
        # dataset 4a - parameters not supported on write
        # dataset 4b - parameters not supported on write

        nrow, ncol, nlay, nper = self.parent.get_nrow_ncol_nlay_nper()

        kpers = list(self.stress_period_data.data.keys())
        kpers.sort()

        # set column lengths for fixed format input files for
        # datasets 6, 8, and 9
        fmt6 = [5, 5, 5, 5, 5, 15, 10, 10, 10, 10]
        fmt8 = [10, 10, 10]
        fmt9 = 5

        for iper in range(nper):
            if iper not in kpers:
                if iper == 0:
                    itmp = 0
                else:
                    itmp = -1
            else:
                tdata = self.stress_period_data[iper]
                sdata = self.segment_data[iper]
                if isinstance(tdata, int):
                    itmp = tdata
                elif tdata is None:
                    itmp = -1
                else:
                    itmp = tdata.shape[0]
            line = (
                "{:10d}".format(itmp)
                + "{:10d}".format(self.irdflg[iper])
                + "{:10d}".format(self.iptflg[iper])
                + "  # stress period {}\n".format(iper + 1)
            )
            f_str.write(line)
            if itmp > 0:
                tdata = np.recarray.copy(tdata)
                # dataset 6
                for line in tdata:
                    line["k"] += 1
                    line["i"] += 1
                    line["j"] += 1
                    ds6 = []
                    for idx, v in enumerate(line):
                        if idx < 10 or idx > 12:
                            ds6.append(v)
                        if idx > 12:
                            fmt6 += [10]
                    f_str.write(write_fixed_var(ds6, ipos=fmt6, free=free))

                # dataset 8
                if self.icalc > 0:
                    for line in tdata:
                        ds8 = []
                        for idx in range(10, 13):
                            ds8.append(line[idx])
                        f_str.write(write_fixed_var(ds8, ipos=fmt8, free=free))

                # dataset 9
                if self.ntrib > 0:
                    for line in sdata:
                        ds9 = []
                        for idx in range(self.ntrib):
                            ds9.append(line[idx])
                        f_str.write(
                            write_fixed_var(ds9, length=fmt9, free=free)
                        )

                # dataset 10
                if self.ndiv > 0:
                    for line in sdata:
                        f_str.write(
                            write_fixed_var([line[-1]], length=10, free=free)
                        )

        # close the str file
        f_str.close()

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
        str : ModflowStr object
            ModflowStr object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> strm = flopy.modflow.ModflowStr.load('test.str', m)

        """
        # set local variables
        free = model.free_format_input
        fmt2 = [10, 10, 10, 10, 10, 10, 10, 10]
        fmt6 = [5, 5, 5, 5, 5, 15, 10, 10, 10, 10]
        type6 = [
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
        ]
        fmt8 = [10, 10, 10]
        fmt9 = [5]

        if model.verbose:
            sys.stdout.write("loading str package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # read dataset 1 - optional parameters
        npstr, mxl = 0, 0
        t = line.strip().split()
        if t[0].lower() == "parameter":
            if model.verbose:
                sys.stdout.write("  loading str dataset 1\n")
            npstr = np.int32(t[1])
            mxl = np.int32(t[2])

            # read next line
            line = f.readline()

        # data set 2
        if model.verbose:
            sys.stdout.write("  loading str dataset 2\n")
        t = read_fixed_var(line, ipos=fmt2, free=free)
        mxacts = np.int32(t[0])
        nss = np.int32(t[1])
        ntrib = np.int32(t[2])
        ndiv = np.int32(t[3])
        icalc = np.int32(t[4])
        const = np.float32(t[5])
        istcb1 = np.int32(t[6])
        istcb2 = np.int32(t[7])
        ipakcb = 0
        try:
            if istcb1 != 0:
                ipakcb = istcb1
                model.add_pop_key_list(istcb1)
        except:
            if model.verbose:
                print("  could not remove unit number {}".format(istcb1))
        try:
            if istcb2 != 0:
                ipakcb = 53
                model.add_pop_key_list(istcb2)
        except:
            if model.verbose:
                print("  could not remove unit number {}".format(istcb2))

        options = []
        aux_names = []
        naux = 0
        if "AUX" in line.upper():
            t = line.strip().split()
            it = 8
            while it < len(t):
                toption = t[it]
                if "aux" in toption.lower():
                    naux += 1
                    options.append(" ".join(t[it : it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1

        # read parameter data
        if npstr > 0:
            dt = ModflowStr.get_empty(1, aux_names=aux_names).dtype
            pak_parms = mfparbc.load(
                f, npstr, dt, model, ext_unit_dict, model.verbose
            )

        if nper is None:
            nper = model.nper

        irdflg = {}
        iptflg = {}
        stress_period_data = {}
        segment_data = {}
        for iper in range(nper):
            if model.verbose:
                print(
                    "   loading "
                    + str(ModflowStr)
                    + " for kper {0:5d}".format(iper + 1)
                )
            line = f.readline()
            if line == "":
                break
            t = line.strip().split()

            # set itmp
            itmp = int(t[0])

            # set irdflg and iptflg - initialize to 0 since this is how
            # MODFLOW would interpret a missing value
            iflg0, iflg1 = 0, 0
            if len(t) > 1:
                iflg0 = int(t[1])
            if len(t) > 2:
                iflg1 = int(t[2])
            irdflg[iper] = iflg0
            iptflg[iper] = iflg1

            if itmp == 0:
                bnd_output = None
                seg_output = None
                current, current_seg = ModflowStr.get_empty(
                    itmp, nss, aux_names=aux_names
                )
            elif itmp > 0:
                if npstr > 0:
                    partype = ["cond"]
                    if model.verbose:
                        print("   reading str dataset 7")
                    for iparm in range(itmp):
                        line = f.readline()
                        t = line.strip().split()
                        pname = t[0].lower()
                        iname = "static"
                        try:
                            tn = t[1]
                            c = tn.lower()
                            instance_dict = pak_parms.bc_parms[pname][1]
                            if c in instance_dict:
                                iname = c
                            else:
                                iname = "static"
                        except:
                            if model.verbose:
                                print(
                                    "  implicit static instance for "
                                    + "parameter {}".format(pname)
                                )

                        par_dict, current_dict = pak_parms.get(pname)
                        data_dict = current_dict[iname]

                        current = ModflowStr.get_empty(
                            par_dict["nlst"], aux_names=aux_names
                        )

                        #  get appropriate parval
                        if model.mfpar.pval is None:
                            parval = float(par_dict["parval"])
                        else:
                            try:
                                parval = float(
                                    model.mfpar.pval.pval_dict[pname]
                                )
                            except:
                                parval = float(par_dict["parval"])

                        # fill current parameter data (par_current)
                        for ibnd, t in enumerate(data_dict):
                            current[ibnd] = tuple(
                                t[: len(current.dtype.names)]
                            )

                else:
                    if model.verbose:
                        print("   reading str dataset 6")
                    current, current_seg = ModflowStr.get_empty(
                        itmp, nss, aux_names=aux_names
                    )
                    for ibnd in range(itmp):
                        line = f.readline()
                        t = read_fixed_var(line, ipos=fmt6, free=free)
                        v = [tt(vv) for tt, vv in zip(type6, t)]
                        ii = len(fmt6)
                        for idx, name in enumerate(current.dtype.names[:ii]):
                            current[ibnd][name] = v[idx]
                        if len(aux_names) > 0:
                            if free:
                                tt = line.strip().split()[len(fmt6) :]
                            else:
                                istart = 0
                                for i in fmt6:
                                    istart += i
                                tt = line[istart:].strip().split()
                            for iaux, name in enumerate(aux_names):
                                current[ibnd][name] = np.float32(tt[iaux])

                # convert indices to zero-based
                current["k"] -= 1
                current["i"] -= 1
                current["j"] -= 1

                # read dataset 8
                if icalc > 0:
                    if model.verbose:
                        print("   reading str dataset 8")
                    for ibnd in range(itmp):
                        line = f.readline()
                        t = read_fixed_var(line, ipos=fmt8, free=free)
                        ipos = 0
                        for idx in range(10, 13):
                            current[ibnd][idx] = np.float32(t[ipos])
                            ipos += 1

                bnd_output = np.recarray.copy(current)

                # read data set 9
                if ntrib > 0:
                    if model.verbose:
                        print("   reading str dataset 9")
                    for iseg in range(nss):
                        line = f.readline()
                        t = read_fixed_var(line, ipos=fmt9 * ntrib, free=free)
                        v = [np.float32(vt) for vt in t]
                        names = current_seg.dtype.names[:ntrib]
                        for idx, name in enumerate(names):
                            current_seg[iseg][idx] = v[idx]

                # read data set 10
                if ndiv > 0:
                    if model.verbose:
                        print("   reading str dataset 10")
                    for iseg in range(nss):
                        line = f.readline()
                        t = read_fixed_var(line, length=10, free=free)
                        current_seg[iseg]["iupseg"] = np.int32(t[0])

                seg_output = np.recarray.copy(current_seg)

            else:
                bnd_output = -1
                seg_output = -1

            if bnd_output is None:
                stress_period_data[iper] = itmp
                segment_data[iper] = itmp
            else:
                stress_period_data[iper] = bnd_output
                segment_data[iper] = seg_output

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowStr._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
            if abs(istcb2) > 0:
                iu, filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(istcb2)
                )

        return cls(
            model,
            mxacts=mxacts,
            nss=nss,
            ntrib=ntrib,
            ndiv=ndiv,
            icalc=icalc,
            const=const,
            ipakcb=ipakcb,
            istcb2=istcb2,
            iptflg=iptflg,
            irdflg=irdflg,
            stress_period_data=stress_period_data,
            segment_data=segment_data,
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "STR"

    @staticmethod
    def _defaultunit():
        return 118
