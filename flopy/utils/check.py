import os
import numpy as np
from numpy.lib import recfunctions
from ..utils.recarray_utils import recarray


class check:
    """
    Check package for common errors

    Parameters
    ----------
    package : object
        Instance of Package class.
    verbose : bool
        Boolean flag used to determine if check method results are
        written to the screen
    level : int
        Check method analysis level. If level=0, summary checks are
        performed. If level=1, full checks are performed.
    property_threshold_values : dict
        hk : tuple
            Reasonable minimum/maximum hydraulic conductivity value; values
            below this will be flagged. Default is (1e-11, 1e5), after Bear,
            1972 (see https://en.wikipedia.org/wiki/Hydraulic_conductivity)
            and Schwartz and Zhang (2003, Table 4.4).
        vka : tuple
            Reasonable minimum/maximum hydraulic conductivity value;
            Default is (1e-11, 1e5), after Bear, 1972 (see
            https://en.wikipedia.org/wiki/Hydraulic_conductivity)
            and Schwartz and Zhang (2003, Table 4.4).
        vkcb : tuple
            Reasonable minimum/maximum hydraulic conductivity value for
            quasi-3D confining bed; Default is (1e-11, 1e5), after Bear, 1972
            (see https://en.wikipedia.org/wiki/Hydraulic_conductivity)
            and Schwartz and Zhang (2003, Table 4.4).
        sy : tuple
            Reasonable minimum/maximum specific yield values;
            Default is (0.01,0.5) after Anderson, Woessner and Hunt
            (2015, Table 5.2).
        sy : tuple
            Reasonable minimum/maximum specific storage values;
            Default is (3.3e-6, 2e-2) after Anderson, Woessner and Hunt
            (2015, Table 5.2).

    Notes
    -----
    Anderson, M.P, Woessner, W.W. and Hunt, R.J., 2015. Applied Groundwater
        Modeling: Simulation of Flow and Advective Transport, Elsevier, 564p.
    Bear, J., 1972. Dynamics of Fluids in Porous Media. Dover Publications.
    Schwartz, F.W. and Zhang, H., 2003. Fundamentals of Groundwater, Wiley,
        583 p.

    """

    bc_stage_names = {"GHB": "bhead", "DRN": "elev"}  # all names in lower case

    # only check packages when level is >= to these values
    # default is 0 (always check package)
    package_check_levels = {"sfr": 1}

    property_threshold_values = {
        "hk": (1e-11, 1e5),
        "k": (1e-11, 1e5),
        "k22": (1e-11, 1e5),
        # after Schwartz and Zhang, table 4.4
        "hani": None,
        "vka": (1e-11, 1e5),
        "k33": (1e-11, 1e5),
        "vkcb": (1e-11, 1e5),
        "ss": (1e-6, 1e-2),
        "sy": (0.01, 0.5),
    }

    # which versions is pks compatible with?
    solver_packages = {
        "mf2k": ["DE4", "SIP", "SOR", "GMG", "PCG", "PCGN"],
        "mf2005": ["DE4", "SIP", "GMG", "PCG", "PCGN"],
        "mfnwt": ["DE4", "SIP", "PCG", "NWT"],
        "mfusg": ["SMS"],
    }

    # cells thickness less than this value will be flagged
    thin_cell_threshold = 1.0

    def __init__(
        self,
        package,
        f=None,
        verbose=True,
        level=1,
        property_threshold_values={},
    ):

        # allow for instantiation with model or package
        # if isinstance(package, BaseModel): didn't work
        if hasattr(package, "parent"):
            self.model = package.parent
            self.prefix = "{} PACKAGE DATA VALIDATION".format(package.name[0])
        else:
            self.model = package
            self.prefix = "{} MODEL DATA VALIDATION SUMMARY".format(
                self.model.name
            )
        self.package = package
        if "structured" in self.model.__dict__:
            self.structured = self.model.structured
        else:
            self.structured = self.model.modelgrid.grid_type == "structured"
        self.verbose = verbose
        self.level = level
        self.passed = []
        self.property_threshold_values.update(property_threshold_values)

        self.summary_array = self._get_summary_array()

        self.f = None
        if f is not None:
            if isinstance(f, str):
                if os.path.split(f)[0] == "":
                    self.summaryfile = os.path.join(self.model.model_ws, f)
                else:  # if a path is supplied with summary file, save there
                    self.summaryfile = f
                self.f = open(self.summaryfile, "w")
            else:
                self.f = f
        self.txt = "\n{}:\n".format(self.prefix)

    def _add_to_summary(
        self,
        type="Warning",
        k=0,
        i=0,
        j=0,
        node=0,
        value=0,
        desc="",
        package=None,
    ):
        if package is None:
            package = self.package.name[0]
        col_list = [type, package]
        col_list += [k, i, j] if self.structured else [node]
        col_list += [value, desc]
        sa = self._get_summary_array(np.array(col_list))
        self.summary_array = np.append(self.summary_array, sa).view(
            np.recarray
        )

    def _boolean_compare(
        self,
        array,
        col1,
        col2,
        level0txt="{} violations encountered.",
        level1txt="Violations:",
        sort_ascending=True,
        print_delimiter=" ",
    ):
        """
        Compare two columns in a record array. For each row,
        tests if value in col1 is greater than col2. If any values
        in col1 are > col2, subsets array to only include rows where
        col1 is greater. Creates another column with differences
        (col1-col2), and prints the array sorted by the differences
        column (diff).

        Parameters
        ----------
        array : record array
            Array with columns to compare.
        col1 : string
            Column name in array.
        col2 : string
            Column name in array.
        sort_ascending : T/F; default True
            If True, printed array will be sorted by differences in
            ascending order.
        print_delimiter : str
            Delimiter for printed array.

        Returns
        -------
        txt : str
            Error messages and printed array (if .level attribute of
            checker is set to 1). Returns an empty string if no
            values in col1 are greater than col2.

        Notes
        -----
        info about appending to record arrays (views vs. copies and upcoming
        changes to numpy):
        http://stackoverflow.com/questions/22865877/how-do-i-write-to-multiple-fields-of-a-structured-array
        """
        txt = ""
        array = array.copy()
        if isinstance(col1, np.ndarray):
            array = recfunctions.append_fields(
                array, names="tmp1", data=col1, asrecarray=True
            )
            col1 = "tmp1"
        if isinstance(col2, np.ndarray):
            array = recfunctions.append_fields(
                array, names="tmp2", data=col2, asrecarray=True
            )
            col2 = "tmp2"
        if isinstance(col1, tuple):
            array = recfunctions.append_fields(
                array, names=col1[0], data=col1[1], asrecarray=True
            )
            col1 = col1[0]
        if isinstance(col2, tuple):
            array = recfunctions.append_fields(
                array, names=col2[0], data=col2[1], asrecarray=True
            )
            col2 = col2[0]

        failed = array[col1] > array[col2]
        if np.any(failed):
            failed_info = array[failed].copy()
            txt += level0txt.format(len(failed_info)) + "\n"
            if self.level == 1:
                diff = failed_info[col2] - failed_info[col1]
                cols = [
                    c
                    for c in failed_info.dtype.names
                    if failed_info[c].sum() != 0
                    and c != "diff"
                    and "tmp" not in c
                ]
                # currently failed_info[cols] results in a warning. Not sure
                # how to do this properly with a recarray.
                failed_info = recfunctions.append_fields(
                    failed_info[cols].copy(),
                    names="diff",
                    data=diff,
                    asrecarray=True,
                )
                failed_info.sort(order="diff", axis=0)
                if not sort_ascending:
                    failed_info = failed_info[::-1]
                txt += level1txt + "\n"
                txt += _print_rec_array(failed_info, delimiter=print_delimiter)
            txt += "\n"
        return txt

    def _get_summary_array(self, array=None):
        dtype = self._get_dtype()
        if array is None:
            return np.recarray((0), dtype=dtype)
        ra = recarray(array, dtype)
        # at = array.transpose()
        # a = np.core.records.fromarrays(at, dtype=dtype)
        return ra

    def _txt_footer(
        self, headertxt, txt, testname, passed=False, warning=True
    ):
        """
        if len(txt) == 0 or passed:
            txt += 'passed.'
            self.passed.append(testname)
        elif warning:
            self.warnings.append(testname)
        else:
            self.errors.append(testname)
        if self.verbose:
            print(txt + '\n')
        self.txt += headertxt + txt + '\n'
        """

    def _stress_period_data_valid_indices(self, stress_period_data):
        """Check that stress period data inds are valid for model grid."""
        spd_inds_valid = self._has_cell_indices(stress_period_data)

        # check for BCs indices that are invalid for grid
        inds = self._get_cell_inds(stress_period_data)
        isvalid = self.isvalid(inds)
        if not np.all(isvalid):
            sa = self._list_spd_check_violations(
                stress_period_data,
                ~isvalid,
                error_name="invalid BC index",
                error_type="Error",
            )
            self.summary_array = np.append(self.summary_array, sa).view(
                np.recarray
            )
            spd_inds_valid = False
            self.remove_passed("BC indices valid")
        if spd_inds_valid:
            self.append_passed("BC indices valid")
        return spd_inds_valid

    def _stress_period_data_nans(self, stress_period_data, nan_excl_list):
        """Check for and list any nans in stress period data."""
        isnan = np.array(
            [
                np.isnan(stress_period_data[c])
                for c in stress_period_data.dtype.names
                if not (stress_period_data.dtype[c].name == "object")
                and c not in nan_excl_list
            ]
        ).transpose()
        if np.any(isnan):
            row_has_nan = np.any(isnan, axis=1)
            sa = self._list_spd_check_violations(
                stress_period_data,
                row_has_nan,
                error_name="Not a number",
                error_type="Error",
            )
            self.summary_array = np.append(self.summary_array, sa).view(
                np.recarray
            )
            self.remove_passed("not a number (Nan) entries")
        else:
            self.append_passed("not a number (Nan) entries")

    def _stress_period_data_inactivecells(self, stress_period_data):
        """Check for and list any stress period data in cells with ibound=0."""
        spd = stress_period_data
        inds = self._get_cell_inds(spd)
        msg = "BC in inactive cell"

        idomain = self.model.modelgrid.idomain
        if idomain is not None:
            ibnd = idomain[inds]

            if np.any(ibnd == 0):
                sa = self._list_spd_check_violations(
                    stress_period_data,
                    ibnd == 0,
                    error_name=msg,
                    error_type="Warning",
                )
                self.summary_array = np.append(self.summary_array, sa).view(
                    np.recarray
                )
                self.remove_passed(msg + "s")
            else:
                self.append_passed(msg + "s")

    def _list_spd_check_violations(
        self,
        stress_period_data,
        criteria,
        col=None,
        error_name="",
        error_type="Warning",
    ):
        """
        If criteria contains any true values, return the error_type, package
        name, k,i,j indices, values, and description of error for each row in
        stress_period_data where criteria=True.
        """
        inds_col = self._get_cell_inds_names()
        # inds = stress_period_data[criteria][inds_col]\
        #    .reshape(stress_period_data[criteria].shape + (-1,))
        # inds = np.atleast_2d(np.squeeze(inds.tolist()))
        inds = stress_period_data[criteria]
        a = self._get_cellid_cols(inds, inds_col)
        inds = a.view(int)
        inds = inds.reshape(stress_period_data[criteria].shape + (-1,))

        if col is not None:
            v = stress_period_data[criteria][col]
        else:
            v = np.zeros(len(stress_period_data[criteria]))
        pn = [self.package.name] * len(v)
        en = [error_name] * len(v)
        tp = [error_type] * len(v)
        return self._get_summary_array(np.column_stack([tp, pn, inds, v, en]))

    @staticmethod
    def _get_cellid_cols(inds, inds_col):
        a = inds[inds_col[0]]
        if len(inds_col) > 1:
            for n in inds_col[1:]:
                a = np.concatenate((a, inds[n]))
        return a

    def append_passed(self, message):
        """Add a check to the passed list if it isn't already in there."""
        self.passed.append(message) if message not in self.passed else None

    def remove_passed(self, message):
        """Remove a check to the passed list if it failed in any stress period."""
        self.passed.remove(message) if message in self.passed else None

    def isvalid(self, inds):
        """Check that indices are valid for model grid

        Parameters
        ----------
        inds : tuple or lists or arrays; or a 1-D array
            (k, i, j) for structured grids; (node) for unstructured.

        Returns
        -------
        isvalid : 1-D boolean array
            True for each index in inds that is valid for the model grid.
        """
        if isinstance(inds, np.ndarray):
            inds = [inds]

        mg = self.model.modelgrid
        if mg.grid_type == "structured" and len(inds) == 3:
            k = inds[0] < mg.nlay
            i = inds[1] < mg.nrow
            j = inds[2] < mg.ncol
            return k & i & j
        elif mg.grid_type == "vertex" and len(inds) == 2:
            lay = inds[0] < mg.nlay
            cpl = inds[1] < mg.ncpl
            return lay & cpl
        elif mg.grid_type == "unstructured" and len(inds) == 1:
            return inds[0] < mg.nnodes
        else:
            return np.zeros(inds[0].shape, dtype=bool)

    def get_active(self, include_cbd=False):
        """Returns a boolean array of active cells for the model.

        Parameters
        ----------
        include_cbd : boolean
            If True, active is of same dimension as the thickness array
            in the DIS module (includes quasi 3-D confining beds).
            Default False.

        Returns
        -------
        active : 3-D boolean array
            True where active.
        """
        mg = self.model.modelgrid
        if mg.grid_type == "structured":
            nlaycbd = mg._StructuredGrid__laycbd.sum() if include_cbd else 0
            inds = (mg.nlay + nlaycbd, mg.nrow, mg.ncol)
        elif mg.grid_type == "vertex":
            inds = (mg.nlay, mg.ncpl)
        else:
            inds = mg.nnodes
            include_cbd = False

        if "BAS6" in self.model.get_package_list():
            if "DIS" in self.model.get_package_list():
                dis = self.model.dis
            else:
                dis = self.model.disu

            # make ibound of same shape as thicknesses/botm for quasi-3D models
            active = self.model.bas6.ibound.array != 0
            if include_cbd and dis.laycbd.sum() > 0:
                laycbd = np.flatnonzero(
                    dis.laycbd.array > 0
                )  # cbd layer index
                active = np.insert(active, laycbd, active[laycbd], axis=0)

        else:  # if bas package is missing
            active = np.ones(inds, dtype=bool)
        return active

    def print_summary(self, cols=None, delimiter=",", float_format="{:.6f}"):
        # strip description column
        sa = self.summary_array.copy()
        desc = self.summary_array.desc
        sa["desc"] = [s.strip() for s in desc]
        return _print_rec_array(
            sa, cols=cols, delimiter=delimiter, float_format=float_format
        )

    def stress_period_data_values(
        self,
        stress_period_data,
        criteria,
        col=None,
        error_name="",
        error_type="Warning",
    ):
        """
        If criteria contains any true values, return the error_type, package
        name, k,i,j indices, values, and description of error for each row in
        stress_period_data where criteria=True.
        """
        # check for valid cell indices
        # self._stress_period_data_valid_indices(stress_period_data)

        # first check for and list nan values
        # self._stress_period_data_nans(stress_period_data)

        # next check for BCs in inactive cells
        # self._stress_period_data_inactivecells(stress_period_data)

        if np.any(criteria):
            # list the values that met the criteria
            sa = self._list_spd_check_violations(
                stress_period_data,
                criteria,
                col,
                error_name=error_name,
                error_type=error_type,
            )
            self.summary_array = np.append(self.summary_array, sa).view(
                np.recarray
            )
            self.remove_passed(error_name)
        else:
            self.append_passed(error_name)

    def values(self, a, criteria, error_name="", error_type="Warning"):
        """
        If criteria contains any true values, return the error_type, package
        name, indices, array values, and description of error for each
        True value in criteria.
        """
        if np.any(criteria):
            inds = np.where(criteria)
            v = a[inds]  # works with structured or unstructured
            pn = [self.package.name] * len(v)
            en = [error_name] * len(v)
            tp = [error_type] * len(v)
            indsT = np.transpose(inds)
            # _get_summary_array requires 3 columns for k, i, j,
            # but indsT will only have two columns if a 2-D array is being compared
            # pad indsT with a column of zeros for k
            if indsT.shape[1] == 2:
                indsT = np.column_stack(
                    [np.zeros(indsT.shape[0], dtype=int), indsT]
                )
            sa = np.column_stack([tp, pn, indsT, v, en])
            sa = self._get_summary_array(sa)
            self.summary_array = np.append(self.summary_array, sa).view(
                np.recarray
            )
            self.remove_passed(error_name)
        else:
            self.append_passed(error_name)

    def view_summary_array_fields(self, fields):
        arr = self.summary_array
        dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
        return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)

    def summarize(self):

        # write the summary array to text file (all levels)
        if self.f is not None:
            self.f.write(self.print_summary())
            self.f.close()

        # print the screen output depending on level
        txt = ""
        # tweak screen output for model-level to report package for each error
        if "MODEL" in self.prefix:  # add package name for model summary output
            packages = self.summary_array.package
            desc = self.summary_array.desc
            self.summary_array["desc"] = [
                "\r    {} package: {}".format(packages[i], d.strip())
                if packages[i] != "model"
                else d
                for i, d in enumerate(desc)
            ]

        for etype in ["Error", "Warning"]:
            a = self.summary_array[self.summary_array.type == etype]
            desc = a.desc
            t = ""
            if len(a) > 0:
                t += "  {} {}s:\n".format(len(a), etype)
                if len(a) == 1:
                    t = t.replace("s", "")  # grammar
                for e in np.unique(desc):
                    n = np.sum(desc == e)
                    if n > 1:
                        t += "    {} instances of {}\n".format(n, e)
                    else:
                        t += "    {} instance of {}\n".format(n, e)
                txt += t
        if txt == "":
            txt += "  No errors or warnings encountered.\n"

        elif (
            self.f is not None
            and self.verbose
            and self.summary_array.shape[0] > 0
        ):
            txt += "  see {} for details.\n".format(self.summaryfile)

        # print checks that passed for higher levels
        if len(self.passed) > 0 and self.level > 0:
            txt += "\n  Checks that passed:\n"
            for chkname in self.passed:
                txt += "    {}\n".format(chkname)
        self.txt += txt

        # for level 2, print the whole summary table at the bottom
        if self.level > 1:
            # kludge to improve screen printing
            self.summary_array["package"] = [
                "{} ".format(s) for s in self.summary_array["package"]
            ]
            self.txt += "\nDETAILED SUMMARY:\n{}".format(
                self.print_summary(float_format="{:.2e}", delimiter="\t")
            )

        if self.verbose:
            print(self.txt)
        elif self.summary_array.shape[0] > 0 and self.level > 0:
            print("Errors and/or Warnings encountered.")
            if self.f is not None:
                print("  see {} for details.\n".format(self.summaryfile))

    # start of older model specific code
    def _has_cell_indices(self, stress_period_data):
        if self.model.has_package("DIS") and {"k", "i", "j"}.intersection(
            set(stress_period_data.dtype.names)
        ) != {"k", "i", "j"}:
            self._add_to_summary(
                type="Error",
                desc="\r    Stress period data missing k, "
                "i, j for structured grid.",
            )
            return False
        elif (
            self.model.has_package("DISU")
            and "node" not in stress_period_data.dtype.names
        ):
            self._add_to_summary(
                type="Error",
                desc="\r    Stress period data missing "
                "node number for unstructured grid.",
            )
            return False
        return True

    def _get_cell_inds(self, spd):
        return (spd.k, spd.i, spd.j) if self.structured else (spd.node)

    def _get_cell_inds_names(self):
        return ["k", "i", "j"] if self.structured else ["node"]

    def _get_dtype(self):
        if self.structured:
            # include node column for structured grids (useful for indexing)
            return np.dtype(
                [
                    ("type", object),
                    ("package", object),
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("value", float),
                    ("desc", object),
                ]
            )
        else:
            return np.dtype(
                [
                    ("type", object),
                    ("package", object),
                    ("node", int),
                    ("value", float),
                    ("desc", object),
                ]
            )


def _fmt_string_list(array, float_format="{}"):
    fmt_string = []
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if vtype == "i":
            fmt_string += ["{:.0f}"]
        elif vtype == "f":
            fmt_string += [float_format]
        elif vtype == "o":
            fmt_string += ["{}"]
        elif vtype == "s":
            raise Exception(
                "MfList error: 'str' type found in dtype."
                + " This gives unpredictable results when "
                + "recarray to file - change to 'object' type"
            )
        else:
            raise Exception(
                "MfList.fmt_string error: unknown vtype " + "in dtype:" + vtype
            )
    return fmt_string


def _print_rec_array(array, cols=None, delimiter=" ", float_format="{:.6f}"):
    """
    Print out a numpy record array to string, with column names.

    Parameters
    ----------
    cols : list of strings
        List of columns to print.
    delimiter : string
        Delimited to use.

    Returns
    -------
    txt : string
        Text string of array.
    """
    txt = ""
    dtypes = list(array.dtype.names)
    if cols is not None:
        cols = [c for c in dtypes if c in cols]
    else:
        cols = dtypes
    # drop columns with no data
    if np.shape(array)[0] > 1:
        cols = [
            c
            for c in cols
            if array["type"].dtype.kind == "O" or array[c].min() > -999999
        ]
    # edit dtypes
    array_cols = fields_view(array, cols)
    fmts = _fmt_string_list(array_cols, float_format=float_format)
    txt += delimiter.join(cols) + "\n"
    array_cols = array_cols.copy().tolist()
    txt += "\n".join([delimiter.join(fmts).format(*r) for r in array_cols])
    return txt


def fields_view(arr, fields):
    """
    creates view of array that only contains the fields in fields.
    http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-
    several-columns-in-numpy-structured-array
    """
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def get_neighbors(a):
    """
    Returns the 6 neighboring values for each value in a.

    Parameters
    ----------
    a : 3-D array
        Model array in layer, row, column order.

    Returns
    -------
    neighbors : 4-D array
        Array of neighbors, where axis 0 contains the 6 neighboring
        values for each value in a, and subsequent axes are in layer, row,
        column order.
        Nan is returned for values at edges.
    """
    nk, ni, nj = a.shape
    tmp = np.empty((nk + 2, ni + 2, nj + 2), dtype=float)
    tmp[:, :, :] = np.nan
    tmp[1:-1, 1:-1, 1:-1] = a[:, :, :]
    neighbors = np.vstack(
        [
            tmp[0:-2, 1:-1, 1:-1].ravel(),  # k-1
            tmp[2:, 1:-1, 1:-1].ravel(),  # k+1
            tmp[1:-1, 0:-2, 1:-1].ravel(),  # i-1
            tmp[1:-1, 2:, 1:-1].ravel(),  # i+1
            tmp[1:-1, 1:-1, :-2].ravel(),  # j-1
            tmp[1:-1, 1:-1, 2:].ravel(),
        ]
    )  # j+1
    return neighbors.reshape(6, nk, ni, nj)


class mf6check(check):
    def __init__(
        self,
        package,
        f=None,
        verbose=True,
        level=1,
        property_threshold_values={},
    ):
        super(mf6check, self).__init__(
            package, f, verbose, level, property_threshold_values
        )
        if hasattr(package, "model_or_sim"):
            self.model = package.model_or_sim

    @staticmethod
    def _get_cellid_cols(inds, inds_col):
        a = inds[inds_col[0]]
        return np.asarray(a.tolist())

    def _get_cell_inds(self, spd):
        hnames = ()
        if "cellid" in spd.dtype.names:
            cellid = spd.cellid
        elif "cellid1" in spd.dtype.names:
            cellid = spd.cellid1
        else:
            return None

        for item in zip(*cellid):
            hnames += (
                np.ndarray(
                    shape=(len(item),), buffer=np.array(item), dtype=np.int32
                ),
            )
        return hnames

    def _get_dtype(self):
        mg = self.model.modelgrid
        if mg.grid_type == "structured":
            return np.dtype(
                [
                    ("type", object),
                    ("package", object),
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("value", float),
                    ("desc", object),
                ]
            )
        elif mg.grid_type == "vertex":
            return np.dtype(
                [
                    ("type", object),
                    ("package", object),
                    ("lay", int),
                    ("cell", int),
                    ("value", float),
                    ("desc", object),
                ]
            )
        else:
            return np.dtype(
                [
                    ("type", object),
                    ("package", object),
                    ("node", int),
                    ("value", float),
                    ("desc", object),
                ]
            )

    def _has_cell_indices(self, stress_period_data):
        mg = self.model.modelgrid
        if (
            mg.grid_type == "structured"
            or mg.grid_type == "vertex"
            or mg.grid_type == "unstructured"
        ):
            if "cellid" not in set(
                stress_period_data.dtype.names
            ) and "cellid1" not in set(stress_period_data.dtype.names):
                self._add_to_summary(
                    type="Error",
                    desc="\r    Stress period data missing " "cellid.",
                )
                return False
        return True

    def _get_cell_inds_names(self):
        return ["cellid"]

    def get_active(self, include_cbd=False):
        """Returns a boolean array of active cells for the model.

        Parameters
        ----------
        include_cbd : boolean
            Does not apply to MF6 models, always false.

        Returns
        -------
        active : 3-D boolean array
            True where active.
        """
        mg = self.model.modelgrid
        idomain = mg.idomain
        if idomain is None:
            return np.ones(shape=mg.shape, dtype=bool)
        else:
            return idomain > 0
