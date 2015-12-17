import os
import numpy as np
from numpy.lib import recfunctions

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
            Reasonable minimum/maximum hydraulic conductivity value; values below this will be flagged.
            Default is (10e-10, 10e5), after Bear, 1972 (see https://en.wikipedia.org/wiki/Hydraulic_conductivity)
        vka : tuple
            Reasonable minimum/maximum hydraulic conductivity value;
            Default is (10e-10, 10e5), after Bear, 1972 (see https://en.wikipedia.org/wiki/Hydraulic_conductivity)
        vkcb : tuple
            Reasonable minimum/maximum hydraulic conductivity value for quasi-3D confining bed;
            Default is (10e-10, 10e5), after Bear, 1972 (see https://en.wikipedia.org/wiki/Hydraulic_conductivity)
        sy : tuple
            Reasonable minimum/maximum specific yield values;
            Default is (0.01,0.5) after Anderson, Woessner and Hunt (2015, Table 5.2).
        sy : tuple
            Reasonable minimum/maximum specific storage values;
            Default is (3.3e-6, 2e-2) after Anderson, Woessner and Hunt (2015, Table 5.2).

    Notes
    -----

    """

    bc_elev_names = {'GHB': 'bhead', # all names in lower case
                     'RIV': 'stage',
                     'DRN': 'elev'}

    property_threshold_values = {'hk': (1e-11, 1e5), # after Schwartz and Zhang, table 4.4
                                 'hani': None,
                                 'vka': (1e-11, 1e5),
                                 'vkcb': (1e-11, 1e5),
                                 'ss': (1e-6, 1e-2),
                                 'sy': (0.01, 0.5)}

    # which versions is pks compatible with?
    solver_packages = {'mf2k': ['DE4', 'SIP', 'SOR', 'GMG', 'PCG', 'PCGN'],
                       'mf2005': ['DE4', 'SIP', 'GMG', 'PCG', 'PCGN'],
                       'mfnwt': ['DE4', 'SIP', 'PCG', 'NWT'],
                       'mfusg': ['SMS']}

    thin_cell_threshold = 1.0 # cells thickness less than this value will be flagged

    def __init__(self, package, f=None, verbose=True, level=1,
                 property_threshold_values={}):

        # allow for instantiation with model or package
        #if isinstance(package, BaseModel): didn't work
        if package.parent is not None:
            self.model = package.parent
            self.prefix = '{} PACKAGE DATA VALIDATION'.format(package.name[0])
        else:
            self.model = package
            self.prefix = '{} MODEL DATA VALIDATION SUMMARY'.format(self.model.name)
        self.package = package
        self.structured = self.model.structured
        self.verbose = verbose
        self.level = level
        self.passed = []
        self.property_threshold_values.update(property_threshold_values)

        self.summary_array = self._get_summary_array()

        self.f = None
        if f is not None:
            if isinstance(f, str):
               self.summaryfile = os.path.join(self.model.model_ws, f)
               self.f = open(self.summaryfile, 'w')
            else:
                self.f = f
        self.txt = '\n{}:\n'.format(self.prefix)

    def _add_to_summary(self, type='Warning', k=0, i=0, j=0, node=0,
                        value=0, desc='', package=None):
        if package is None:
            package = self.package.name[0]
        col_list = [type, package]
        col_list += [k, i, j] if self.structured else [node]
        col_list += [value, desc]
        sa = self._get_summary_array(np.array(col_list))
        self.summary_array = np.append(self.summary_array, sa).view(np.recarray)

    def _boolean_compare(self, array, col1, col2,
                         level0txt='{} violations encountered.',
                         level1txt='Violations:',
                         sort_ascending=True, print_delimiter=' '):
        """Compare two columns in a record array. For each row,
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
        info about appending to record arrays (views vs. copies and upcoming changes to numpy):
        http://stackoverflow.com/questions/22865877/how-do-i-write-to-multiple-fields-of-a-structured-array
        """
        txt = ''
        array = array.copy()
        if isinstance(col1, np.ndarray):
            array = recfunctions.append_fields(array, names='tmp1', data=col1,
                                               asrecarray=True)
            col1 = 'tmp1'
        if isinstance(col2, np.ndarray):
            array = recfunctions.append_fields(array, names='tmp2', data=col2,
                                               asrecarray=True)
            col2 = 'tmp2'
        if isinstance(col1, tuple):
            array = recfunctions.append_fields(array, names=col1[0], data=col1[1],
                                               asrecarray=True)
            col1 = col1[0]
        if isinstance(col2, tuple):
            array = recfunctions.append_fields(array, names=col2[0], data=col2[1],
                                               asrecarray=True)
            col2 = col2[0]

        failed = array[col1] > array[col2]
        if np.any(failed):
            failed_info = array[failed].copy()
            txt += level0txt.format(len(failed_info)) + '\n'
            if self.level == 1:
                diff = failed_info[col2] - failed_info[col1]
                cols = [c for c in failed_info.dtype.names if failed_info[c].sum() != 0
                        and c != 'diff'
                        and 'tmp' not in c]
                # currently failed_info[cols] results in a warning. Not sure
                # how to do this properly with a recarray.
                failed_info = recfunctions.append_fields(failed_info[cols].copy(),
                                                         names='diff',
                                                         data=diff,
                                                         asrecarray=True)
                failed_info.sort(order='diff', axis=0)
                if not sort_ascending:
                    failed_info = failed_info[::-1]
                txt += level1txt + '\n'
                txt += _print_rec_array(failed_info, delimiter=print_delimiter)
            txt += '\n'
        return txt

    def _get_summary_array(self, array=None):

        if self.structured:
            # include node column for structured grids (useful for indexing)
            dtype = np.dtype([('type', np.object),
                              ('package', np.object),
                              ('k', np.int),
                              ('i', np.int),
                              ('j', np.int),
                              ('value', np.float),
                              ('desc', np.object)
                              ])
        else:
            dtype = np.dtype([('type', np.object),
                              ('package', np.object),
                              ('node', np.int),
                              ('value', np.float),
                              ('desc', np.object)
                              ])
        if array is None:
            array = np.empty((0, len(dtype)), dtype=dtype)
        return np.core.records.fromarrays(array.transpose(), dtype=dtype)

    def _txt_footer(self, headertxt, txt, testname, passed=False, warning=True):
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

    def _stress_period_data_nans(self, stress_period_data):
        """Check for and list any nans in stress period data."""
        isnan = np.array([np.isnan(stress_period_data[c])
                          for c in stress_period_data.dtype.names]).transpose()
        if np.any(isnan):
            row_has_nan = np.any(isnan, axis=1)
            col = self.bc_elev_names[self.package.name[0]]
            sa = self._list_spd_check_violations(stress_period_data,
                                                 row_has_nan, col,
                                                 error_name='Not a number',
                                                 error_type='Error')
            self.summary_array = np.append(self.summary_array, sa).view(np.recarray)
        else:
            self.passed.append('not a number (Nan) entries')

    def _stress_period_data_inactivecells(self, stress_period_data):
        """Check for and list any stress period data in cells with ibound=0."""
        ibnd = self.package.parent.bas6.ibound.array[stress_period_data.k,
                                                     stress_period_data.i,
                                                     stress_period_data.j]
        if np.any(ibnd == 0):
            col = self.bc_elev_names[self.package.name[0]]
            sa = self._list_spd_check_violations(stress_period_data,
                                                 ibnd == 0, col,
                                                 error_name='BC in inactive cell',
                                                 error_type='Warning')
            self.summary_array = np.append(self.summary_array, sa).view(np.recarray)
        else:
            self.passed.append('BCs in inactive cells')

    def _list_spd_check_violations(self, stress_period_data, criteria, col,
                                  error_name='', error_type='Warning'):
        """If criteria contains any true values, return the error_type, package name, k,i,j indicies,
        values, and description of error for each row in stress_period_data where criteria=True.
        """
        inds = stress_period_data[criteria][['k', 'i', 'j']].view(int)\
            .reshape(stress_period_data[criteria].shape + (-1,))
        v = stress_period_data[criteria][col]
        pn = [self.package.name] * len(v)
        en = [error_name] * len(v)
        tp = [error_type] * len(v)
        return self._get_summary_array(np.column_stack([tp, pn, inds, v, en]))

    def print_summary(self, cols=None, delimiter=',', float_format='{:.6f}'):
        # strip description column
        self.summary_array['desc'] = [s.strip() for s in self.summary_array.desc]
        return _print_rec_array(self.summary_array, cols=cols, delimiter=delimiter,
                                float_format=float_format)

    def stress_period_data(self, stress_period_data, criteria, col,
                                  error_name='', error_type='Warning'):
        """If criteria contains any true values, return the error_type, package name, k,i,j indicies,
        values, and description of error for each row in stress_period_data where criteria=True.
        """
        # first check for and list nan values
        self._stress_period_data_nans(stress_period_data)

        # next check for BCs in inactive cells
        self._stress_period_data_inactivecells(stress_period_data)

        if np.any(criteria):
            # list the values that met the criteria
            sa = self._list_spd_check_violations(stress_period_data, criteria, col,
                                                 error_name=error_name, error_type=error_type)
            self.summary_array = np.append(self.summary_array, sa).view(np.recarray)
        else:
            self.passed.append(error_name)

    def values(self, a, criteria, error_name='', error_type='Warning'):
        """If criteria contains any true values, return the error_type, package name, indices,
        array values, and description of error for each True value in criteria."""
        if np.any(criteria):
            inds = np.where(criteria)
            v = a[inds] # works with structured or unstructured
            pn = [self.package.name] * len(v)
            en = [error_name] * len(v)
            tp = [error_type] * len(v)
            sa = self._get_summary_array(np.column_stack([tp, pn, np.transpose(inds), v, en]))
            self.summary_array = np.append(self.summary_array, sa).view(np.recarray)
        else:
            self.passed.append(error_name)

    def summarize(self):

        # write the summary array to text file (all levels)
        if self.f is not None:
            self.f.write(self.print_summary())
            self.f.close()

        # print the screen output depending on level
        txt = ''
        if 'MODEL' in self.prefix: # add package name for model summary output
            self.summary_array['desc'] = \
                ['\r    {} package: {}'.format(self.summary_array.package[i], d.strip())
                 if self.summary_array.package[i] != 'model' else d
                 for i, d in enumerate(self.summary_array.desc)]

        for etype in ['Error', 'Warning']:
            a = self.summary_array[self.summary_array.type == etype]
            t = ''
            if len(a) > 0:
                t += '  {} {}s:\n'.format(len(a), etype)
                if len(a) == 1:
                    t = t.replace('s', '') #grammer
                for e in np.unique(a.desc):
                    n = len(a[a.desc == e])
                    if n > 1:
                        t += '    {} instances of {}\n'.format(n, e)
                    else:
                        t += '    {} instance of {}\n'.format(n, e)
                txt += t
        if txt == '':
            txt += '  No errors or warnings encountered.\n'
        elif self.f is not None and self.verbose and self.summary_array.shape[0] > 0:
            txt += '  see {} for details.\n'.format(self.summaryfile)

        if len(self.passed) > 0 and self.level > 0:
            txt += '\n  Checks that passed:\n'
            for chkname in self.passed:
                txt += '    {}\n'.format(chkname)
        self.txt += txt

        if self.level > 1:
            # kludge to improve screen printing
            self.summary_array['package'] = ['{} '.format(s) for s in self.summary_array['package']]
            self.txt += '\nDETAILED SUMMARY:\n{}'.format(self.print_summary(float_format='{:.2e}', delimiter='\t'))

        if self.verbose:
            print(self.txt)



def _fmt_string_list(array, float_format='{}'):
    fmt_string = []
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if (vtype == 'i'):
            fmt_string += ['{:.0f}']
        elif (vtype == 'f'):
            fmt_string += [float_format]
        elif (vtype == 'o'):
            fmt_string += ['{}']
        elif (vtype == 's'):
            raise Exception("MfList error: '\str\' type found it dtype." + \
                            " This gives unpredictable results when " + \
                            "recarray to file - change to \'object\' type")
        else:
            raise Exception("MfList.fmt_string error: unknown vtype " + \
                            "in dtype:" + vtype)
    return fmt_string

def _print_rec_array(array, cols=None, delimiter=' ', float_format='{:.6f}'):
    """Print out a numpy record array to string, with column names.

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
    txt = ''
    if cols is not None:
        cols = [c for c in array.dtype.names if c in cols]
    else:
        cols = list(array.dtype.names)
    # drop columns with no data
    if np.shape(array)[0] > 1:
        cols = [c for c in cols if array['type'].dtype.kind == 'O' or array[c].min() > -999999]
    # add _fmt_string call here
    fmts = _fmt_string_list(array[cols], float_format=float_format)
    txt += delimiter.join(cols) + '\n'
    txt += '\n'.join([delimiter.join(fmts).format(*r) for r in array[cols].copy().tolist()])
    return txt

def get_neighbors(a):
    """Returns the 6 neighboring values for each value in a.

    Parameters
    ----------
    a : 3-D array
        Model array in layer, row, column order.

    Returns
    -------
    neighbors : 4-D array
        Array of neighbors, where axis 0 contains the 6 neighboring
        values for each value in a, and subsequent axes are in layer, row, column order.
        Nan is returned for values at edges.
    """
    nk, ni, nj = a.shape
    tmp = np.empty((nk+2, ni+2, nj+2), dtype=float)
    tmp[:, :, :] = np.nan
    tmp[1:-1, 1:-1, 1:-1] = a[:, :, :]
    neighbors = np.vstack([tmp[0:-2, 1:-1, 1:-1].ravel(),
                           tmp[2:, 1:-1, 1:-1].ravel(),
                           tmp[1:-1, 0:-2, 1:-1].ravel(),
                           tmp[1:-1, 2:, 1:-1].ravel(),
                           tmp[1:-1, 1:-1, :-2].ravel(),
                           tmp[1:-1, 1:-1, 2:].ravel()])
    return neighbors.reshape(6, nk, ni, nj)