"""
mfwel module.  Contains the ModflowWel class. Note that the user can access
the ModflowWel class as `flopy.modflow.ModflowWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?wel.htm>`_.

"""

import sys
import numpy as np
from ..utils import MfList
from ..pakbase import Package
from ..utils.recarray_utils import create_empty_recarray


class ModflowWel(Package):
    """
    MODFLOW Well Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    stress_period_data : list of boundaries, or recarray of boundaries, or
        dictionary of boundaries
        Each well is defined through definition of
        layer (int), row (int), column (int), flux (float).
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of:

            stress_period_data =
            {0: [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ],
            1:  [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ], ...
            kper:
                [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ]
            }

        Note that if the number of lists is smaller than the number of stress
        periods, then the last list of wells will apply until the end of the
        simulation. Full details of all options to specify stress_period_data
        can be found in the flopy3 boundaries Notebook in the basic
        subdirectory of the examples directory
    dtype : custom datatype of stress_period_data.
        If None the default well datatype will be applied (default is None).
    extension : string
        Filename extension (default is 'wel')
    options : list of strings
        Package options (default is None).
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
    mxactw : int
        Maximum number of wells for a stress period.  This is calculated
        automatically by FloPy based on the information in
        stress_period_data.

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
    >>> lrcq = {0:[[2, 3, 4, -100.]], 1:[[2, 3, 4, -100.]]}
    >>> wel = flopy.modflow.ModflowWel(m, stress_period_data=lrcq)

    """

    def __init__(self, model, ipakcb=None, stress_period_data=None, dtype=None,
                 extension='wel', options=None, binary=False,
                 unitnumber=None, filenames=None):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowWel.defaultunit()

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
            model.add_output_file(ipakcb, fname=fname,
                                  package=ModflowWel.ftype())
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowWel.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
        self.url = 'wel.htm'

        self.ipakcb = ipakcb
        self.np = 0
        if options is None:
            options = []
        self.specify = False
        for idx, opt in enumerate(options):
            if 'specify' in opt:
                t = opt.strip().split()
                self.specify = True
                self.phiramp = np.float(t[1])
                self.phiramp_unit = np.int(t[2])
                options.pop(idx)
                break
        #self.options = options
        #self.parent.add_package(self)

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured)

        # determine if any aux variables in dtype
        dt = self.get_default_dtype(structured=self.parent.structured)
        if len(self.dtype.names) > len(dt.names):
            for name in self.dtype.names[len(dt.names):]:
                ladd = True
                for option in options:
                    if name.lower() in option.lower():
                        ladd = False
                        break
                if ladd:
                    options.append('aux {} '.format(name))

        # add options
        self.options = options

        # initialize MfList
        self.stress_period_data = MfList(self, stress_period_data,
                                         binary=binary)

        self.parent.add_package(self)

    def ncells(self):
        # Returns the  maximum number of cells that have a well
        # (developed for MT3DMS SSM package)
        return self.stress_period_data.mxact

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f_wel = open(self.fn_path, 'w')
        f_wel.write('%s\n' % self.heading)
        line = (
        ' {0:9d} {1:9d}'.format(self.stress_period_data.mxact, self.ipakcb))

        for opt in self.options:
            line += ' ' + str(opt)
        line += '\n'
        f_wel.write(line)

        if self.specify and self.parent.version == 'mfnwt':
            f_wel.write('SPECIFY {0:10.5g} {1:10d}\n'.format(self.phiramp,
                                                             self.phiramp_unit))

        self.stress_period_data.write_transient(f_wel)
        f_wel.close()

    def add_record(self, kper, index, values):
        try:
            self.stress_period_data.add_record(kper, index, values)
        except Exception as e:
            raise Exception("mfwel error adding record to list: " + str(e))

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype([("k", np.int), ("i", np.int),
                              ("j", np.int), ("flux", np.float32)])
        else:
            dtype = np.dtype([("node", np.int), ("flux", np.float32)])
        return dtype

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowWel.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0E+10)

    @staticmethod
    def load(f, model, nper=None, ext_unit_dict=None, check=True):
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
        wel : ModflowWel object
            ModflowWel object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflow.ModflowWel.load('test.wel', m)

        """

        if model.verbose:
            sys.stdout.write('loading wel package file...\n')

        return Package.load(model, ModflowWel, f, nper, check=check,
                            ext_unit_dict=ext_unit_dict)

    @staticmethod
    def ftype():
        return 'WEL'

    @staticmethod
    def defaultunit():
        return 20
