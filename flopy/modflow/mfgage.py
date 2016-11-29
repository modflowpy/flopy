"""
mfgage module.  Contains the ModflowGage class. Note that the user can access
the ModflowGage class as `flopy.modflow.ModflowGage`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/gage.htm>`_.

"""
import os
import sys

import numpy as np

from ..pakbase import Package
from ..utils import read_fixed_var, write_fixed_var


class ModflowGage(Package):
    """
    MODFLOW Gage Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    options : list of strings
        Package options. (default is None).
    extension : string
        Filename extension (default is 'str')
    unitnumber : int
        File unit number (default is 118).

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
    >>> gages = [[-1,  -26, 1], [-2,  -27, 1]]
    >>> files = ['gage1.go', 'gage2.go']
    >>> gage = flopy.modflow.ModflowGage(m, numgage=2,
    >>>                                  gage_data=gages, files=files)

    """

    def __init__(self, model, numgage=0, gage_data=None, files=None,
                 extension='gage', unitnumber=None, options=None,
                 filenames=None, **kwargs):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowGage.defaultunit()

        # initialize file information for gage ourput
        name = [ModflowGage.ftype()]
        units = [unitnumber]
        extra = ['']
        extension = [extension]

        # process gage output files
        dtype = ModflowGage.get_default_dtype()
        if numgage > 0:
            # check the provided file entries
            if filenames is None:
                if files is None:
                    err = "a list of output gage 'files' must be provided"
                    raise Exception(err)
                if isinstance(files, np.ndarray):
                    files = files.flatten().aslist()
                elif isinstance(files, str):
                    files = [files]
                elif isinstance(files, int) or isinstance(files, float):
                    files = ['{}.go'.format(files)]
                if len(files) < numgage:
                    err = 'a filename needs to be provided ' + \
                          'for {} gages '.format(numgage) + \
                          '- {} filenames were provided'.format(len(files))
                    raise Exception(err)
            else:
                if len(filenames) < numgage + 1:
                    err = "filenames must have a " + \
                          "length of {} ".format(numgage+1) + \
                          "the length provided is {}".format(len(filenames))
                    raise Exception(err)

            # convert gage_data to a recarry, if necessary
            if isinstance(gage_data, np.ndarray):
                if not gage_data.dtype == dtype:
                    gage_data = np.core.records.fromarrays(
                        gage_data.transpose(),
                        dtype=dtype)
            elif isinstance(gage_data, list):
                d = ModflowGage.get_empty(ncells=numgage)
                for n in range(len(gage_data)):
                    t = gage_data[n]
                    gageloc = int(t[0])
                    if gageloc < 0:
                        gagerch = 0
                        iu = int(t[1])
                        outtype = 0
                        if iu < 0:
                            outtype = int(t[2])
                    else:
                        gagerch = int(t[1])
                        iu = int(t[2])
                        outtype = int(t[3])

                    d['gageloc'][n] = gageloc
                    d['gagerch'][n] = gagerch
                    d['unit'][n] = iu
                    d['outtype'][n] = outtype
                gage_data = d
            else:
                err = 'gage_data must be a numpy record array, numpy array' + \
                      'or a list'
                raise Exception(err)

            # fill gage output information for package initiation
            for n in range(numgage):
                iu = gage_data['unit'][n]
                gage_data['unit'][n] = iu
                name.append('DATA')
                units.append(abs(iu))
                extension.append('got')
                extra.append('REPLACE')

        # Call parent init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=filenames)

        # # reset file name with filepths passed from load method
        # if files is not None:
        #     for idx, pth in enumerate(files):
        #         if pth is None:
        #             continue
        #         self.file_name[idx+1] = pth

        vn = model.version_types[model.version]
        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, generated by Flopy.'.format(vn)
        self.url = 'gage.htm'

        if options is None:
            options = []
        self.options = options
        self.numgage = numgage

        self.dtype = self.get_default_dtype()

        self.gage_data = gage_data

        self.parent.add_package(self)

        return

    @staticmethod
    def get_default_dtype(structured=True):
        dtype = np.dtype([("gageloc", np.int), ("gagerch", np.int),
                          ("unit", np.int), ("outtype", np.int)])
        return dtype

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recaray that correponds to dtype
        dtype = ModflowGage.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((ncells, len(dtype)), dtype=dtype)
        d[:, :] = -1.0E+10
        return np.core.records.fromarrays(d.transpose(), dtype=dtype)

    def ncells(self):
        # Return 0 for the gage package
        # (developed for MT3DMS SSM package)
        return 0

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, 'w')

        # # dataset 0
        # vn = self.parent.version_types[self.parent.version]
        # self.heading = '# {} package for '.format(self.name[0]) + \
        #                '{}, generated by Flopy.'.format(vn)
        # f.write('{0}\n'.format(self.heading))

        # dataset 1
        f.write(write_fixed_var([self.numgage], free=True))

        # dataset 2
        for n in range(self.numgage):
            gageloc = self.gage_data['gageloc'][n]
            gagerch = self.gage_data['gagerch'][n]
            iu = self.gage_data['unit'][n]
            outtype = self.gage_data['outtype'][n]
            t = [gageloc]
            if gageloc < 0:
                t.append(iu)
                if iu < 0:
                    t.append(outtype)
            else:
                t.append(gagerch)
                t.append(iu)
                t.append(outtype)
            f.write(write_fixed_var(t, free=True))

        # close the gage file
        f.close()

    @staticmethod
    def load(f, model, nper=None, ext_unit_dict=None):
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
        >>> gage = flopy.modflow.ModflowGage.load('test.gage', m)

        """

        if model.verbose:
            sys.stdout.write('loading gage package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            if sys.version_info[0] == 2:
                f = open(filename, 'r')
            elif sys.version_info[0] == 3:
                f = open(filename, 'r', errors='replace')

        # dataset 0 -- header
        while True:
            line = f.readline().rstrip()
            if line[0] != '#':
                break

        # read dataset 1
        if model.verbose:
            print("   reading gage dataset 1")
        t = read_fixed_var(line, free=True)
        numgage = int(t[0])

        if numgage == 0:
            gage_data = None
            files = None
        else:
            # read dataset 2
            if model.verbose:
                print("   reading gage dataset 2")

            gage_data = ModflowGage.get_empty(ncells=numgage)
            files = []

            for n in range(numgage):
                line = f.readline().rstrip()
                t = read_fixed_var(line, free=True)
                gageloc = int(t[0])
                if gageloc < 0:
                    gagerch = 0
                    iu = int(t[1])
                    outtype = 0
                    if iu < 0:
                        outtype = int(t[2])
                else:
                    gagerch = int(t[1])
                    iu = int(t[2])
                    outtype = int(t[3])
                gage_data['gageloc'][n] = gageloc
                gage_data['gagerch'][n] = gagerch
                gage_data['unit'][n] = iu
                gage_data['outtype'][n] = outtype

                for key, value in ext_unit_dict.items():
                    if key == abs(iu):
                        model.add_pop_key_list(abs(iu))
                        relpth = os.path.relpath(value.filename,
                                                 model.model_ws)
                        files.append(relpth)
                        break

        # determine specified unit number
        unitnumber = None
        filenames = []
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowGage.ftype():
                    unitnumber = key
                    filenames.append(os.path.basename(value.filename))
        for file in files:
            filenames.append(os.path.basename(file))

        gagepak = ModflowGage(model, numgage=numgage,
                              gage_data=gage_data, filenames=filenames,
                              unitnumber=unitnumber)
        return gagepak

    @staticmethod
    def ftype():
        return 'GAGE'

    @staticmethod
    def defaultunit():
        return 120
