"""
mffhb module.  Contains the ModflowFhb class. Note that the user can access
the ModflowFhb class as `flopy.modflow.ModflowFhb`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?fhb.htm>`_.

"""
import sys

import numpy as np

from ..pakbase import Package
from ..utils.recarray_utils import create_empty_recarray

class ModflowFhb(Package):
    """
    MODFLOW Flow and Head Boundary Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.ModflowFhb`) to
        which this package will be added.
    nbdtim : int
        The number of times at which flow and head will be specified for all
        selected cells. (default is 1)
    nflw : int
        Number of cells at which flows will be specified. (default is 0)
    nhed: int
        Number of cells at which heads will be specified. (default is 0)
    ifhbss : int
        FHB steady-state option flag. If the simulation includes any
        transient-state stress periods, the flag is read but not used; in
        this case, specified-flow, specified-head, and auxiliary-variable
        values will be interpolated for steady-state stress periods in the
        same way that values are interpolated for transient stress periods.
        If the simulation includes only steady-state stress periods, the flag
        controls how flow, head, and auxiliary-variable values will be
        computed for each steady-state solution. (default is 0)
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is None).
    nfhbx1 : int
        Number of auxiliary variables whose values will be computed for each
        time step for each specified-flow cell. Auxiliary variables are
        currently not supported. (default is 0)
    nfhbx2 : int
        Number of auxiliary variables whose values will be computed for each
        time step for each specified-head cell. Auxiliary variables are
        currently not supported. (default is 0)
    ifhbpt : int
        Flag for printing values of data list. Applies to datasets 4b, 5b, 6b,
        7b, and 8b. If ifhbpt > 0, datasets read at the beginning of the
        simulation will be printed. Otherwise, the datasets will not be
        printed. (default is 0).
    bdtimecnstm : float
        A constant multiplier for data list bdtime. (default is 1.0)
    bdtime : float or list of floats
        Simulation time at which values of specified flow and (or) values of
        specified head will be read. nbdtim values are required.
        (default is 0.0)
    cnstm5 : float
        A constant multiplier for data list flwrat. (default is 1.0)
    ds5 : list or numpy array or recarray
        Each FHB flwrat cell (dataset 5) is defined through definition of
        layer(int), row(int), column(int), iaux(int), flwrat[nbdtime](float).
        There are nflw entries. (default is None)
        The simplest form is a list of lists with the FHB flow boundaries.
        This gives the form of::

            ds5 =
            [
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)]
            ]

        Note there should be nflw rows in ds7.

    cnstm7 : float
        A constant multiplier for data list sbhedt. (default is 1.0)
    ds7 : list or numpy array or recarray
        Each FHB sbhed cell (dataset 7) is defined through definition of
        layer(int), row(int), column(int), iaux(int), sbhed[nbdtime](float).
        There are nflw entries. (default is None)
        The simplest form is a list of lists with the FHB flow boundaries.
        This gives the form of::

            ds5 =
            [
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)]
            ]

        Note there should be nhed rows in ds7.

    extension : string
        Filename extension (default is 'fhb')
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
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> fhb = flopy.modflow.ModflowFhb(m)

    """

    def __init__(self, model, nbdtim=1, nflw=0, nhed=0, ifhbss=0, ipakcb=None,
                 nfhbx1=0, nfhbx2=0, ifhbpt=0, bdtimecnstm=1.0, bdtime=[0.],
                 cnstm5=1.0, ds5=None, cnstm7=1.0, ds7=None, extension='fhb',
                 unitnumber=None, filenames=None):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowFhb.defaultunit()

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
                                  package=ModflowFhb.ftype())
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowFhb.ftype()]
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
        self.url = 'flow_and_head_boundary_packag2.htm'

        self.nbdtim = nbdtim
        self.nflw = nflw
        self.nhed = nhed
        self.ifhbss = ifhbss
        self.ipakcb = ipakcb
        if nfhbx1 != 0:
            nfhbx1 = 0
        self.nfhbx1 = nfhbx1
        if nfhbx2 != 0:
            nfhbx2 = 0
        self.nfhbx2 = nfhbx2
        self.ifhbpt = ifhbpt
        self.bdtimecnstm = bdtimecnstm
        if isinstance(bdtime, float):
            bdtime = [bdtime]
        self.bdtime = bdtime
        self.cnstm5 = cnstm5
        self.cnstm7 = cnstm7

        # check the type of dataset 5
        if ds5 is not None:
            dtype = ModflowFhb.get_default_dtype(nbdtim=nbdtim, head=False,
                                                 structured=model.structured)
            if isinstance(ds5, (float, int, str)):
                msg = 'dataset 5 must be a list of lists or a numpy array'
                raise TypeError(msg)
            elif isinstance(ds5, list):
                ds5 = np.array(ds5)
            # convert numpy array to a rec array
            if ds5.dtype != dtype:
                ds5 = np.core.records.fromarrays(ds5.transpose(), dtype=dtype)

        # assign dataset 5
        self.ds5 = ds5

        # check the type of dataset 7
        if ds7 is not None:
            dtype = ModflowFhb.get_default_dtype(nbdtim=nbdtim, head=True,
                                                 structured=model.structured)
            if isinstance(ds7, (float, int, str)):
                msg = 'dataset 7 must be a list of lists or a numpy array'
                raise TypeError(msg)
            elif isinstance(ds7, list):
                ds7 = np.array(ds7)
            # convert numpy array to a rec array
            if ds7.dtype != dtype:
                ds7 = np.core.records.fromarrays(ds7.transpose(), dtype=dtype)

        # assign dataset 7
        self.ds7 = ds7

        # perform some simple verification
        if len(self.bdtime) != self.nbdtim:
            msg = 'bdtime has {} entries '.format(len(self.bdtime)) + \
                  'but requires {} entries.'.format(self.nbdtim)
            raise ValueError(msg)

        if self.nflw > 0:
            if self.ds5 is None:
                msg = 'dataset 5 is not specified but ' + \
                      'nflw > 0 ({})'.format(self.nflw)
                raise TypeError(msg)

            if self.ds5.shape[0] != self.nflw:
                msg = 'dataset 5 has {} rows '.format(self.ds5.shape[0]) + \
                      'but requires {} rows.'.format(self.nflw)
                raise ValueError(msg)
            nc = self.nbdtim
            if model.structured:
                nc += 4
            else:
                nc += 2
            if len(self.ds5.dtype.names) != nc:
                msg = 'dataset 5 has {} '.format(len(self.ds5.dtype.names)) + \
                      'columns but requires {} columns.'.format(nc)
                raise ValueError(msg)

        if self.nhed > 0:
            if self.ds7 is None:
                msg = 'dataset 7 is not specified but ' + \
                      'nhed > 0 ({})'.format(self.nhed)
                raise TypeError(msg)
            if self.ds7.shape[0] != self.nhed:
                msg = 'dataset 7 has {} rows '.format(self.ds7.shape[0]) + \
                      'but requires {} rows.'.format(self.nhed)
                raise ValueError(msg)
            nc = self.nbdtim
            if model.structured:
                nc += 4
            else:
                nc += 2
            if len(self.ds7.dtype.names) != nc:
                msg = 'dataset 7 has {} '.format(len(self.ds7.dtype.names)) + \
                      'columns but requires {} columns.'.format(nc)
                raise ValueError(msg)

        self.parent.add_package(self)

    @staticmethod
    def get_empty(ncells=0, nbdtim=1, structured=True, head=False):
        # get an empty recarray that correponds to dtype
        dtype = ModflowFhb.get_default_dtype(nbdtim=nbdtim,
                                             structured=structured, head=head)
        return create_empty_recarray(ncells, dtype, default_value=-1.0E+10)

    @staticmethod
    def get_default_dtype(nbdtim=1, structured=True, head=False):
        if structured:
            dtype = [("k", np.int), ("i", np.int), ("j", np.int)]
        else:
            dtype = [("node", np.int)]
        dtype.append(("iaux", np.int))
        for n in range(nbdtim):
            if head:
                name = ("sbhed{}".format(n + 1))
            else:
                name = ("flwrat{}".format(n + 1))
            dtype.append((name, np.float32))
        return np.dtype(dtype)

    def ncells(self):
        # Return the  maximum number of cells that have a fhb flow or
        # head boundary. (developed for MT3DMS SSM package)
        return self.nflw + self.nhed

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        f = open(self.fn_path, 'w')
        # f.write('{0:s}\n'.format(self.heading))

        # Data set 1
        f.write('{} '.format(self.nbdtim))
        f.write('{} '.format(self.nflw))
        f.write('{} '.format(self.nhed))
        f.write('{} '.format(self.ifhbss))
        f.write('{} '.format(self.ipakcb))
        f.write('{} '.format(self.nfhbx1))
        f.write('{}\n'.format(self.nfhbx2))

        # Dataset 2 - flow auxiliary names

        # Dataset 3 - head auxiliary names

        # Dataset 4a IFHBUN CNSTM IFHBPT
        f.write('{} '.format(self.unit_number[0]))
        f.write('{} '.format(self.bdtimecnstm))
        f.write('{}\n'.format(self.ifhbpt))

        # Dataset 4b
        for n in range(self.nbdtim):
            f.write('{} '.format(self.bdtime[n]))
        f.write('\n')

        # Dataset 5 and 6
        if self.nflw > 0:
            # Dataset 5a IFHBUN CNSTM IFHBPT
            f.write('{} '.format(self.unit_number[0]))
            f.write('{} '.format(self.cnstm5))
            f.write('{}\n'.format(self.ifhbpt))

            # Dataset 5b
            for n in range(self.nflw):
                for name in self.ds5.dtype.names:
                    v = self.ds5[n][name]
                    if name in ['k', 'i', 'j', 'node']:
                        v += 1
                    f.write('{} '.format(v))
                f.write('\n')

            # Dataset 6a and 6b - flow auxiliary data
            if self.nfhbx1 > 0:
                i = 0

        # Dataset 7
        if self.nhed > 0:
            # Dataset 7a IFHBUN CNSTM IFHBPT
            f.write('{} '.format(self.unit_number[0]))
            f.write('{} '.format(self.cnstm7))
            f.write('{}\n'.format(self.ifhbpt))

            # Dataset 7b IFHBUN CNSTM IFHBPT
            for n in range(self.nhed):
                for name in self.ds7.dtype.names:
                    v = self.ds7[n][name]
                    if name in ['k', 'i', 'j', 'node']:
                        v += 1
                    f.write('{} '.format(v))
                f.write('\n')

            # Dataset 8a and 8b - head auxiliary data
            if self.nfhbx2 > 0:
                i = 1

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
        fhb : ModflowFhb object
            ModflowFhb object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> fhb = flopy.modflow.ModflowFhb.load('test.fhb', m)

        """
        if model.verbose:
            sys.stdout.write('loading fhb package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # determine package unit number
        iufhb = None
        if ext_unit_dict is not None:
            iufhb, fname = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowFhb.ftype())

        # Dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # dataset 1
        if model.verbose:
            sys.stdout.write('loading fhb dataset 1\n')
        raw = line.strip().split()
        nbdtim = int(raw[0])
        nflw = int(raw[1])
        nhed = int(raw[2])
        ifhbss = int(raw[3])
        ipakcb = int(raw[4])
        nfhbx1 = int(raw[5])
        nfhbx2 = int(raw[6])

        ifhbpt = 0

        # Dataset 2
        flow_aux = []
        if nfhbx1 > 0:
            if model.verbose:
                sys.stdout.write('loading fhb dataset 2\n')
            msg = 'dataset 2 will not be preserved ' + \
                  'in the created hfb object.\n'
            sys.stdout.write(msg)
            for idx in range(nfhbx1):
                line = f.readline()
                raw = line.strip().split()
                varnam = raw[0]
                if len(varnam) > 16:
                    varnam = varnam[0:16]
                weight = float(raw[1])
                flow_aux.append([varnam, weight])

        # Dataset 3
        head_aux = []
        if nfhbx2 > 0:
            if model.verbose:
                sys.stdout.write('loading fhb dataset 3\n')
            msg = 'dataset 3 will not be preserved ' + \
                  'in the created hfb object.\n'
            sys.stdout.write(msg)
            for idx in range(nfhbx2):
                line = f.readline()
                raw = line.strip().split()
                varnam = raw[0]
                if len(varnam) > 16:
                    varnam = varnam[0:16]
                weight = float(raw[1])
                head_aux.append([varnam, weight])

        # Dataset 4a IFHBUN CNSTM IFHBPT
        if model.verbose:
            sys.stdout.write('loading fhb dataset 4a\n')
        line = f.readline()
        raw = line.strip().split()
        ifhbun = int(raw[0])
        if ifhbun != iufhb:
            msg = 'fhb dataset 4a must be in the fhb file '
            msg += '(unit={}) '.format(iufhb)
            msg += 'fhb data is specified in unit={}'.format(ifhbun)
            raise ValueError(msg)
        bdtimecnstm = float(raw[1])
        ifhbpt = max(ifhbpt, int(raw[2]))

        # Dataset 4b
        if model.verbose:
            sys.stdout.write('loading fhb dataset 4b\n')
        line = f.readline()
        raw = line.strip().split()
        bdtime = []
        for n in range(nbdtim):
            bdtime.append(float(raw[n]))

        # Dataset 5 and 6
        cnstm5 = None
        ds5 = None
        cnstm6 = None
        ds6 = None
        if nflw > 0:
            if model.verbose:
                sys.stdout.write('loading fhb dataset 5a\n')
            # Dataset 5a IFHBUN CNSTM IFHBPT
            line = f.readline()
            raw = line.strip().split()
            ifhbun = int(raw[0])
            if ifhbun != iufhb:
                msg = 'fhb dataset 5a must be in the fhb file '
                msg += '(unit={}) '.format(iufhb)
                msg += 'fhb data is specified in unit={}'.format(ifhbun)
                raise ValueError(msg)
            cnstm5 = float(raw[1])
            ifhbpt = max(ifhbpt, int(raw[2]))

            if model.verbose:
                sys.stdout.write('loading fhb dataset 5b\n')
            dtype = ModflowFhb.get_default_dtype(nbdtim=nbdtim, head=False,
                                                 structured=model.structured)
            ds5 = ModflowFhb.get_empty(ncells=nflw, nbdtim=nbdtim, head=False,
                                       structured=model.structured)
            for n in range(nflw):
                line = f.readline()
                raw = line.strip().split()
                ds5[n] = tuple(raw[:len(dtype.names)])

            if model.structured:
                ds5['k'] -= 1
                ds5['i'] -= 1
                ds5['j'] -= 1
            else:
                ds5['node'] -= 1

            # Dataset 6
            if nfhbx1 > 0:
                cnstm6 = []
                ds6 = []
                dtype = []
                for name, weight in flow_aux:
                    dtype.append((name, np.float32))
                for naux in range(nfhbx1):
                    if model.verbose:
                        sys.stdout.write('loading fhb dataset 6a - aux ' +
                                         '{}\n'.format(naux + 1))
                    msg = 'dataset 6a will not be preserved in ' + \
                          'the created hfb object.\n'
                    sys.stdout.write(msg)
                    # Dataset 6a IFHBUN CNSTM IFHBPT
                    line = f.readline()
                    raw = line.strip().split()
                    ifhbun = int(raw[0])
                    if ifhbun != iufhb:
                        msg = 'fhb dataset 6a must be in the fhb file '
                        msg += '(unit={}) '.format(iufhb)
                        msg += 'fhb data is specified in ' + \
                               'unit={}'.format(ifhbun)
                        raise ValueError(msg)
                    cnstm6.append(float(raw[1]))
                    ifhbpt = max(ifhbpt, int(raw[2]))

                    if model.verbose:
                        sys.stdout.write('loading fhb dataset 6b - aux ' +
                                         '{}\n'.format(naux + 1))
                    msg = 'dataset 6b will not be preserved in ' + \
                          'the created hfb object.\n'
                    sys.stdout.write(msg)
                    current = np.recarray(nflw, dtype=dtype)
                    for n in range(nflw):
                        line = f.readline()
                        raw = line.strip().split()
                        current[n] = tuple(raw[:len(dtype.names)])
                    ds6.append(current.copy())

        # Dataset 7
        cnstm7 = None
        ds7 = None
        cnstm8 = None
        ds8 = None
        if nhed > 0:
            if model.verbose:
                sys.stdout.write('loading fhb dataset 7a\n')
            # Dataset 7a IFHBUN CNSTM IFHBPT
            line = f.readline()
            raw = line.strip().split()
            ifhbun = int(raw[0])
            if ifhbun != iufhb:
                msg = 'fhb dataset 7a must be in the fhb file '
                msg += '(unit={}) '.format(iufhb)
                msg += 'fhb data is specified in unit={}'.format(ifhbun)
                raise ValueError(msg)
            cnstm7 = float(raw[1])
            ifhbpt = max(ifhbpt, int(raw[2]))

            if model.verbose:
                sys.stdout.write('loading fhb dataset 7b\n')
            dtype = ModflowFhb.get_default_dtype(nbdtim=nbdtim, head=True,
                                                 structured=model.structured)
            ds7 = ModflowFhb.get_empty(ncells=nhed, nbdtim=nbdtim, head=True,
                                       structured=model.structured)
            for n in range(nhed):
                line = f.readline()
                raw = line.strip().split()
                ds7[n] = tuple(raw[:len(dtype.names)])

            if model.structured:
                ds7['k'] -= 1
                ds7['i'] -= 1
                ds7['j'] -= 1
            else:
                ds7['node'] -= 1

            # Dataset 8
            if nfhbx2 > 0:
                cnstm8 = []
                ds8 = []
                dtype = []
                for name, weight in head_aux:
                    dtype.append((name, np.float32))
                for naux in range(nfhbx1):
                    if model.verbose:
                        sys.stdout.write('loading fhb dataset 8a - aux ' +
                                         '{}\n'.format(naux + 1))
                    msg = 'dataset 8a will not be preserved in ' + \
                          'the created hfb object.\n'
                    sys.stdout.write(msg)
                    # Dataset 6a IFHBUN CNSTM IFHBPT
                    line = f.readline()
                    raw = line.strip().split()
                    ifhbun = int(raw[0])
                    if ifhbun != iufhb:
                        msg = 'fhb dataset 8a must be in the fhb file '
                        msg += '(unit={}) '.format(iufhb)
                        msg += 'fhb data is specified in ' + \
                               'unit={}'.format(ifhbun)
                        raise ValueError(msg)
                    cnstm8.append(float(raw[1]))
                    ifhbpt6 = int(raw[2])
                    ifhbpt = max(ifhbpt, ifhbpt6)

                    if model.verbose:
                        sys.stdout.write('loading fhb dataset 8b - aux ' +
                                         '{}\n'.format(naux + 1))
                    msg = 'dataset 8b will not be preserved in ' + \
                          'the created hfb object.'
                    sys.stdout.write(msg)
                    current = np.recarray(nflw, dtype=dtype)
                    for n in range(nhed):
                        line = f.readline()
                        raw = line.strip().split()
                        current[n] = tuple(raw[:len(dtype.names)])
                    ds8.append(current.copy())

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowFhb.ftype())
        if ipakcb > 0:
            iu, filenames[1] = \
                model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
            model.add_pop_key_list(ipakcb)

        # auxillary data are not passed to load instantiation
        nfhbx1 = 0
        nfhbx2 = 0

        fhb = ModflowFhb(model, nbdtim=nbdtim, nflw=nflw, nhed=nhed,
                         ifhbss=ifhbss, ipakcb=ipakcb,
                         nfhbx1=nfhbx1, nfhbx2=nfhbx2, ifhbpt=ifhbpt,
                         bdtimecnstm=bdtimecnstm, bdtime=bdtime,
                         cnstm5=cnstm5, ds5=ds5, cnstm7=cnstm7, ds7=ds7,
                         unitnumber=unitnumber, filenames=filenames)

        # return fhb object
        return fhb

    @staticmethod
    def ftype():
        return 'FHB'

    @staticmethod
    def defaultunit():
        return 40
