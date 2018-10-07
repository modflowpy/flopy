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
from ..utils.recarray_utils import create_empty_recarray

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
    stress_period_data : dictionary of reach data
        Each dictionary contains a list of str reach data for a stress period.

        Each stress period in the dictionary data contains data for
        datasets 6 and 8.

        The value for stress period data for a stress period can be an integer
        (-1 or 0), a list of lists, a numpy array, or a numpy recarry. If
        stress period data for a stress period contains an integer, a -1 denotes
        data from the previous stress period will be reused and a 0 indicates
        there are no str reaches for this stress period.

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
        If data are specified for dataset 6 for a given stress period and icalc>0,
        then stream width, stream slope, and roughness coefficients should be
        appropriately set.

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
        (-1 or 0), a list of lists, a numpy array, or a numpy recarry. If
        segment data for a stress period contains an integer, a -1 denotes
        data from the previous stress period will be reused and a 0 indicates
        there are no str segments for this stress period.

        Otherwise stress period data for a stress period should contain nss
        rows of data containing data for each segment. Segment data are
        specified through definition of itrib (int) data for up to 10 tributaries
        and iupseg (int) data.

        If ntrib=0 is specified, itrib values are not used and can be any value
        for each stress period. If data are specified for dataset 6 for a given
        stress period and ntrib>0, then itrib data should be specified for columns
        0:ntrib.

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
        Package options. (default is None).
    extension : string
        Filename extension (default is 'str')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output and sfr output name will be
        created using the model name and .cbc the .sfr.bin/.sfr.out extensions
        (for example, modflowtest.cbc, and modflowtest.sfr.bin), if ipakcbc and
        istcb2 are numbers greater than zero. If a single string is passed
        the package will be set to the string and cbc and sf routput names
        will be created using the model name and .cbc and .sfr.bin/.sfr.out
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
    >>> strd[0] = [[2, 3, 4, 15.6, 1050., -4]]  #this river boundary will be
    >>>                                         #applied to all stress periods
    >>> str8 = flopy.modflow.ModflowStr(m, stress_period_data=strd)

    """

    def __init__(self, model, mxacts=0, nss=0, ntrib=0, ndiv=0, icalc=0,
                 const=86400., ipakcb=None, istcb2=None,
                 dtype=None, stress_period_data=None, segment_data=None,
                 extension='str', unitnumber=None, filenames=None,
                 options=None, **kwargs):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowStr.defaultunit()

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
            model.add_output_file(ipakcb, fname=fname,
                                  package=ModflowStr.ftype())
        else:
            ipakcb = 0

        if istcb2 is not None:
            fname = filenames[2]
            model.add_output_file(istcb2, fname=fname,
                                  package=ModflowStr.ftype())
        else:
            ipakcb = 0


        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowStr.ftype()]
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
        self.url = 'str.htm'
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
            raise Exception('ModflowStr error: ntrib must be less that 10: ' +
                            'specified value = {}'.format(ntrib))

        if options is None:
            options = []
        self.options = options

        # parameters are not supported
        self.npstr = 0

        # determine dtype for dataset 6
        if dtype is not None:
            self.dtype = dtype[0]
            self.dtype2 = dtype[1]
        else:
            auxnames = []
            if len(options) > 0:
                auxnames = []
                it = 0
                while True:
                    if 'aux' in options[it].lower():
                        aux_names.append(options[it + 1].lower())
                        it += 1
                    it += 1
                    if it > len(options):
                        break
            if len(auxnames) < 1:
                auxnames = None
            d, d2 = self.get_empty(1, 1, aux_names=auxnames,
                                   structured=self.parent.structured)
            self.dtype = d.dtype
            self.dtype2 = d2.dtype

        # convert stress_period_data for datasets 6 and 8 to a recarray if necessary
        if stress_period_data is not None:
            for key, d in stress_period_data.items():
                if isinstance(d, list):
                    d = np.array(d)
                if isinstance(d, np.recarray):
                    assert d.dtype == self.dtype, 'ModflowStr error: recarray dtype: ' + \
                                                  str(
                                                      d.dtype) + ' does not match ' + \
                                                  'self dtype: ' + str(
                        self.dtype)
                elif isinstance(d, np.ndarray):
                    d = np.core.records.fromarrays(d.transpose(),
                                                   dtype=self.dtype)
                elif isinstance(d, int):
                    if model.verbose:
                        if d < 0:
                            print(
                                '   reusing str data from previous stress period')
                        elif d == 0:
                            print('   no str data for stress period {}'.format(
                                key))
                else:
                    raise Exception(
                        'ModflowStr error: unsupported data type: ' +
                        str(type(d)) + ' at kper ' +
                        '{0:d}'.format(key))
        # add stress_period_data to package
        self.stress_period_data = MfList(self, stress_period_data)

        # convert segment_data for datasets 9 and 10 to a recarray if necessary
        if segment_data is not None:
            for key, d in segment_data.items():
                if isinstance(d, list):
                    d = np.array(d)
                if isinstance(d, np.recarray):
                    assert d.dtype == self.dtype2, 'ModflowStr error: recarray dtype: ' + \
                                                   str(
                                                       d.dtype) + ' does not match ' + \
                                                   'self dtype: ' + str(
                        self.dtype2)
                elif isinstance(d, np.ndarray):
                    d = np.core.records.fromarrays(d.transpose(),
                                                   dtype=self.dtype2)
                elif isinstance(d, int):
                    if model.verbose:
                        if d < 0:
                            print(
                                '   reusing str segment data from previous stress period')
                        elif d == 0:
                            print(
                                '   no str segment data for stress period {}'.format(
                                    key))
                else:
                    raise Exception(
                        'ModflowStr error: unsupported data type: ' +
                        str(type(d)) + ' at kper ' +
                        '{0:d}'.format(key))
        # add stress_period_data to package
        self.segment_data = segment_data

        self.parent.add_package(self)
        return

    @staticmethod
    def get_empty(ncells=0, nss=0, aux_names=None, structured=True):
        # get an empty recarray that correponds to dtype
        dtype, dtype2 = ModflowStr.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        #return (create_empty_recarray(ncells, dtype=dtype, default_value=-1.0E+10),
        #        create_empty_recarray(nss, dtype=dtype, default_value=0))
        return (create_empty_recarray(ncells, dtype=dtype, default_value=-1.0E+10),
                create_empty_recarray(nss, dtype=dtype2, default_value=0))

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype([("k", np.int), ("i", np.int), ("j", np.int),
                              ("segment", np.int), ("reach", np.int),
                              ("flow", np.float32), ("stage", np.float32),
                              ("cond", np.float32), ("sbot", np.float32),
                              ("stop", np.float32),
                              ("width", np.float32), ("slope", np.float32),
                              ("rough", np.float32)])
        else:
            dtype = np.dtype([("node", np.int),
                              ("segment", np.int), ("reach", np.int),
                              ("flow", np.float32), ("stage", np.float32),
                              ("cond", np.float32), ("sbot", np.float32),
                              ("stop", np.float32),
                              ("width", np.float32), ("slope", np.float32),
                              ("rough", np.float32)])

        dtype2 = np.dtype([("itrib01", np.int), ("itrib02", np.int),
                           ("itrib03", np.int), ("itrib04", np.int),
                           ("itrib05", np.int), ("itrib06", np.int),
                           ("itrib07", np.int), ("itrib08", np.int),
                           ("itrib09", np.int), ("itrib10", np.int),
                           ("iupseg", np.int)])
        return dtype, dtype2

    def ncells(self):
        # Return the  maximum number of cells that have a stream
        # (developed for MT3DMS SSM package)
        return self.mxacts

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f_str = open(self.fn_path, 'w')
        # dataset 0
        f_str.write('{0}\n'.format(self.heading))
        # dataset 1 - parameters not supported on write
        # dataset 2
        line = '{:10d}{:10d}{:10d}{:10d}{:10d}{:10.3f}{:10d}{:10d}'.format(
            self.mxacts, self.nss,
            self.ntrib, self.ndiv,
            self.icalc, self.const,
            self.ipakcb, self.istcb2)
        for opt in self.options:
            line += ' ' + str(opt)
        line += '\n'
        f_str.write(line)

        # dataset 3  - parameters not supported on write
        # dataset 4a - parameters not supported on write
        # dataset 4b - parameters not supported on write

        nrow, ncol, nlay, nper = self.parent.get_nrow_ncol_nlay_nper()

        kpers = list(self.stress_period_data.data.keys())
        kpers.sort()

        if self.parent.bas6.ifrefm:
            fmt6 = ['{:5d} ', '{:5d} ', '{:5d} ', '{:5d} ', '{:5d} ',
                    '{:15.7f} ', '{:15.7f} ', '{:15.7f} ', '{:15.7f} ',
                    '{:15.7f} ']
            fmt8 = '{:15.7} '
            fmt9 = '{:10d} '
        else:
            fmt6 = ['{:5d}', '{:5d}', '{:5d}', '{:5d}', '{:5d}',
                    '{:15.4f}', '{:10.3f}', '{:10.3f}', '{:10.3f}', '{:10.3f}']
            fmt8 = '{:10.4g}'
            fmt9 = '{:5d}'

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
            line = '{:10d}{:10d}{:10d}  # stress period {}\n'.format(itmp, 0,
                                                                     0, iper+1)
            f_str.write(line)
            if itmp > 0:
                tdata = np.recarray.copy(tdata)
                # dataset 6
                for line in tdata:
                    line['k'] += 1
                    line['i'] += 1
                    line['j'] += 1
                    for idx, v in enumerate(line):
                        if idx < 10:
                            f_str.write(fmt6[idx].format(v))
                        elif idx > 12:
                            f_str.write('{} '.format(v))
                    f_str.write('\n')
                # dataset 8
                if self.icalc > 0:
                    for line in tdata:
                        for idx in range(10, 13):
                            f_str.write(fmt8.format(line[idx]))
                        f_str.write('\n')
                # dataset 9
                if self.ntrib > 0:
                    for line in sdata:
                        #for idx in range(3):
                        for idx in range(self.ntrib):
                            f_str.write(fmt9.format(line[idx]))
                        f_str.write('\n')
                # dataset 10
                if self.ndiv > 0:
                    for line in sdata:
                        #f_str.write('{:10d}\n'.format(line[3]))
                        f_str.write('{:10d}\n'.format(line[self.ntrib]))

        # close the str file
        f_str.close()

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
        >>> strm = flopy.modflow.ModflowStr.load('test.str', m)

        """

        if model.verbose:
            sys.stdout.write('loading str package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # read dataset 1 - optional parameters
        npstr, mxl = 0, 0
        t = line.strip().split()
        if t[0].lower() == 'parameter':
            if model.verbose:
                sys.stdout.write('  loading str dataset 1\n')
            npstr = int(t[1])
            mxl = int(t[2])

            # read next line
            line = f.readline()

        # data set 2
        if model.verbose:
            sys.stdout.write('  loading str dataset 2\n')
        t = line.strip().split()
        mxacts = int(t[0])
        nss = int(t[1])
        ntrib = int(t[2])
        ndiv = int(t[3])
        icalc = int(t[4])
        const = float(t[5])
        istcb1 = int(t[6])
        istcb2 = int(t[7])
        ipakcb = 0
        try:
            if istcb1 != 0:
                ipakcb = istcb1
                model.add_pop_key_list(istcb1)
        except:
            pass
        try:
            if istcb2 != 0:
                ipakcb = 53
                model.add_pop_key_list(istcb2)
        except:
            pass

        options = []
        aux_names = []
        if len(t) > 8:
            it = 8
            while it < len(t):
                toption = t[it]
                if 'aux' in toption.lower():
                    options.append(' '.join(t[it:it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1

        # read parameter data
        if npstr > 0:
            dt = ModflowStr.get_empty(1, aux_names=aux_names).dtype
            pak_parms = mfparbc.load(f, npstr, dt, model.verbose)

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        stress_period_data = {}
        segment_data = {}
        for iper in range(nper):
            if model.verbose:
                print("   loading " + str(
                    ModflowStr) + " for kper {0:5d}".format(iper + 1))
            line = f.readline()
            if line == '':
                break
            t = line.strip().split()
            itmp = int(t[0])
            irdflg, iptflg = 0, 0
            if len(t) > 1:
                irdflg = int(t[1])
            if len(t) > 2:
                iptflg = int(t[2])

            if itmp == 0:
                bnd_output = None
                seg_output = None
                current, current_seg = ModflowStr.get_empty(itmp, nss,
                                                            aux_names=aux_names)
            elif itmp > 0:
                if npstr > 0:
                    partype = ['cond']
                    if model.verbose:
                        print("   reading str dataset 7")
                    for iparm in range(itmp):
                        line = f.readline()
                        t = line.strip().split()
                        pname = t[0].lower()
                        iname = 'static'
                        try:
                            tn = t[1]
                            c = tn.lower()
                            instance_dict = pak_parms.bc_parms[pname][1]
                            if c in instance_dict:
                                iname = c
                            else:
                                iname = 'static'
                        except:
                            pass
                        par_dict, current_dict = pak_parms.get(pname)
                        data_dict = current_dict[iname]

                        current = ModflowStr.get_empty(par_dict['nlst'],
                                                       aux_names=aux_names)

                        #  get appropriate parval
                        if model.mfpar.pval is None:
                            parval = np.float(par_dict['parval'])
                        else:
                            try:
                                parval = np.float(
                                    model.mfpar.pval.pval_dict[pname])
                            except:
                                parval = np.float(par_dict['parval'])

                        # fill current parameter data (par_current)
                        for ibnd, t in enumerate(data_dict):
                            current[ibnd] = tuple(t[:len(current.dtype.names)])

                else:
                    if model.verbose:
                        print("   reading str dataset 6")
                    current, current_seg = ModflowStr.get_empty(itmp, nss,
                                                                aux_names=aux_names)
                    for ibnd in range(itmp):
                        line = f.readline()
                        t = []
                        if model.free_format_input:
                            tt = line.strip().split()
                            # current[ibnd] = tuple(t[:len(current.dtype.names)])
                            for idx, v in enumerate(tt[:10]):
                                t.append(v)
                            for ivar in range(3):
                                t.append(-1.0E+10)
                            if len(aux_names) > 0:
                                for idx, v in enumerate(t[10:]):
                                    t.append(v)
                            if len(tt) != len(current.dtype.names) - 3:
                                raise Exception
                        else:
                            ipos = [5, 5, 5, 5, 5, 15, 10, 10, 10, 10]
                            istart = 0
                            for ivar in range(len(ipos)):
                                istop = istart + ipos[ivar]
                                txt = line[istart:istop]
                                try:
                                    t.append(float(txt))
                                except:
                                    t.append(0.)
                                istart = istop
                            for ivar in range(3):
                                t.append(-1.0E+10)
                            if len(aux_names) > 0:
                                tt = line[istart:].strip().split()
                                for ivar in range(len(aux_names)):
                                    t.append(tt[ivar])
                        current[ibnd] = tuple(t[:len(current.dtype.names)])

                # convert indices to zero-based
                current['k'] -= 1
                current['i'] -= 1
                current['j'] -= 1

                # read dataset 8
                if icalc > 0:
                    if model.verbose:
                        print("   reading str dataset 8")
                    for ibnd in range(itmp):
                        line = f.readline()
                        if model.free_format_input:
                            t = line.strip().split()
                            v = [float(vt) for vt in t[:3]]
                        else:
                            v = []
                            ipos = [10, 10, 10]
                            istart = 0
                            for ivar in range(len(ipos)):
                                istop = istart + ipos[ivar]
                                v.append(float(line[istart:istop]))
                                istart = istop + 1
                        ipos = 0
                        for idx in range(10, 13):
                            current[ibnd][idx] = v[ipos]
                            ipos += 1

                bnd_output = np.recarray.copy(current)

                # read data set 9
                if ntrib > 0:
                    if model.verbose:
                        print("   reading str dataset 9")
                    for iseg in range(nss):
                        line = f.readline()
                        if model.free_format_input:
                            t = line.strip().split()
                            v = [float(vt) for vt in t[:ntrib]]
                        else:
                            v = []
                            ipos = 5
                            istart = 0
                            for ivar in range(ntrib):
                                istop = istart + ipos
                                try:
                                    v.append(float(line[istart:istop]))
                                except:
                                    v.append(0.)
                                istart = istop
                        for idx in range(ntrib):
                            current_seg[iseg][idx] = v[idx]

                # read data set 10
                if ndiv > 0:
                    if model.verbose:
                        print("   reading str dataset 10")
                    for iseg in range(nss):
                        line = f.readline()
                        if model.free_format_input:
                            t = line.strip().split()
                            v = float(t[0])
                        else:
                            ipos = 10
                            istart = 0
                            for ivar in range(ndiv):
                                istop = istart + ipos
                                v = float(line[istart:istop])
                                istart = istop
                        current_seg[iseg][10] = v

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


        # determine specified unit number
        unitnumber = None
        filenames = [None, None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowStr.ftype())
            if ipakcb > 0:
                iu, filenames[1] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
            if abs(istcb2) > 0:
                iu, filenames[2] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=abs(istcb2))

        strpak = ModflowStr(model, mxacts=mxacts, nss=nss,
                            ntrib=ntrib, ndiv=ndiv, icalc=icalc,
                            const=const, ipakcb=ipakcb, istcb2=istcb2,
                            stress_period_data=stress_period_data,
                            segment_data=segment_data,
                            options=options, unitnumber=unitnumber,
                            filenames=filenames)
        return strpak

    @staticmethod
    def ftype():
        return 'STR'

    @staticmethod
    def defaultunit():
        return 118
