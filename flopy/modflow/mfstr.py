"""
mfstr module.  Contains the ModflowStr class. Note that the user can access
the ModflowStr class as `flopy.modflow.ModflowStr`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/str.htm>`_.

"""
import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils.util_list import mflist


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
        is a flag and a unit number. (default is 0)
    dtype : numpy dtype
        is the dtype for dataset 6 data in stress_period_data dictionary.
        (default is None)
    stress_period_data : dictionary of boundaries
        Each dictionary contains a list of str data for a stress period.

        Each stress period in the dictionary data contains data for
        datasets 6, 8, 9, and 10.

        Data for dataset 6 can be an integer (-1 or 0), a list of lists,
        a numpy array, or a numpy recarry. If dataset 6 contains an integer
        a -1 denotes data from the previous stress period will be reused
        and a 0 indicates there are no str reaches for this stress period.

        Otherwise dataset 6 data should contain mxacts or fewer rows of data
        containing data for each reach. Reach data are specified through
        definition of layer (int), row (int), column (int), segment number
        (int), sequential reach number (int), flow entering a segment (float),
        stream stage (float), streambed hydraulic conductance (float),
        streambed bottom elevation (float), streambed top elevation (float),
        auxiliary variable data for auxiliary variables defined in options
        (float).

        If icalc=0 is used for str dataset 8 should be None for each stress
        period. If data are specified for dataset 6 for a given stress period
        and icalc>0, then dataset 8 data should be a numpy array or list of
        lists with a shape of (itmp, 3).

        If ntrib=0, str dataset 9 should be None for each stress period.
        If data are specified for dataset 6 for a given stress period and
        ntrib>0, then dataset 9 data should be a numpy array or list of
        lists with a shape of (nss, ntrib).

        If ndiv=0, str dataset 10 should be None for each stress period.
        If data are specified for dataset 6 for a given stress period and
        ntrib>0, then dataset 10 data should be a numpy array or list of
        lists with a shape of (nss).

        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. For example, if mxacts=3, nss=2, icalc=0, ntrib=1, and ndiv=1
        this gives the form of
            stress_period_data =
            {0: [
                [[lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop]],
                [[width, slope, rough],
                 [width, slope, rough],
                 [width, slope, rough]],
                [[itrib],
                 [itib]],
                [[iupseg],
                 [iupseg]]
                ],
            1:  [
                [[lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop]],
                [[width, slope, rough],
                 [width, slope, rough],
                 [width, slope, rough]],
                [[itrib],
                 [itib]],
                [[iupseg],
                 [iupseg]]
                ], ...
            kper:
                [
                [[lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop],
                 [lay, row, col, seg, reach, flow, stage, cond, sbot, stop]],
                [[width, slope, rough],
                 [width, slope, rough],
                 [width, slope, rough]],
                [[itrib],
                 [itib]],
                [[iupseg],
                 [iupseg]]
                ]
            }

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
    >>> strd = {}
    >>> strd[0] = [[2, 3, 4, 15.6, 1050., -4]]  #this river boundary will be
    >>>                                         #applied to all stress periods
    >>> str8 = flopy.modflow.ModflowStr(m, stress_period_data=lrcd)

    """

    def __init__(self, model, mxacts=0, nss=0, ntrib=0, ndiv=0, icalc=0,
                 const=86400., ipakcb=0,
                 dtype=None, stress_period_data=None,
                 extension='str', unitnumber=118, options=None, **kwargs):
        """
        Package constructor.

        """
        # Call parent init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension, 'STR', unitnumber)
        self.heading = '# STR for MODFLOW, generated by Flopy.'
        self.url = 'str.htm'
        self.mxacts = mxacts
        self.nss = nss
        self.icalc = icalc
        self.ntrib = ntrib
        self.ndiv = ndiv
        self.const = const
        self.ipakcb = ipakcb

        if options is None:
            options = []
        self.options = options

        # parameters are not supported
        self.npstr = 0

        # determine dtype for dataset 6
        if dtype is not None:
            self.dtype = dtype
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
            d = self.get_empty(1, aux_names=auxnames, structured=self.parent.structured)
            self.dtype = d.dtype

        # convert stress_period_data for dataset 6 to a recarray if necessary
        if stress_period_data is not None:
            for key, val in stress_period_data.items():
                d = val[0]
                if isinstance(d, list):
                    d = np.array(d)
                if isinstance(d, np.recarray):
                    assert d.dtype == self.dtype, 'ModflowStr error: recarray dtype: ' + \
                                                   str(d.dtype) + ' does not match ' + \
                                                   'self dtype: ' + str(self.dtype)
                elif isinstance(d, np.ndarray):
                    val[0] = np.core.records.fromarrays(d.transpose(),
                                                        dtype=self.dtype)
                elif isinstance(d, int):
                    if model.verbose:
                        if d < 0:
                            print('   reusing str data from previous stress period')
                        elif d == 0:
                            print('   no str data for stress period {}'.format(key))
                else:
                    raise Exception('ModflowStr error: unsupported data type: ' +
                                    str(type(d)) + ' at kper ' +
                                    '{0:d}'.format(key))


        self.stress_period_data = stress_period_data

        self.parent.add_package(self)

    def __repr__(self):
        return 'Stream class'

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that correponds to dtype
        dtype = ModflowStr.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        d = np.zeros((ncells, len(dtype)), dtype=dtype)
        d[:, :] = -1.0E+10
        return np.core.records.fromarrays(d.transpose(), dtype=dtype)

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype([("k", np.int), ("i", np.int), ("j", np.int),
                              ("segment", np.int), ("reach", np.int),
                              ("flow", np.float32), ("stage", np.float32),
                              ("cond", np.float32), ("sbot", np.float32),
                              ("stop", np.float32)])
        else:
            dtype = np.dtype([("node", np.int),
                              ("segment", np.int), ("reach", np.int),
                              ("flow", np.float32), ("stage", np.float32),
                              ("cond", np.float32), ("sbot", np.float32),
                              ("stop", np.float32)])

        return dtype

    def ncells(self):
        # Return the  maximum number of cells that have a stream
        # (developed for MT3DMS SSM package)
        return self.mxacts

    def write_file(self):
        """
        Write the file.

        """
        f_str = open(self.fn_path, 'w')
        # dataset 0
        f_str.write('{0}\n'.format(self.heading))
        # dataset 1 - parameters not supported on write
        # dataset 2
        line = '{:10d}{:10d}{:10d}{:10d}{:10d}{:10.3f}{:10d}{:10d}'.format(self.mxacts, self.nss,
                                                                           self.ntrib, self.ndiv,
                                                                           self.icalc, self.const,
                                                                           self.ipakcb, self.ipakcb)
        for opt in self.options:
            line += ' ' + str(opt)
        line += '\n'
        f_str.write(line)

        # dataset 3  - parameters not supported on write
        # dataset 4a - parameters not supported on write
        # dataset 4b - parameters not supported on write

        nrow, ncol, nlay, nper = self.parent.get_nrow_ncol_nlay_nper()

        kpers = list(self.stress_period_data.keys())
        kpers.sort()

        if self.parent.bas6.ifrefm:
            fmt6 = ['{:5d} ', '{:5d} ', '{:5d} ', '{:5d} ', '{:5d} ',
                    '{:15.7f} ', '{:15.7f} ', '{:15.7f} ', '{:15.7f} ', '{:15.7f} ']
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
                if isinstance(tdata[0], int):
                    itmp = tdata[0]
                else:
                    itmp = tdata[0].shape[0]
            line = '{:10d}{:10d}{:10d}  # stress period {}\n'.format(itmp, 0, 0, iper)
            f_str.write(line)
            if itmp > 0:
                # dataset 6
                for line in tdata[0]:
                    line['k'] += 1
                    line['i'] += 1
                    line['j'] += 1
                    for idx, v in enumerate(line):
                        if idx < 10:
                            f_str.write(fmt6[idx].format(v))
                        else:
                            f_str.write('{} '.format(v))
                    f_str.write('\n')
                # dataset 8
                if self.icalc > 0:
                    for line in tdata[1]:
                        for v in line:
                            f_str.write(fmt8.format(v))
                        f_str.write('\n')
                # dataset 9
                if self.ntrib > 0:
                    for line in tdata[2]:
                        for v in line:
                            f_str.write(fmt9.format(v))
                        f_str.write('\n')
                # dataset 10
                if self.ndiv > 0:
                    for v in tdata[3]:
                        f_str.write('{:10d}\n'.format(v[0]))

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
                ipakcb = 53
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
        for iper in range(nper):
            if model.verbose:
                print("   loading " + str(ModflowStr) + " for kper {0:5d}".format(iper + 1))
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
                current = ModflowStr.get_empty(itmp, aux_names=aux_names)
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

                        current = ModflowStr.get_empty(par_dict['nlst'], aux_names=aux_names)

                        #  get appropriate parval
                        if model.mfpar.pval is None:
                            parval = np.float(par_dict['parval'])
                        else:
                            try:
                                parval = np.float(model.mfpar.pval.pval_dict[pname])
                            except:
                                parval = np.float(par_dict['parval'])

                        # fill current parameter data (par_current)
                        for ibnd, t in enumerate(data_dict):
                            current[ibnd] = tuple(t[:len(current.dtype.names)])

                else:
                    if model.verbose:
                        print("   reading str dataset 6")
                    current = ModflowStr.get_empty(itmp, aux_names=aux_names)
                    for ibnd in range(itmp):
                        line = f.readline()
                        if "open/close" in line.lower():
                            #raise NotImplementedError("load() method does not support \'open/close\'")
                            oc_filename = os.path.join(model.model_ws, line.strip().split()[1])
                            assert os.path.exists(oc_filename), "Package.load() error: open/close filename " + \
                                                                oc_filename + " not found"
                            try:
                                current = np.genfromtxt(oc_filename, dtype=current.dtype)
                                current = current.view(np.recarray)
                            except Exception as e:
                                raise Exception("Package.load() error loading open/close file " + oc_filename + \
                                                " :" + str(e))
                            assert current.shape[0] == itmp, "Package.load() error: open/close rec array from file " + \
                                                             oc_filename + " shape (" + str(current.shape) + \
                                                             ") does not match itmp: {0:d}".format(itmp)
                            break
                        try:
                            t = line.strip().split()
                            current[ibnd] = tuple(t[:len(current.dtype.names)])
                        except:
                            t = []
                            ipos = [5, 5, 5, 5, 5, 15, 10, 10, 10, 10]
                            istart = 0
                            for ivar in range(len(ipos)):
                                istop = istart + ipos(ivar)
                                t.append(line[istart:istop])
                                istart = istop + 1
                            if len(aux_names) > 0:
                                tt = line[istart:].strip().split()
                                for ivar in len(aux_names):
                                    t.append(tt[ivar])
                            current[ibnd] = tuple(t[:len(current.dtype.names)])

                # convert indices to zero-based
                current['k'] -= 1
                current['i'] -= 1
                current['j'] -= 1
                bnd_output = np.recarray.copy(current)

                # read dataset 8
                if icalc > 0:
                    if model.verbose:
                        print("   reading str dataset 8")
                    tds8 = np.zeros((itmp, 3), dtype=np.float)
                    for ibnd in range(itmp):
                        line = f.readline()
                        try:
                            t = line.strip().split()
                            v = [float(vt) for vt in t[:3]]
                        except:
                            v = []
                            ipos = [10, 10, 10]
                            istart = 0
                            for ivar in range(len(ipos)):
                                istop = istart + ipos(ivar)
                                v.append(float(line[istart:istop]))
                                istart = istop + 1
                        tds8[ibnd, :] = np.array(v)
                    rch_data = tds8.copy()
                else:
                    rch_data = None

                # read data set 9
                if ntrib > 0:
                    if model.verbose:
                        print("   reading str dataset 9")
                    tds9 = np.zeros((nss, ntrib), dtype=np.int)
                    for iseg in range(nss):
                        line = f.readline()
                        try:
                            t = line.strip().split()
                            v = [float(vt) for vt in t[:ntrib]]
                        except:
                            v = []
                            ipos = 10
                            istart = 0
                            for ivar in range(ntrib):
                                istop = istart + ipos
                                v.append(float(line[istart:istop]))
                                istart = istop + 1
                        tds9[iseg, :] = np.array(v)
                    seg_data = tds9.copy()
                else:
                    seg_data = None

                # read data set 10
                if ndiv > 0:
                    if model.verbose:
                        print("   reading str dataset 10")
                    tds10 = np.zeros((nss), dtype=np.int)
                    for iseg in range(nss):
                        line = f.readline()
                        try:
                            t = line.strip().split()
                            v = float(t[0])
                        except:
                            ipos = 10
                            istart = 0
                            for ivar in range(ntrib):
                                istop = istart + ipos
                                v = float(line[istart:istop])
                                istart = istop + 1
                        tds10[iseg] = np.array(v)
                    useg_data = tds10.copy()
                else:
                    useg_data = None

            else:
                #bnd_output = np.recarray.copy(current)
                bnd_output = -1
                if icalc > 0:
                    #rch_data = tds8.copy()
                    rch_data = -1
                else:
                    rch_data = None
                if ntrib > 0:
                    #seg_data = tds9.copy()
                    seg_data = -1
                else:
                    seg_data = None
                if ndiv > 0:
                    #useg_data = tds10.copy()
                    useg_data = -1
                else:
                    useg_data = None


            if bnd_output is None:
                stress_period_data[iper] = [itmp, itmp, itmp, itmp]
            else:
                stress_period_data[iper] = [bnd_output, rch_data, seg_data, useg_data]


        strpak = ModflowStr(model, mxacts=mxacts, nss=nss,
                            ntrib=ntrib, ndiv=ndiv, icalc=icalc,
                            const=const, ipakcb=ipakcb,
                            stress_period_data=stress_period_data, options=options)
        return strpak
