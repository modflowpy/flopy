import sys
import numpy as np
import warnings
from ..pakbase import Package
from ..utils import Util2d, MfList, Transient2d

# Note: Order matters as first 6 need logical flag on line 1 of SSM file
SsmLabels = ['WEL', 'DRN', 'RCH', 'EVT', 'RIV', 'GHB', 'BAS6', 'CHD', 'PBC']


class SsmPackage(object):
    def __init__(self, label='', instance=None, needTFstr=False):
        self.label = label
        self.instance = instance
        self.needTFstr = needTFstr
        self.TFstr = ' F'
        if self.instance is not None:
            self.TFstr = ' T'


class Mt3dSsm(Package):
    """
    MT3DMS Source and Sink Mixing Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to which
        this package will be added.
    crch : Transient2d, scalar, array of floats, or dictionary
        CRCH is the concentration of recharge for species 1.
        If the recharge flux is positive, it acts as a source whose
        concentration can be specified as desired. If the recharge flux is
        negative, it acts as a sink (discharge) whose concentration is always
        set equal to the concentration of groundwater at the cell where
        discharge occurs. Note that the location and flow rate of
        recharge/discharge are obtained from the flow model directly through
        the unformatted flow-transport link file.  crch can be specified as
        an array, if the array is constant for the entire simulation.  If
        crch changes by stress period, then the user must provide a
        dictionary, where the key is the stress period number (zero based) and
        the value is the recharge array.  The recharge concentration
        can be specified for additional species by passing additional
        arguments to the Mt3dSsm constructor.  For example, to specify the
        recharge concentration for species two one could use
        crch2={0: 0., 1: 10*np.ones((nrow, ncol), dtype=np.float)} as
        and additional keyword argument that is passed to Mt3dSsm when making
        the ssm object.
    cevt : Transient2d, scalar, array of floats, or dictionary
        is the concentration of evapotranspiration flux for species 1.
        Evapotranspiration is the only type of sink whose
        concentration may be specified externally. Note that the
        concentration of a sink cannot be greater than that of the aquifer at
        the sink cell. Thus, if the sink concentration is specified greater
        than that of the aquifer, it is automatically set equal to the
        concentration of the aquifer. Also note that the location and flow
        rate of evapotranspiration are obtained from the flow model directly
        through the unformatted flow-transport link file.  For multi-species
        simulations, see crch for a description of how to specify
        additional concentrations arrays for each species.
    stress_period_data : dictionary
        Keys in the dictionary are stress zero-based stress period numbers;
        values in the dictionary are recarrays of SSM boundaries.  The
        dtype for the recarray can be obtained using ssm.dtype (after the
        ssm package has been created).  The default dtype for the recarray is
        np.dtype([('k', np.int), ("i", np.int), ("j", np.int),
        ("css", np.float32), ("itype", np.int),
        ((cssms(n), np.float), n=1, ncomp)])
        If there are more than one component species, then additional entries
        will be added to the dtype as indicated by cssm(n).
        Note that if the number of dictionary entries is less than the number
        of stress periods, then the last recarray of boundaries will apply
        until the end of the simulation. Full details of all options to
        specify stress_period_data can be found in the
        flopy3_multi-component_SSM ipython notebook in the Notebook
        subdirectory of the examples directory.
        css is the specified source concentration or mass-loading rate,
        depending on the value of ITYPE, in a single-species simulation,
        (For a multispecies simulation, CSS is not used, but a dummy value
        still needs to be entered here.)
        Note that for most types of sources, CSS is interpreted as the
        source concentration with the unit of mass per unit volume (ML-3),
        which, when multiplied by its corresponding flow rate (L3T-1) from
        the flow model, yields the mass-loading rate (MT-1) of the source.
        For a special type of sources (ITYPE = 15), CSS is taken directly as
        the mass-loading rate (MT-1) of the source so that no flow rate is
        required from the flow model.
        Furthermore, if the source is specified as a constant-concentration
        cell (itype = -1), the specified value of CSS is assigned directly as
        the concentration of the designated cell. If the designated cell is
        also associated with a sink/source term in the flow model, the flow
        rate is not used.
        itype is an integer indicating the type of the point source.  An itype
        dictionary can be retrieved from the ssm object as
        itype = mt3d.Mt3dSsm.itype_dict()
        (CSSMS(n), n=1, NCOMP) defines the concentrations of a point source
        for multispecies simulation with NCOMP>1. In a multispecies
        simulation, it is necessary to define the concentrations of all
        species associated with a point source. As an example, if a chemical
        of a certain species is injected into a multispecies system, the
        concentration of that species is assigned a value greater than zero
        while the concentrations of all other species are assigned zero.
        CSSMS(n) can be entered in free format, separated by a comma or space
        between values.
        Several important notes on assigning concentration for the
        constant-concentration condition (ITYPE = -1) are listed below:
        The constant-concentration condition defined in this input file takes
        precedence to that defined in the Basic Transport Package input file.
        In a multiple stress period simulation, a constant-concentration
        cell, once defined, will remain a constant- concentration cell in the
        duration of the simulation, but its concentration value can be
        specified to vary in different stress periods.
        In a multispecies simulation, if it is only necessary to define
        different constant-concentration conditions for selected species at
        the same cell location, specify the desired concentrations for those
        species, and assign a negative value for all other species. The
        negative value is a flag used by MT3DMS to skip assigning the
        constant-concentration condition for the designated species.
    dtype : np.dtype
        dtype to use for the recarray of boundaries.  If left as None (the
        default) then the dtype will be automatically constructed.
    extension : string
        Filename extension (default is 'ssm')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.

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
    >>> m = flopy.mt3d.Mt3dms()
    >>> itype = mt3d.Mt3dSsm.itype_dict()
    >>> ssm_data = {}
    >>> ssm_data[0] = [(4, 4, 4, 1.0, itype['GHB'], 1.0, 100.0)]
    >>> ssm_data[5] = [(4, 4, 4, 0.5, itype['GHB'], 0.5, 200.0)]
    >>> ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

    """

    def __init__(self, model, crch=None, cevt=None, mxss=None,
                 stress_period_data=None, dtype=None,
                 extension='ssm', unitnumber=None, filenames=None,
                 **kwargs):

        if unitnumber is None:
            unitnumber = Mt3dSsm.defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dSsm.reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dSsm.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        deprecated_kwargs = ['criv', 'cghb', 'cibd', 'cchd', 'cpbc', 'cwel']
        for key in kwargs:
            if (key in deprecated_kwargs):
                warnings.warn("Deprecation Warning: Keyword argument '" + key +
                              "' no longer supported. Use " +
                              "'stress_period_data' instead.")

        # Set dimensions
        mf = self.parent.mf
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp

        # Create a list of SsmPackage (class defined above)
        self.__SsmPackages = []
        if mf is not None:
            for i, label in enumerate(SsmLabels):
                mfpack = mf.get_package(label)
                ssmpack = SsmPackage(label, mfpack, (i < 6))
                self.__SsmPackages.append(
                    ssmpack)  # First 6 need T/F flag in file line 1

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        if stress_period_data is None:
            self.stress_period_data = None
        else:
            self.stress_period_data = MfList(self, model=model,
                                             data=stress_period_data,
                                             list_free_format=False)

        if mxss is None and mf is None:
            warnings.warn('SSM Package: mxss is None and modflowmodel is ' +
                          'None.  Cannot calculate max number of sources ' +
                          'and sinks.  Estimating from stress_period_data. ')

        if mxss is None:
            # Need to calculate max number of sources and sinks
            self.mxss = 0
            if self.stress_period_data is not None:
                self.mxss += np.sum(
                    self.stress_period_data.data[0].itype == -1)
                self.mxss += np.sum(
                    self.stress_period_data.data[0].itype == -15)

            if isinstance(self.parent.btn.icbund, np.ndarray):
                self.mxss += (self.parent.btn.icbund < 0).sum()

            for p in self.__SsmPackages:
                if ((p.label == 'BAS6') and (p.instance != None)):
                    self.mxss += (p.instance.ibound.array < 0).sum()
                elif p.instance != None:
                    self.mxss += p.instance.ncells()
        else:
            self.mxss = mxss

        # Note: list is used for multi-species, NOT for stress periods!
        self.crch = None
        try:
            if crch is None and model.mf.rch is not None:
                print("found 'rch' in modflow model, resetting crch to 0.0")
                crch = 0.0
        except:
            pass
        if crch is not None:

            self.crch = []
            t2d = Transient2d(model, (nrow, ncol), np.float32,
                              crch, name='crch1',
                              locat=self.unit_number[0],
                              array_free_format=False)
            self.crch.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    val = 0.0
                    name = "crch" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print("SSM: setting crch for component " + \
                              str(icomp) + " to zero. kwarg name " + \
                              name)
                    t2d = Transient2d(model, (nrow, ncol), np.float32,
                                      val, name=name,
                                      locat=self.unit_number[0],
                                      array_free_format=False)
                    self.crch.append(t2d)
        # else:
        #     try:
        #         if model.mf.rch is not None:
        #             print("found 'rch' in modflow model, resetting crch to 0.0")
        #             self.crch = [Transient2d(model, (nrow, ncol), np.float32,
        #                           0, name='crch1',
        #                           locat=self.unit_number[0],
        #                           array_free_format=False)]
        #
        #         else:
        #             self.crch = None
        #     except:
        #         self.crch = None

        self.cevt = None
        try:
            if cevt is None and (model.mf.evt is not None or model.mf.ets is not None):
                print("found 'ets'/'evt' in modflow model, resetting cevt to 0.0")
                cevt = 0.0
        except:
            pass
        if cevt is not None:
            self.cevt = []
            t2d = Transient2d(model, (nrow, ncol), np.float32,
                              cevt, name='cevt1',
                              locat=self.unit_number[0],
                              array_free_format=False)
            self.cevt.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    val = 0.0
                    name = "cevt" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs[name]
                        kwargs.pop(name)
                    else:
                        print("SSM: setting cevt for component " + \
                              str(icomp) + " to zero, kwarg name " + \
                              name)
                    t2d = Transient2d(model, (nrow, ncol), np.float32,
                                      val, name=name,
                                      locat=self.unit_number[0],
                                      array_free_format=False)
                    self.cevt.append(t2d)

        # else:
        #     try:
        #         if model.mf.evt is not None or model.mf.ets is not None:
        #             print("found 'ets'/'evt' in modflow model, resetting cevt to 0.0")
        #             self.cevt = [Transient2d(model, (nrow, ncol), np.float32,
        #                                     0, name='cevt1',
        #                                     locat=self.unit_number[0],
        #                                     array_free_format=False)]
        #
        #         else:
        #             self.cevt = None
        #     except:
        #         self.cevt = None

        if len(list(kwargs.keys())) > 0:
            raise Exception("SSM error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))

        # Add self to parent and return
        self.parent.add_package(self)
        return

    def from_package(self, package, ncomp_aux_names):
        """
        read the point source and sink info from a package
        ncomp_aux_names (list): the aux variable names in the package
        that are the component concentrations
        """
        raise NotImplementedError()

    @staticmethod
    def itype_dict():
        itype = {}
        itype["CHD"] = 1
        itype["BAS6"] = 1
        itype["PBC"] = 1
        itype["WEL"] = 2
        itype["DRN"] = 3
        itype["RIV"] = 4
        itype["GHB"] = 5
        itype["MAS"] = 15
        itype["CC"] = -1
        return itype

    @staticmethod
    def get_default_dtype(ncomp=1):
        """
        Construct a dtype for the recarray containing the list of sources
        and sinks
        """
        type_list = [("k", np.int), ("i", np.int), ("j", np.int),
                     ("css", np.float32), ("itype", np.int)]
        if ncomp > 1:
            for comp in range(1, ncomp + 1):
                comp_name = "cssm({0:02d})".format(comp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
        return dtype

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_ssm = open(self.fn_path, 'w')
        for p in self.__SsmPackages:
            if p.needTFstr:
                f_ssm.write(p.TFstr)

        f_ssm.write(' F F F F F F F F F F\n')

        f_ssm.write('{:10d}\n'.format(self.mxss))

        # Loop through each stress period and write ssm information
        nper = self.parent.nper
        for kper in range(nper):
            if f_ssm.closed == True:
                f_ssm = open(f_ssm.name, 'a')

            # Distributed sources and sinks (Recharge and Evapotranspiration)
            if self.crch is not None:
                # If any species need to be written, then all need to be
                # written
                incrch = -1
                for t2d in self.crch:
                    incrchicomp, file_entry = t2d.get_kper_entry(kper)
                    incrch = max(incrch, incrchicomp)
                    if incrch == 1:
                        break
                f_ssm.write('{:10d}\n'.format(incrch))
                if incrch == 1:
                    for t2d in self.crch:
                        u2d = t2d[kper]
                        file_entry = u2d.get_file_entry()
                        f_ssm.write(file_entry)

            if self.cevt is not None:
                # If any species need to be written, then all need to be
                # written
                incevt = -1
                for t2d in self.cevt:
                    incevticomp, file_entry = t2d.get_kper_entry(kper)
                    incevt = max(incevt, incevticomp)
                    if incevt == 1:
                        break
                f_ssm.write('{:10d}\n'.format(incevt))
                if incevt == 1:
                    for t2d in self.cevt:
                        u2d = t2d[kper]
                        file_entry = u2d.get_file_entry()
                        f_ssm.write(file_entry)

            # List of sources
            if self.stress_period_data is not None:
                self.stress_period_data.write_transient(f_ssm, single_per=kper)
            else:
                f_ssm.write('{}\n'.format(0))

        f_ssm.close()
        return

    @staticmethod
    def load(f, model, nlay=None, nrow=None, ncol=None, nper=None,
             ncomp=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        ssm :  Mt3dSsm object
            Mt3dSsm object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> ssm = flopy.mt3d.Mt3dSsm.load('test.ssm', mt)

        """

        if model.verbose:
            sys.stdout.write('loading ssm package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set modflow model and dimensions if necessary
        mf = model.mf
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol
        if nper is None:
            nper = model.nper
        if ncomp is None:
            ncomp = model.ncomp

        # dtype
        dtype = Mt3dSsm.get_default_dtype(ncomp)

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # Item D1: Dummy input line - line already read above
        if model.verbose:
            print(
                '   loading FWEL, FDRN, FRCH, FEVT, FRIV, FGHB, (FNEW(n), n=1,4)...')
        fwel = line[0:2]
        fdrn = line[2:4]
        frch = line[4:6]
        fevt = line[6:8]
        friv = line[8:10]
        fghb = line[10:12]
        if len(line) >= 14:
            fnew1 = line[12:14]
        else:
            fnew1 = 'F'
        if len(line) >= 16:
            fnew2 = line[14:16]
        else:
            fnew2 = 'F'
        if len(line) >= 18:
            fnew3 = line[16:18]
        else:
            fnew3 = 'F'
        if len(line) >= 20:
            fnew4 = line[18:20]
        else:
            fnew4 = 'F'
        if model.verbose:
            print('   FWEL {}'.format(fwel))
            print('   FDRN {}'.format(fdrn))
            print('   FRCH {}'.format(frch))
            print('   FEVT {}'.format(fevt))
            print('   FRIV {}'.format(friv))
            print('   FGHB {}'.format(fghb))
            print('   FNEW1 {}'.format(fnew1))
            print('   FNEW2 {}'.format(fnew2))
            print('   FNEW3 {}'.format(fnew3))
            print('   FNEW4 {}'.format(fnew4))

        # Override the logical settings at top of ssm file using the
        # modflowmodel, if it is attached to parent
        if mf is not None:
            rchpack = mf.get_package('RCH')
            if rchpack is not None:
                frch = 't'
            evtpack = mf.get_package('EVT')
            if evtpack is not None:
                fevt = 't'

        # Item D2: MXSS, ISSGOUT
        mxss = None
        if model.verbose:
            print('   loading MXSS, ISSGOUT...')
        line = f.readline()
        mxss = int(line[0:10])
        try:
            issgout = int(line[10:20])
        except:
            issgout = 0
        if model.verbose:
            print('   MXSS {}'.format(mxss))
            print('   ISSGOUT {}'.format(issgout))

        # kwargs needed to construct crch2, crch3, etc. for multispecies
        kwargs = {}

        crch = None
        if 't' in frch.lower():
            t2d = Transient2d(model, (nrow, ncol), np.float32,
                              0.0, name='crch', locat=0,
                              array_free_format=False)
            crch = {0: t2d}
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "crch" + str(icomp)
                    t2d = Transient2d(model, (nrow, ncol), np.float32,
                                      0.0, name=name, locat=0,
                                      array_free_format=False)
                    kwargs[name] = {0: t2d}

        cevt = None
        if 't' in fevt.lower():
            t2d = Transient2d(model, (nrow, ncol), np.float32,
                              0.0, name='cevt', locat=0,
                              array_free_format=False)
            cevt = {0: t2d}
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "cevt" + str(icomp)
                    t2d = Transient2d(model, (nrow, ncol), np.float32,
                                      0.0, name=name, locat=0,
                                      array_free_format=False)
                    kwargs[name] = {0: t2d}

        stress_period_data = {}

        for iper in range(nper):

            if model.verbose:
                print("   loading ssm for kper {0:5d}".format(iper + 1))

            # Item D3: INCRCH
            incrch = -1
            if 't' in frch.lower():
                if model.verbose:
                    print('   loading INCRCH...')
                line = f.readline()
                incrch = int(line[0:10])

            # Item D4: CRCH
            if incrch >= 0:
                if model.verbose:
                    print('   loading CRCH...')
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'crch',
                                ext_unit_dict, array_format="mt3d")
                crch[iper] = t
                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = "crch" + str(icomp)
                        if model.verbose:
                            print('   loading {}...'.format(name))
                        t = Util2d.load(f, model, (nrow, ncol),
                                        np.float32, name, ext_unit_dict,
                                        array_format="mt3d")
                        crchicomp = kwargs[name]
                        crchicomp[iper] = t

            # Item D5: INCEVT
            incevt = -1
            if 't' in fevt.lower():
                if model.verbose:
                    print('   loading INCEVT...')
                line = f.readline()
                incevt = int(line[0:10])

            # Item D6: CEVT
            if incevt >= 0:
                if model.verbose:
                    print('   loading CEVT...')
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'cevt',
                                ext_unit_dict, array_format="mt3d")
                cevt[iper] = t
                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = "cevt" + str(icomp)
                        if model.verbose:
                            print('   loading {}...'.format(name))
                        t = Util2d.load(f, model, (nrow, ncol),
                                        np.float32, name, ext_unit_dict,
                                        array_format="mt3d")
                        cevticomp = kwargs[name]
                        cevticomp[iper] = t

            # Item D7: NSS
            if model.verbose:
                print('   loading NSS...')
            line = f.readline()
            nss = int(line[0:10])

            # Item D8: KSS, ISS, JSS, CSS, ITYPE, (CSSMS(n),n=1,NCOMP)
            if model.verbose:
                print(
                    '   loading KSS, ISS, JSS, CSS, ITYPE, (CSSMS(n),n=1,NCOMP)...')
            current = 0
            if nss > 0:
                current = np.empty((nss), dtype=dtype)
                for ibnd in range(nss):
                    line = f.readline()
                    t = []
                    for ivar in range(5):
                        istart = ivar * 10
                        istop = istart + 10
                        t.append(line[istart:istop])
                    ncssms = len(current.dtype.names) - 5
                    if ncssms > 0:
                        tt = line[istop:].strip().split()
                        for ivar in range(ncssms):
                            t.append(tt[ivar])
                    current[ibnd] = tuple(t[:len(current.dtype.names)])
                # convert indices to zero-based
                current['k'] -= 1
                current['i'] -= 1
                current['j'] -= 1
                current = current.view(np.recarray)
            stress_period_data[iper] = current

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=Mt3dSsm.ftype())

        # Construct and return ssm package
        ssm = Mt3dSsm(model, crch=crch, cevt=cevt, mxss=mxss,
                      stress_period_data=stress_period_data,
                      unitnumber=unitnumber, filenames=filenames, **kwargs)
        return ssm

    @staticmethod
    def ftype():
        return 'SSM'

    @staticmethod
    def defaultunit():
        return 34

    @staticmethod
    def reservedunit():
        return 4
