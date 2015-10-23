import sys
import numpy as np
import warnings
from flopy.mbase import Package
from flopy.utils import util_2d
from flopy.utils.util_list import mflist
from flopy.utils.util_array import transient_2d

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
    crch : array of floats
        CRCH is the concentration of recharge flux for a particular species.
        If the recharge flux is positive, it acts as a source whose
        concentration can be specified as desired. If the recharge flux is
        negative, it acts as a sink (discharge) whose concentration is always
        set equal to the concentration of groundwater at the cell where
        discharge occurs. Note that the location and flow rate of
        recharge/discharge are obtained from the flow model directly through
        the unformatted flow-transport link file.
    cevt : array of floats
        is the concentration of evapotranspiration flux for a particular
        species. Evapotranspiration is the only type of sink whose
        concentration may be specified externally. Note that the
        concentration of a sink cannot be greater than that of the aquifer at
        the sink cell. Thus, if the sink concentration is specified greater
        than that of the aquifer, it is automatically set equal to the
        concentration of the aquifer. Also note that the location and flow
        rate of evapotranspiration are obtained from the flow model directly
        through the unformatted flow-transport link file.
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
        File unit number (default is 34).

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
    unitnumber = 34
    def __init__(self, model, crch=None, cevt=None, mxss=None,
                 modflowmodel=None, stress_period_data=None, dtype=None,
                 extension='ssm', unitnumber=None, **kwargs):

        if unitnumber is None:
            unitnumber = self.unitnumber
        Package.__init__(self, model, extension, 'SSM', unitnumber)

        deprecated_kwargs = ['criv', 'cghb', 'cibd', 'cchd', 'cpbc', 'cwel'] 
        for key in kwargs:
            if (key in deprecated_kwargs):
                warnings.warn("Deprecation Warning: Keyword argument '" + key +
                              "' no longer supported. Use " +
                              "'stress_period_data' instead.")

        # Set dimensions
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp

        self.__SsmPackages = []
        if modflowmodel is not None:
            for i, label in enumerate(SsmLabels):
                self.__SsmPackages.append(SsmPackage(label,
                                   modflowmodel.get_package(label),
                                   (i < 6))) # First 6 need T/F flag in file line 1

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        self.stress_period_data = mflist(self, model=model,
                                                 data=stress_period_data)

        if mxss is None and modflowmodel is None:
            warnings.warn('SSM Package: mxss is None and modflowmodel is ' +
                          'None.  Cannot calculate max number of sources ' +
                          'and sinks.  Estimating from stress_period_data. ')

        if mxss is None:
            # Need to calculate max number of sources and sinks
            self.mxss = np.sum(self.stress_period_data.data[0].itype == -1)
            self.mxss += np.sum(self.stress_period_data.data[0].itype == -15)

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
        if crch is not None:
            self.crch = []
            t2d = transient_2d(model, (nrow, ncol), np.float32,
                               crch, name='crch1',
                               locat=self.unit_number[0])
            self.crch.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = "crch" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs[name]
                        kwargs.pop(name)
                    else:
                        print("SSM: setting crch for component " +\
                              str(icomp) + " to zero. kwarg name " +\
                              name)
                    t2d = transient_2d(model, (nrow, ncol), np.float32,
                                       val, name=name,
                                       locat=self.unit_number[0])
                    self.crch.append(t2d)
        else:
            self.crch = None

        if (cevt != None):
            self.cevt = []
            t2d = transient_2d(model, (nrow, ncol), np.float32,
                               cevt, name='cevt1',
                               locat=self.unit_number[0])
            self.cevt.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = "cevt" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs[name]
                        kwargs.pop(name)
                    else:
                        print("SSM: setting cevt for component " +\
                              str(icomp) + " to zero, kwarg name " +\
                              name)
                    t2d = transient_2d(model, (nrow, ncol), np.float32,
                                       val, name=name,
                                       locat=self.unit_number[0])
                    self.cevt.append(t2d)

        else:
            self.cevt = None

        if len(list(kwargs.keys())) > 0:
            raise Exception("SSM error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))


        #Add self to parent and return
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
            for comp in range(1,ncomp+1):
                comp_name = "cssm({0:02d})".format(comp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
        return dtype

    def write_file(self):
        """
        Write the SSM file

        """

        # Open file for writing
        f_ssm = open(self.fn_path, 'w')
        for p in self.__SsmPackages:
            if p.needTFstr:
                f_ssm.write(p.TFstr)
        f_ssm.write(' F F F F\n')
        f_ssm.write('{:10d}\n'.format(self.mxss))
        
        # Loop through each stress period and write ssm information
        nper = self.parent.nper
        for kper in range(nper):
            if f_ssm.closed == True:
                f_ssm = open(f_ssm.name,'a')

            # Distributed sources and sinks (Recharge and Evapotranspiration)
            if self.crch is not None:
                for c, t2d in enumerate(self.crch):
                    incrch, file_entry = t2d.get_kper_entry(kper)
                    if (c == 0):
                        f_ssm.write('{:10i}\n'.format(incrch))
                    f_ssm.write(file_entry)

            if (self.cevt != None):
                for c, t2d in enumerate(self.cevt):
                    incevt, file_entry = t2d.get_kper_entry(kper)
                    if (c == 0):
                        f_ssm.write('{:10i}\n'.format(incevt))
                    f_ssm.write(file_entry)

            # List of sources
            self.stress_period_data.write_transient(f_ssm, single_per=kper)

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
        >>> ssm = flopy.mt3d.Mt3dSsm.load('test.ssm', m)

        """

        if model.verbose:
            sys.stdout.write('loading ssm package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set dimensions if necessary
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
            print('   loading FWEL, FDRN, FRCH, FEVT, FRIV, FGHB, (FNEW(n), n=1,4)...')
        fwel = line[0:2]
        fdrn = line[4:6]
        frch = line[6:8]
        fevt = line[8:10]
        friv = line[10:12]
        fghb = line[12:14]
        fnew1 = line[14:16]
        fnew2 = line[16:18]
        fnew3 = line[18:20]
        fnew4 = line[20:22]
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


        crch = None
        if 't' in frch.lower():
            crch = {0:0}

        cevt = None
        if 't' in fevt.lower():
            cevt = {0:0}

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
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'crch',
                                 ext_unit_dict)
                crch[iper] = t

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
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'cevt',
                                 ext_unit_dict)
                cevt[iper] = t

            # Item D7: NSS
            if model.verbose:
                print('   loading NSS...')
            line = f.readline()
            nss = int(line[0:10])

            # Item D8: KSS, ISS, JSS, CSS, ITYPE, (CSSMS(n),n=1,NCOMP)
            if model.verbose:
                print('   loading KSS, ISS, JSS, CSS, ITYPE, (CSSMS(n),n=1,NCOMP)...')
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

        # Construct and return ssm package
        ssm = Mt3dSsm(model, crch=crch, cevt=cevt, mxss=mxss,
                      stress_period_data=stress_period_data)
        return ssm


