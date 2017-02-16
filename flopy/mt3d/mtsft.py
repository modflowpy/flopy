__author__ = 'emorway'

import sys
import numpy as np

from ..pakbase import Package
from flopy.utils import Util2d, Util3d, read1d, MfList
class Mt3dSft(Package):
    """
    MT3D-USGS StreamFlow Transport package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    nsfinit : int
        Is the number of simulated stream reaches (in SFR2, the number of
        stream reaches is greater than or equal to the number of stream
        segments).  This is equal to NSTRM found on the first line of the
        SFR2 input file.  If NSFINIT > 0 then surface-water transport is
        solved in the stream network while taking into account groundwater
        exchange and precipitation and evaporation sources and sinks.
        Otherwise, if NSFINIT < 0, the surface-water network as represented
        by the SFR2 flow package merely acts as a boundary condition to the
        groundwater transport problem; transport in the surface-water
        network is not simulated.
    mxsfbc : int
        Is the maximum number of stream boundary conditions.
    icbcsf : int
        Is an integer value that directs MT3D-USGS to write reach-by-reach
        concentration information to unit ICBCSF.
    ioutobs : int
        Is the unit number of the output file for simulated concentrations at
        specified gage locations.  The NAM file must also list the unit
        number to which observation information will be written.
    ietsfr : int
        Specifies whether or not mass will exit the surface-water network
        with simulated evaporation.  If IETSFR = 0, then mass does not leave
        via stream evaporation.  If IETSFR > 0, then mass is allowed to exit
        the simulation with the simulated evaporation.
    isfsolv : int
        Specifies the numerical technique that will be used to solve the
        transport problem in the surface water network.  The first release
        of MT3D-USGS (version 1.0) only allows for a finite-difference
        formulation and regardless of what value the user specifies, the
        variable defaults to 1, meaning the finite-difference solution is
        invoked.
    wimp : float
        Is the stream solver time weighting factor.  Ranges between 0.0 and
        1.0.  Values of 0.0, 0.5, or 1.0 correspond to explicit,
        Crank-Nicolson, and fully implicit schemes, respectively.
    wups : float
        Is the space weighting factor employed in the stream network solver.
        Ranges between 0.0 and 1.0.  Values of 0.0 and 1.0 correspond to a
        central-in-space and upstream weighting factors, respectively.
    cclosesf : float
        Is the closure criterion for the SFT solver
    mxitersf : int
        Limits the maximum number of iterations the SFT solver can use to
        find a solution of the stream transport problem.
    crntsf : float
        Is the Courant constraint specific to the SFT time step, its value
        has no bearing upon the groundwater transport solution time step.
    iprtxmd : int
        A flag to print SFT solution information to the standard output file.
        IPRTXMD = 0 means no SFT solution information is printed;
        IPRTXMD = 1 means SFT solution summary information is printed at the
        end of every MT3D-USGS outer iteration; and IPRTXMD = 2 means SFT
        solution details are written for each SFT outer iteration that
        calls the xMD solver that solved SFT equations.
    coldsf : array of floats
        Represents the initial concentrations in the surface water network.
        The length of the array is equal to the number of stream reaches and
        starting concentration values should be entered in the same order
        that individual reaches are entered for record set 2 in the SFR2
        input file.
    dispsf : array of floats
        Is the dispersion coefficient [L2 T-1] for each stream reach in the
        simulation and can vary for each simulated component of the
        simulation.  That is, the length of the array is equal to the number
        of simulated stream reaches times the number of simulated components.
        Values of dispersion for each reach should be entered in the same
        order that individual reaches are entered for record set 2 in the
        SFR2 input file.  The first NSTRM entries correspond to NCOMP = 1,
        with subsequent entries for each NCOMP simulated species.
    nobssf : int
        Specifies the number of surface flow observation points for
        monitoring simulated concentrations in streams.
    isobs : int
        The segment number for each stream flow concentration observation
        point.
    irobs : int
        The reach number for each stream flow concentration observation point.
    ntmp : int
        The number of specified stream boundary conditions to follow.  For
        the first stress period, this value must be greater than or equal to
        zero, but may be less than zero in subsequent stress periods.
    isegbc : int
        Is the segment number for which the current boundary condition will
        be applied.
    irchbc : int
        Is the reach number for which the current boundary condition will be
        applied.
    isfbctyp : int
        Specifies, for ISEGBC/IRCHBC, what the boundary condition type is
           0   A headwater boundary.  That is, for streams entering at the
               boundary of the simulated domain that need a specified
               concentration, use ISFBCTYP = 0
           1   a precipitation boundary. If precipitation directly to
               channels is simulated in the flow model and a non-zero
               concentration (default is zero) is desired, use ISFBCTYP = 1
           2   a runoff boundary condition that is not the same thing as
               runoff simulated in the UZF1 package and routed to a stream
               (or lake) using the IRNBND array.  Users who specify runoff
               in the SFR2 input via the RUNOFF variable appearing in either
               record sets 4b or 6a and want to assign a non-zero
               concentration (default is zero) associated with this specified
               source, use ISFBCTYP=2;
           3   a constant-concentration boundary.  Any ISEGBC/IRCHBC
               combination may set equal to a constant concentration boundary
               condition.
           4   a pumping boundary condition.
           5   an evaporation boundary condition.  In models where
               evaporation is simulated directly from the surface of the
               channel, users can use this boundary condition to specify a
               non-zero concentration (default is zero) associated with the
               evaporation losses.
    cbcsf : float
        Is the specified concentration associated with the current boundary
        condition entry.  Repeat CBCSF for each simulated species (NCOMP).

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

    >>> mf = flopy.modflow.Modflow.load('CrnkNic_mf.nam',
    >>>                                  load_only=['dis', 'bas6'])
    >>> sfr = flopy.modflow.ModflowSfr2.load('CrnkNic.sfr2', mf)
    >>> chk = sfr.check()

    >>> # initialize an MT3D-USGS model
    >>> mt = flopy.mt3d.Mt3dms.load('CrnkNic_mt.nam',
    >>>        exe_name = 'mt3d-usgs_1.0.00.exe',
    >>>        model_ws = r'.\CrnkNic',
    >>>        load_only='btn')
    >>> sft = flopy.mt3d.Mt3dSft.load(mt, 'CrnkNic.sft')


    """

    unitnumber = 46
    def __init__(self, model, nsfinit=0, mxsfbc=0, icbcsf=0, ioutobs=None,
                 ietsfr=0, isfsolv=1, wimp=0.50, wups=1.00, cclosesf=1.0E-6,
                 mxitersf=10, crntsf=1.0, iprtxmd=0, coldsf=0.0, dispsf=0.0,
                 nobssf=0, obs_sf=None, sf_stress_period_data = None,
                 filenames=None, dtype=None, extension='sft',unit_number=None, **kwargs):

        #unit number
        # set default unit number of one is not specified
        if unit_number is None:
            unit_number = Mt3dSft.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 3:
                for idx in range(len(filenames), 3):
                    filenames.append(None)

        Package.__init__(self, model, extension, 'SFT', unit_number)

        # Set dimensions
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp
        mcomp = model.mcomp

        # Set package specific parameters
        self.nsfinit = nsfinit
        self.mxsfbc = mxsfbc
        self.icbcsf = icbcsf
        self.ioutobs = ioutobs


        # add sft observation output file
        if ioutobs is not None:
            if abs(ioutobs) > 0:
                ext = 'sftobs'
                #if ioutobs < 0:  # no support for this yet in MT3D-USGS
                #    binflag = True
                #    ext = 'bin'
                fname = filenames[2]
                model.add_output_file(abs(ioutobs), fname=fname, extension=ext,
                                      binflag=False, package=Mt3dSft.ftype())
        else:
            ioutobs = 0

        self.ietsfr = ietsfr
        self.isfsolv = isfsolv
        self.wimp = wimp
        self.wups = wups
        self.cclosesf = cclosesf
        self.mxitersf = mxitersf
        self.crntsf = crntsf
        self.iprtxmd = iprtxmd

        # Set 1D array values
        if isinstance(coldsf, (list, np.ndarray)):
            locat = 19
        else:
            locat = self.unit_number[0]

        self.coldsf = Util2d(model, (nsfinit,), np.float32, coldsf,
                             name='coldsf', locat=locat)

        if isinstance(dispsf, (list, np.ndarray)):
            locat = 19
        else:
            locat = self.unit_number[0]

        self.dispsf = Util2d(model, (nsfinit,), np.float32, dispsf,
                             name='dispsf', locat=locat)

        # Set streamflow observation locations
        self.nobssf = nobssf
        self.obs_sf = obs_sf

        # Read and set transient data
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        if sf_stress_period_data is None:
            self.sf_stress_period_data = None
        else:
            self.sf_stress_period_data = MfList(self, model=model,
                                                data=sf_stress_period_data)

        self.parent.add_package(self)
        return

    @staticmethod
    def get_default_dtype(ncomp=1):
        """
        Construct a dtype for the recarray containing the list of surface
        water boundary conditions.
        """
        type_list = [("isegbc", np.int), ("irchbc", np.int), \
                     ("isfbctyp", np.float32)]
        if ncomp > 1:
            for comp in range(1,ncomp+1):
                comp_name = "cbcsf{0:d}".format(comp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
        return dtype

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        Examples
        --------
        >>> mt = flopy.mt3d.Mt3dms.load('CrnkNic_mt.nam',
        >>>      exe_name = 'mt3d-usgs_1.0.00.exe',
        >>>      model_ws = r'C:\EDM_LT\GitHub\mt3d-usgs\autotest\temp\CrnkNic',
        >>>      verbose=True)
        >>> mt.name = 'CrnkNic_rewrite'
        >>>
        >>> mt.sft.dispsf.fmtin = '(10F12.2)'
        >>> mt.write_input()

        """

        # Open file for writing
        f_sft = open(self.fn_path, 'w')

        # Item 1
        f_sft.write('{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}'.format(self.nsfinit,
                    self.mxsfbc, self.icbcsf, self.ioutobs, self.ietsfr) +
                    '                              # nsfinit, mxsfbc, icbcsf, ioutobs, ietsfr\n')

        # Item 2
        f_sft.write('{0:10d}{1:10.5f}{2:10.5f}{3:10.7f}{4:10d}{5:10.5f}{6:10d}'
                    .format(self.isfsolv, self.wimp, self.wups, self.cclosesf,
                            self.mxitersf, self.crntsf, self.iprtxmd) +
                    ' # isfsolv, wimp, wups, cclosesf, mxitersf, crntsf, ' \
                    'iprtxmd\n')

        # Item 3
        f_sft.write(self.coldsf.get_file_entry())

        # Item 4
        f_sft.write(self.dispsf.get_file_entry())

        # Item 5
        f_sft.write('{0:10d}                 # nobssf\n'.format(self.nobssf))

        # Item 6
        if self.nobssf != 0:
            for iobs in range(self.nobssf):
                f_sft.write('{0:10d}                          # location of obs as given by position in list of irch\n'
                            .format(self.obs_sf[iobs]))

        # Items 7, 8
        # Loop through each stress period and assign source & sink concentrations to stream features
        nper = self.parent.nper
        for kper in range(nper):
            if f_sft.closed == True:
                f_sft = open(f_sft.name, 'a')

            # List of concentrations associated with various boundaries
            # interacting with the stream network.
            if self.sf_stress_period_data is not None:
                self.sf_stress_period_data.write_transient(f_sft,
                                                           single_per=kper)
            else:
                f_sft.write('{0:10d}       # ntmp - SP {1:5d}'.format(0, kper))

        f_sft.close()
        return

    @staticmethod
    def load(f, model, nsfinit=None, nper=None, ncomp=None,
             ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        nsfinit : int
            number of simulated stream reaches in the surface-water transport
            process.
        isfsolv : int
            Specifies the numerical technique that will be used to solve the
            transport problem in the surface water network.  The first release
            of MT3D-USGS (version 1.0) only allows for a finite-difference
            formulation and regardless of what value the user specifies, the
            variable defaults to 1, meaning the finite-difference solution is
            invoked.
        wimp : float
            Is the stream solver time weighting factor.  Ranges between 0.0
            and 1.0.  Values of 0.0, 0.5, or 1.0 correspond to explicit,
            Crank-Nicolson, and fully implicit schemes, respectively.
        wups : float
            Is the space weighting factor employed in the stream network
            solver. Ranges between 0.0 and 1.0.  Values of 0.0 and 1.0
            correspond to a central-in-space and upstream weighting factors,
            respectively.
        cclosesf : float
            Is the closure criterion for the SFT solver
        mxitersf : int
            Limits the maximum number of iterations the SFT solver can use to
            find a solution of the stream transport problem.
        crntsf : float
            Is the Courant constraint specific to the SFT time step, its value
            has no bearing upon the groundwater transport solution time step.
        iprtxmd : int
            a flag to print SFT solution information to the standard output
            file. IPRTXMD can equal 0, 1, or 2, and will write increasing
            amounts of solver information to the standard output file,
            respectively.

        Returns
        -------
        sft : MT3D-USGS object
            MT3D-USGS object

        Examples
        --------

        >>> import os
        >>> import flopy

        >>> os.chdir(r'C:\temp\flopy_test\sfr_test')
        >>> mf = flopy.modflow.Modflow.load('CrnkNic_mf.nam', load_only=['dis', 'bas6'])
        >>> sfr = flopy.modflow.ModflowSfr2.load('CrnkNic.sfr2', mf)
        >>> chk = sfr.check()

        >>> mt = flopy.mt3d.Mt3dms.load('CrnkNic_mt.nam', exe_name = 'mt3d-usgs_1.0.00.exe', load_only='btn')
        >>> sft = flopy.mt3d.Mt3dSft.load('CrnkNic.sft', mt)

        """
        if model.verbose:
            sys.stdout.write('loading sft package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set default nlay values
        nlay = None
        nrow = None
        ncol = None

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

        dtype = Mt3dSft.get_default_dtype(ncomp)

        # Item 1 (NSFINIT, MXSFBC, ICBCSF, IOUTOBS, IETSFR)
        line = f.readline()
        if line[0] == '#':
            if model.verbose:
                print('   SFT package currently does not support comment ' \
                      'lines...')
                sys.exit()

        if model.verbose:
            print('   loading nsfinit, mxsfbc, icbcsf, ioutobs, ietsfr...')
        vals = line.strip().split()

        nsfinit = int(vals[0])
        mxsfbc = int(vals[1])
        icbcsf = int(vals[2])
        ioutobs = int(vals[3])
        ietsfr = int(vals[4])

        if model.verbose:
            print('   NSFINIT {}'.format(nsfinit))
            print('   MXSFBC {}'.format(mxsfbc))
            print('   ICBCSF {}'.format(icbcsf))
            print('   IOUTOBS {}'.format(ioutobs))
            print('   IETSFR {}'.format(ietsfr))
            if ietsfr == 0:
                print('   Mass does not exit the model via simulated ' \
                      'stream evaporation ')
            else:
                print('   Mass exits the stream network via simulated ' \
                      'stream evaporation ')

        # Item 2 (ISFSOLV, WIMP, WUPS, CCLOSESF, MXITERSF, CRNTSF, IPRTXMD)
        line = f.readline()
        if model.verbose:
            print('   loading isfsolv, wimp, wups, cclosesf, mxitersf, ' \
                  'crntsf, iprtxmd...')

        vals = line.strip().split()

        if len(vals) < 7 and model.verbose:
            print('   not enough values specified in item 2 of SFT input \
                      file, exiting...')
            sys.exit()
        else:
            isfsolv = int(vals[0])
            wimp = float(vals[1])
            wups = float(vals[2])
            cclosesf = float(vals[3])
            mxitersf = int(vals[4])
            crntsf = float(vals[5])
            iprtxmd = int(vals[6])
        if isfsolv != 1:
            isfsolv = 1
            print('   Resetting isfsolv to 1')
            print('   In version 1.0 of MT3D-USGS, isfsov=1 is only option')

        if model.verbose:
            print('   ISFSOLV {}'.format(isfsolv))
            print('   WIMP {}'.format(wimp))
            print('   WUPS {}'.format(wups))
            print('   CCLOSESF {}'.format(cclosesf))
            print('   MXITERSF {}'.format(mxitersf))
            print('   CRNTSF {}'.format(crntsf))
            print('   IPRTXMD {}'.format(iprtxmd))

        # Item 3 (COLDSF(NRCH)) Initial concentration
        if model.verbose:
            print('   loading NSFINIT...')

            if model.free_format:
                print('   Using MODFLOW style array reader utilities to ' \
                      'read NSFINIT')
            elif model.array_format == 'mt3d':
                print('   Using historic MT3DMS array reader utilities to ' \
                      'read NSFINIT')

        coldsf = Util2d.load(f, model, (1, nsfinit), np.float32, 'nsfinit',
                             ext_unit_dict, array_format=model.array_format)

        # Item 4 (DISPSF(NRCH)) Reach-by-reach dispersion
        if model.verbose:
            if model.free_format:
                print('   Using MODFLOW style array reader utilities to ' \
                      'read DISPSF')
            elif model.array_format == 'mt3d':
                print('   Using historic MT3DMS array reader utilities to ' \
                      'read DISPSF')

        dispsf = Util2d.load(f, model, (1, nsfinit), np.float32, 'dispsf',
                                 ext_unit_dict, array_format=model.array_format)

        # Item 5 NOBSSF
        if model.verbose:
            print('   loading NOBSSF...')
        line = f.readline()
        m_arr = line.strip().split()
        nobssf = int(m_arr[0])
        if model.verbose:
            print('   NOBSSF {}'.format(nobssf))

        # If NOBSSF > 0, store observation segment & reach (Item 6)
        obs_sf = []
        if nobssf > 0:
            if model.verbose:
                print('   loading {} observation locations given by ISOBS, '\
                          'IROBS...'.format(nobssf))
            for i in range(nobssf):
                line = f.readline()
                m_arr = line.strip().split()
                obs_sf.append([int(m_arr[0])])
            obs_sf = np.array(obs_sf)
            if model.verbose:
                print('   Surface water concentration observation locations:')
                print('   {}',format(obs_sf))
        else:
            if model.verbose:
                print('   No observation points specified.')

        sf_stress_period_data = {}

        for iper in range(nper):

            # Item 7 NTMP (Transient data)
            if model.verbose:
                print('   loading NTMP...')
            line = f.readline()
            m_arr = line.strip().split()
            ntmp = int(m_arr[0])

            # Item 8 ISEGBC, IRCHBC, ISFBCTYP, CBCSF
            if model.verbose:
                print('   loading {} instances of ISEGBC, IRCHBC, ' \
                        'ISFBCTYP, CBCSF...'.format(ntmp))
            current_sf = 0
            if ntmp > 0:
                current_sf = np.empty((ntmp), dtype=dtype)
                for ibnd in range(ntmp):
                    line = f.readline()
                    m_arr = line.strip().split()
                    t = []
                    for ivar in range(3):  # First three terms are not variable
                        t.append(m_arr[ivar])
                    cbcsf = len(current_sf.dtype.names) - 3
                    if cbcsf > 0:
                        for ivar in range(cbcsf):
                            t.append(m_arr[ivar + 3])
                    current_sf[ibnd] = tuple(map(float, t[:len(current_sf.dtype.names)]))
                # Convert ISEG IRCH indices to zero-based
                current_sf['isegbc'] -= 1
                current_sf['irchbc'] -= 1
                current_sf = current_sf.view(np.recarray)
                sf_stress_period_data[iper] = current_sf
            else:
                if model.verbose:
                    print('   No transient boundary conditions specified')
                pass

        # Construct and return SFT package
        sft = Mt3dSft(model, nsfinit=nsfinit, mxsfbc=mxsfbc, icbcsf=icbcsf,
                      ioutobs=ioutobs, ietsfr=ietsfr, isfsolv=isfsolv,
                      wimp=wimp, cclosesf=cclosesf, mxitersf=mxitersf,
                      crntsf=crntsf, iprtxmd=iprtxmd, coldsf=coldsf,
                      dispsf=dispsf, nobssf=nobssf, obs_sf=obs_sf,
                      sf_stress_period_data=sf_stress_period_data)
        return sft


    @staticmethod
    def ftype():
        return 'SFT'


    @staticmethod
    def defaultunit():
        return 19