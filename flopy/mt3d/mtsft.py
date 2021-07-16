import sys
import numpy as np

from ..pakbase import Package
from ..utils import Util2d, MfList

__author__ = "emorway"


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
        input file. To specify starting concentrations for other species in a
        multi-species simulation, include additional keywords, such as
        coldsf2, coldsf3, and so forth.
    dispsf : array of floats
        Is the dispersion coefficient [L2 T-1] for each stream reach in the
        simulation and can vary for each simulated component of the
        simulation.  That is, the length of the array is equal to the number
        of simulated stream reaches times the number of simulated components.
        Values of dispersion for each reach should be entered in the same
        order that individual reaches are entered for record set 2 in the
        SFR2 input file.  To specify dispsf for other species in a
        multi-species simulation, include additional keywords, such as
        dispsf2, dispsf3, and so forth.
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
    extension : string
        Filename extension (default is 'sft')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the sfr output name will be created using
        the model name and lake concentration observation extension
        (for example, modflowtest.cbc and modflowtest.sftcobs.out), if ioutobs
        is a number greater than zero. If a single string is passed the
        package will be set to the string and sfr concentration observation
        output name will be created using the model name and .sftcobs.out
        extension, if ioutobs is a number greater than zero. To define the
        names for all package files (input and output) the length of the list
        of strings should be 2. Default is None.

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
    >>> datadir = 'examples/data/mt3d_test/mfnwt_mt3dusgs/sft_crnkNic'
    >>> mf = flopy.modflow.Modflow.load(
    ...     'CrnkNic.nam', model_ws=datadir, load_only=['dis', 'bas6'])
    >>> sfr = flopy.modflow.ModflowSfr2.load('CrnkNic.sfr2', mf)
    >>> chk = sfr.check()
    >>> # initialize an MT3D-USGS model
    >>> mt = flopy.mt3d.Mt3dms.load(
    ...     'CrnkNic.mtnam', exe_name='mt3d-usgs_1.0.00.exe',
    >>>     model_ws=datadir, load_only='btn')
    >>> sft = flopy.mt3d.Mt3dSft.load(mt, 'CrnkNic.sft')

    """

    def __init__(
        self,
        model,
        nsfinit=0,
        mxsfbc=0,
        icbcsf=0,
        ioutobs=0,
        ietsfr=0,
        isfsolv=1,
        wimp=0.50,
        wups=1.00,
        cclosesf=1.0e-6,
        mxitersf=10,
        crntsf=1.0,
        iprtxmd=0,
        coldsf=0.0,
        dispsf=0.0,
        nobssf=0,
        obs_sf=None,
        sf_stress_period_data=None,
        unitnumber=None,
        filenames=None,
        dtype=None,
        extension="sft",
        **kwargs
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = Mt3dSft._defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dSft._reservedunit()

        # set filenames
        if filenames is None:  # if filename not passed
            filenames = [None, None]  # setup filenames
            if abs(ioutobs) > 0:
                filenames[1] = model.name
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                for idx in range(len(filenames), 2):
                    filenames.append(None)

        if ioutobs is not None:
            ext = "sftcobs.out"
            if filenames[1] is not None:
                if (
                    len(filenames[1].split(".", 1)) > 1
                ):  # already has extension
                    fname = "{}.{}".format(*filenames[1].split(".", 1))
                else:
                    fname = "{}.{}".format(filenames[1], ext)
            else:
                fname = "{}.{}".format(model.name, ext)
            model.add_output_file(
                abs(ioutobs),
                fname=fname,
                extension=None,
                binflag=False,
                package=Mt3dSft._ftype(),
            )
        else:
            ioutobs = 0

        # Fill namefile items
        name = [Mt3dSft._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

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

        self.ietsfr = ietsfr
        self.isfsolv = isfsolv
        self.wimp = wimp
        self.wups = wups
        self.cclosesf = cclosesf
        self.mxitersf = mxitersf
        self.crntsf = crntsf
        self.iprtxmd = iprtxmd

        # Set 1D array values
        self.coldsf = [
            Util2d(
                model,
                (nsfinit,),
                np.float32,
                coldsf,
                name="coldsf",
                locat=self.unit_number[0],
                array_free_format=False,
            )
        ]

        self.dispsf = [
            Util2d(
                model,
                (nsfinit,),
                np.float32,
                dispsf,
                name="dispsf",
                locat=self.unit_number[0],
                array_free_format=False,
            )
        ]
        ncomp = model.ncomp
        # handle the miult
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                for base_name, attr in zip(
                    ["coldsf", "dispsf"], [self.coldsf, self.dispsf]
                ):
                    name = "{0}{1}".format(base_name, icomp)
                    if name in kwargs:
                        val = kwargs.pop(name)
                    else:
                        print(
                            "SFT: setting {0} for component {1} to zero, kwarg name {2}".format(
                                base_name, icomp, name
                            )
                        )
                        val = 0.0
                    u2d = Util2d(
                        model,
                        (nsfinit,),
                        np.float32,
                        val,
                        name=name,
                        locat=self.unit_number[0],
                        array_free_format=model.free_format,
                    )
                    attr.append(u2d)

        # Set streamflow observation locations
        self.nobssf = nobssf
        self.obs_sf = obs_sf

        # Read and set transient data
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        if sf_stress_period_data is None or len(sf_stress_period_data) == 0:
            self.sf_stress_period_data = None
        else:
            self.sf_stress_period_data = MfList(
                self, model=model, data=sf_stress_period_data
            )
            self.sf_stress_period_data.list_free_format = True
        self.parent.add_package(self)
        return

    @staticmethod
    def get_default_dtype(ncomp=1):
        """
        Construct a dtype for the recarray containing the list of surface
        water boundary conditions.
        """
        type_list = [
            ("node", int),
            ("isfbctyp", int),
            ("cbcsf0", np.float32),
        ]
        if ncomp > 1:
            for icomp in range(1, ncomp):
                comp_name = "cbcsf{0:d}".format(icomp)
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
        >>> import flopy
        >>> datadir = .examples/data/mt3d_test/mfnwt_mt3dusgs/sft_crnkNic
        >>> mt = flopy.mt3d.Mt3dms.load(
        ...     'CrnkNic.mtnam', exe_name='mt3d-usgs_1.0.00.exe',
        ...     model_ws=datadir, verbose=True)
        >>> mt.name = 'CrnkNic_rewrite'
        >>> mt.sft.dispsf.fmtin = '(10F12.2)'
        >>> mt.write_input()

        """

        # Open file for writing
        f = open(self.fn_path, "w")

        # Item 1
        f.write(
            "{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}".format(
                self.nsfinit,
                self.mxsfbc,
                self.icbcsf,
                self.ioutobs,
                self.ietsfr,
            )
            + 30 * " "
            + "# nsfinit, mxsfbc, icbcsf, ioutobs, ietsfr\n"
        )

        # Item 2
        f.write(
            "{0:10d}{1:10.5f}{2:10.5f}{3:10.7f}{4:10d}{5:10.5f}{6:10d}".format(
                self.isfsolv,
                self.wimp,
                self.wups,
                self.cclosesf,
                self.mxitersf,
                self.crntsf,
                self.iprtxmd,
            )
            + " # isfsolv, wimp, wups, cclosesf, mxitersf, crntsf, "
            + "iprtxmd\n"
        )

        # Item 3
        for coldsf in self.coldsf:
            f.write(coldsf.get_file_entry())

        # Item 4
        for dispsf in self.dispsf:
            f.write(dispsf.get_file_entry())

        # Item 5
        f.write("{0:10d}                 # nobssf\n".format(self.nobssf))

        # Item 6
        if self.nobssf != 0:
            for iobs in self.obs_sf:
                line = (
                    "{0:10d}".format(iobs)
                    + 26 * " "
                    + "# location of obs as given by position in irch list\n"
                )
                f.write(line)

        # Items 7, 8
        # Loop through each stress period and assign source & sink concentrations to stream features
        nper = self.parent.nper
        for kper in range(nper):
            if f.closed == True:
                f = open(f.name, "a")

            # List of concentrations associated with various boundaries
            # interacting with the stream network.
            if self.sf_stress_period_data is not None:
                self.sf_stress_period_data.write_transient(f, single_per=kper)
            else:
                f.write("{0:10d}       # ntmp - SP {1:5d}\n".format(0, kper))

        f.close()
        return

    @classmethod
    def load(
        cls, f, model, nsfinit=None, nper=None, ncomp=None, ext_unit_dict=None
    ):
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
        >>> mf = flopy.modflow.Modflow.load('CrnkNic_mf.nam',
        ...                                 load_only=['dis', 'bas6'])
        >>> sfr = flopy.modflow.ModflowSfr2.load('CrnkNic.sfr2', mf)
        >>> mt = flopy.mt3d.Mt3dms.load('CrnkNic_mt.nam', load_only='btn')
        >>> sft = flopy.mt3d.Mt3dSft.load('CrnkNic.sft', mt)

        """
        if model.verbose:
            sys.stdout.write("loading sft package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

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
        if line[0] == "#":
            raise ValueError("SFT package does not support comment lines")

        if model.verbose:
            print("   loading nsfinit, mxsfbc, icbcsf, ioutobs, ietsfr...")
        vals = line.strip().split()

        nsfinit = int(vals[0])
        mxsfbc = int(vals[1])
        icbcsf = int(vals[2])
        ioutobs = int(vals[3])
        ietsfr = int(vals[4])

        if model.verbose:
            print("   NSFINIT {}".format(nsfinit))
            print("   MXSFBC {}".format(mxsfbc))
            print("   ICBCSF {}".format(icbcsf))
            print("   IOUTOBS {}".format(ioutobs))
            print("   IETSFR {}".format(ietsfr))
            if ietsfr == 0:
                print(
                    "   Mass does not exit the model via simulated "
                    "stream evaporation "
                )
            else:
                print(
                    "   Mass exits the stream network via simulated "
                    "stream evaporation "
                )

        # Item 2 (ISFSOLV, WIMP, WUPS, CCLOSESF, MXITERSF, CRNTSF, IPRTXMD)
        line = f.readline()
        if model.verbose:
            print(
                "   loading isfsolv, wimp, wups, cclosesf, mxitersf, "
                "crntsf, iprtxmd..."
            )

        vals = line.strip().split()

        if len(vals) < 7:
            raise ValueError("expected 7 values for item 2 of SFT input file")
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
            print("   Resetting isfsolv to 1")
            print("   In version 1.0 of MT3D-USGS, isfsov=1 is only option")

        if model.verbose:
            print("   ISFSOLV {}".format(isfsolv))
            print("   WIMP {}".format(wimp))
            print("   WUPS {}".format(wups))
            print("   CCLOSESF {}".format(cclosesf))
            print("   MXITERSF {}".format(mxitersf))
            print("   CRNTSF {}".format(crntsf))
            print("   IPRTXMD {}".format(iprtxmd))

        # Item 3 (COLDSF(NRCH)) Initial concentration
        if model.verbose:
            print("   loading COLDSF...")

            if model.free_format:
                print(
                    "   Using MODFLOW style array reader utilities to "
                    "read COLDSF"
                )
            elif model.array_format == "mt3d":
                print(
                    "   Using historic MT3DMS array reader utilities to "
                    "read COLDSF"
                )

        coldsf = Util2d.load(
            f,
            model,
            (np.abs(nsfinit),),
            np.float32,
            "coldsf1",
            ext_unit_dict,
            array_format=model.array_format,
        )

        kwargs = {}
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "coldsf" + str(icomp)
                if model.verbose:
                    print("   loading {}...".format(name))
                u2d = Util2d.load(
                    f,
                    model,
                    (nsfinit,),
                    np.float32,
                    name,
                    ext_unit_dict,
                    array_format=model.array_format,
                )
                kwargs[name] = u2d

        # Item 4 (DISPSF(NRCH)) Reach-by-reach dispersion
        if model.verbose:
            if model.free_format:
                print(
                    "   Using MODFLOW style array reader utilities to "
                    "read DISPSF"
                )
            elif model.array_format == "mt3d":
                print(
                    "   Using historic MT3DMS array reader utilities to "
                    "read DISPSF"
                )

        dispsf = Util2d.load(
            f,
            model,
            (np.abs(nsfinit),),
            np.float32,
            "dispsf1",
            ext_unit_dict,
            array_format=model.array_format,
        )
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "dispsf" + str(icomp)
                if model.verbose:
                    print("   loading {}...".format(name))
                u2d = Util2d.load(
                    f,
                    model,
                    (np.abs(nsfinit),),
                    np.float32,
                    name,
                    ext_unit_dict,
                    array_format=model.array_format,
                )
                kwargs[name] = u2d

        # Item 5 NOBSSF
        if model.verbose:
            print("   loading NOBSSF...")
        line = f.readline()
        m_arr = line.strip().split()
        nobssf = int(m_arr[0])
        if model.verbose:
            print("   NOBSSF {}".format(nobssf))

        # If NOBSSF > 0, store observation segment & reach (Item 6)
        obs_sf = []
        if nobssf > 0:
            if model.verbose:
                print(
                    "   loading {} observation locations given by ISOBS, "
                    "IROBS...".format(nobssf)
                )
            for i in range(nobssf):
                line = f.readline()
                m_arr = line.strip().split()
                obs_sf.append(int(m_arr[0]))
            obs_sf = np.array(obs_sf)
            if model.verbose:
                print("   Surface water concentration observation locations:")
                text = ""
                for o in obs_sf:
                    text += "{} ".format(o)
                print("   {}\n".format(text))
        else:
            if model.verbose:
                print("   No observation points specified.")

        sf_stress_period_data = {}

        for iper in range(nper):

            # Item 7 NTMP (Transient data)
            if model.verbose:
                print(
                    "   loading NTMP...stress period {} of {}".format(
                        iper + 1, nper
                    )
                )
            line = f.readline()
            m_arr = line.strip().split()
            ntmp = int(m_arr[0])

            # Item 8 ISEGBC, IRCHBC, ISFBCTYP, CBCSF
            if model.verbose:
                print(
                    "   loading {} instances of ISEGBC, IRCHBC, "
                    "ISFBCTYP, CBCSF...stress period {} of {}".format(
                        ntmp, iper + 1, nper
                    )
                )
            current_sf = 0
            if ntmp > 0:
                current_sf = np.empty(ntmp, dtype=dtype)
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
                    current_sf[ibnd] = tuple(
                        map(float, t[: len(current_sf.dtype.names)])
                    )
                # Convert node IRCH indices to zero-based
                current_sf["node"] -= 1
                current_sf = current_sf.view(np.recarray)
                sf_stress_period_data[iper] = current_sf
            else:
                if model.verbose:
                    print("   No transient boundary conditions specified")
                pass

        if openfile:
            f.close()

        # 1 item for SFT input file, 1 item for SFTOBS file
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=Mt3dSft._ftype()
            )
            if abs(ioutobs) > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(ioutobs)
                )
                model.add_pop_key_list(abs(ioutobs))

        # Construct and return SFT package
        return cls(
            model,
            nsfinit=nsfinit,
            mxsfbc=mxsfbc,
            icbcsf=icbcsf,
            ioutobs=ioutobs,
            ietsfr=ietsfr,
            isfsolv=isfsolv,
            wimp=wimp,
            cclosesf=cclosesf,
            mxitersf=mxitersf,
            crntsf=crntsf,
            iprtxmd=iprtxmd,
            coldsf=coldsf,
            dispsf=dispsf,
            nobssf=nobssf,
            obs_sf=obs_sf,
            sf_stress_period_data=sf_stress_period_data,
            unitnumber=unitnumber,
            filenames=filenames,
            **kwargs
        )

    @staticmethod
    def _ftype():
        return "SFT"

    @staticmethod
    def _defaultunit():
        return 19

    @staticmethod
    def _reservedunit():
        return 19
