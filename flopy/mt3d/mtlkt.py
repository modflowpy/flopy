import sys
import numpy as np

from ..pakbase import Package
from ..utils import Util2d, MfList

__author__ = "emorway"


class Mt3dLkt(Package):
    """
    MT3D-USGS LaKe Transport package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    nlkinit : int
        is equal to the number of simulated lakes as specified in the flow
        simulation
    mxlkbc : int
        must be greater than or equal to the sum total of boundary conditions
        applied to each lake
    icbclk : int
        is equal to the unit number on which lake-by-lake transport information
        will be printed.  This unit number must appear in the NAM input file
        required for every MT3D-USGS simulation.
    ietlak : int
        specifies whether or not evaporation as simulated in the flow solution
        will act as a mass sink.
        = 0, Mass does not exit the model via simulated lake evaporation
        != 0, Mass may leave the lake via simulated lake evaporation
    coldlak : array of floats
        is a vector of real numbers representing the initial concentrations in
        the simulated lakes.  The length of the vector is equal to the number
        of simulated lakes, NLKINIT.  Initial lake concentrations should be
        in the same order as the lakes appearing in the LAK input file
        corresponding to the MODFLOW simulation.
    ntmp : int
        is an integer value corresponding to the number of specified lake
        boundary conditions to follow.  For the first stress period, this
        value must be greater than or equal to zero, but may be less than
        zero in subsequent stress periods.
    ilkbc : int
        is the lake number for which the current boundary condition will be
        specified
    ilkbctyp : int
        specifies what the boundary condition type is for ilakbc
           1   a precipitation boundary. If precipitation directly to lakes
               is simulated in the flow model and a non-zero concentration
               (default is zero) is desired, use ISFBCTYP = 1;
           2   a runoff boundary condition that is not the same thing as
               runoff simulated in the UZF1 package and routed to a lake (or
               stream) using the IRNBND array.  Users who specify runoff in
               the LAK input via the RNF variable appearing in record set 9a
               and want to assign a non-zero concentration (default is zero)
               associated with this specified source, use ISFBCTYP=2;
           3   a Pump boundary condition.  Users who specify a withdrawal
               from a lake via the WTHDRW variable appearing in record set 9a
               and want to assign a non-zero concentration (default is zero)
               associated with this specified source, use ISFBCTYP=2;
           4   an evaporation boundary condition.  In models where evaporation
               is simulated directly from the surface of the lake, users can use
               this boundary condition to specify a non-zero concentration
               (default is zero) associated with the evaporation losses.
    extension : string
        Filename extension (default is 'lkt')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the lake output name will be created using
        the model name and lake concentration observation extension
        (for example, modflowtest.cbc and modflowtest.lkcobs.out), if icbclk
        is a number greater than zero. If a single string is passed the
        package will be set to the string and lake concentration observation
        output name will be created using the model name and .lkcobs.out
        extension, if icbclk is a number greater than zero. To define the
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
    >>> mt = flopy.mt3d.Mt3dms()
    >>> lkt = flopy.mt3d.Mt3dLkt(mt)

    """

    def __init__(
        self,
        model,
        nlkinit=0,
        mxlkbc=0,
        icbclk=None,
        ietlak=0,
        coldlak=0.0,
        lk_stress_period_data=None,
        dtype=None,
        extension="lkt",
        unitnumber=None,
        filenames=None,
        iprn=-1,
        **kwargs
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = Mt3dLkt._reservedunit()
        elif unitnumber == 0:
            unitnumber = Mt3dLkt._reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
            if abs(icbclk) > 0:
                filenames[1] = model.name
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                for idx in range(len(filenames), 2):
                    filenames.append(None)

        if icbclk is not None:
            ext = "lkcobs.out"
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
                icbclk,
                fname=fname,
                extension=None,
                binflag=False,
                package=Mt3dLkt._ftype(),
            )
        else:
            icbclk = 0

        # Fill namefile items
        name = [Mt3dLkt._ftype()]
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

        # Set package specific parameters
        self.nlkinit = nlkinit
        self.mxlkbc = mxlkbc
        self.icbclk = icbclk
        self.ietlak = ietlak

        # Set initial lake concentrations
        self.coldlak = []
        u2d = Util2d(
            self.parent,
            (nlkinit,),
            np.float32,
            coldlak,
            name="coldlak",
            locat=self.unit_number[0],
            array_free_format=False,
            iprn=iprn,
        )
        self.coldlak.append(u2d)

        # next, handle multi-species when appropriate
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                for base_name, attr in zip(["coldlak"], [self.coldlak]):
                    name = "{0}{1}".format(base_name, icomp)
                    if name in kwargs:
                        val = kwargs.pop(name)
                    else:
                        print(
                            "LKT: setting {0} for component {1} to zero, kwarg name {2}".format(
                                base_name, icomp, name
                            )
                        )
                        val = 0.0
                    u2d = Util2d(
                        model,
                        (nlkinit,),
                        np.float32,
                        val,
                        name=name,
                        locat=self.unit_number[0],
                        array_free_format=model.free_format,
                    )
                    self.coldlak.append(u2d)

        # Set transient data
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        if lk_stress_period_data is None:
            self.lk_stress_period_data = None
        else:
            self.lk_stress_period_data = MfList(
                self, model=model, data=lk_stress_period_data
            )

        # Check to make sure that all kwargs have been consumed
        if len(list(kwargs.keys())) > 0:
            raise Exception(
                "LKT error: unrecognized kwargs: "
                + " ".join(list(kwargs.keys()))
            )

        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """

        # Open file for writing
        f_lkt = open(self.fn_path, "w")

        # Item 1
        f_lkt.write(
            "{0:10d}{1:10d}{2:10}{3:10}          ".format(
                self.nlkinit, self.mxlkbc, self.icbclk, self.ietlak
            )
            + "# NLKINIT, MXLKBC, ICBCLK, IETLAK\n"
        )

        # Item 2
        for s in range(len(self.coldlak)):
            f_lkt.write(self.coldlak[s].get_file_entry())

        # Items 3-4
        # (Loop through each stress period and write LKT information)
        nper = self.parent.nper
        for kper in range(nper):
            if f_lkt.closed == True:
                f_lkt = open(f_lkt.name, "a")

            # List of concentrations associated with fluxes in/out of lake
            # (Evap, precip, specified runoff into the lake, specified
            # withdrawal directly from the lake
            if self.lk_stress_period_data is not None:
                self.lk_stress_period_data.write_transient(
                    f_lkt, single_per=kper
                )
            else:
                f_lkt.write("{}\n".format(0))

        f_lkt.close()
        return

    @classmethod
    def load(
        cls, f, model, nlak=None, nper=None, ncomp=None, ext_unit_dict=None
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
        nlak : int
            number of lakes to be simulated
        nper : int
            number of stress periods
        ncomp : int
            number of species to be simulated
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        lkt :  MT3D-USGS object
            MT3D-USGS object.

        Examples
        --------

        >>> import flopy
        >>> datadir = 'examples/data/mt3d_test/mfnwt_mt3dusgs/lkt'
        >>> mt = flopy.mt3d.Mt3dms.load(
        ...     'lkt_mt.nam', exe_name='mt3d-usgs_1.0.00.exe',
        ...     model_ws=datadir, load_only='btn')
        >>> lkt = flopy.mt3d.Mt3dLkt.load('test.lkt', mt)

        """
        if model.verbose:
            sys.stdout.write("loading lkt package file...\n")

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

        # Item 1 (NLKINIT,MXLKBC,ICBCLK,IETLAK)
        line = f.readline()
        if line[0] == "#":
            raise ValueError("LKT package does not support comment lines")

        if model.verbose:
            print("   loading nlkinit,mxlkbc,icbclk,ietlak   ")
        vals = line.strip().split()

        nlkinit = int(vals[0])
        mxlkbc = int(vals[1])
        icbclk = int(vals[2])
        ietlak = int(vals[3])

        if model.verbose:
            print("   NLKINIT {}".format(nlkinit))
            print("   MXLKBC {}".format(mxlkbc))
            print("   ICBCLK {}".format(icbclk))
            print("   IETLAK {}".format(ietlak))
            if ietlak == 0:
                print(
                    "   Mass does not exit the model via simulated lake evaporation   "
                )
            else:
                print(
                    "   Mass exits the lake via simulated lake evaporation   "
                )

        # Item 2 (COLDLAK - Initial concentration in this instance)
        if model.verbose:
            print("   loading initial concentration (COLDLAK)  ")
            if model.free_format:
                print(
                    "   Using MODFLOW style array reader utilities to "
                    "read COLDLAK"
                )
            elif model.array_format == "mt3d":
                print(
                    "   Using historic MT3DMS array reader utilities to "
                    "read COLDLAK"
                )

        kwargs = {}
        coldlak = Util2d.load(
            f,
            model,
            (nlkinit,),
            np.float32,
            "coldlak1",
            ext_unit_dict,
            array_format=model.array_format,
        )

        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "coldlak" + str(icomp)
                if model.verbose:
                    print("   loading {}...".format(name))
                u2d = Util2d.load(
                    f,
                    model,
                    (nlkinit,),
                    np.float32,
                    name,
                    ext_unit_dict,
                    array_format=model.array_format,
                )
                kwargs[name] = u2d

        # dtype
        dtype = Mt3dLkt.get_default_dtype(ncomp)

        # Items 3-4
        lk_stress_period_data = {}

        for iper in range(nper):
            if model.verbose:
                print(
                    "   loading lkt boundary condition data for kper {0:5d}".format(
                        iper + 1
                    )
                )

            # Item 3: NTMP: An integer value corresponding to the number of
            #         specified lake boundary conditions to follow.
            #         For the first stress period, this value must be greater
            #         than or equal to zero, but may be less than zero in
            #         subsequent stress periods.
            line = f.readline()
            vals = line.strip().split()
            ntmp = int(vals[0])
            if model.verbose:
                print(
                    "   {0:5d} lkt boundary conditions specified ".format(ntmp)
                )
                if (iper == 0) and (ntmp < 0):
                    print("   ntmp < 0 not allowed for first stress period   ")
                if (iper > 0) and (ntmp < 0):
                    print(
                        "   use lkt boundary conditions specified in last stress period   "
                    )

            # Item 4: Read ntmp boundary conditions
            if ntmp > 0:
                current_lk = np.empty((ntmp), dtype=dtype)
                for ilkbnd in range(ntmp):
                    line = f.readline()
                    m_arr = line.strip().split()  # These items are free format
                    t = []
                    for ivar in range(2):
                        t.append(m_arr[ivar])
                    cbclk = len(current_lk.dtype.names) - 2
                    if cbclk > 0:
                        for ilkvar in range(cbclk):
                            t.append(m_arr[ilkvar + 2])
                    current_lk[ilkbnd] = tuple(
                        t[: len(current_lk.dtype.names)]
                    )
                # Convert ILKBC (node) index to zero-based
                current_lk["node"] -= 1
                current_lk = current_lk.view(np.recarray)
                lk_stress_period_data[iper] = current_lk
            else:
                if model.verbose:
                    print("   No transient boundary conditions specified")
                pass

        if openfile:
            f.close()

        if len(lk_stress_period_data) == 0:
            lk_stress_period_data = None

        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=Mt3dLkt._ftype()
            )
            if icbclk > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=icbclk
                )
                model.add_pop_key_list(icbclk)

        # Construct and return LKT package
        return cls(
            model,
            nlkinit=nlkinit,
            mxlkbc=mxlkbc,
            icbclk=icbclk,
            ietlak=ietlak,
            coldlak=coldlak,
            lk_stress_period_data=lk_stress_period_data,
            unitnumber=unitnumber,
            filenames=filenames,
            **kwargs
        )

    @staticmethod
    def get_default_dtype(ncomp=1):
        """
        Construct a dtype for the recarray containing the list of boundary
        conditions interacting with the lake (i.e., pumps, specified runoff...)
        """
        type_list = [
            ("node", int),
            ("ilkbctyp", int),
            ("cbclk0", np.float32),
        ]
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                comp_name = "cbclk({0:02d})".format(icomp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
        return dtype

    @staticmethod
    def _ftype():
        return "LKT"

    @staticmethod
    def _defaultunit():
        return 45

    @staticmethod
    def _reservedunit():
        return 18
