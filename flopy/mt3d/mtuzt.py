__author__ = "emorway"

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d, Transient2d


class Mt3dUzt(Package):
    """
    MT3D-USGS Unsaturated-Zone Transport package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    icbcuz : int
        Is the unit number to which unsaturated-zone concentration will be
        written out.
    iet : int
        Is a flag that indicates whether or not ET is being simulated in the
        UZF1 flow package (=0 indicates that ET is not being simulated).
        If ET is not being simulated, IET informs FMI package not to look
        for UZET and GWET arrays in the flow-transport link file.
    iuzfbnd : array of ints
        Specifies which row/column indices variably-saturated transport will
        be simulated in.
           >0  indicates variably-saturated transport will be simulated;
           =0  indicates variably-saturated transport will not be simulated;
           <0  Corresponds to IUZFBND < 0 in the UZF1 input package, meaning
               that user-supplied values for FINF are specified recharge and
               therefore transport through the unsaturated zone is not
               simulated.
    incuzinf : int
        (This value is repeated for each stress period as explained next) A
        flag indicating whether an array containing the concentration of
        infiltrating water (FINF) for each simulated species (ncomp) will be
        read for the current stress period.  If INCUZINF >= 0, an array
        containing the concentration of infiltrating flux for each species
        will be read.  If INCUZINF < 0, the concentration of infiltrating flux
        will be reused from the previous stress period.  If INCUZINF < 0 is
        specified for the first stress period, then by default the
        concentration of positive infiltrating flux (source) is set equal to
        zero.  There is no possibility of a negative infiltration flux being
        specified.  If infiltrating water is rejected due to an infiltration
        rate exceeding the vertical hydraulic conductivity, or because
        saturation is reached in the unsaturated zone and the water table is
        therefore at land surface, the concentration of the runoff will be
        equal to CUZINF specified next.  The runoff is routed if IRNBND is
        specified in the MODFLOW simulation.
    cuzinf : array of floats
        Is the concentration of the infiltrating flux for a particular species.
        An array for each species will be read.
    incuzet : int
        (This value is repeated for each stress period as explained next) A
        flag indicating whether an array containing the concentration of
        evapotranspiration flux originating from the unsaturated zone will be
        read for the current stress period.  If INCUZET >= 0, an array
        containing the concentration of evapotranspiration flux originating
        from the unsaturated zone for each species will be read.  If
        INCUZET < 0, the concentration of evapotranspiration flux for each
        species will be reused from the last stress period.  If INCUZET < 0
        is specified for the first stress period, then by default, the
        concentration of negative evapotranspiration flux (sink) is set
        equal to the aquifer concentration, while the concentration of
        positive evapotranspiration flux (source) is set to zero.
    cuzet : array of floats
        Is the concentration of ET fluxes originating from the unsaturated
        zone.  As a default, this array is set equal to 0 and only overridden
        if the user specifies INCUZET > 1.  If empirical evidence suggest
        volatilization of simulated constituents from the unsaturated zone,
        this may be one mechanism for simulating this process, though it would
        depend on the amount of simulated ET originating from the unsaturated
        zone.  An array for each species will be read.
    incgwet : int
        (This value is repeated for each stress period as explained next) Is
        a flag indicating whether an array containing the concentration of
        evapotranspiration flux originating from the saturated zone will be
        read for the current stress period.  If INCGWET >= 0, an array
        containing the concentration of evapotranspiration flux originating
        from the saturated zone for each species will be read.  If
        INCGWET < 0, the concentration of evapotranspiration flux for each
        species will be reused from the last stress period.  If INCUZET < 0
        is specified for the first stress period, then by default, the
        concentration of negative evapotranspiration flux (sink) is set to
        the aquifer concentration, while the concentration of positive
        evapotranspiration flux (source) is set to zero.
    cgwet : array of floats
        Is the concentration of ET fluxes originating from the saturated zone.
        As a default, this array is set equal to 0 and only overridden if the
        user specifies INCUZET > 1.  An array for each species will be read.
    extension : string
        Filename extension (default is 'uzt')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the uzf output name will be created using
        the model name and uzf concentration observation extension
        (for example, modflowtest.cbc and modflowtest.uzcobs.out), if icbcuz
        is a number greater than zero. If a single string is passed the
        package will be set to the string and uzf concentration observation
        output name will be created using the model name and .uzcobs.out
        extension, if icbcuz is a number greater than zero. To define the
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
    >>> datadir = 'examples/data/mt3d_test/mfnwt_mt3dusgs/keat_uzf'
    >>> mt = flopy.mt3d.Mt3dms.load(
    ...     'Keat_UZF_mt.nam', exe_name='mt3d-usgs_1.0.00.exe',
    ...     model_ws=datadir, load_only='btn')
    >>> uzt = flopy.mt3d.Mt3dUzt('Keat_UZF.uzt', mt)

    """

    def __init__(
        self,
        model,
        icbcuz=None,
        iet=0,
        iuzfbnd=None,
        cuzinf=None,
        cuzet=None,
        cgwet=None,
        extension="uzt",
        unitnumber=None,
        filenames=None,
        **kwargs
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = Mt3dUzt._defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dUzt._reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                for idx in range(len(filenames), 2):
                    filenames.append(None)

        if icbcuz is not None:
            fname = filenames[1]
            extension = "uzcobs.out"
            model.add_output_file(
                icbcuz,
                fname=fname,
                extension=extension,
                binflag=False,
                package=Mt3dUzt._ftype(),
            )
        else:
            icbcuz = 0

        # Fill namefile items
        name = [Mt3dUzt._ftype()]
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
        self.heading1 = "# UZT for MT3D-USGS, generated by Flopy"
        self.icbcuz = icbcuz
        self.iet = iet

        if iuzfbnd is not None:
            self.iuzfbnd = Util2d(
                self.parent,
                (nrow, ncol),
                np.int32,
                iuzfbnd,
                name="iuzfbnd",
                locat=self.unit_number[0],
            )
        # set iuzfbnd based on UZF input file
        else:
            arr = np.zeros((nrow, ncol), dtype=np.int32)
            self.iuzfbnd = Util2d(
                self.parent,
                (nrow, ncol),
                np.int32,
                arr,
                name="iuzfbnd",
                locat=self.unit_number[0],
            )

        # Note: list is used for multi-species, NOT for stress periods!
        if cuzinf is not None:
            self.cuzinf = []
            t2d = Transient2d(
                model,
                (nrow, ncol),
                np.float32,
                cuzinf,
                name="cuzinf1",
                locat=self.unit_number[0],
            )
            self.cuzinf.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    val = 0.0
                    name = "cuzinf" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print(
                            "UZT: setting cuzinf for component "
                            + str(icomp)
                            + " to zero. kwarg name "
                            + name
                        )

                    t2d = Transient2d(
                        model,
                        (nrow, ncol),
                        np.float32,
                        val,
                        name=name,
                        locat=self.unit_number[0],
                    )
                    self.cuzinf.append(t2d)

        if cuzet is not None:
            self.cuzet = []
            t2d = Transient2d(
                model,
                (nrow, ncol),
                np.float32,
                cuzet,
                name="cuzet1",
                locat=self.unit_number[0],
            )
            self.cuzet.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    val = 0.0
                    name = "cuzet" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print(
                            "UZT: setting cuzet for component "
                            + str(icomp)
                            + " to zero. kwarg name "
                            + name
                        )

                    t2d = Transient2d(
                        model,
                        (nrow, ncol),
                        np.float32,
                        val,
                        name=name,
                        locat=self.unit_number[0],
                    )
                    self.cuzet.append(t2d)

        if cgwet is not None:
            self.cgwet = []
            t2d = Transient2d(
                model,
                (nrow, ncol),
                np.float32,
                cgwet,
                name="cgwet1",
                locat=self.unit_number[0],
            )
            self.cgwet.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    val = 0.0
                    name = "cgwet" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print(
                            "UZT: setting cgwet for component "
                            + str(icomp)
                            + " to zero. kwarg name "
                            + name
                        )

                    t2d = Transient2d(
                        model,
                        (nrow, ncol),
                        np.float32,
                        val,
                        name=name,
                        locat=self.unit_number[0],
                    )
                    self.cgwet.append(t2d)

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
        f_uzt = open(self.fn_path, "w")

        # Write header
        f_uzt.write("#{0:s}\n".format(self.heading1))

        # Item 2
        f_uzt.write(
            "{0:10d}{1:10d}                    #ICBCUZ, IET\n".format(
                self.icbcuz, self.iet
            )
        )

        # Item 3
        f_uzt.write(self.iuzfbnd.get_file_entry())

        # Items 4-9
        # (Loop through each stress period and write uzt information)
        nper = self.parent.nper
        for kper in range(nper):
            if f_uzt.closed == True:
                f_uzt = open(f_uzt.name, "a")

            # Concentrations associated with distributed stresses (Infil, ET)
            if self.cuzinf is not None:
                # If any species needs to be written, then all need to be
                # written
                incuzinf = -1
                for t2d in self.cuzinf:
                    incuzinficomp, file_entry = t2d.get_kper_entry(kper)
                    incuzinf = max(incuzinf, incuzinficomp)
                    if incuzinf == 1:
                        break
                f_uzt.write(
                    "{0:10d}          # INCUZINF - SP {1:5d}\n".format(
                        incuzinf, kper + 1
                    )
                )
                if incuzinf == 1:
                    for t2d in self.cuzinf:
                        u2d = t2d[kper]
                        file_entry = u2d.get_file_entry()
                        f_uzt.write(file_entry)

            if self.iet != 0:
                if self.cuzet is not None:
                    # If any species needs to be written, then all need to be
                    # written
                    incuzet = -1
                    for t2d in self.cuzet:
                        incuzeticomp, file_entry = t2d.get_kper_entry(kper)
                        incuzet = max(incuzet, incuzeticomp)
                        if incuzet == 1:
                            break
                    f_uzt.write(
                        "{0:10d}          # INCUZET - SP {1:5d}\n".format(
                            incuzet, kper + 1
                        )
                    )
                    if incuzet == 1:
                        for t2d in self.cuzet:
                            u2d = t2d[kper]
                            file_entry = u2d.get_file_entry()
                            f_uzt.write(file_entry)

                if self.cgwet is not None:
                    # If any species needs to be written, then all need to be
                    # written
                    incgwet = -1
                    for t2d in self.cgwet:
                        incgweticomp, file_entry = t2d.get_kper_entry(kper)
                        incgwet = max(incgwet, incgweticomp)
                        if incgwet == 1:
                            break
                    f_uzt.write(
                        "{0:10d}          # INCGWET - SP {1:5d}\n".format(
                            incgwet, kper + 1
                        )
                    )
                    if incgwet == 1:
                        for t2d in self.cgwet:
                            u2d = t2d[kper]
                            file_entry = u2d.get_file_entry()
                            f_uzt.write(file_entry)

        f_uzt.write("\n")
        f_uzt.close()
        return

    @classmethod
    def load(
        cls,
        f,
        model,
        nlay=None,
        nrow=None,
        ncol=None,
        nper=None,
        ncomp=None,
        ext_unit_dict=None,
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
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        uzt :  Mt3dSsm object
            Mt3dUzt object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> uzt = flopy.mt3d.Mt3dUzt.load('test.uzt', mt)

        """

        if model.verbose:
            print("loading uzt package file...\n")

        # Open file if necessary
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

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

        # Item 1 (comments, must be preceded by '#')
        if model.verbose:
            print("   Reading off comment lines...")
        line = f.readline()
        while line[0:1] == "#":
            i = 1
            if model.verbose:
                print("   Comment Line " + str(i) + ": ".format(line.strip()))
                i += 1
            line = f.readline()

        # Item 2 (ICBCUZ, IET)
        if line[0:1] != "#":
            # Don't yet read the next line because the current line
            # contains the values in item 2
            m_arr = line.strip().split()
            icbcuz = int(m_arr[0])
            iet = int(m_arr[1])

        # Item 3 [IUZFBND(NROW,NCOL) (one array for each layer)]
        if model.verbose:
            print("   loading IUZFBND...")
        iuzfbnd = Util2d.load(
            f, model, (nrow, ncol), np.int32, "iuzfbnd", ext_unit_dict
        )

        # kwargs needed to construct cuzinf2, cuzinf3, etc. for multispecies
        kwargs = {}

        cuzinf = None
        # At least one species being simulated, so set up a place holder
        t2d = Transient2d(
            model, (nrow, ncol), np.float32, 0.0, name="cuzinf", locat=0
        )
        cuzinf = {0: t2d}
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "cuzinf" + str(icomp)
                t2d = Transient2d(
                    model, (nrow, ncol), np.float32, 0.0, name=name, locat=0
                )
                kwargs[name] = {0: t2d}

        # Repeat cuzinf initialization procedure for cuzet only if iet != 0
        if iet != 0:
            cuzet = None
            t2d = Transient2d(
                model, (nrow, ncol), np.float32, 0.0, name="cuzet", locat=0
            )
            cuzet = {0: t2d}
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "cuzet" + str(icomp)
                    t2d = Transient2d(
                        model,
                        (nrow, ncol),
                        np.float32,
                        0.0,
                        name=name,
                        locat=0,
                    )
                    kwargs[name] = {0: t2d}

            # Repeat cuzinf initialization procedures for cgwet
            cgwet = None
            t2d = Transient2d(
                model, (nrow, ncol), np.float32, 0.0, name="cgwet", locat=0
            )
            cgwet = {0: t2d}
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "cgwet" + str(icomp)
                    t2d = Transient2d(
                        model,
                        (nrow, ncol),
                        np.float32,
                        0.0,
                        name=name,
                        locat=0,
                    )
                    kwargs[name] = {0: t2d}
        elif iet == 0:
            cuzet = None
            cgwet = None

        # Start of transient data
        for iper in range(nper):

            if model.verbose:
                print("   loading UZT data for kper {0:5d}".format(iper + 1))

            # Item 4 (INCUZINF)
            line = f.readline()
            m_arr = line.strip().split()
            incuzinf = int(m_arr[0])

            # Item 5 (CUZINF)
            if incuzinf >= 0:
                if model.verbose:
                    print(
                        "   Reading CUZINF array for kper "
                        "{0:5d}".format(iper + 1)
                    )
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "cuzinf", ext_unit_dict
                )
                cuzinf[iper] = t

                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = "cuzinf" + str(icomp)
                        if model.verbose:
                            print("   loading {}...".format(name))
                        t = Util2d.load(
                            f,
                            model,
                            (nrow, ncol),
                            np.float32,
                            name,
                            ext_unit_dict,
                        )
                        cuzinficomp = kwargs[name]
                        cuzinficomp[iper] = t

            elif incuzinf < 0 and iper == 0:
                if model.verbose:
                    print(
                        "   INCUZINF < 0 in first stress period. Setting "
                        "CUZINF to default value of 0.00 for all calls"
                    )
                    # This happens implicitly and is taken care of my
                    # existing functionality within flopy.  This elif
                    # statement exist for the purpose of printing the message
                    # above
                pass

            elif incuzinf < 0 and iper > 0:
                if model.verbose:
                    print(
                        "   Reusing CUZINF array from kper "
                        "{0:5d}".format(iper) + " in kper "
                        "{0:5d}".format(iper + 1)
                    )

            if iet != 0:
                # Item 6 (INCUZET)
                line = f.readline()
                m_arr = line.strip().split()
                incuzet = int(m_arr[0])

                # Item 7 (CUZET)
                if incuzet >= 0:
                    if model.verbose:
                        print(
                            "   Reading CUZET array for kper "
                            "{0:5d}".format(iper + 1)
                        )
                    t = Util2d.load(
                        f,
                        model,
                        (nrow, ncol),
                        np.float32,
                        "cuzet",
                        ext_unit_dict,
                    )
                    cuzet[iper] = t

                    # Load each multispecies array
                    if ncomp > 1:
                        for icomp in range(2, ncomp + 1):
                            name = "cuzet" + str(icomp)
                            if model.verbose:
                                print("   loading {}".format(name))
                            t = Util2d.load(
                                f,
                                model,
                                (nrow, ncol),
                                np.float32,
                                name,
                                ext_unit_dict,
                            )
                            cuzeticomp = kwargs[name]
                            cuzeticomp[iper] = t

                elif incuzet < 0 and iper == 0:
                    if model.verbose:
                        print(
                            "   INCUZET < 0 in first stress period. Setting "
                            "CUZET to default value of 0.00 for all calls"
                        )
                        # This happens implicitly and is taken care of my
                        # existing functionality within flopy.  This elif
                        # statement exist for the purpose of printing the message
                        # above
                    pass
                else:
                    if model.verbose:
                        print(
                            "   Reusing CUZET array from kper "
                            "{0:5d}".format(iper) + " in kper "
                            "{0:5d}".format(iper + 1)
                        )

                # Item 8 (INCGWET)
                line = f.readline()
                m_arr = line.strip().split()
                incgwet = int(m_arr[0])

                # Item 9 (CGWET)
                if model.verbose:
                    if incuzet >= 0:
                        print(
                            "   Reading CGWET array for kper "
                            "{0:5d}".format(iper + 1)
                        )
                    t = Util2d.load(
                        f,
                        model,
                        (nrow, ncol),
                        np.float32,
                        "cgwet",
                        ext_unit_dict,
                    )
                    cgwet[iper] = t

                    # Load each multispecies array
                    if ncomp > 1:
                        for icomp in range(2, ncomp + 1):
                            name = "cgwet" + str(icomp)
                            if model.verbose:
                                print("   loading {}...".format(name))
                            t = Util2d.load(
                                f,
                                model,
                                (nrow, ncol),
                                np.float32,
                                name,
                                ext_unit_dict,
                            )
                            cgweticomp = kwargs[name]
                            cgweticomp[iper] = t

                elif incuzet < 0 and iper == 0:
                    if model.verbose:
                        print(
                            "   INCGWET < 0 in first stress period. Setting "
                            "CGWET to default value of 0.00 for all calls"
                        )
                        # This happens implicitly and is taken care of my
                        # existing functionality within flopy.  This elif
                        # statement exist for the purpose of printing the
                        # message above
                        pass

                elif incgwet < 0 and iper > 0:
                    if model.verbose:
                        print(
                            "   Reusing CGWET array from kper "
                            "{0:5d}".format(iper) + " in kper "
                            "{0:5d}".format(iper + 1)
                        )

        if openfile:
            f.close()

        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=Mt3dUzt._ftype()
            )
            if icbcuz > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=icbcuz
                )
                model.add_pop_key_list(icbcuz)

        # Construct and return uzt package
        return cls(
            model,
            icbcuz=icbcuz,
            iet=iet,
            iuzfbnd=iuzfbnd,
            cuzinf=cuzinf,
            cuzet=cuzet,
            cgwet=cgwet,
            unitnumber=unitnumber,
            filenames=filenames,
            **kwargs
        )

    @staticmethod
    def _ftype():
        return "UZT2"

    @staticmethod
    def _defaultunit():
        return 7

    @staticmethod
    def _reservedunit():
        return 7
