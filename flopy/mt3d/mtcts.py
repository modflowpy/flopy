from ..pakbase import Package

__author__ = "emorway"


class Mt3dCts(Package):
    """
    MT3D-USGS Contaminant Treatment System package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    mxcts : int
        The maximum number of contaminant transport systems implemented in a
        simulation.
    ictsout : int
        The unit number on which well-by-well output information is written.
        The default file extension assigned to the output file is TSO
    mxext : int
        The maximum number of extraction wells specified as part of a
        contaminant treatment system
    mxinj: int
        The maximum number of injection wells specified as part of a
        contaminant treatment system
    mxwel : int
        The maximum number of wells in the flow model. MXWEL is recommended
        to be set equal to MXWEL as specified in the WEL file
    iforce : int
        A flag to force concentration in treatment systems to satisfy
        specified concentration/mass values based on the treatment option
        selected without considering whether treatment is necessary or not.
        This flag is ignored if 'no treatment' option is selected.
           0   Concentration for all injection wells is set to satisfy
               treatment levels only if blended concentration exceeds
               the desired concentration/mass level for a treatment system.
               If the blended concentration in a treatment system is less
               than the specified concentration/mass level, then injection
               wells inject water with blended concentrations.
           1   Concentration for all injection wells is forced to satisfy
               specified concentration/mass values.
    ncts : int
        The number of contaminant treatment systems.  If NCTS >= 0,
        NCTS is the number of contaminant treatment systems. If NCTS = -1,
        treatment system information from the previous stress period is reused
        for the current stress period.
    icts : int
        The contaminant treatment system index number.
    next : int
        The number of extraction wells for the treatment system number ICTS.
    ninj : int
        The number of injection wells for the treatment system number ICTS.
    itrtinj : int
        Is the level of treatment provided for the treatment system number
        ICTS. Each treatment system blends concentration collected from all
        extraction wells contributing to the treatment system and assigns a
        treated concentration to all injection wells associated with that
        treatment system based on the treatment option selected
            0   no treatment is provided
            1   same level of treatment is provided to all injection wells.
            2   different level of treatment can be provided to each
                individual injection well.
    qincts : float
        The external flow entering a treatment system. External flow may be
        flow entering a treatment system that is not a part of the model
        domain but plays an important role in influencing the blended
        concentration of a treatment system
    cincts : float
        The concentration with which the external flow enters a treatment
        system
    ioptinj : int
        Is a treatment option. Negative values indicate removal of
        concentration/mass and positive values indicate addition of
        concentration/mass.
            1   Percentage concentration/mass addition/removal is performed.
                Percentages must be specified as fractions. Example, for 50%
                concentration/mass removal is desired, -0.5 must be specified.
            2   Concentration is added/removed from the blended concentration.
                Specified concentration CMCHGINJ is added to the blended
                concentration. If the specified concentration removal,
                CMCHGINJ, is greater than the blended concentration, the
                treated concentration is set to zero.
            3   Mass is added/removed from the blended concentration.
                Specified mass CMCHGINJ is added to the blended concentration.
                If the specified mass removal, CMCHGINJ, is greater than the
                blended total mass, the treated concentration is set to zero.
            4   Specified concentration is set equal to the entered value
                CMCHGINJ. A positive value is expected for CMCHGINJ with this
                option.
    cmchginj : float
        Is the addition, removal, or specified concentration/mass values set
        for the treatment system. Concentration/mass is added, removed, or
        used as specified concentrations depending on the treatment option
        IOPTINJ.
        Note that concentration/mass values as specified by CMCHGINJ are
        enforced if the option IFORCE is set to 1. If IFORCE is set to 0,
        then CMCHGINJ is enforced only when the blended concentration exceeds
        the specified concentration CNTE.
    cnte : float
        The concentration that is not to be exceeded for a treatment system.
        Treatment is applied to blended concentration only if it exceeds
        CNTE, when IFORCE is set to 0.
    kinj : int
        Layer index for a CTS injection well
    iinj : int
        Row index for a CTS injection well
    jinj : int
        Column index for a CTS injection well
    iwinj : int
        The well index number. This number corresponds to the well number as
        it appears in the WEL file of the flow model.
    qoutcts : float
        the flow rate of outflow from a treatment system to an external sink.
        This flow rate must be specified to maintain an overall treatment
        system mass balance. QOUTCTS must be set equal to total inflow into a
        treatment system minus total outflow to all injection wells for a
        treatment system

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

    >>>
    >>>
    >>>
    >>>

    """

    def __init__(
        self,
    ):
        raise NotImplementedError()
        # # unit number
        # if unitnumber is None:
        #     unitnumber = self.unitnumber
        # Package.__init__(self, model, extension, 'CTS', self.unitnumber)
        #
        # # Set dimensions
        # nrow = model.nrow
        # ncol = model.ncol
        # nlay = model.nlay
        # ncomp = model.ncomp
        # mcomp = model.mcomp

        # Set package specific parameters

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
        cts : Mt3dCts object
            Mt3dCts object

        Examples
        --------

        >>>

        """

        raise NotImplementedError()

        # if model.verbose:
        #     sys.stdout.write('loading cts package file...\n')
        #
        # # Open file, if necessary
        # openfile = not hasattr(f, 'read')
        # if openfile:
        #     filename = f
        #     f = open(filename, 'r')
        #
        # # Set dimensions if necessary
        # if nlay is None:
        #     nlay = model.nlay
        # if nrow is None:
        #     nrow = model.nrow
        # if ncol is None:
        #     ncol = model.ncol
        # if nper is None:
        #     nper = model.nper
        # if ncomp is None:
        #     ncomp = model.ncomp
        #
        # # Item 1 (MXCTS, ICTSOUT, MXEXT, MXINJ, MXWEL, IFORCE)
        # line = f.readline()
        # if line[0] == '#':
        #     raise ValueError('CTS package does not support comment lines')
        # if model.verbose:
        #     print('   loading MXCTS, ICTSOUT, MXEXT, MXINJ, MXWEL, IFORCE...')
        #
        # m_arr = line.strip().split()
        # mxcts = int(m_arr[0])
        # ictsout = int(m_arr[1])
        # mxext = int(m_arr[2])
        # mxinj = int(m_arr[3])
        # mxwel = int(m_arr[4])
        # iforce = int(m_arr[5])
        #
        # # Start of transient data
        # for iper in range(nper):
        #
        #     if model.verbose:
        #         print('   loading CTS data for kper {0:5d}'.format(iper + 1))
        #
        #     # Item 2 (NCTS)
        #     line = f.readline()
        #     m_arr = line.strip().split()
        #     ncts = int(m_arr[0])
        #
        #     # Start of information for each CTS
        #     for icts in range(ncts):
        #
        #         if model.verbose:
        #             print('   loading data for system #{0:5d}'
        #                   .format(icts + 1))
        #         # Item 3 (ICTS, NEXT, NINJ, ITRTINJ)
        #         line = f.readline()
        #         m_arr = line.strip().split()
        #         icts = int(m_arr[0])
        #         next = int(m_arr[1])
        #         ninj = int(m_arr[2])
        #         itrtinj = int(m_arr[3])
        #
        # if openfile:
        #     f.close()

    @staticmethod
    def get_default_CTS_dtype(ncomp=1, iforce=0):
        """
        Construct a dtype for the recarray containing the list of cts systems
        """

        raise NotImplementedError()

        # # Item 3
        # type_list = [("icts", int), ("next", int), ("ninj", int),
        #              ("itrtinj", int)]
        #
        # # Create a list for storing items 5, 6, & 9
        # items_5_6_7_9_list = []
        # if ncomp > 1:
        #     # Item 5 in CTS input
        #     for comp in range(1, ncomp+1):
        #         qincts_name = "qincts{0:d}".format(comp)
        #         cincts_name = "cincts{0:d}".format(comp)
        #         items_5_6_7_9_list.append((qincts_name, np.float32))
        #         items_5_6_7_9_list.append((cincts_name, np.float32))
        #
        #     # Item 6 in CTS input
        #     for comp in range(1, ncomp+1):
        #         ioptinj_name = "ioptinj{0:d}".format(comp)
        #         cmchginj_name = "cmchginj{0:d}".format(comp)
        #         items_5_6_7_9_list.append((ioptinj_name, int))
        #         items_5_6_7_9_list.append((cmchginj_name, np.float32))
        #
        #     if iforce == 0:
        #         for comp in range(1, ncomp+1):
        #             cnte_name = "cnte{0:d}".format(comp)
        #             items_5_6_7_9_list.append(cnte_name, np.float32)
        #
        #     # Item 9 in CTS input
        #     items_5_6_7_9_list.append(("qoutcts", np.float32))
        #
        # type_list.append(items_5_6_7_9_list)
        #
        # # Now create a list for the records in Item 4
        # ext_wels_list = [("kext", int), ("iext", int), ("jext", int),
        #                  ("iwext", int)]
        #
        # type_list.append(ext_wels_list)
        #
        # # Now create a list for the records in Item 8
        # inj_wels_list = [("kinj", int), ("iinj", int), ("jinj", int),
        #                  ("iwinj", int)]
        # type_list.append(inj_wels_list)
        #
        # #
        #
        # dtype = np.dtype(type_list)
        # dtype = dtype

    @staticmethod
    def _ftype():
        return "CTS"

    @staticmethod
    def _defaultunit():
        return 5

    @staticmethod
    def _reservedunit():
        return 5
