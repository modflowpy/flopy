import sys
from ..pakbase import Package


class Mt3dAdv(Package):
    """
    MT3DMS Advection Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to which
        this package will be added.
    mixelm : int
        MIXELM is an integer flag for the advection solution option.
        MIXELM = 0, the standard finite-difference method with upstream or
        central-in-space weighting, depending on the value of NADVFD;
        = 1, the forward-tracking method of characteristics (MOC);
        = 2, the backward-tracking modified method of characteristics (MMOC);
        = 3, the hybrid method of characteristics (HMOC) with MOC or MMOC
        automatically and dynamically selected;
        = -1, the third-order TVD scheme (ULTIMATE).
    percel : float
        PERCEL is the Courant number (i.e., the number of cells, or a
        fraction of a cell) advection will be allowed in any direction in one
        transport step.
        For implicit finite-difference or particle-tracking-based schemes,
        there is no limit on PERCEL, but for accuracy reasons, it is generally
        not set much greater than one. Note, however, that the PERCEL limit is
        checked over the entire model grid. Thus, even if PERCEL > 1,
        advection may not be more than one cell's length at most model
        locations.
        For the explicit finite-difference or the third-order TVD scheme,
        PERCEL is also a stability constraint which must not exceed one and
        will be automatically reset to one if a value greater than one is
        specified.
    mxpart : int
        MXPART is the maximum total number of moving particles allowed and is
        used only when MIXELM = 1 or 3.
    nadvfd : int
        NADVFD is an integer flag indicating which weighting scheme should be
        used; it is needed only when the advection term is solved using the
        implicit finite- difference method.
        NADVFD = 0 or 1, upstream weighting (default); = 2,central-in-space
        weighting.
    itrack : int
        ITRACK is a flag indicating which particle-tracking algorithm is
        selected for the Eulerian-Lagrangian methods.
        ITRACK = 1, the first-order Euler algorithm is used.
        = 2, the fourth-order Runge-Kutta algorithm is used; this option is
        computationally demanding and may be needed only when PERCEL is set
        greater than one.
        = 3, the hybrid first- and fourth-order algorithm is used; the
        Runge-Kutta algorithm is used in sink/source cells and the cells next
        to sinks/sources while the Euler algorithm is used elsewhere.
    wd : float
        is a concentration weighting factor between 0.5 and 1. It is used for
        operator splitting in the particle- tracking-based methods. The value
        of 0.5 is generally adequate. The value of WD may be adjusted to
        achieve better mass balance. Generally, it can be increased toward
        1.0 as advection becomes more dominant.
    dceps : float
        is a small Relative Cell Concentration Gradient below which advective
        transport is considered
    nplane : int
        NPLANE is a flag indicating whether the random or
        fixed pattern is selected for initial placement of moving particles.
        If NPLANE = 0, the random pattern is selected for initial placement.
        Particles are distributed randomly in both the horizontal and vertical
        directions by calling a random number generator (Figure 18b). This
        option is usually preferred and leads to smaller mass balance
        discrepancy in nonuniform or diverging/converging flow fields.
        If NPLANE > 0, the fixed pattern is selected for initial placement.
        The value of NPLANE serves as the number of vertical 'planes' on
        which initial particles are placed within each cell block (Figure 18a).
        The fixed pattern may work better than the random pattern only in
        relatively uniform flow fields. For two-dimensional simulations in
        plan view, set NPLANE = 1. For cross sectional or three-dimensional
        simulations, NPLANE = 2 is normally adequate. Increase NPLANE if more
        resolution in the vertical direction is desired.
    npl : int
        NPL is the number of initial particles per cell to be placed at cells
        where the Relative Cell Concentration Gradient is less than or equal
        to DCEPS. Generally, NPL can be set to zero since advection is
        considered insignificant when the Relative Cell Concentration Gradient
        is less than or equal to DCEPS. Setting NPL equal to NPH causes a
        uniform number of particles to be placed in every cell over the entire
        grid (i.e., the uniform approach).
    nph : int
        NPH is the number of initial particles per cell to be placed at cells
        where the Relative Cell Concentration Gradient is greater than DCEPS.
        The selection of NPH depends on the nature of the flow field and also
        the computer memory limitation. Generally, a smaller number should be
        used in relatively uniform flow fields and a larger number should be
        used in relatively nonuniform flow fields. However, values exceeding
        16 in two-dimensional simulation or 32 in three- dimensional
        simulation are rarely necessary. If the random pattern is chosen, NPH
        particles are randomly distributed within the cell block. If the fixed
        pattern is chosen, NPH is divided by NPLANE to yield the number of
        particles to be placed per vertical plane, which is rounded to one of
        the values shown in Figure 30.
    npmin : int
        is the minimum number of particles allowed per cell. If the number of
        particles in a cell at the end of a transport step is fewer than
        NPMIN, new particles are inserted into that cell to maintain a
        sufficient number of particles. NPMIN can be set to zero in relatively
        uniform flow fields and to a number greater than zero in
        diverging/converging flow fields. Generally, a value between zero and
        four is adequate.
    npmax : int
        NPMAX is the maximum number of particles allowed per cell. If the
        number of particles in a cell exceeds NPMAX, all particles are removed
        from that cell and replaced by a new set of particles equal to NPH to
        maintain mass balance. Generally, NPMAX can be set to approximately
        two times of NPH.
    interp : int
        is a flag indicating the concentration interpolation method for use in
        the MMOC scheme. Currently, only linear interpolation is implemented.
    nlsink : int
        s a flag indicating whether the random or fixed pattern is selected
        for initial placement of particles to approximate sink cells in the
        MMOC scheme. The convention is the same as that for NPLANE. It is
        generally adequate to set NLSINK equivalent to NPLANE.
    npsink : int
        is the number of particles used to approximate sink cells in the MMOC
        scheme. The convention is the same as that for NPH. It is generally
        adequate to set NPSINK equivalent to NPH.
    dchmoc : float
        DCHMOC is the critical Relative Concentration Gradient for
        controlling the selective use of either MOC or MMOC in the HMOC
        solution scheme.
        The MOC solution is selected at cells where the Relative
        Concentration Gradient is greater than DCHMOC.
        The MMOC solution is selected at cells where the Relative
        Concentration Gradient is less than or equal to DCHMOC.
    extension : string
        Filename extension (default is 'adv')
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
    >>> adv = flopy.mt3d.Mt3dAdv(m)

    """

    def __init__(
        self,
        model,
        mixelm=3,
        percel=0.75,
        mxpart=800000,
        nadvfd=1,
        itrack=3,
        wd=0.5,
        dceps=1e-5,
        nplane=2,
        npl=10,
        nph=40,
        npmin=5,
        npmax=80,
        nlsink=0,
        npsink=15,
        dchmoc=0.0001,
        extension="adv",
        unitnumber=None,
        filenames=None,
    ):

        if unitnumber is None:
            unitnumber = Mt3dAdv._defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dAdv._reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dAdv._ftype()]
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

        self.mixelm = mixelm
        self.percel = percel
        self.mxpart = mxpart
        self.nadvfd = nadvfd
        self.mixelm = mixelm
        self.itrack = itrack
        self.wd = wd
        self.dceps = dceps
        self.nplane = nplane
        self.npl = npl
        self.nph = nph
        self.npmin = npmin
        self.npmax = npmax
        self.interp = 1  # Command-line 'interp' might once be needed if MT3DMS is updated to include other interpolation method
        self.nlsink = nlsink
        self.npsink = npsink
        self.dchmoc = dchmoc
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        f_adv = open(self.fn_path, "w")
        f_adv.write(
            "%10i%10f%10i%10i\n"
            % (self.mixelm, self.percel, self.mxpart, self.nadvfd)
        )
        if self.mixelm > 0:
            f_adv.write("%10i%10f\n" % (self.itrack, self.wd))
        if (self.mixelm == 1) or (self.mixelm == 3):
            f_adv.write(
                "%10.4e%10i%10i%10i%10i%10i\n"
                % (
                    self.dceps,
                    self.nplane,
                    self.npl,
                    self.nph,
                    self.npmin,
                    self.npmax,
                )
            )
        if (self.mixelm == 2) or (self.mixelm == 3):
            f_adv.write(
                "%10i%10i%10i\n" % (self.interp, self.nlsink, self.npsink)
            )
        if self.mixelm == 3:
            f_adv.write("%10f\n" % (self.dchmoc))
        f_adv.close()
        return

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
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
        adv :  Mt3dAdv object
            Mt3dAdv object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> adv = flopy.mt3d.Mt3dAdv.load('test.adv', m)

        """

        if model.verbose:
            sys.stdout.write("loading adv package file...\n")

        # Open file, if necessary
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # Item B1: MIXELM, PERCEL, MXPART, NADVFD - line already read above
        if model.verbose:
            print("   loading MIXELM, PERCEL, MXPART, NADVFD...")
        mixelm = int(line[0:10])
        percel = float(line[10:20])
        mxpart = 0
        if mixelm == 1 or mixelm == 3:
            if len(line[20:30].strip()) > 0:
                mxpart = int(line[20:30])
        nadvfd = 0
        if mixelm == 0:
            if len(line[30:40].strip()) > 0:
                nadvfd = int(line[30:40])
        if model.verbose:
            print("   MIXELM {}".format(mixelm))
            print("   PERCEL {}".format(nadvfd))
            print("   MXPART {}".format(mxpart))
            print("   NADVFD {}".format(nadvfd))

        # Item B2: ITRACK WD
        itrack = None
        wd = None
        if mixelm == 1 or mixelm == 2 or mixelm == 3:
            if model.verbose:
                print("   loading ITRACK, WD...")
            line = f.readline()
            itrack = int(line[0:10])
            wd = float(line[10:20])
            if model.verbose:
                print("   ITRACK {}".format(itrack))
                print("   WD {}".format(wd))

        # Item B3: DCEPS, NPLANE, NPL, NPH, NPMIN, NPMAX
        dceps = None
        nplane = None
        npl = None
        nph = None
        npmin = None
        npmax = None
        if mixelm == 1 or mixelm == 3:
            if model.verbose:
                print("   loading DCEPS, NPLANE, NPL, NPH, NPMIN, NPMAX...")
            line = f.readline()
            dceps = float(line[0:10])
            nplane = int(line[10:20])
            npl = int(line[20:30])
            nph = int(line[30:40])
            npmin = int(line[40:50])
            npmax = int(line[50:60])
            if model.verbose:
                print("   DCEPS {}".format(dceps))
                print("   NPLANE {}".format(nplane))
                print("   NPL {}".format(npl))
                print("   NPH {}".format(nph))
                print("   NPMIN {}".format(npmin))
                print("   NPMAX {}".format(npmax))

        # Item B4: INTERP, NLSINK, NPSINK
        interp = None
        nlsink = None
        npsink = None
        if mixelm == 2 or mixelm == 3:
            if model.verbose:
                print("   loading INTERP, NLSINK, NPSINK...")
            line = f.readline()
            interp = int(line[0:10])
            nlsink = int(line[10:20])
            npsink = int(line[20:30])
            if model.verbose:
                print("   INTERP {}".format(interp))
                print("   NLSINK {}".format(nlsink))
                print("   NPSINK {}".format(npsink))

        # Item B5: DCHMOC
        dchmoc = None
        if mixelm == 3:
            if model.verbose:
                print("   loading DCHMOC...")
            line = f.readline()
            dchmoc = float(line[0:10])
            if model.verbose:
                print("   DCHMOC {}".format(dchmoc))

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=Mt3dAdv._ftype()
            )

        # Construct and return adv package
        return cls(
            model,
            mixelm=mixelm,
            percel=percel,
            mxpart=mxpart,
            nadvfd=nadvfd,
            itrack=itrack,
            wd=wd,
            dceps=dceps,
            nplane=nplane,
            npl=npl,
            nph=nph,
            npmin=npmin,
            npmax=npmax,
            nlsink=nlsink,
            npsink=npsink,
            dchmoc=dchmoc,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "ADV"

    @staticmethod
    def _defaultunit():
        return 32

    @staticmethod
    def _reservedunit():
        return 2
