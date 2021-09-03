"""
mfpcg module.  Contains the ModflowPcg class. Note that the user can access
the ModflowPcg class as `flopy.modflow.ModflowPcg`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/pcg.htm>`_.

"""
from ..pakbase import Package
from ..utils.flopy_io import line_parse


class ModflowPcg(Package):
    """
    MODFLOW Pcg Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mxiter : int
        maximum number of outer iterations. (default is 50)
    iter1 : int
        maximum number of inner iterations. (default is 30)
    npcond : int
        flag used to select the matrix conditioning method. (default is 1).
        specify npcond = 1 for Modified Incomplete Cholesky.
        specify npcond = 2 for Polynomial.
    hclose : float
        is the head change criterion for convergence. (default is 1e-5).
    rclose : float
        is the residual criterion for convergence. (default is 1e-5)
    relax : float
        is the relaxation parameter used with npcond = 1. (default is 1.0)
    nbpol : int
        is only used when npcond = 2 to indicate whether the estimate of the
        upper bound on the maximum eigenvalue is 2.0, or whether the estimate
        will be calculated. nbpol = 2 is used to specify the value is 2.0;
        for any other value of nbpol, the estimate is calculated. Convergence
        is generally insensitive to this parameter. (default is 0).
    iprpcg : int
        solver print out interval. (default is 0).
    mutpcg : int
        If mutpcg = 0, tables of maximum head change and residual will be
        printed each iteration.
        If mutpcg = 1, only the total number of iterations will be printed.
        If mutpcg = 2, no information will be printed.
        If mutpcg = 3, information will only be printed if convergence fails.
        (default is 3).
    damp : float
        is the steady-state damping factor. (default is 1.)
    dampt : float
        is the transient damping factor. (default is 1.)
    ihcofadd : int
        is a flag that determines what happens to an active cell that is
        surrounded by dry cells.  (default is 0). If ihcofadd=0, cell
        converts to dry regardless of HCOF value. This is the default, which
        is the way PCG2 worked prior to the addition of this option. If
        ihcofadd<>0, cell converts to dry only if HCOF has no head-dependent
        stresses or storage terms.
    extension : list string
        Filename extension (default is 'pcg')
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
    >>> m = flopy.modflow.Modflow()
    >>> pcg = flopy.modflow.ModflowPcg(m)

    """

    def __init__(
        self,
        model,
        mxiter=50,
        iter1=30,
        npcond=1,
        hclose=1e-5,
        rclose=1e-5,
        relax=1.0,
        nbpol=0,
        iprpcg=0,
        mutpcg=3,
        damp=1.0,
        dampt=1.0,
        ihcofadd=0,
        extension="pcg",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowPcg._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowPcg._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        # check if a valid model version has been specified
        if model.version == "mfusg":
            err = "Error: cannot use {} package with model version {}".format(
                self.name, model.version
            )
            raise Exception(err)

        self._generate_heading()
        self.url = "pcg.htm"
        self.mxiter = mxiter
        self.iter1 = iter1
        self.npcond = npcond
        self.hclose = hclose
        self.rclose = rclose
        self.relax = relax
        self.nbpol = nbpol
        self.iprpcg = iprpcg
        self.mutpcg = mutpcg
        self.damp = damp
        self.dampt = dampt
        self.ihcofadd = ihcofadd
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, "w")
        f.write(f"{self.heading}\n")
        ifrfm = self.parent.get_ifrefm()
        if ifrfm:
            f.write(f"{self.mxiter} ")
            f.write(f"{self.iter1} ")
            f.write(f"{self.npcond} ")
            f.write(f"{self.ihcofadd}")
            f.write("\n")
            f.write(f"{self.hclose} ")
            f.write(f"{self.rclose} ")
            f.write(f"{self.relax} ")
            f.write(f"{self.nbpol} ")
            f.write(f"{self.iprpcg} ")
            f.write(f"{self.mutpcg} ")
            f.write(f"{self.damp} ")
            if self.damp < 0:
                f.write(f"{self.dampt}")
            f.write("\n")
        else:
            f.write(f" {self.mxiter:9d}")
            f.write(f" {self.iter1:9d}")
            f.write(f" {self.npcond:9d}")
            f.write(f" {self.ihcofadd:9d}")
            f.write("\n")
            f.write(f" {self.hclose:9.3e}")
            f.write(f" {self.rclose:9.3e}")
            f.write(f" {self.relax:9.3e}")
            f.write(f" {self.nbpol:9d}")
            f.write(f" {self.iprpcg:9d}")
            f.write(f" {self.mutpcg:9d}")
            f.write(f" {self.damp:9.3e}")
            if self.damp < 0:
                f.write(f" {self.dampt:9.3e}")
            f.write("\n")
        f.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        pcg : ModflowPcg object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> pcg = flopy.modflow.ModflowPcg.load('test.pcg', m)

        """

        if model.verbose:
            print("loading pcg package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        # dataset 1
        ifrfm = model.get_ifrefm()
        if model.version != "mf2k":
            ifrfm = True
        ihcofadd = 0
        dampt = 0.0

        # free format
        if ifrfm:
            t = line_parse(line)
            # t = line.strip().split()
            mxiter = int(t[0])
            iter1 = int(t[1])
            npcond = int(t[2])
            try:
                ihcofadd = int(t[3])
            except:
                if model.verbose:
                    print("   explicit ihcofadd in file")

            # dataset 2
            try:
                line = f.readline()
                t = line_parse(line)
                # t = line.strip().split()
                hclose = float(t[0])
                rclose = float(t[1])
                relax = float(t[2])
                nbpol = int(t[3])
                iprpcg = int(t[4])
                mutpcg = int(t[5])
                damp = float(t[6])
                if damp < 0.0:
                    dampt = float(t[7])
            except ValueError:
                hclose = float(line[0:10].strip())
                rclose = float(line[10:20].strip())
                relax = float(line[20:30].strip())
                nbpol = int(line[30:40].strip())
                iprpcg = int(line[40:50].strip())
                mutpcg = int(line[50:60].strip())
                damp = float(line[60:70].strip())
                if damp < 0.0:
                    dampt = float(line[70:80].strip())
        # fixed format
        else:
            mxiter = int(line[0:10].strip())
            iter1 = int(line[10:20].strip())
            npcond = int(line[20:30].strip())
            try:
                ihcofadd = int(line[30:40].strip())
            except:
                if model.verbose:
                    print("   explicit ihcofadd in file")

            # dataset 2
            line = f.readline()
            hclose = float(line[0:10].strip())
            rclose = float(line[10:20].strip())
            relax = float(line[20:30].strip())
            nbpol = int(line[30:40].strip())
            iprpcg = int(line[40:50].strip())
            mutpcg = int(line[50:60].strip())
            damp = float(line[60:70].strip())
            if damp < 0.0:
                dampt = float(line[70:80].strip())

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowPcg._ftype()
            )

        return cls(
            model,
            mxiter=mxiter,
            iter1=iter1,
            npcond=npcond,
            ihcofadd=ihcofadd,
            hclose=hclose,
            rclose=rclose,
            relax=relax,
            nbpol=nbpol,
            iprpcg=iprpcg,
            mutpcg=mutpcg,
            damp=damp,
            dampt=dampt,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "PCG"

    @staticmethod
    def _defaultunit():
        return 27
