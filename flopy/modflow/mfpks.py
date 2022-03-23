"""
mfpks module.  Contains the ModflowPks class. Note that the user can access
the ModflowPks class as `flopy.modflow.ModflowPks`.

"""
from ..pakbase import Package


class ModflowPks(Package):
    """
    MODFLOW Pks Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mxiter : int
        maximum number of outer iterations. (default is 100)
    innerit : int
        maximum number of inner iterations. (default is 30)
    hclose : float
        is the head change criterion for convergence. (default is 1.e-3).
    rclose : float
        is the residual criterion for convergence. (default is 1.e-1)
    relax : float
        is the relaxation parameter used with npcond = 1. (default is 1.0)
    .
    .
    .
    iprpks : int
        solver print out interval. (default is 0).
    mutpks : int
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
    extension : list string
        Filename extension (default is 'pks')
    unitnumber : int
        File unit number (default is 27).
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
    >>> pks = flopy.modflow.ModflowPks(m)

    """

    def __init__(
        self,
        model,
        mxiter=100,
        innerit=50,
        isolver=1,
        npc=2,
        iscl=0,
        iord=0,
        ncoresm=1,
        ncoresv=1,
        damp=1.0,
        dampt=1.0,
        relax=0.97,
        ifill=0,
        droptol=0.0,
        hclose=1e-3,
        rclose=1e-1,
        l2norm=None,
        iprpks=0,
        mutpks=3,
        mpi=False,
        partopt=0,
        novlapimpsol=1,
        stenimpsol=2,
        verbose=0,
        partdata=None,
        extension="pks",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowPks._defaultunit()

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )
        # check if a valid model version has been specified
        if model.version == "mf2k" or model.version == "mfnwt":
            err = "Error: cannot use {} package with model version {}".format(
                self.name, model.version
            )
            raise Exception(err)

        self._generate_heading()
        self.mxiter = mxiter
        self.innerit = innerit
        self.isolver = isolver
        self.npc = npc
        self.iscl = iscl
        self.iord = iord
        self.ncoresm = ncoresm
        self.ncoresv = ncoresv
        self.damp = damp
        self.dampt = dampt
        self.relax = relax
        self.ifill = ifill
        self.droptol = droptol
        self.hclose = hclose
        self.rclose = rclose
        self.l2norm = l2norm
        self.iprpks = iprpks
        self.mutpks = mutpks
        # MPI
        self.mpi = mpi
        self.partopt = partopt
        self.novlapimpsol = novlapimpsol
        self.stenimpsol = stenimpsol
        self.verbose = verbose
        self.partdata = partdata

        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing
        f = open(self.fn_path, "w")
        f.write(f"{self.heading}\n")
        f.write(f"MXITER {self.mxiter}\n")
        f.write(f"INNERIT {self.innerit}\n")
        f.write(f"ISOLVER {self.isolver}\n")
        f.write(f"NPC {self.npc}\n")
        f.write(f"ISCL {self.iscl}\n")
        f.write(f"IORD {self.iord}\n")
        if self.ncoresm > 1:
            f.write(f"NCORESM {self.ncoresm}\n")
        if self.ncoresv > 1:
            f.write(f"NCORESV {self.ncoresv}\n")
        f.write(f"DAMP {self.damp}\n")
        f.write(f"DAMPT {self.dampt}\n")
        if self.npc > 0:
            f.write(f"RELAX {self.relax}\n")
        if self.npc == 3:
            f.write(f"IFILL {self.ifill}\n")
            f.write(f"DROPTOL {self.droptol}\n")
        f.write(f"HCLOSEPKS {self.hclose}\n")
        f.write(f"RCLOSEPKS {self.rclose}\n")
        if self.l2norm != None:
            if self.l2norm.lower() == "l2norm" or self.l2norm == "1":
                f.write("L2NORM\n")
            elif self.l2norm.lower() == "rl2norm" or self.l2norm == "2":
                f.write("RELATIVE-L2NORM\n")
        f.write(f"IPRPKS {self.iprpks}\n")
        f.write(f"MUTPKS {self.mutpks}\n")
        # MPI
        if self.mpi:
            f.write(f"PARTOPT {self.partopt}\n")
            f.write(f"NOVLAPIMPSOL {self.novlapimpsol}\n")
            f.write(f"STENIMPSOL {self.stenimpsol}\n")
            f.write(f"VERBOSE {self.verbose}\n")
            if self.partopt == 1 | 2:
                pass
                # to be implemented

        f.write("END\n")
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
        pks : ModflowPks object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> pks = flopy.modflow.ModflowPks.load('test.pks', m)

        """

        if model.verbose:
            print("loading pks package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header

        print(
            "   Warning: "
            "load method not completed. default pks object created."
        )

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowPks._ftype()
            )

        return cls(model, unitnumber=unitnumber, filenames=filenames)

    @staticmethod
    def _ftype():
        return "PKS"

    @staticmethod
    def _defaultunit():
        return 27
