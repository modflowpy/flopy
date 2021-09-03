"""
mfsip module.  Contains the ModflowSip class. Note that the user can access
the ModflowSip class as `flopy.modflow.ModflowSip`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?sip.htm>`_.

"""
from ..pakbase import Package


class ModflowSip(Package):
    """
    MODFLOW Strongly Implicit Procedure Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:flopy.modflow.mf.Modflow) to which
        this package will be added.
    mxiter : integer
        The maximum number of times through the iteration loop in one time
        step in an attempt to solve the system of finite-difference equations.
        (default is 200)
    nparm : integer
        The number of iteration variables to be used.
        Five variables are generally sufficient. (default is 5)
    accl : float
        The acceleration variable, which must be greater than zero
        and is generally equal to one. If a zero is entered,
        it is changed to one. (default is 1)
    hclose : float > 0
        The head change criterion for convergence. When the maximum absolute
        value of head change from all nodes during an iteration is less than
        or equal to hclose, iteration stops. (default is 1e-5)
    ipcalc : 0 or 1
        A flag indicating where the seed for calculating iteration variables
        will come from. 0 is the seed entered by the user will be used.
        1 is the seed will be calculated at the start of the simulation from
        problem variables. (default is 0)
    wseed : float > 0
        The seed for calculating iteration variables. wseed is always read,
        but is used only if ipcalc is equal to zero. (default is 0)
    iprsip : integer > 0
        the printout interval for sip. iprsip, if equal to zero, is changed
        to 999. The maximum head change (positive or negative) is printed for
        each iteration of a time step whenever the time step is an even
        multiple of iprsip. This printout also occurs at the end of each
        stress period regardless of the value of iprsip. (default is 0)
    extension : string
        Filename extension (default is 'sip')
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
    >>> ml = flopy.modflow.Modflow()
    >>> sip = flopy.modflow.ModflowSip(ml, mxiter=100, hclose=0.0001)

    """

    def __init__(
        self,
        model,
        mxiter=200,
        nparm=5,
        accl=1,
        hclose=1e-5,
        ipcalc=1,
        wseed=0,
        iprsip=0,
        extension="sip",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowSip._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowSip._ftype()]
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
            raise Exception(
                f"Error: cannot use {self.name} package "
                f"with model version {model.version}"
            )

        self._generate_heading()
        self.url = "sip.htm"

        self.mxiter = mxiter
        self.nparm = nparm
        self.accl = accl
        self.hclose = hclose
        self.ipcalc = ipcalc
        self.wseed = wseed
        self.iprsip = iprsip
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
        ifrfm = self.parent.get_ifrefm()
        if ifrfm:
            f.write(f"{self.mxiter} {self.nparm}\n")
            f.write(
                f"{self.accl} {self.hclose} {self.ipcalc} {self.wseed} {self.iprsip}\n"
            )
        else:
            f.write(f"{self.mxiter:10d}{self.nparm:10d}\n")
            f.write(
                "{:10.3f}{:10.3g}{:10d}{:10.3f}{:10d}\n".format(
                    self.accl,
                    self.hclose,
                    self.ipcalc,
                    self.wseed,
                    self.iprsip,
                )
            )
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
        sip : ModflowSip object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> sip = flopy.modflow.ModflowSip.load('test.sip', m)

        """

        if model.verbose:
            print("loading sip package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        ifrfm = model.get_ifrefm()
        # dataset 1
        if ifrfm:
            t = line.strip().split()
            mxiter = int(t[0])
            nparm = int(t[1])
        else:
            mxiter = int(line[0:10].strip())
            nparm = int(line[10:20].strip())
        # dataset 2
        line = f.readline()
        if ifrfm:
            t = line.strip().split()
            accl = float(t[0])
            hclose = float(t[1])
            ipcalc = int(t[2])
            wseed = float(t[3])
            iprsip = int(t[4])
        else:
            accl = float(line[0:10].strip())
            hclose = float(line[10:20].strip())
            ipcalc = int(line[20:30].strip())
            wseed = float(line[30:40].strip())
            iprsip = int(line[40:50].strip())

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowSip._ftype()
            )

        return cls(
            model,
            mxiter=mxiter,
            nparm=nparm,
            accl=accl,
            hclose=hclose,
            ipcalc=ipcalc,
            wseed=wseed,
            iprsip=iprsip,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "SIP"

    @staticmethod
    def _defaultunit():
        return 25
