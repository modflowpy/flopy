import sys
from flopy.mbase import Package

class ModflowSip(Package):
    """
    MODFLOW Strongly Implicit Procedure Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:flopy.modflow.mf.Modflow) to which this package will be added.
    mxiter : integer
        The maximum number of times through the iteration loop in one time step in an attempt to solve the
        system of finite-difference equations. (default is 200)
    nparm : integer
        The number of iteration variables to be used.
        Five variables are generally sufficient. (default is 5)
    accl : float
        The acceleration variable, which must be greater than zero
        and is generally equal to one. If a zero is entered,
        it is changed to one. (default is 1)
    hclose : float > 0
        The head change criterion for convergence. When the maximum absolute value of head change from all nodes
        during an iteration is less than or equal to HCLOSE, iteration stops. (default is 1e-5)
    ipcalc : 0 or 1
        A flag indicating where the seed for calculating iteration variables will come from.
            0 is the seed entered by the user will be used.
            1 is the seed will be calculated at the start of the simulation from problem variables.
        (default is 0)
    wseed : float > 0
        The seed for calculating iteration variables. WSEED is always read,
        but is used only if IPCALC is equal to zero. (default is 0)
    iprsip : integer > 0
        the printout interval for SIP. IPRSIP, if equal to zero, is changed to 999.
        The maximum head change (positive or negative) is printed for each iteration of
        a time step whenever the time step is an even multiple of IPRSIP. This printout
        also occurs at the end of each stress period regardless of the value of IPRSIP.
        (default is 0)
    extension : string
        Filename extension (default is 'sip')
    unitnumber : int
        File unit number (default is 25).

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

    def __init__(self, model, mxiter=200, nparm=5, \
                 accl=1, hclose=1e-5, ipcalc=1, wseed=0, iprsip=0, extension='sip', unitnumber=25):
        """
        Package constructor.

        """
        Package.__init__(self, model, extension, 'SIP', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.url = 'sip.htm'
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
        Write the package input file.

        """
        # Open file for writing
        f = open(self.fn_path, 'w')
        f.write('{:10d}{:10d}\n'.format(self.mxiter, self.nparm))
        f.write('{:10.3f}{:10.3f}{:10d}{:10.3f}{:10d}\n'.format(self.accl, self.hclose, self.ipcalc, self.wseed, self.iprsip))
        f.close()

    @staticmethod
    def load(f, model, ext_unit_dict=None):
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
            sys.stdout.write('loading sip package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header

        print('   Warning: load method not completed. default sip object created.')

        # close the open file
        f.close()

        sip = ModflowSip(model)
        return sip
