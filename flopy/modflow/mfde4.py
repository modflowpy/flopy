"""
mfde4 module.  Contains the ModflowDe4 class. Note that the user can access
the ModflowDe4 class as `flopy.modflow.ModflowDe4`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/de4.htm>`_.

"""
import sys
from ..pakbase import Package


class ModflowDe4(Package):
    """
    MODFLOW DE4 - Direct Solver Package
    
    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    itmx : int
        Maximum number of iterations for each time step. Specify ITMAX = 1 if 
        iteration is not desired. Ideally iteration would not be required for 
        direct solution. However, it is necessary to iterate if the flow 
        equation is nonlinear or if computer precision limitations result in 
        inaccurate calculations as indicated by a large water budget error 
        (default is 50).
    mxup : int
        Maximum number of equations in the upper part of the equations to be 
        solved. This value impacts the amount of memory used by the DE4 
        Package. If specified as 0, the program will calculate MXUP as half 
        the number of cells in the model, which is an upper limit (default 
        is 0).
    mxlow : int
        Maximum number of equations in the lower part of equations to be 
        solved. This value impacts the amount of memory used by the DE4 
        Package. If specified as 0, the program will calculate MXLOW as half 
        the number of cells in the model, which is an upper limit (default is 
        0).
    mxbw : int
        Maximum band width plus 1 of the lower part of the head coefficients 
        matrix. This value impacts the amount of memory used by the DE4 
        Package. If specified as 0, the program will calculate MXBW as the 
        product of the two smallest grid dimensions plus 1, which is an 
        upper limit (default is 0).
    ifreq : int
        Flag indicating the frequency at which coefficients in head matrix 
        change.
        IFREQ = 1 indicates that the flow equations are linear and that 
        coefficients of simulated head for all stress terms are constant 
        for all stress periods. 
        IFREQ = 2 indicates that the flow equations are linear, but 
        coefficients of simulated head for some stress terms may change 
        at the start of each stress period.
        IFREQ = 3 indicates that a nonlinear flow equation is being solved, 
        which means that some terms in the head coefficients matrix depend 
        on simulated head (default is 3).
    mutd4 : int
        Flag that indicates the quantity of information that is printed when 
        convergence information is printed for a time step.
        MUTD4 = 0 indicates that the number of iterations in the time step 
        and the maximum head change each iteration are printed.
        MUTD4 = 1 indicates that only the number of iterations in the time 
        step is printed.
        MUTD4 = 2 indicates no information is printed (default is 0).
    accl : int
        Multiplier for the computed head change for each iteration. Normally 
        this value is 1. A value greater than 1 may be useful for improving 
        the rate of convergence when using external iteration to solve 
        nonlinear problems (default is 1).
    hclose : float
        Head change closure criterion. If iterating (ITMX > 1), iteration 
        stops when the absolute value of head change at every node is less 
        than or equal to HCLOSE. HCLOSE is not used if not iterating, but a 
        value must always be specified (default is 1e-5).
    iprd4 : int
        Time step interval for printing out convergence information when 
        iterating (ITMX > 1). If IPRD4 is 2, convergence information is 
        printed every other time step. A value must always be specified 
        even if not iterating (default is 1).
    extension : string
        Filename extension (default is 'de4')
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
    >>> de4 = flopy.modflow.ModflowDe4(m)

    """

    def __init__(self, model, itmx=50, mxup=0, mxlow=0, mxbw=0,
                 ifreq=3, mutd4=0, accl=1., hclose=1e-5, iprd4=1,
                 extension='de4', unitnumber=None, filenames=None):
        """
        Package constructor.

        """

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDe4.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowDe4.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        # check if a valid model version has been specified
        if model.version == 'mfusg':
            err = 'Error: cannot use {} package with model version {}'.format(
                self.name, model.version)
            raise Exception(err)

        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
        self.url = 'de4.htm'

        self.itmx = itmx
        self.mxup = mxup
        self.mxlow = mxlow
        self.mxbw = mxbw
        self.ifreq = ifreq
        self.mutd4 = mutd4
        self.accl = accl
        self.hclose = hclose
        self.iprd4 = iprd4
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing
        f = open(self.fn_path, 'w')
        f.write('%s\n' % self.heading)
        ifrfm = self.parent.get_ifrefm()
        if ifrfm:
            f.write('{0} '.format(self.itmx))
            f.write('{0} '.format(self.mxup))
            f.write('{0} '.format(self.mxlow))
            f.write('{0} '.format(self.mxbw))
            f.write('\n')
            f.write('{0} '.format(self.ifreq))
            f.write('{0} '.format(self.mutd4))
            f.write('{0} '.format(self.accl))
            f.write('{0} '.format(self.hclose))
            f.write('{0} '.format(self.iprd4))
            f.write('\n')
        else:
            f.write('{0:10d}'.format(self.itmx))
            f.write('{0:10d}'.format(self.mxup))
            f.write('{0:10d}'.format(self.mxlow))
            f.write('{0:10d}'.format(self.mxbw))
            f.write('\n')
            f.write('{0:10d}'.format(self.ifreq))
            f.write('{0:10d}'.format(self.mutd4))
            f.write('{0:9.4e} '.format(self.accl))
            f.write('{0:9.4e} '.format(self.hclose))
            f.write('{0:10d}'.format(self.iprd4))
            f.write('\n')
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
        de4 : ModflowDe4 object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> de4 = flopy.modflow.ModflowDe4.load('test.de4', m)

        """

        if model.verbose:
            sys.stdout.write('loading de4 package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        # read dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        # read dataset 1
        ifrfm = model.get_ifrefm()
        if model.version != 'mf2k':
            ifrfm = True
        ifreq = 1
        if ifrfm:
            t = line.strip().split()
            itmx = int(t[0])
            mxup = int(t[1])
            mxlow = int(t[2])
            mxbw = int(t[3])
            line = f.readline()
            t = line.strip().split()
            ifreq = int(t[0])
            mutd4 = int(t[1])
            accl = float(t[2])
            hclose = float(t[3])
            iprd4 = int(t[4])
        else:
            itmx = int(line[0:10].strip())
            mxup = int(line[10:20].strip())
            mxlow = int(line[20:30].strip())
            mxbw = int(tline[30:40].strip())
            line = f.readline()
            ifreq = int(line[0:10].strip())
            mutd4 = int(line[10:20].strip())
            accl = float(line[20:30].strip())
            hclose = float(line[30:40].strip())
            iprd4 = int(line[40:50].strip())

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowDe4.ftype())

        de4 = ModflowDe4(model, itmx=itmx, mxup=mxup, mxlow=mxlow, mxbw=mxbw,
                         ifreq=ifreq, mutd4=mutd4, accl=accl, hclose=hclose,
                         iprd4=iprd4, unitnumber=unitnumber,
                         filenames=filenames)
        return de4


    @staticmethod
    def ftype():
        return 'DE4'


    @staticmethod
    def defaultunit():
        return 28
