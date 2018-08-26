"""
mfswi2 module.  Contains the ModflowSwi2 class. Note that the user can access
the ModflowSwi2 class as `flopy.modflow.ModflowSwi2`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/swi2_seawater_intrusion_pack.htm>`_.

"""
import sys

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d


class ModflowSwi2(Package):
    """
    MODFLOW SWI2 Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    nsrf : int
        number of active surfaces (interfaces). This equals the number of zones
        minus one. (default is 1).
    istrat : int
        flag indicating the density distribution. (default is 1).
    iswizt : int
        unit number for zeta output. (default is None).
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is None).
    iswiobs : int
        flag and unit number SWI2 observation output. (default is 0).
    options : list of strings
        Package options. If 'adaptive' is one of the options adaptive SWI2 time 
        steps will be used. (default is None).
    nsolver : int
        DE4 solver is used if nsolver=1. PCG solver is used if nsolver=2.
        (default is 1).
    iprsol : int
        solver print out interval. (default is 0).
    mutsol : int
        If MUTSOL = 0, tables of maximum head change and residual will be
        printed each iteration.
        If MUTSOL = 1, only the total number of iterations will be printed.
        If MUTSOL = 2, no information will be printed.
        If MUTSOL = 3, information will only be printed if convergence fails.
        (default is 3).
    solver2parameters : dict
        only used if nsolver = 2

        mxiter : int
            maximum number of outer iterations. (default is 100)

        iter1 : int
            maximum number of inner iterations. (default is 20)

        npcond : int
            flag used to select the matrix conditioning method. (default is 1).
            specify NPCOND = 1 for Modified Incomplete Cholesky.
            specify NPCOND = 2 for Polynomial.

        zclose : float
            is the ZETA change criterion for convergence. (default is 1e-3).

        rclose : float
            is the residual criterion for convergence. (default is 1e-4)

        relax : float
            is the relaxation parameter used with NPCOND = 1. (default is 1.0)

        nbpol : int
            is only used when NPCOND = 2 to indicate whether the estimate of
            the upper bound on the maximum eigenvalue is 2.0, or whether the
            estimate will be calculated. NBPOL = 2 is used to specify the
            value is 2.0; for any other value of NBPOL, the estimate is
            calculated. Convergence is generally insensitive to this
            parameter. (default is 2).

        damp : float
            is the steady-state damping factor. (default is 1.)

        dampt : float
            is the transient damping factor. (default is 1.)

    toeslope : float
        Maximum slope of toe cells. (default is 0.05)
    tipslope : float
        Maximum slope of tip cells. (default is 0.05)
    alpha : float
        fraction of threshold used to move the tip and toe to adjacent empty cells
        when the slope exceeds user-specified TOESLOPE and TIPSLOPE values. (default is None)
    beta : float
        Fraction of threshold used to move the toe to adjacent non-empty cells when the
        surface is below a minimum value defined by the user-specified TOESLOPE value.
        (default is 0.1).
    napptmx : int
        only used if adaptive is True. Maximum number of SWI2 time steps per MODFLOW
        time step. (default is 1).
    napptmn : int
        only used if adaptive is True. Minimum number of SWI2 time steps per MODFLOW
        time step. (default is 1).
    adptfct : float
        is the factor used to evaluate tip and toe thicknesses and control the number
        of SWI2 time steps per MODFLOW time step. When the maximum tip or toe thickness
        exceeds the product of TOESLOPE or TIPSLOPE the cell size and ADPTFCT, the number
        of SWI2 time steps are increased to a value less than or equal to NADPT.
        When the maximum tip or toe thickness is less than the product of TOESLOPE or
        TIPSLOPE the cell size and ADPTFCT, the number of SWI2 time steps is decreased
        in the next MODFLOW time step to a value greater than or equal to 1. ADPTFCT
        must be greater than 0.0 and is reset to 1.0 if NADPTMX is equal to NADPTMN.
        (default is 1.0).
    nu : array of floats
        if istart = 1, density of each zone (nsrf + 1 values). if istrat = 0,
        density along top of layer, each surface, and bottom of layer
        (nsrf + 2 values). (default is 0.025)
    zeta : list of floats or list of array of floats [(nlay, nrow, ncol),
        (nlay, nrow, ncol)] initial elevations of the active surfaces. The
        list should contain an entry for each surface and be of size nsrf.
        (default is [0.])
    ssz : float or array of floats (nlay, nrow, ncol)
        effective porosity. (default is 0.25)
    isource : integer or array of integers (nlay, nrow, ncol)
        Source type of any external sources or sinks, specified with any outside package
        (i.e. WEL Package, RCH Package, GHB Package). (default is 0).
        If ISOURCE > 0 sources and sinks have the same fluid density as the
        zone ISOURCE. If such a zone is not present in the cell, sources and
        sinks have the same fluid density as the active zone at the top of
        the aquifer. If ISOURCE = 0 sources and sinks have the same fluid
        density as the active zone at the top of the aquifer. If ISOURCE < 0
        sources have the same fluid density as the zone with a number equal
        to the absolute value of ISOURCE. Sinks have the same fluid density
        as the active zone at the top of the aquifer. This option is useful
        for the modeling of the ocean bottom where infiltrating water is
        salt, yet exfiltrating water is of the same type as the water at the
        top of the aquifer.
    obsnam : list of strings
        names for nobs observations.
    obslrc : list of lists
        zero-based [layer, row, column] lists for nobs observations.
    extension : string
        Filename extension (default is 'swi2')
    npln : int
        Deprecated - use nsrf instead.
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the zeta, cbc, obs output files.
        If filenames=None the package name will be created using the model name
        and package extension and the output file names will be created using
        the model name and output extensions. If a single string is passed the
        package will be set to the string and output names will be created
        using the model name and zeta, cbc, and observation extensions. To
        define the names for all package files (input and output) the length
        of the list of strings should be 4. Default is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> swi2 = flopy.modflow.ModflowSwi2(m)

    """

    def __init__(self, model, nsrf=1, istrat=1, nobs=0, iswizt=None,
                 ipakcb=None, iswiobs=0, options=None,
                 nsolver=1, iprsol=0, mutsol=3,
                 solver2params={'mxiter': 100, 'iter1': 20, 'npcond': 1,
                                'zclose': 1e-3, 'rclose': 1e-4, 'relax': 1.0,
                                'nbpol': 2, 'damp': 1.0, 'dampt': 1.0},
                 toeslope=0.05, tipslope=0.05, alpha=None, beta=0.1, nadptmx=1,
                 nadptmn=1, adptfct=1.0, nu=0.025, zeta=[0.0], ssz=0.25,
                 isource=0, obsnam=None, obslrc=None, npln=None,
                 extension='swi2', unitnumber=None, filenames=None):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowSwi2.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 4:
                for idx in range(len(filenames), 4):
                    filenames.append(None)

        # update external file information with zeta output, if necessary
        if iswizt is not None:
            fname = filenames[1]
            model.add_output_file(iswizt, fname=fname, extension='zta',
                                  package=ModflowSwi2.ftype())
        else:
            iswizt = 0

        # update external file information with swi2 cell-by-cell output,
        # if necessary
        if ipakcb is not None:
            fname = filenames[2]
            model.add_output_file(ipakcb, fname=fname,
                                  package=ModflowSwi2.ftype())
        else:
            ipakcb = 0

        # Process observations
        if nobs != 0:
            print('ModflowSwi2: specification of nobs is deprecated.')
        nobs = 0
        if obslrc is not None:
            if isinstance(obslrc, list) or isinstance(obslrc, tuple):
                obslrc = np.array(obslrc, dtype=np.int32)
            if isinstance(obslrc, np.ndarray):
                if obslrc.ndim == 1 and obslrc.size == 3:
                    obslrc = obslrc.reshape((1, 3))
            else:
                errmsg = 'ModflowSwi2: obslrc must be a tuple or ' + \
                         'list of tuples.'
                raise Exception(errmsg)
            nobs = obslrc.shape[0]

            if obsnam is None:
                obsnam = []
                for n in range(nobs):
                    obsnam.append('Obs{:03}'.format(n + 1))
            else:
                if not isinstance(obsnam, list):
                    obsnam = [obsnam]
                if len(obsnam) != nobs:
                    errmsg = 'ModflowSwi2: obsnam must be a list with a ' + \
                             'length of {} not {}.'.format(nobs, len(obsnam))
                    raise Exception(errmsg)

        if nobs > 0:
            binflag = False
            ext = 'zobs.out'
            fname = filenames[3]
            if iswiobs is not None:
                if iswiobs < 0:
                    binflag = True
                    ext = 'zobs.bin'
            else:
                iswiobs = 1053
            # update external file information with swi2 observation output,
            # if necessary
            model.add_output_file(iswiobs, fname=fname, binflag=binflag,
                                  extension=ext, package=ModflowSwi2.ftype())
        else:
            iswiobs = 0

        # Fill namefile items
        name = [ModflowSwi2.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'

        # options
        self.fsssopt, self.adaptive = False, False
        if isinstance(options, list):
            if len(options) < 1:
                self.options = None
            else:
                self.options = options
                for o in self.options:
                    if o.lower() == 'fsssopt':
                        self.fsssopt = True
                    elif o.lower() == 'adaptive':
                        self.adaptive = True
        else:
            self.options = None

        if npln is not None:
            print('npln keyword is deprecated. use the nsrf keyword')
            nsrf = npln

        self.nsrf, self.istrat, self.nobs, self.iswizt, self.iswiobs = nsrf, istrat, nobs, \
                                                                       iswizt, iswiobs
        self.ipakcb = ipakcb

        #
        self.nsolver, self.iprsol, self.mutsol = nsolver, iprsol, mutsol
        #
        self.solver2params = solver2params
        #
        self.toeslope, self.tipslope, self.alpha, self.beta = toeslope, tipslope, alpha, beta
        self.nadptmx, self.nadptmn, self.adptfct = nadptmx, nadptmn, adptfct
        # Create arrays so that they have the correct size
        if self.istrat == 1:
            self.nu = Util2d(model, (self.nsrf + 1,), np.float32, nu,
                             name='nu')
        else:
            self.nu = Util2d(model, (self.nsrf + 2,), np.float32, nu,
                             name='nu')
        self.zeta = []
        for i in range(self.nsrf):
            self.zeta.append(Util3d(model, (nlay, nrow, ncol), np.float32,
                                    zeta[i], name='zeta_' + str(i + 1)))
        self.ssz = Util3d(model, (nlay, nrow, ncol), np.float32, ssz,
                          name='ssz')
        self.isource = Util3d(model, (nlay, nrow, ncol), np.int32, isource,
                              name='isource')
        #
        self.obsnam = obsnam
        self.obslrc = obslrc
        if nobs != 0:
            self.nobs = self.obslrc.shape[0]
        #
        self.parent.add_package(self)

    def write_file(self, check=True,f=None):
        """
        Write the package file.

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        if f is None:
            f = open(self.fn_path, 'w')
        # First line: heading
        f.write('{}\n'.format(
            self.heading))  # Writing heading not allowed in SWI???
        # write dataset 1
        f.write('# Dataset 1\n')
        f.write(
            '{:10d}{:10d}{:10d}{:10d}{:10d}{:10d}'.format(self.nsrf,
                                                          self.istrat,
                                                          self.nobs,
                                                          self.iswizt,
                                                          self.ipakcb,
                                                          self.iswiobs))
        # write SWI2 options
        if self.options != None:
            for o in self.options:
                f.write(' {}'.format(o))
        f.write('\n')
        # write dataset 2a
        f.write('# Dataset 2a\n')
        f.write('{:10d}{:10d}{:10d}\n'.format(self.nsolver, self.iprsol,
                                              self.mutsol))
        # write dataset 2b
        if self.nsolver == 2:
            f.write('# Dataset 2b\n')
            f.write('{:10d}'.format(self.solver2params['mxiter']))
            f.write('{:10d}'.format(self.solver2params['iter1']))
            f.write('{:10d}'.format(self.solver2params['npcond']))
            f.write('{:14.6g}'.format(self.solver2params['zclose']))
            f.write('{:14.6g}'.format(self.solver2params['rclose']))
            f.write('{:14.6g}'.format(self.solver2params['relax']))
            f.write('{:10d}'.format(self.solver2params['nbpol']))
            f.write('{:14.6g}'.format(self.solver2params['damp']))
            f.write('{:14.6g}\n'.format(self.solver2params['dampt']))
        # write dataset 3a
        f.write('# Dataset 3a\n')
        f.write('{:14.6g}{:14.6g}'.format(self.toeslope, self.tipslope))
        if self.alpha is not None:
            f.write('{:14.6g}{:14.6g}'.format(self.alpha, self.beta))
        f.write('\n')
        # write dataset 3b
        if self.adaptive is True:
            f.write('# Dataset 3b\n')
            f.write('{:10d}{:10d}{:14.6g}\n'.format(self.nadptmx,
                                                    self.nadptmn,
                                                    self.adptfct))
        # write dataset 4
        f.write('# Dataset 4\n')
        f.write(self.nu.get_file_entry())
        # write dataset 5
        f.write('# Dataset 5\n')
        for isur in range(self.nsrf):
            for ilay in range(nlay):
                f.write(self.zeta[isur][ilay].get_file_entry())
        # write dataset 6
        f.write('# Dataset 6\n')
        f.write(self.ssz.get_file_entry())
        # write dataset 7
        f.write('# Dataset 7\n')
        f.write(self.isource.get_file_entry())
        # write dataset 8
        if self.nobs > 0:
            f.write('# Dataset 8\n')
            for i in range(self.nobs):
                # f.write(self.obsnam[i] + 3 * '%10i' % self.obslrc + '\n')
                f.write('{} '.format(self.obsnam[i]))
                for v in self.obslrc[i, :]:
                    f.write('{:10d}'.format(v + 1))
                f.write('\n')

        # close swi2 file
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
        swi2 : ModflowSwi2 object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> swi2 = flopy.modflow.ModflowSwi2.load('test.swi2', m)

        """

        if model.verbose:
            sys.stdout.write('loading swi2 package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        # determine problem dimensions
        nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # --read dataset 1
        if model.verbose:
            sys.stdout.write('  loading swi2 dataset 1\n')
        t = line.strip().split()
        nsrf = int(t[0])
        istrat = int(t[1])
        nobs = int(t[2])
        if int(t[3]) > 0:
            model.add_pop_key_list(int(t[3]))
            iswizt = int(t[3])
        if int(t[4]) > 0:
            model.add_pop_key_list(int(t[4]))
            ipakcb = int(t[4])
        else:
            ipakcb = 0
        iswiobs = 0
        if int(t[5]) > 0:
            model.add_pop_key_list(int(t[5]))
            iswiobs = int(t[5])
        options = []
        adaptive = False
        for idx in range(6, len(t)):
            if '#' in t[idx]:
                break
            options.append(t[idx])
            if 'adaptive' in t[idx].lower():
                adaptive = True

        # read dataset 2a
        if model.verbose:
            sys.stdout.write('  loading swi2 dataset 2a\n')
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        t = line.strip().split()
        nsolver = int(t[0])
        iprsol = int(t[1])
        mutsol = int(t[2])

        # read dataset 2b
        solver2params = {}
        if nsolver == 2:
            if model.verbose:
                sys.stdout.write('  loading swi2 dataset 2b\n')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            t = line.strip().split()
            solver2params['mxiter'] = int(t[0])
            solver2params['iter1'] = int(t[1])
            solver2params['npcond'] = int(t[2])
            solver2params['zclose'] = float(t[3])
            solver2params['rclose'] = float(t[4])
            solver2params['relax'] = float(t[5])
            solver2params['nbpol'] = int(t[6])
            solver2params['damp'] = float(t[7])
            solver2params['dampt'] = float(t[8])

        # read dataset 3a
        if model.verbose:
            sys.stdout.write('  loading swi2 dataset 3a\n')
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        t = line.strip().split()
        toeslope = float(t[0])
        tipslope = float(t[1])
        alpha = None
        beta = 0.1
        if len(t) > 2:
            try:
                alpha = float(t[2])
                beta = float(t[3])
            except:
                pass

        # read dataset 3b
        nadptmx, nadptmn, adptfct = None, None, None
        if adaptive:
            if model.verbose:
                sys.stdout.write('  loading swi2 dataset 3b\n')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            t = line.strip().split()
            nadptmx = int(t[0])
            nadptmn = int(t[1])
            adptfct = float(t[2])

        # read dataset 4
        if model.verbose:
            print('   loading nu...')
        if istrat == 1:
            nnu = nsrf + 1
        else:
            nnu = nsrf + 2
        while True:
            ipos = f.tell()
            line = f.readline()
            if line[0] != '#':
                f.seek(ipos)
                break
        nu = Util2d.load(f, model, (nnu,), np.float32, 'nu',
                         ext_unit_dict)

        # read dataset 5
        if model.verbose:
            print('   loading initial zeta surfaces...')
        while True:
            ipos = f.tell()
            line = f.readline()
            if line[0] != '#':
                f.seek(ipos)
                break
        zeta = []
        for n in range(nsrf):
            ctxt = 'zeta_surf{:02d}'.format(n + 1)
            zeta.append(Util3d.load(f, model, (nlay, nrow, ncol),
                                    np.float32, ctxt, ext_unit_dict))

        # read dataset 6
        if model.verbose:
            print('   loading initial ssz...')
        while True:
            ipos = f.tell()
            line = f.readline()
            if line[0] != '#':
                f.seek(ipos)
                break
        ssz = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                          'ssz', ext_unit_dict)

        # read dataset 7
        if model.verbose:
            print('   loading initial isource...')
        while True:
            ipos = f.tell()
            line = f.readline()
            if line[0] != '#':
                f.seek(ipos)
                break
        isource = Util3d.load(f, model, (nlay, nrow, ncol), np.int32,
                              'isource', ext_unit_dict)

        # read dataset 8
        obsname = []
        obslrc = []
        if nobs > 0:
            if model.verbose:
                print('   loading observation locations...')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            for i in range(nobs):
                if i > 0:
                    try:
                        line = f.readline()
                    except:
                        break
                t = line.strip().split()
                obsname.append(t[0])
                kk = int(t[1]) - 1
                ii = int(t[2]) - 1
                jj = int(t[3]) - 1
                obslrc.append([kk, ii, jj])
                nobs = len(obsname)


        # determine specified unit number
        unitnumber = None
        filenames = [None, None, None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowSwi2.ftype())
            if iswizt > 0:
                iu, filenames[1] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=iswizt)
            if ipakcb > 0:
                iu, filenames[2] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
            if abs(iswiobs) > 0:
                iu, filenames[3] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=abs(iswiobs))

        # create swi2 instance
        swi2 = ModflowSwi2(model, nsrf=nsrf, istrat=istrat,
                           iswizt=iswizt, ipakcb=ipakcb,
                           iswiobs=iswiobs, options=options,
                           nsolver=nsolver, iprsol=iprsol, mutsol=mutsol,
                           solver2params=solver2params,
                           toeslope=toeslope, tipslope=tipslope, alpha=alpha,
                           beta=beta,
                           nadptmx=nadptmx, nadptmn=nadptmn, adptfct=adptfct,
                           nu=nu, zeta=zeta, ssz=ssz, isource=isource,
                           obsnam=obsname, obslrc=obslrc,
                           unitnumber=unitnumber, filenames=filenames)

        # return swi2 instance
        return swi2

    @staticmethod
    def ftype():
        return 'SWI2'

    @staticmethod
    def defaultunit():
        return 29
