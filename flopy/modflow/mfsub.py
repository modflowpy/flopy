"""
mfswi2 module.  Contains the ModflowSwi2 class. Note that the user can access
the ModflowSwi2 class as `flopy.modflow.ModflowSwi2`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/swi2_seawater_intrusion_pack.htm>`_.

"""
import sys
import copy
import numpy as np
# from numpy import ones, zeros, empty
from flopy.mbase import Package
from flopy.utils import util_2d, util_3d, read1d

class ModflowSub(Package):
    """
    MODFLOW SUB Package Class.

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
    nobs : int
        number of observation locations. (default is 0).
    iswizt : int
        unit number for zeta output. (default is 55).
    iswibd : int
        unit number for SWI2 Package budget output. (default is 56).
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
        If MUTSOL = 0, tables of maximum head change and residual will be printed each iteration.
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
            is only used when NPCOND = 2 to indicate whether the estimate of the
            upper bound on the maximum eigenvalue is 2.0, or whether the estimate
            will be calculated. NBPOL = 2 is used to specify the value is 2.0;
            for any other value of NBPOL, the estimate is calculated. Convergence
            is generally insensitive to this parameter. (default is 2).
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
        if istart = 1, density of each zone (nsrf + 1 values). if istrat = 0, density along
        top of layer, each surface, and bottom of layer (nsrf + 2 values). (default is 0.025)
    zeta : list of floats or list of array of floats [(nlay, nrow, ncol), (nlay, nrow, ncol)]
        initial elevations of the active surfaces. (default is 0.)
    ssz : float or array of floats (nlay, nrow, ncol)
        effective porosity. (default is 0.25)
    isource : integer or array of integers (nlay, nrow, ncol)
        Source type of any external sources or sinks, specified with any outside package
        (i.e. WEL Package, RCH Package, GHB Package). (default is 0).
            If ISOURCE > 0 sources and sinks have the same fluid density as the zone
                ISOURCE. If such a zone is not present in the cell, sources and sinks
                have the same fluid density as the active zone at the top of the aquifer.
            If ISOURCE = 0 sources and sinks have the same fluid density as the active
                zone at the top of the aquifer.
            If ISOURCE < 0 sources have the same fluid density as the zone with a
                number equal to the absolute value of ISOURCE. Sinks have the same
                fluid density as the active zone at the top of the aquifer. This
                option is useful for the modeling of the ocean bottom where infiltrating
                water is salt, yet exfiltrating water is of the same type as the water
                at the top of the aquifer.
    obsnam : list of strings
        names for nobs observations.
    obslrc : list of lists
        [layer, row, column] lists for nobs observations.
    naux : int
        number of auxiliary variables
    extension : list string
        Filename extension (default is ['swi2', 'zta', 'swb'])
    unitnumber : int
        File unit number (default is 29).
    npln : int
        Deprecated - use nsrf instead.

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
    >>> sub = flopy.modflow.ModflowSub(m)

    """

    def __init__(self, model, isubcb=0, isuboc=0, idsave=0, idrest=0,
                 nndb=1, ndb=1, nmz=1, nn=20, ac1=0., ac2=0.2, itmin=5,
                 ln=0, ldn=0, rnb=1,
                 hc=100000., sfe=1.e-4, sfv=1.e-3, com=0., dp=[1.e-6, 6.e-6, 6.e-4],
                 dstart=1., dhc=100000., dcom=0., dz=1., nz=1,
                 ids15=None, ids16=None,
                 extension=['sub', 'rst', 'rst'], unit_number=29):
        """
        Package constructor.

        """
        name = ['SUB', 'DATA(BINARY)', 'DATA(BINARY)']
        units = [unit_number, idsave, idrest]
        extra = ['', 'REPLACE', 'OLD']

        Package.__init__(self, model, extension=extension, name=name, unit_number=units,
                         extra=extra)  # Call ancestor's init to set self.parent, extension, name and unit number

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self.heading = '# Subsidence (SUB) package file for {}, generated by Flopy.'.format(model.version)
        self.url = 'sub.htm'

        self.isubc = isubcb
        self.isuboc = isuboc
        self.idsave = idsave
        self.idrest = idrest
        self.nndb = nndb
        self.ndb = ndb
        self.nmz = nmz
        self.nn = nn
        self.ac1 = ac1
        self.ac2 = ac2
        self.itmin = itmin
        # no-delay bed data
        self.ln = None
        self.hc = None
        self.sfe = None
        self.sfv = None
        if nndb > 0:
            self.ln = util_2d(model, (nndb,), np.int, ln, name='ln')
            self.hc = util_3d(model, (nndb, nrow, ncol), np.float32, hc, name='hc',
                              locat=self.unit_number[0])
            self.sfe = util_3d(model, (nndb, nrow, ncol), np.float32, sfe, name='sfe',
                               locat=self.unit_number[0])
            self.sfv = util_3d(model, (nndb, nrow, ncol), np.float32, sfv, name='sfv',
                               locat=self.unit_number[0])
            self.com = util_3d(model, (nndb, nrow, ncol), np.float32, com, name='com',
                               locat=self.unit_number[0])
        # delay bed data
        self.ldn = None
        self.rnb = None
        self.dstart = None
        self.dhc = None
        self.dz = None
        self.nz = None
        if ndb > 0:
            self.ldn = util_2d(model, (ndb,), np.int, ldn, name='ldn')
            self.rnb = util_3d(model, (ndb, nrow, ncol), np.float32, rnb, name='rnb',
                            locat=self.unit_number[0])
            self.dstart = util_3d(model, (ndb, nrow, ncol), np.float32, dstart, name='dstart',
                                  locat=self.unit_number[0])
            self.dhc = util_3d(model, (ndb, nrow, ncol), np.float32, dhc, name='dhc',
                               locat=self.unit_number[0])
            self.dz = util_3d(model, (ndb, nrow, ncol), np.float32, dz, name='dz',
                              locat=self.unit_number[0])
            self.nz = util_3d(model, (ndb, nrow, ncol), np.int, nz, name='nz',
                              locat=self.unit_number[0])
        # material zone data
        if isinstance(dp, list):
            dp = np.array(dp)
        self.dp = dp

        # output data
        if isuboc > 0:
            if ids15 is None:
                ids15 = None
            else:
                if isinstance(ids15, list):
                    ids15 = np.array(ids15)
            self.ids15 = ids15
            if ids16 is None:
                ids16 = None
            else:
                if isinstance(ids16, list):
                    ids16 = np.array(ids15)
            self.ids16 = ids16

        # add package to model
        self.parent.add_package(self)

    def __repr__(self):
        return 'Subsidence (SUB) package class'

    def write_file(self):
        """
        Write the package input file.

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # # Open file for writing
        # f = open(self.fn_path, 'w')
        # # First line: heading
        # f.write('{}\n'.format(self.heading))
        # # write dataset 1
        # f.write('# Dataset 1\n')
        # f.write(
        #     '{:10d}{:10d}{:10d}{:10d}{:10d}{:10d}'.format(self.nsrf, self.istrat, self.nobs, self.iswizt, self.iswibd,
        #                                                   self.iswiobs))
        # # write SWI2 options
        # if self.options != None:
        #     for o in self.options:
        #         f.write(' {}'.format(o))
        # f.write('\n')
        # # write dataset 2a
        # f.write('# Dataset 2a\n')
        # f.write('{:10d}{:10d}{:10d}\n'.format(self.nsolver, self.iprsol, self.mutsol))
        # # write dataset 2b
        # if self.nsolver == 2:
        #     f.write('# Dataset 2b\n')
        #     f.write('{:10d}'.format(self.solver2params['mxiter']))
        #     f.write('{:10d}'.format(self.solver2params['iter1']))
        #     f.write('{:10d}'.format(self.solver2params['npcond']))
        #     f.write('{:14.6g}'.format(self.solver2params['zclose']))
        #     f.write('{:14.6g}'.format(self.solver2params['rclose']))
        #     f.write('{:14.6g}'.format(self.solver2params['relax']))
        #     f.write('{:10d}'.format(self.solver2params['nbpol']))
        #     f.write('{:14.6g}'.format(self.solver2params['damp']))
        #     f.write('{:14.6g}\n'.format(self.solver2params['dampt']))
        # # write dataset 3a
        # f.write('# Dataset 3a\n')
        # f.write('{:14.6g}{:14.6g}'.format(self.toeslope, self.tipslope))
        # if self.alpha is not None:
        #     f.write('{:14.6g}{:14.6g}'.format(self.alpha, self.beta))
        # f.write('\n')
        # # write dataset 3b
        # if self.adaptive is True:
        #     f.write('# Dataset 3b\n')
        #     f.write('{:10d}{:10d}{:14.6g}\n'.format(self.nadptmx, self.nadptmn, self.adptfct))
        # # write dataset 4
        # f.write('# Dataset 4\n')
        # f.write(self.nu.get_file_entry())
        # # write dataset 5
        # f.write('# Dataset 5\n')
        # for isur in range(self.nsrf):
        #     for ilay in range(nlay):
        #         f.write(self.zeta[isur][ilay].get_file_entry())
        # # write dataset 6
        # f.write('# Dataset 6\n')
        # f.write(self.ssz.get_file_entry())
        # # write dataset 7
        # f.write('# Dataset 7\n')
        # f.write(self.isource.get_file_entry())
        # # write dataset 8
        # if self.nobs > 0:
        #     f.write('# Dataset 8\n')
        #     for i in range(self.nobs):
        #         #f.write(self.obsnam[i] + 3 * '%10i' % self.obslrc + '\n')
        #         f.write('{} '.format(self.obsnam[i]))
        #         for v in self.obslrc[i, :]:
        #             f.write('{:10d}'.format(v))
        #         f.write('\n')
        #
        # # close swi2 file
        # f.close()


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
        >>> sub = flopy.modflow.ModflowSub.load('test.sub', m)

        """

        if model.verbose:
            sys.stdout.write('loading sub package file...\n')

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

        # read dataset 1
        if model.verbose:
            sys.stdout.write('  loading sub dataset 1\n')
        t = line.strip().split()
        isubcb, isuboc, nndb, ndb, nmz, nn = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])
        ac1, ac2 = float(t[6]), float(t[7])
        itmin, idsave, idrest = int(t[8]), int(t[9]), int(t[10])

        if isubcb > 0:
            isubcb = 53
        if idsave > 0:
            idsave = 2051
        if idrest > 0:
            idrest = 2052

        ln = None
        if nndb > 0:
            if model.verbose:
                sys.stdout.write('  loading sub dataset 2\n')
            ln = np.empty((nndb), dtype=np.int)
            ln = read1d(f, ln) - 1
        ldn = None
        if ndb > 0:
            if model.verbose:
                sys.stdout.write('  loading sub dataset 3\n')
            ldn = np.empty((ndb), dtype=np.int)
            ldn = read1d(f, ldn) - 1
        rnb = None
        if ndb > 0:
            if model.verbose:
                sys.stdout.write('  loading sub dataset 4\n')
            rnb = util_3d.load(f, model, (ndb, nrow, ncol), np.float32,
                               'rnb', ext_unit_dict)
        hc = None
        sfe = None
        sfv = None
        com = None
        if nndb > 0:
            hc = [0] * nndb
            sfe = [0] * nndb
            sfv = [0] * nndb
            com = [0] * nndb
            for k in range(nndb):
                kk = ln[k] + 1
                # hc
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 5 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'hc',
                                 ext_unit_dict)
                hc[k] = t
                # sfe
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 6 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'sfe',
                                 ext_unit_dict)
                sfe[k] = t
                # sfv
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 7 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'sfv',
                                 ext_unit_dict)
                sfv[k] = t
                # com
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 8 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'com',
                                 ext_unit_dict)
                com[k] = t

        # dp
        dp = None
        if ndb > 0:
            dp = np.zeros((nmz, 3), dtype=np.float32)
            for k in range(nmz):
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 9 for material zone {}\n'.format(k+1))
                line = f.readline()
                t = line.strip().split()
                dp[k, :] = float(t[0]), float(t[1]), float(t[2])

        dstart = None
        dhc = None
        dcom = None
        dz = None
        nz = None
        if ndb > 0:
            dstart = [0] * ndb
            dhc = [0] * ndb
            dcom = [0] * ndb
            dz = [0] * ndb
            nz = [0] * ndb
            for k in range(ndb):
                kk = ldn[k] + 1
                # dstart
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 10 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'dstart',
                                 ext_unit_dict)
                dstart[k] = t
                # dhc
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 11 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'dhc',
                                 ext_unit_dict)
                dhc[k] = t
                # dcom
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 12 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'dcom',
                                 ext_unit_dict)
                dcom[k] = t
                # dz
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 13 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'dz',
                                 ext_unit_dict)
                dz[k] = t
                # nz
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 14 for layer {}\n'.format(kk))
                t = util_2d.load(f, model, (nrow, ncol), np.int, 'nz',
                                 ext_unit_dict)
                nz[k] = t

        ids15 = None
        ids16 = None
        if isuboc > 0:
            # dataset 15
            if model.verbose:
                sys.stdout.write('  loading sub dataset 15 for layer {}\n'.format(kk))
            ids15 = np.empty((12), dtype=np.int)
            ids15 = read1d(f, ids15)
            for k in range(1, 12, 2):
                ids15[k] = 53  # all subsidence data sent to unit 53
            # dataset 16
            ids16 = [0] * isuboc
            for k in range(isuboc):
                if model.verbose:
                    sys.stdout.write('  loading sub dataset 16 for isuboc {}\n'.format(k+1))
                t = np.empty((17), dtype=np.int)
                t = read1d(f, t)
                ids16[k] = t


        # close file
        f.close()

        # create sub instance
        sub = ModflowSub(model, isubcb=isubcb, isuboc=isuboc, idsave=idsave, idrest=idrest,
                 nndb=nndb, ndb=ndb, nmz=nmz, nn=nn, ac1=ac1, ac2=ac2, itmin=itmin,
                 ln=ln, ldn=ldn, rnb=rnb,
                 hc=hc, sfe=sfe, sfv=sfv, com=com, dp=dp,
                 dstart=dstart, dhc=dhc, dcom=dcom, dz=dz, nz=nz,
                 ids15=ids15, ids16=ids16)
        # return sub instance
        return sub
