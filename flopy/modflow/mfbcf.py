import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d,util_3d

class ModflowBcf(Package):
    """
    MODFLOW Block Centered Flow Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    ibcfcb : int
        A flag and unit number. (default is 53)
    intercellt : int
        Intercell transmissivities, harmonic mean (0), arithmetic mean (1),
        logarithmetic mean (2), combination (3). (default is 0)
    laycon : int
        Layer type, confined (0), unconfined (1), constant T, variable S (2),
        variable T, variable S (default is 3)
    trpy : float or array of floats (nlay)
        horizontal anisotropy ratio (default is 1.0)
    hdry : float
        head assigned when cell is dry - used as indicator(default is -1E+30)
    iwdflg : int
        flag to indicate if wetting is inactive (0) or not (non zero)
        (default is 0)
    wetfct : float
        factor used when cell is converted from dry to wet (default is 0.1)
    iwetit : int
        iteration interval in wetting/drying algorithm (default is 1)
    ihdwet : int
        flag to indicate how initial head is computd for cells that become
        wet (default is 0)
    tran : float or array of floats (nlay, nrow, ncol), optional
        transmissivity (only read if laycon is 0 or 2) (default is 1.0)
    hy : float or array of floats (nlay, nrow, ncol)
        hydraulic conductivity (only read if laycon is 1 or 3)
        (default is 1.0)
    vcont : float or array of floats (nlay-1, nrow, ncol)
        vertical leakance between layers (default is 1.0)
    sf1 : float or array of floats (nlay, nrow, ncol)
        specific storage (confined) or storage coefficient (unconfined),
        read when there is at least one transient stress period.
        (default is 1e-5)
    sf2 : float or array of floats (nrow, ncol)
        specific yield, only read when laycon is 2 or 3 and there is at least
        one transient stress period (default is 0.15)
    wetdry : float
        a combination of the wetting threshold and a flag to indicate which
        neighboring cells can cause a cell to become wet (default is -0.01)
    extension : string
        Filename extension (default is 'bcf')
    unitnumber : int
        File unit number (default is 15).

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
    >>> bcf = flopy.modflow.ModflowBcf(ml)

    """

    def __init__(self, model, ibcfcb=53, intercellt=0,laycon=3, trpy=1.0,
                 hdry=-1E+30, iwdflg=0, wetfct=0.1, iwetit=1, ihdwet=0,
                 tran=1.0, hy=1.0, vcont=1.0, sf1=1e-5, sf2=0.15, wetdry=-0.01,
                 extension='bcf', unitnumber=15):
        Package.__init__(self, model, extension, 'BCF6', unitnumber)
        self.url = 'bcf.htm'
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Set values of all parameters
        self.intercellt = util_2d(model, (nlay,), np.int,intercellt,
                                  name='laycon',locat=self.unit_number[0])
        self.laycon = util_2d(model, (nlay,), np.int,laycon, name='laycon',
                              locat=self.unit_number[0])
        self.trpy = util_2d(model, (nlay,), np.int, trpy,
                            name='Anisotropy factor',locat=self.unit_number[0])
        self.ibcfcb = ibcfcb
        self.hdry = hdry
        self.iwdflg = iwdflg
        self.wetfct = wetfct
        self.iwetit = iwetit
        self.ihdwet = ihdwet
        self.tran = util_3d(model, (nlay,nrow,ncol), np.float32, tran,
                            'Transmissivity', locat=self.unit_number[0])
        self.hy = util_3d(model, (nlay,nrow,ncol), np.float32, hy,
                          'Horizontal Hydraulic Conductivity',
                          locat=self.unit_number[0])
        self.vcont = util_3d(model, (nlay-1,nrow,ncol), np.float32, vcont,
                             'Vertical Conductance', locat=self.unit_number[0])
        self.sf1 = util_3d(model, (nlay,nrow,ncol), np.float32, sf1,
                           'Primary Storage Coefficient',
                           locat=self.unit_number[0])
        self.sf2 = util_3d(model, (nlay,nrow,ncol), np.float32, sf2,
                           'Secondary Storage Coefficient',
                           locat=self.unit_number[0])
        self.wetdry = util_3d(model, (nlay,nrow,ncol), np.float32, wetdry,
                              'WETDRY', locat=self.unit_number[0])
        self.parent.add_package(self)
    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        f_bcf = open(self.fn_path, 'w')
        # Item 1: IBCFCB, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET
        f_bcf.write('%10d%10.6G%10d%10f%10d%10d\n' % (self.ibcfcb, self.hdry, self.iwdflg, self.wetfct, self.iwetit, self.ihdwet))
        # LAYCON array
        for k in range(nlay):            
            f_bcf.write('{0:1d}{1:1d} '.format(self.intercellt[k],self.laycon[k]))
        f_bcf.write('\n')
        f_bcf.write(self.trpy.get_file_entry())
        transient = not self.parent.get_package('DIS').steady.all()
        for k in range(nlay):
            if (transient == True):
                #comment = 'Sf1() = Confined storage coefficient of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.sf1[k], self.unit_number[0], True, 13, ncol, comment,ext_base='sf1' )
                f_bcf.write(self.sf1[k].get_file_entry())
            if ((self.laycon[k] == 0) or (self.laycon[k] == 2)):
                #comment = 'TRANS() = Transmissivity of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.tran[k], self.unit_number[0], True, 13, ncol, comment,ext_base='tran' )
                f_bcf.write(self.tran[k].get_file_entry())
            else:
                #comment = 'HY() = Hydr. Conductivity of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.hy[k], self.unit_number[0], True, 13, ncol, comment,ext_base='hy')
                f_bcf.write(self.hy[k].get_file_entry())
            if k < nlay - 1:
                #comment = 'VCONT() = Vert. leakance of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.vcont[k], self.unit_number[0], True, 13, ncol, comment,ext_base='vcont' )
                f_bcf.write(self.vcont[k].get_file_entry())
            if ((transient == True) and ((self.laycon[k] == 2) or (self.laycon[k] == 3))):
                #comment = 'Sf2() = Specific yield of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.sf2[k], self.unit_number[0], True, 13, ncol, comment,ext_base='sf2' )
                f_bcf.write(self.sf2[k].get_file_entry())
            if ((self.iwdflg != 0) and ((self.laycon[k] == 1) or (self.laycon[k] == 3))):
                #comment = 'Wetdry() = Wetdry array of layer ' + str(k + 1)
                #self.parent.write_array( f_bcf, self.wetdry[k], self.unit_number[0], True, 13, ncol, comment,ext_base='wetdry' )
                f_bcf.write(self.wetdry[k].get_file_entry())
        f_bcf.close()

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
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        wel : ModflowBcf object
            ModflowBcf object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflow.ModflowBcf.load('test.bcf', m)

        """

        if model.verbose:
            sys.stdout.write('loading bcf package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        # determine problem dimensions
        nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
        # Item 1: IBCFCB, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET - line already read above
        if model.verbose:
            print('   loading IBCFCB, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET...')
        t = line.strip().split()
        ibcfcb,hdry,iwdflg,wetfct,iwetit,ihdwet = int(t[0]),float(t[1]),int(t[2]),float(t[3]),int(t[4]),int(t[5])
        if ibcfcb != 0:
            model.add_pop_key_list(ibcfcb)
            ibcfcb = 53
        # LAYCON array
        if model.verbose:
            print('   loading LAYCON...')
        line = f.readline()
        t = line.strip().split()
        intercellt = np.zeros(nlay, dtype=np.int)
        laycon = np.zeros(nlay, dtype=np.int)
        for k in range(nlay):
            ival = int(t[k])
            if ival > 9:
                intercellt[k] = int(t[k][0])
                laycon[k] = int(t[k][1])
            else:
                laycon[k] = ival
        # TRPY array
        if model.verbose:
            print('   loading TRPY...')
        trpy = util_2d.load(f, model, (1, nlay), np.float32, 'trpy', ext_unit_dict)
        trpy = trpy.array.reshape( (nlay) )
        # property data for each layer based on options
        transient = not model.get_package('DIS').steady.all()
        sf1 = np.empty((nlay,nrow,ncol), dtype=np.float)
        tran = np.empty((nlay,nrow,ncol), dtype=np.float)
        hy = np.empty((nlay,nrow,ncol), dtype=np.float)
        if nlay > 1:
            vcont = np.empty((nlay-1,nrow,ncol), dtype=np.float)
        else:
            vcont = 1.0
        sf2 = np.empty((nlay,nrow,ncol), dtype=np.float)
        wetdry = np.empty((nlay,nrow,ncol), dtype=np.float)
        for k in range(nlay):
            if transient == True:
                if model.verbose:
                    print('   loading sf1 layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'sf1', ext_unit_dict)
                sf1[k,:,:] = t.array
            if ((laycon[k] == 0) or (laycon[k] == 2)):
                if model.verbose:
                    print('   loading tran layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'tran', ext_unit_dict)
                tran[k,:,:] = t.array
            else:
                if model.verbose:
                    print('   loading hy layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'hy', ext_unit_dict)
                hy[k,:,:] = t.array
            if k < (nlay - 1):
                if model.verbose:
                    print('   loading vcont layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'vcont', ext_unit_dict)
                vcont[k,:,:] = t.array
            if ((transient == True) and ((laycon[k] == 2) or (laycon[k] == 3))):
                if model.verbose:
                    print('   loading sf2 layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'sf2', ext_unit_dict)
                sf2[k,:,:] = t.array
            if ((iwdflg != 0) and ((laycon[k] == 1) or (laycon[k] == 3))):
                if model.verbose:
                    print('   loading sf2 layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'wetdry', ext_unit_dict)
                wetdry[k,:,:] = t.array

        # create instance of bcf object
        bcf = ModflowBcf(model, ibcfcb=ibcfcb, intercellt=intercellt, laycon=laycon, trpy=trpy, hdry=hdry,
                         iwdflg=iwdflg, wetfct=wetfct, iwetit=iwetit, ihdwet=ihdwet,
                         tran=tran, hy=hy, vcont=vcont, sf1=sf1, sf2=sf2, wetdry=wetdry)

        # return bcf object
        return bcf
