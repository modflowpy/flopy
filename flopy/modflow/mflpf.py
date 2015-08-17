"""
mflpf module.  Contains the ModflowLpf class. Note that the user can access
the ModflowLpf class as `flopy.modflow.ModflowLpf`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?lpf.htm>`_.

"""

import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d, util_3d, read1d
from flopy.modflow.mfpar import ModflowPar as mfpar

class ModflowLpf(Package):
    """
    MODFLOW Discretization Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ilpfcb : int
        A flag and unit number. (default is 53).
    hdry : float
        Is the head that is assigned to cells that are converted to dry during
        a simulation. Although this value plays no role in the model
        calculations, it is useful as an indicator when looking at the
        resulting heads that are output from the model. HDRY is thus similar
        to HNOFLO in the Basic Package, which is the value assigned to cells
        that are no-flow cells at the start of a model simulation. (default
        is -1.e30).
    laytyp : int or array of ints (nlay)
        Layer type (default is 0).
    layavg : int or array of ints (nlay)
        Layer average (default is 0).
        0 is harmonic mean
        1 is logarithmic mean
        2 is arithmetic mean of saturated thickness and logarithmic mean of
        of hydraulic conductivity
    chani : float or array of floats (nlay)
        contains a value for each layer that is a flag or the horizontal
        anisotropy. If CHANI is less than or equal to 0, then variable HANI
        defines horizontal anisotropy. If CHANI is greater than 0, then CHANI
        is the horizontal anisotropy for the entire layer, and HANI is not
        read. If any HANI parameters are used, CHANI for all layers must be
        less than or equal to 0. Use as many records as needed to enter a
        value of CHANI for each layer. The horizontal anisotropy is the ratio
        of the hydraulic conductivity along columns (the Y direction) to the
        hydraulic conductivity along rows (the X direction).
    layvka : float or array of floats (nlay)
        a flag for each layer that indicates whether variable VKA is vertical
        hydraulic conductivity or the ratio of horizontal to vertical
        hydraulic conductivity.
    laywet : float or array of floats (nlay)
        contains a flag for each layer that indicates if wetting is active.
    wetfct : float
        is a factor that is included in the calculation of the head that is
        initially established at a cell when it is converted from dry to wet.
        (default is 0.1).
    iwetit : int
        is the iteration interval for attempting to wet cells. Wetting is
        attempted every IWETIT iteration. If using the PCG solver
        (Hill, 1990), this applies to outer iterations, not inner iterations.
        If IWETIT  less than or equal to 0, it is changed to 1.
        (default is 1).
    ihdwet : int
        is a flag that determines which equation is used to define the
        initial head at cells that become wet. (default is 0)
    hk : float or array of floats (nlay, nrow, ncol)
        is the hydraulic conductivity along rows. HK is multiplied by
        horizontal anisotropy (see CHANI and HANI) to obtain hydraulic
        conductivity along columns. (default is 1.0).
    hani : float or array of floats (nlay, nrow, ncol)
        is the ratio of hydraulic conductivity along columns to hydraulic
        conductivity along rows, where HK of item 10 specifies the hydraulic
        conductivity along rows. Thus, the hydraulic conductivity along
        columns is the product of the values in HK and HANI.
        (default is 1.0).
    vka : float or array of floats (nlay, nrow, ncol)
        is either vertical hydraulic conductivity or the ratio of horizontal
        to vertical hydraulic conductivity depending on the value of LAYVKA.
        (default is 1.0).
    ss : float or array of floats (nlay, nrow, ncol)
        is specific storage unless the STORAGECOEFFICIENT option is used.
        When STORAGECOEFFICIENT is used, Ss is confined storage coefficient.
        (default is 1.e-5).
    sy : float or array of floats (nlay, nrow, ncol)
        is specific yield. (default is 0.15).
    vkcb : float or array of floats (nlay, nrow, ncol)
        is the vertical hydraulic conductivity of a Quasi-three-dimensional
        confining bed below a layer. (default is 0.0).
    wetdry : float or array of floats (nlay, nrow, ncol)
        is a combination of the wetting threshold and a flag to indicate
        which neighboring cells can cause a cell to become wet.
        (default is -0.01).
    storagecoefficient : boolean
        indicates that variable Ss and SS parameters are read as storage
        coefficient rather than specific storage. (default is False).
    constantcv : boolean
         indicates that vertical conductance for an unconfined cell is
         computed from the cell thickness rather than the saturated thickness.
         The CONSTANTCV option automatically invokes the NOCVCORRECTION
         option. (default is False).
    thickstrt : boolean
        indicates that layers having a negative LAYTYP are confined, and their
        cell thickness for conductance calculations will be computed as
        STRT-BOT rather than TOP-BOT. (default is False).
    nocvcorrection : boolean
        indicates that vertical conductance is not corrected when the vertical
        flow correction is applied. (default is False).
    novfc : boolean
         turns off the vertical flow correction under dewatered conditions.
         This option turns off the vertical flow calculation described on p.
         5-8 of USGS Techniques and Methods Report 6-A16 and the vertical
         conductance correction described on p. 5-18 of that report.
         (default is False).
    extension : string
        Filename extension (default is 'lpf')
    unitnumber : int
        File unit number (default is 15).


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
    >>> lpf = flopy.modflow.ModflowLpf(m)

    """

    'Layer-property flow package class\n'
    def __init__(self, model, laytyp=0, layavg=0, chani=1.0, layvka=0,
                 laywet=0, ilpfcb = 53, hdry=-1E+30, iwdflg=0, wetfct=0.1,
                 iwetit=1, ihdwet=0, hk=1.0, hani=1.0, vka=1.0, ss=1e-5,
                 sy=0.15, vkcb=0.0, wetdry=-0.01, storagecoefficient=False,
                 constantcv=False, thickstrt=False, nocvcorrection=False,
                 novfc=False, extension='lpf', unitnumber=15):
        Package.__init__(self, model, extension, 'LPF', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.heading = '# LPF for MODFLOW, generated by Flopy.'
        self.url = 'lpf.htm'
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # item 1
        self.ilpfcb = ilpfcb # Unit number for file with cell-by-cell flow terms
        self.hdry = hdry # Head in cells that are converted to dry during a simulation
        self.nplpf = 0 # number of LPF parameters
        self.laytyp = util_2d(model,(nlay,),np.int,laytyp,name='laytyp')
        self.layavg = util_2d(model,(nlay,),np.int,layavg,name='layavg')
        self.chani = util_2d(model,(nlay,),np.float32,chani,name='chani')
        self.layvka = util_2d(model,(nlay,),np.int,layvka,name='layvka')
        self.laywet = util_2d(model,(nlay,),np.int,laywet,name='laywet')
        self.wetfct = wetfct # Factor that is included in the calculation of the head when a cell is converted from dry to wet
        self.iwetit = iwetit # Iteration interval for attempting to wet cells
        self.ihdwet = ihdwet # Flag that determines which equation is used to define the initial head at cells that become wet
        self.options = ' '
        if storagecoefficient:
            self.options = self.options + 'STORAGECOEFFICIENT '
        if constantcv: self.options = self.options + 'CONSTANTCV '
        if thickstrt: self.options = self.options + 'THICKSTRT '
        if nocvcorrection: self.options = self.options + 'NOCVCORRECTION '
        if novfc: self.options = self.options + 'NOVFC '
        self.hk = util_3d(model, (nlay,nrow,ncol), np.float32, hk, name='hk',
                          locat=self.unit_number[0])
        self.hani = util_3d(model, (nlay,nrow,ncol), np.float32, hani,
                            name='hani', locat=self.unit_number[0])
        self.vka = util_3d(model, (nlay,nrow,ncol), np.float32, vka,
                           name='vka', locat=self.unit_number[0])
        self.ss = util_3d(model, (nlay,nrow,ncol), np.float32, ss, name='ss',
                          locat=self.unit_number[0])
        self.sy = util_3d(model, (nlay,nrow,ncol), np.float32, sy, name='sy',
                          locat=self.unit_number[0])
        self.vkcb = util_3d(model, (nlay,nrow,ncol), np.float32, vkcb,
                            name='vkcb', locat=self.unit_number[0])
        self.wetdry = util_3d(model, (nlay,nrow,ncol), np.float32, wetdry,
                              name='wetdry', locat=self.unit_number[0])
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the file.

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        f_lpf = open(self.fn_path, 'w')
        # Item 0: text
        f_lpf.write('%s\n' % self.heading)
        # Item 1: IBCFCB, HDRY, NPLPF        
        f_lpf.write('{0:10d}{1:10.6G}{2:10d} {3:s}\n'.format(self.ilpfcb,
                                                             self.hdry,
                                                             self.nplpf,
                                                             self.options))
        # LAYTYP array
        f_lpf.write(self.laytyp.string)
        # LAYAVG array
        f_lpf.write(self.layavg.string)
        # CHANI array
        f_lpf.write(self.chani.string)
        # LAYVKA array
        f_lpf.write(self.layvka.string)
        # LAYWET array
        f_lpf.write(self.laywet.string)
        # Item 7: WETFCT, IWETIT, IHDWET
        iwetdry = self.laywet.sum()
        if iwetdry > 0:            
            f_lpf.write('{0:10f}{1:10d}{2:10d}\n'.format(self.wetfct,
                                                         self.iwetit,
                                                         self.ihdwet))
        transient = not self.parent.get_package('DIS').steady.all()
        for k in range(nlay):           
            f_lpf.write(self.hk[k].get_file_entry())
            if self.chani[k] < 0:
                f_lpf.write(self.hani[k].get_file_entry())            
            f_lpf.write(self.vka[k].get_file_entry())
            if transient == True:                
                f_lpf.write(self.ss[k].get_file_entry())
                if self.laytyp[k] !=0:                                        
                    f_lpf.write(self.sy[k].get_file_entry())
            if self.parent.get_package('DIS').laycbd[k] > 0:                
                f_lpf.write(self.vkcb[k].get_file_entry())
            if (self.laywet[k] != 0 and self.laytyp[k] != 0):               
                f_lpf.write(self.wetdry[k].get_file_entry())
        f_lpf.close()
        return

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
        dis : ModflowLpf object
            ModflowLpf object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> lpf = flopy.modflow.ModflowLpf.load('test.lpf', m)

        """

        if model.verbose:
            sys.stdout.write('loading lpf package file...\n')

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
        # Item 1: IBCFCB, HDRY, NPLPF - line already read above
        if model.verbose:
            print('   loading IBCFCB, HDRY, NPLPF...')
        t = line.strip().split()
        ilpfcb, hdry, nplpf = int(t[0]), float(t[1]), int(t[2])
        if ilpfcb != 0:
            model.add_pop_key_list(ilpfcb)
            ilpfcb = 53
        # options
        storagecoefficient = False
        constantcv = False
        thickstrt = False
        nocvcorrection = False
        novfc = False
        if len(t) > 3:
            for k in range(3,len(t)):
                if 'STORAGECOEFFICIENT' in t[k].upper():
                    storagecoefficient = True
                elif 'CONSTANTCV' in t[k].upper():
                    constantcv = True
                elif 'THICKSTRT' in t[k].upper():
                    thickstrt = True
                elif 'NOCVCORRECTION' in t[k].upper():
                    nocvcorrection = True
                elif 'NOVFC' in t[k].upper():
                    novfc = True
        # LAYTYP array
        if model.verbose:
            print('   loading LAYTYP...')
        laytyp = np.empty((nlay), dtype=np.int)
        laytyp = read1d(f, laytyp)
        # LAYAVG array
        if model.verbose:
            print('   loading LAYAVG...')
        layavg = np.empty((nlay), dtype=np.int)
        layavg = read1d(f, layavg)
        # CHANI array
        if model.verbose:
            print('   loading CHANI...')
        chani = np.empty((nlay), dtype=np.float32)
        chani = read1d(f, chani)
        # LAYVKA array
        if model.verbose:
            print('   loading LAYVKA...')
        layvka = np.empty((nlay), dtype=np.float32)
        layvka = read1d(f, layvka)
        # LAYWET array
        if model.verbose:
            print('   loading LAYWET...')
        laywet = np.empty((nlay), dtype=np.int)
        laywet = read1d(f, laywet)
        # Item 7: WETFCT, IWETIT, IHDWET
        wetfct,iwetit,ihdwet = None, None, None
        iwetdry = laywet.sum()
        if iwetdry > 0:
            if model.verbose:
                print('   loading WETFCT, IWETIT, IHDWET...')
            line = f.readline()
            t = line.strip().split()
            wetfct, iwetit, ihdwet = float(t[0]), int(t[1]), int(t[2])

        # parameters data
        par_types = []
        if nplpf > 0:
            par_types, parm_dict = mfpar.load(f, nplpf, model.verbose)
            #print parm_dict

        # non-parameter data
        transient = not model.get_package('DIS').steady.all()
        hk = [0] * nlay
        hani = [0] * nlay
        vka = [0] * nlay
        ss = [0] * nlay
        sy = [0] * nlay
        vkcb = [0] * nlay
        wetdry = [0] * nlay
        for k in range(nlay):
            if model.verbose:
                print('   loading hk layer {0:3d}...'.format(k+1))
            if 'hk' not in par_types:
                t = util_2d.load(f, model, (nrow,ncol), np.float32, 'hk',
                                 ext_unit_dict)
            else:
                line = f.readline()
                t = mfpar.parameter_fill(model, (nrow, ncol), 'hk', parm_dict, findlayer=k)
            hk[k] = t
            if chani[k] < 0:
                if model.verbose:
                    print('   loading hani layer {0:3d}...'.format(k+1))
                if 'hani' not in par_types:
                    t = util_2d.load(f, model, (nrow, ncol), np.float32, 'hani',
                                     ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'hani', parm_dict, findlayer=k)
                hani[k] = t
            if model.verbose:
                print('   loading vka layer {0:3d}...'.format(k+1))
            if 'vka' not in par_types and 'vani' not in par_types:
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'vka',
                                 ext_unit_dict)
            else:
                line = f.readline()
                key = 'vka'
                if 'vani' in par_types:
                    key = 'vani'
                t = mfpar.parameter_fill(model, (nrow, ncol), key, parm_dict, findlayer=k)
            vka[k] = t
            if transient:
                if model.verbose:
                    print('   loading ss layer {0:3d}...'.format(k+1))
                if 'ss' not in par_types:
                    t = util_2d.load(f, model, (nrow, ncol), np.float32, 'ss',
                                     ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'ss', parm_dict, findlayer=k)
                ss[k] = t
                if laytyp[k] != 0:
                    if model.verbose:
                        print('   loading sy layer {0:3d}...'.format(k+1))
                    if 'sy' not in par_types:
                        t = util_2d.load(f, model, (nrow, ncol), np.float32, 'sy',
                                         ext_unit_dict)
                    else:
                        line = f.readline()
                        t = mfpar.parameter_fill(model, (nrow, ncol), 'sy', parm_dict, findlayer=k)
                    sy[k] = t
            #if self.parent.get_package('DIS').laycbd[k] > 0:
            if model.get_package('DIS').laycbd[k] > 0:
                if model.verbose:
                    print('   loading vkcb layer {0:3d}...'.format(k+1))
                if 'vkcb' not in par_types:
                    t = util_2d.load(f, model, (nrow, ncol), np.float32, 'vkcb',
                                     ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'vkcb', parm_dict, findlayer=k)
                vkcb[k] = t
            if (laywet[k] != 0 and laytyp[k] != 0):
                if model.verbose:
                    print('   loading wetdry layer {0:3d}...'.format(k+1))
                t = util_2d.load(f, model, (nrow, ncol), np.float32, 'wetdry',
                                 ext_unit_dict)
                wetdry[k] = t

        # create instance of lpf class
        lpf = ModflowLpf(model, ilpfcb=ilpfcb, laytyp=laytyp, layavg=layavg, chani=chani,
                         layvka=layvka, laywet=laywet, hdry=hdry, iwdflg=iwetdry,
                         wetfct=wetfct, iwetit=iwetit, ihdwet=ihdwet,
                         hk=hk, hani=hani, vka=vka, ss=ss, sy=sy, vkcb=vkcb,
                         wetdry=wetdry, storagecoefficient=storagecoefficient,
                         constantcv=constantcv,thickstrt=thickstrt,novfc=novfc)
        return lpf

