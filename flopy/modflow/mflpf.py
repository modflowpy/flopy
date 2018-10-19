"""
mflpf module.  Contains the ModflowLpf class. Note that the user can access
the ModflowLpf class as `flopy.modflow.ModflowLpf`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?lpf.htm>`_.

"""

import sys

import numpy as np
from .mfpar import ModflowPar as mfpar

from ..pakbase import Package
from ..utils import Util2d, Util3d, read1d
from ..utils.flopy_io import line_parse


class ModflowLpf(Package):
    """
    MODFLOW Layer Property Flow Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 53)
    hdry : float
        Is the head that is assigned to cells that are converted to dry during
        a simulation. Although this value plays no role in the model
        calculations, it is useful as an indicator when looking at the
        resulting heads that are output from the model. HDRY is thus similar
        to HNOFLO in the Basic Package, which is the value assigned to cells
        that are no-flow cells at the start of a model simulation. 
        (default is -1.e30).
    laytyp : int or array of ints (nlay)
        Layer type, contains a flag for each layer that specifies the layer type.
        0 confined
        >0 convertible
        <0 convertible unless the THICKSTRT option is in effect.
        (default is 0).
    layavg : int or array of ints (nlay)
        Layer average 
        0 is harmonic mean
        1 is logarithmic mean
        2 is arithmetic mean of saturated thickness and logarithmic mean of
        of hydraulic conductivity
        (default is 0).
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
        (default is 1).
    layvka : int or array of ints (nlay)
        a flag for each layer that indicates whether variable VKA is vertical
        hydraulic conductivity or the ratio of horizontal to vertical
        hydraulic conductivity.
        0: VKA is vertical hydraulic conductivity
        not 0: VKA is the ratio of horizontal to vertical hydraulic conductivity
        (default is 0).
    laywet : int or array of ints (nlay)
        contains a flag for each layer that indicates if wetting is active.
        0 wetting is inactive
        not 0 wetting is active
        (default is 0).
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
        initial head at cells that become wet. 
        (default is 0)
    hk : float or array of floats (nlay, nrow, ncol)
        is the hydraulic conductivity along rows. HK is multiplied by
        horizontal anisotropy (see CHANI and HANI) to obtain hydraulic
        conductivity along columns. 
        (default is 1.0).
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
        is specific yield. 
        (default is 0.15).
    vkcb : float or array of floats (nlay, nrow, ncol)
        is the vertical hydraulic conductivity of a Quasi-three-dimensional
        confining bed below a layer. (default is 0.0).  Note that if an array
        is passed for vkcb it must be of size (nlay, nrow, ncol) even though
        the information for the bottom layer is not needed.
    wetdry : float or array of floats (nlay, nrow, ncol)
        is a combination of the wetting threshold and a flag to indicate
        which neighboring cells can cause a cell to become wet.
        (default is -0.01).
    storagecoefficient : boolean
        indicates that variable Ss and SS parameters are read as storage
        coefficient rather than specific storage. 
        (default is False).
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
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output name will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
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
    >>> lpf = flopy.modflow.ModflowLpf(m)

    """

    'Layer-property flow package class\n'

    def __init__(self, model, laytyp=0, layavg=0, chani=1.0, layvka=0,
                 laywet=0, ipakcb=None, hdry=-1E+30, iwdflg=0, wetfct=0.1,
                 iwetit=1, ihdwet=0, hk=1.0, hani=1.0, vka=1.0, ss=1e-5,
                 sy=0.15, vkcb=0.0, wetdry=-0.01, storagecoefficient=False,
                 constantcv=False, thickstrt=False, nocvcorrection=False,
                 novfc=False, extension='lpf',
                 unitnumber=None, filenames=None):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowLpf.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(ipakcb, fname=fname,
                                  package=ModflowLpf.ftype())
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowLpf.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
        self.url = 'lpf.htm'
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        # item 1
        self.ipakcb = ipakcb
        self.hdry = hdry  # Head in cells that are converted to dry during a simulation
        self.nplpf = 0  # number of LPF parameters
        self.laytyp = Util2d(model, (nlay,), np.int32, laytyp, name='laytyp')
        self.layavg = Util2d(model, (nlay,), np.int32, layavg, name='layavg')
        self.chani = Util2d(model, (nlay,), np.float32, chani, name='chani')
        self.layvka = Util2d(model, (nlay,), np.int32, layvka, name='layvka')
        self.laywet = Util2d(model, (nlay,), np.int32, laywet, name='laywet')
        self.wetfct = wetfct  # Factor that is included in the calculation of the head when a cell is converted from dry to wet
        self.iwetit = iwetit  # Iteration interval for attempting to wet cells
        self.ihdwet = ihdwet  # Flag that determines which equation is used to define the initial head at cells that become wet
        self.options = ' '
        if storagecoefficient:
            self.options = self.options + 'STORAGECOEFFICIENT '
        if constantcv: self.options = self.options + 'CONSTANTCV '
        if thickstrt: self.options = self.options + 'THICKSTRT '
        if nocvcorrection: self.options = self.options + 'NOCVCORRECTION '
        if novfc: self.options = self.options + 'NOVFC '
        self.hk = Util3d(model, (nlay, nrow, ncol), np.float32, hk, name='hk',
                         locat=self.unit_number[0])
        self.hani = Util3d(model, (nlay, nrow, ncol), np.float32, hani,
                           name='hani', locat=self.unit_number[0])
        keys = []
        for k in range(nlay):
            key = 'vka'
            if self.layvka[k] != 0:
                key = 'vani'
            keys.append(key)
        self.vka = Util3d(model, (nlay, nrow, ncol), np.float32, vka,
                          name=keys, locat=self.unit_number[0])
        tag = 'ss'
        if storagecoefficient:
            tag = 'storage'
        self.ss = Util3d(model, (nlay, nrow, ncol), np.float32, ss, name=tag,
                         locat=self.unit_number[0])
        self.sy = Util3d(model, (nlay, nrow, ncol), np.float32, sy, name='sy',
                         locat=self.unit_number[0])
        self.vkcb = Util3d(model, (nlay, nrow, ncol), np.float32, vkcb,
                           name='vkcb', locat=self.unit_number[0])
        self.wetdry = Util3d(model, (nlay, nrow, ncol), np.float32, wetdry,
                             name='wetdry', locat=self.unit_number[0])
        self.parent.add_package(self)
        return

    def write_file(self, check=True, f=None):
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
        if check:  # allows turning off package checks when writing files at model level
            self.check(f='{}.chk'.format(self.name[0]),
                       verbose=self.parent.verbose, level=1)

        # get model information
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        dis = self.parent.get_package('DIS')
        if dis is None:
            dis = self.parent.get_package('DISU')

        # Open file for writing
        if f is None:
            f = open(self.fn_path, 'w')

        # Item 0: text
        f.write('{}\n'.format(self.heading))

        # Item 1: IBCFCB, HDRY, NPLPF
        f.write('{0:10d}{1:10.6G}{2:10d} {3:s}\n'.format(self.ipakcb,
                                                         self.hdry,
                                                         self.nplpf,
                                                         self.options))
        # LAYTYP array
        f.write(self.laytyp.string)
        # LAYAVG array
        f.write(self.layavg.string)
        # CHANI array
        f.write(self.chani.string)
        # LAYVKA array
        f.write(self.layvka.string)
        # LAYWET array
        f.write(self.laywet.string)
        # Item 7: WETFCT, IWETIT, IHDWET
        iwetdry = self.laywet.sum()
        if iwetdry > 0:
            f.write('{0:10f}{1:10d}{2:10d}\n'.format(self.wetfct,
                                                     self.iwetit,
                                                     self.ihdwet))
        transient = not dis.steady.all()
        for k in range(nlay):
            f.write(self.hk[k].get_file_entry())
            if self.chani[k] <= 0.:
                f.write(self.hani[k].get_file_entry())
            f.write(self.vka[k].get_file_entry())
            if transient == True:
                f.write(self.ss[k].get_file_entry())
                if self.laytyp[k] != 0:
                    f.write(self.sy[k].get_file_entry())
            if dis.laycbd[k] > 0:
                f.write(self.vkcb[k].get_file_entry())
            if (self.laywet[k] != 0 and self.laytyp[k] != 0):
                f.write(self.wetdry[k].get_file_entry())
        f.close()
        return


    @staticmethod
    def load(f, model, ext_unit_dict=None, check=True):
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        lpf : ModflowLpf object
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
        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # determine problem dimensions
        nr, nc, nlay, nper = model.get_nrow_ncol_nlay_nper()
        dis = model.get_package('DIS')
        if dis is None:
            dis = model.get_package('DISU')


        # Item 1: IBCFCB, HDRY, NPLPF - line already read above
        if model.verbose:
            print('   loading IBCFCB, HDRY, NPLPF...')
        t = line_parse(line)
        ipakcb, hdry, nplpf = int(t[0]), float(t[1]), int(t[2])
        #if ipakcb != 0:
        #    model.add_pop_key_list(ipakcb)
        #    ipakcb = 53
        # options
        storagecoefficient = False
        constantcv = False
        thickstrt = False
        nocvcorrection = False
        novfc = False
        if len(t) > 3:
            for k in range(3, len(t)):
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
        laytyp = np.empty((nlay), dtype=np.int32)
        laytyp = read1d(f, laytyp)

        # LAYAVG array
        if model.verbose:
            print('   loading LAYAVG...')
        layavg = np.empty((nlay), dtype=np.int32)
        layavg = read1d(f, layavg)

        # CHANI array
        if model.verbose:
            print('   loading CHANI...')
        chani = np.empty((nlay), dtype=np.float32)
        chani = read1d(f, chani)

        # LAYVKA array
        if model.verbose:
            print('   loading LAYVKA...')
        layvka = np.empty((nlay,), dtype=np.int32)
        layvka = read1d(f, layvka)

        # LAYWET array
        if model.verbose:
            print('   loading LAYWET...')
        laywet = np.empty((nlay), dtype=np.int32)
        laywet = read1d(f, laywet)

        # Item 7: WETFCT, IWETIT, IHDWET
        wetfct, iwetit, ihdwet = None, None, None
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
            # print parm_dict

        # non-parameter data
        transient = not dis.steady.all()
        hk = [0] * nlay
        hani = [0] * nlay
        vka = [0] * nlay
        ss = [0] * nlay
        sy = [0] * nlay
        vkcb = [0] * nlay
        wetdry = [0] * nlay

        # load by layer
        for k in range(nlay):

            # allow for unstructured changing nodes per layer
            if nr is None:
                nrow = 1
                ncol = nc[k]
            else:
                nrow = nr
                ncol = nc

            # hk
            if model.verbose:
                print('   loading hk layer {0:3d}...'.format(k + 1))
            if 'hk' not in par_types:
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'hk',
                                ext_unit_dict)
            else:
                line = f.readline()
                t = mfpar.parameter_fill(model, (nrow, ncol), 'hk', parm_dict,
                                         findlayer=k)
            hk[k] = t

            # hani
            if chani[k] <= 0.:
                if model.verbose:
                    print('   loading hani layer {0:3d}...'.format(k + 1))
                if 'hani' not in par_types:
                    t = Util2d.load(f, model, (nrow, ncol), np.float32, 'hani',
                                    ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'hani',
                                             parm_dict, findlayer=k)
                hani[k] = t

            # vka
            if model.verbose:
                print('   loading vka layer {0:3d}...'.format(k + 1))
            key = 'vk'
            if layvka[k] != 0:
                key = 'vani'
            if 'vk' not in par_types and 'vani' not in par_types:
                t = Util2d.load(f, model, (nrow, ncol), np.float32, key,
                                ext_unit_dict)
            else:
                line = f.readline()
                key = 'vk'
                if 'vani' in par_types:
                    key = 'vani'
                t = mfpar.parameter_fill(model, (nrow, ncol), key, parm_dict,
                                         findlayer=k)
            vka[k] = t

            # storage properties
            if transient:

                # ss
                if model.verbose:
                    print('   loading ss layer {0:3d}...'.format(k + 1))
                if 'ss' not in par_types:
                    t = Util2d.load(f, model, (nrow, ncol), np.float32, 'ss',
                                    ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'ss',
                                             parm_dict, findlayer=k)
                ss[k] = t

                # sy
                if laytyp[k] != 0:
                    if model.verbose:
                        print('   loading sy layer {0:3d}...'.format(k + 1))
                    if 'sy' not in par_types:
                        t = Util2d.load(f, model, (nrow, ncol), np.float32,
                                        'sy',
                                        ext_unit_dict)
                    else:
                        line = f.readline()
                        t = mfpar.parameter_fill(model, (nrow, ncol), 'sy',
                                                 parm_dict, findlayer=k)
                    sy[k] = t

            # vkcb
            if dis.laycbd[k] > 0:
                if model.verbose:
                    print('   loading vkcb layer {0:3d}...'.format(k + 1))
                if 'vkcb' not in par_types:
                    t = Util2d.load(f, model, (nrow, ncol), np.float32, 'vkcb',
                                    ext_unit_dict)
                else:
                    line = f.readline()
                    t = mfpar.parameter_fill(model, (nrow, ncol), 'vkcb',
                                             parm_dict, findlayer=k)
                vkcb[k] = t

            # wetdry
            if (laywet[k] != 0 and laytyp[k] != 0):
                if model.verbose:
                    print('   loading wetdry layer {0:3d}...'.format(k + 1))
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'wetdry',
                                ext_unit_dict)
                wetdry[k] = t

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowLpf.ftype())
            if ipakcb > 0:
                iu, filenames[1] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
                model.add_pop_key_list(ipakcb)

        # create instance of lpf class
        lpf = ModflowLpf(model, ipakcb=ipakcb, laytyp=laytyp, layavg=layavg,
                         chani=chani, layvka=layvka, laywet=laywet, hdry=hdry,
                         iwdflg=iwetdry,  wetfct=wetfct, iwetit=iwetit,
                         ihdwet=ihdwet, hk=hk, hani=hani, vka=vka, ss=ss,
                         sy=sy, vkcb=vkcb, wetdry=wetdry,
                         storagecoefficient=storagecoefficient,
                         constantcv=constantcv, thickstrt=thickstrt,
                         novfc=novfc,
                         unitnumber=unitnumber, filenames=filenames)
        if check:
            lpf.check(f='{}.chk'.format(lpf.name[0]),
                      verbose=lpf.parent.verbose, level=0)
        return lpf

    @staticmethod
    def ftype():
        return 'LPF'

    @staticmethod
    def defaultunit():
        return 15
