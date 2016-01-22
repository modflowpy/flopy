__author__ = 'emorway'

import sys
import numpy as np

from ..pakbase import Package
from flopy.utils import Util2d, Util3d, read1d, MfList, Transient2d
class Mt3dUzt(Package):
    """
    MT3D-USGS Unsaturated-Zone Transport package class
    
    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    mxuzcon : int
        Is the maximum number of UZF1 connections and is equal to the number 
        of non-zero entries in the IRNBND array found in the UZF1 input file 
        for MODFLOW.  Keep in mind there is potential for every cell with a 
        non-zero IRNBND entry to pass water to either a lake or stream segment
    icbcuz : int
        Is the unit number to which unsaturated-zone concentration will be 
        written out.
    iet : int
        Is a flag that indicates whether or not ET is being simulated in the 
        UZF1 flow package.  If ET is not being simulated, IET informs FMI 
        package not to look for UZET and GWET arrays in the flow-tranpsort 
        link file.
    iuzfbnd : array of ints
        Specifies which row/column indices variably-saturated transport will 
        be simulated in.
           >0  indicates variably-saturated transport will be simulated;
           =0  indicates variably-saturated transport will not be simulated;
           <0  Corresponds to IUZFBND < 0 in the UZF1 input package, meaning 
               that user-supplied values for FINF are specified recharge and
               therefore transport through the unsaturated zone is not 
               simulated.
    wc : array of floats
        Starting water content.  For cells above the water tables, this value 
        can range between residual and saturated water contents.  In cells 
        below the water table, this value will be eqal to saturated water 
        content (i.e., effective porosity).  For cells containing the water 
        table, a volume average approach needs to be used to calculate an 
        equivalent starting water content.
    sdh : array of floats
        Starting saturated thickness for each cell in the simulation.  For 
        cells residing above the starting water table, SDH=0. In completely 
        saturated cells, SDH is equal to total thickness.  For cells 
        containing the water table, SDH equals the water table elevation minus 
        the cell bottom elevation.
    incuzinf : int
        (This value is repeated for each stress period as explained next) A 
        flag indicating whether an array containing the concentration of 
        infiltrating water (FINF) for each simulated species (ncomp) will be 
        read for the current stress period.  If INCUZINF >= 0, an array 
        containing the concentration of infiltrating flux for each species 
        will be read.  If INCUZINF < 0, the concentration of infiltrating flux 
        will be reused from the previous stress period.  If INCUZINF < 0 is 
        specified for the first stress period, then by default the 
        concentration of positive infiltrating flux (source) is set equal to 
        zero.  There is no possibility of a negative infiltration flux being 
        specified.  If infiltrating water is rejected due to an infiltration 
        rate exceeding the vertical hydraulic conductivity, or because 
        saturation is reached in the unsaturated zone and the water table is 
        therefore at land surface, the concentration of the runoff will be 
        equal to CUZINF specified next.  The runoff is routed if IRNBND is 
        specified in the MODFLOW simulation.
    cuzinf : array of floats
        Is the concentration of the infiltrating flux for a particular species.
        An array for each species will be read.
    incuzet : int
        (This value is repeated for each stress period as explained next) A 
        flag indicating whether an array containing the concentration of 
        evapotranspiration flux originating from the unsaturated zone will be 
        read for the current stress period.  If INCUZET >= 0, an array 
        containing the concentration of evapotranspiration flux originating 
        from the unsaturated zone for each species will be read.  If 
        INCUZET < 0, the concentration of evapotranspiration flux for each 
        species will be reused from the last stress period.  If INCUZET < 0 
        is specified for the first stress period, then by default, the 
        concentration of negative evapotranspiration flux (sink) is set 
        equal to the aquifer concentration, while the concentration of 
        positive evapotranspiration flux (source) is set to zero.
    cuzet : array of floats
        Is the concentration of ET fluxes originating from the unsaturated 
        zone.  As a default, this array is set equal to 0 and only overridden 
        if the user specifies INCUZET > 1.  If empirical evidence suggest 
        volatilization of simulated constituents from the unsaturated zone, 
        this may be one mechanism for simulating this process, though it would 
        depend on the amount of simulated ET originating from the unsaturated 
        zone.  An array for each species will be read.
    incgwet : int
        (This value is repeated for each stress period as explained next) Is 
        a flag indicating whether an array containing the concentration of 
        evapotranspiration flux originating from the saturated zone will be 
        read for the current stress period.  If INCGWET >= 0, an array 
        containing the concentration of evapotranspiration flux originating 
        from the saturated zone for each species will be read.  If 
        INCGWET < 0, the concentration of evapotranspiration flux for each 
        species will be reused from the last stress period.  If INCUZET < 0 
        is specified for the first stress period, then by default, the 
        concentration of negative evapotranspiration flux (sink) is set to 
        the aquifer concentration, while the concentration of positive 
        evapotranspiration flux (source) is set to zero.
    cgwet : array of floats
        Is the concentration of ET fluxes originating from the saturated zone. 
        As a default, this array is set equal to 0 and only overridden if the 
        user specifies INCUZET > 1.  An array for each species will be read.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> mt = flopy.mt3d.Mt3dms()
    >>> lkt = flopy.mt3d.Mt3dUzt(mt)

    """

    unitnumber = 46
    def __init__(self, model, mxuzcon=0, icbcuz=0, iet=0, iuzfbnd=None,
                 wc=0., sdh=0., cuzinf=None, cuzet=None, cgwet=None,
                 extension='uzt', unitnumber=None, **kwargs):
        # unit number
        if unitnumber is None:
            unitnumber = self.unitnumber
        Package.__init__(self, model, extension, 'UZT', self.unitnumber)

        # Set dimensions
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp
        mcomp = model.mcomp

        # Set package specific parameters
        self.mxuzcon = mxuzcon
        self.icbcuz = icbcuz
        self.iet = iet
        if iuzfbnd is not None:
            self.iuzfbnd = Util3d(self.parent, (nlay, nrow, ncol), np.int,
                                  iuzfbnd, name='iuzfbnd',
                                  locat=self.unit_number[0])
        else:  # the else statement should instead set iuzfbnd based on UZF input file.
            arr = np.zeros((nlay, nrow, ncol), dtype=np.int)
            self.iuzfbnd = Util3d(self.parent, (nlay, nrow, ncol), np.int,
                                  arr, name='iuzfbnd',
                                  locat=self.unit_number[0])

        self.wc = Util3d(model, (self.nlay, self.nrow, self.ncol),
                         np.float32, wc, name='wc',
                         locat=self.unit_number[0])

        self.sdh = Util3d(model, (self.nlay, self.nrow, self.ncol),
                          np.float32, sdh, name='sdh',
                          locat=self.unit_number[0])

        # Note: list is used for multi-species, NOT for stress periods!
        if cuzinf is not None:
            self.cuzinf = []
            t2d = Transient2d(model, (nrow, ncol), np.float32, cuzinf,
                              name='cuzinf1', locat=self.unit_number[0])
            self.cuzinf.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = 'cuzinf' + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print('UZT: setting cuzinf for component ' + \
                              str(icomp) + ' to zero. kwarg name ' + name)

                    t2d = Transient2d(model, (nrow, ncol), np.float32, val,
                                      name=name, locat=self.unit_number[0])
                    self.cuzinf.append(t2d)

        if cuzet is not None:
            self.cuzet = []
            t2d = Transient2d(model, (nrow, ncol), np.float32, cuzet,
                              name='cuzet1', locat=self.unit_number[0])
            self.cuzet.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = 'cuzet' + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print('UZT: setting cuzet for component ' + \
                              str(icomp) + ' to zero. kwarg name ' + name)

                    t2d = Transient2d(model, (nrow, ncol), np.float32, val,
                                      name=name, locat=self.unit_number[0])
                    self.cuzet.append(t2d)

        if cgwet is not None:
            self.cgwet = []
            t2d = Transient2d(model, (nrow, ncol), np.float32, cgwet,
                              name='cgwet1', locat=self.unit_number[0])
            self.cgwet.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = 'cgwet' + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs.pop(name)
                    else:
                        print('UZT: setting cgwet for component ' + \
                              str(icomp) + ' to zero. kwarg name ' + name)

                    t2d = Transient2d(model, (nrow, ncol), np.float32, val,
                                      name=name, locat=self.unit_number[0])
                    self.cgwet.append(t2d)

    @staticmethod
    def load(f, model, nlay=None, nrow=None, ncol=None, nper=None,
             ncomp=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        uzt :  Mt3dSsm object
            Mt3dUzt object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> azt = flopy.mt3d.Mt3dUzt.load('test.uzt', mt)

        """

        if model.verbose:
            print('loading uzt package file...\n')

        # Open file if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set dimensions if necessary
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol
        if nper is None:
            nper = model.nper
        if ncomp is None:
            ncomp = model.ncomp

        # Item 1 (comments, must be preceded by '#')
        if model.verbose:
            print('   Reading off comment lines...')
        line = f.readline()
        while line[0:1] == '#':
            i = 1
            if model.verbose:
                print('   Comment Line ' + str(i) + ': '.format(line.strip()))
                i += 1
            line = f.readline()

        # Item 2 (MXUZCON, ICBCUZ, IET)
        if line[0:1] != '#':
            # Don't yet read the next line because the current line because it
            # contains the values in item 2
            m_arr = line.strip().split()
            mxuzcon = m_arr[0]
            icbcuz = m_arr[1]
            iet = m_arr[2]

        # Item 3 [IUZFBND(NROW,NCOL) (one array for each layer)]
        if model.verbose:
            print('   loading IUZFBND...')
        iuzfbnd = Util3d.load(f, model (nlay, nrow, ncol), np.int, 'iuzfbnd',
                              ext_unit_dict)

        # Item 4 [WC(NROW,NCOL) (one array for each layer)]
        if model.verbose:
            print('   loading WC...')
        wc = Util3d.load(f, model, (nlay, nrow, ncol), np.float32, 'wc',
                         ext_unit_dict)

        # Item 5 [SDH(NROW,NCOL) (one array for each layer)]
        if model.verbose:
            print('   loading SDH...')
        sdh = Util3d.load(f, model, (nlay, nrow, ncol), np.float32, 'sdh',
                          ext_unit_dict)

        # kwargs needed to construct cuzinf2, cuzinf3, etc. for multispecies
        kwargs = {}

        cuzinf = None
        # At least one species being simulated, so set up a place holder
        t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0, name='cuzinf',
                          locat=0)
        cuzinf = {0 : t2d}
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = 'cuzinf' + str(icomp)
                t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0,
                                  name=name, locat=0)
                kwargs[name] = {0 : t2d}

        # Repeat cuzinf initialization procedure for cuzet
        cuzet = None
        t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0, name='cuzet',
                          locat=0)
        cuzet = {0 : t2d}
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = 'cuzet' + str(icomp)
                t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0,
                                  name=name, locat=0)
                kwargs[name] = {0 : t2d}

        # Repeat cuzinf initialization procedures for cgwet
        cgwet = None
        t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0, name='cgwet',
                          locat=0)
        cgwet = {0 : t2d}
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = 'cgwet' + str(icomp)
                t2d = Transient2d(model, (nrow, ncol), np.float32, 0.0,
                                  name=name, locat=0)
                kwargs[name] = {0 : t2d}

        # Start of transient data
        for iper in range(nper):

            if model.verbose:
                print('   loading UZT data for kper {0:5d}'.format(iper + 1))

            # Item 6 (INCUZINF)
            line = f.readline()
            m_arr = line.strip().split()
            incuzinf = m_arr[0]

            # Item 7 (CUZINF)
            if incuzinf >= 0:
                if model.verbose:
                    print('   Reading CUZINF array for kper ' \
                          '{0:5d}'.format(iper + 1))
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'cuzinf',
                                ext_unit_dict)
                cuzinf[iper] = t

                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = 'cuzinf' + str(icomp)
                        if model.verbose:
                            print('   loading {}...'.format(name))
                        t = Util2d.load(f, model, (nrow, ncol), np.float32,
                                        name, ext_unit_dict)
                        cuzinficomp = kwargs[name]
                        cuzinficomp[iper] = t

            elif incuzinf < 0 and iper == 0:
                if model.verbose:
                    print('   INCUZINF < 0 in first stress period. Setting ' \
                          'CUZINF to default value of 0.00 for all calls')
                    # This happens implicitly and is taken care of my
                    # existing functionality within flopy.  This elif
                    # statement exist for the purpose of printing the message
                    # above
                pass

            elif incuzinf < 0 and iper > 0:
                if model.verbose:
                    print('   Reusing CUZINF array from kper ' \
                          '{0:5d}'.format(iper) + ' in kper ' \
                          '{0:5d}'.format(iper + 1))

            # Item 8 (INCUZET)
            line = f.readline()
            m_arr = line.strip().split()
            incuzet = m_arr[0]

            # Item 9 (CUZET)
            if incuzet >= 0:
                if model.verbose:
                    print('   Reading CUZET array for kper ' \
                          '{0:5d}'.format(iper + 1))
                t = Util2d.load(f, model, (nrow, ncol), np.float32, 'cuzet',
                                ext_unit_dict)
                cuzet[iper] = t

                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = 'cuzet' + str(icomp)
                        if model.verbose:
                            print('   loading {}'.format(name))
                        t = Util2d.load(f, model, (nrow, ncol), np.float32,
                                        name, ext_unit_dict)
                        cuzeticomp = kwargs[name]
                        cuzeticomp[iper] = t

            elif incuzet < 0 and iper == 0:
                if model.verbose:
                    print('   INCUZET < 0 in first stress period. Setting ' \
                          'CUZET to default value of 0.00 for all calls')
                    # This happens implicitly and is taken care of my
                    # existing functionality within flopy.  This elif
                    # statement exist for the purpose of printing the message
                    # above
                pass
            else:
                if model.verbose:
                    print('   Reusing CUZET array from kper ' \
                          '{0:5d}'.format(iper) + ' in kper ' \
                          '{0:5d}'.format(iper + 1))

            # Item 10 (INCGWET)
            line = f.readline()
            m_arr = line.strip().split()
            incgwet = m_arr[0]

            # Item 11 (CGWET)
            if model.verbose:
                if incuzet >= 0:
                    print('   Reading CGWET array for kper ' \
                          '{0:5d}'.format(iper + 1))
                t = Util2d.load(f, model, (nrow,ncol), np.float32, 'cgwet',
                                ext_unit_dict)
                cgwet[iper] = t

                # Load each multispecies array
                if ncomp > 1:
                    for icomp in range(2, ncomp + 1):
                        name = 'cgwet' + str(icomp)
                        if model.verbose:
                            print('   loading {}...'.format(name))
                        t = Util2d.load(f, model, (nrow, ncol), np.float32,
                                        name, ext_unit_dict)
                        cgweticomp = kwargs[name]
                        cgweticomp[iper] = t

            elif incuzet < 0 and iper == 0:
                if model.verbose:
                    print('   INCGWET < 0 in first stress period. Setting ' \
                          'CGWET to default value of 0.00 for all calls')
                    # This happens implicitly and is taken care of my
                    # existing functionality within flopy.  This elif
                    # statement exist for the purpose of printing the message
                    # above
                    pass

            elif incgwet < 0 and iper > 0:
                if model.verbose:
                    print('   Reusing CGWET array from kper ' \
                          '{0:5d}'.format(iper) + ' in kper ' \
                          '{0:5d}'.format(iper + 1))


        # Construct and return uzt package
        uzt = Mt3dUzt(model, mxuzcon=mxuzcon, icbcuz=icbcuz, iet=iet,
                      iuzfbnd=iuzfbnd, wc=wc, sdh=sdh, cuzinf=cuzinf,
                      cuzet=cuzet, cgwet=cgwet)
        return uzt