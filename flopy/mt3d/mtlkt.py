__author__ = 'emorway'

import sys
import numpy as np

from ..pakbase import Package
from flopy.utils import Util2d, Util3d, read1d, MfList
class Mt3dLkt(Package):
    """
    MT3D-USGS LaKe Transport package class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    nlkinit : int
        is equal to the number of simulated lakes as specified in the flow
        simulation
    mxlkbc : int
        must be greater than or equal to the sum total of boundary conditions 
        applied to each lake
    icbclk : int 
        is equal to the unit number on which lake-by-lake transport information
        will be printed.  This unit number must appear in the NAM input file 
        required for every MT3D-USGS simulation.
    ietlak : int 
        specifies whether or not evaporation as simulated in the flow solution 
        will act as a mass sink.
        = 0, Mass does not exit the model via simulated lake evaporation
        != 0, Mass may leave the lake via simulated lake evaporation
    coldlak : array of floats
        is a vector of real numbers representing the initial concentrations in 
        the simulated lakes.  The length of the vector is equal to the number 
        of simulated lakes, NLKINIT.  Initial lake concentrations should be 
        in the same order as the lakes appearing in the LAK input file 
        corresponding to the MODFLOW simulation.
    ntmp : int
        is an integer value corresponding to the number of specified lake 
        boundary conditions to follow.  For the first stress period, this 
        value must be greater than or equal to zero, but may be less than 
        zero in subsequent stress periods.
    ilkbc : int
        is the lake number for which the current boundary condition will be 
        specified
    ilkbctyp : int
        specifies what the boundary condition type is for ilakbc
           1   a precipitation boundary. If precipitation directly to lakes 
               is simulated in the flow model and a non-zero concentration 
               (default is zero) is desired, use ISFBCTYP = 1;
           2   a runoff boundary condition that is not the same thing as 
               runoff simulated in the UZF1 package and routed to a lake (or 
               stream) using the IRNBND array.  Users who specify runoff in 
               the LAK input via the RNF variable appearing in record set 9a 
               and want to assign a non-zero concentration (default is zero) 
               associated with this specified source, use ISFBCTYP=2;
           3   a Pump boundary condition.  Users who specify a withdrawl
               from a lake via the WTHDRW variable appearing in record set 9a 
               and want to assign a non-zero concentration (default is zero) 
               associated with this specified source, use ISFBCTYP=2;
           4   an evaporation boundary condition.  In models where evaporation 
               is simulated directly from the surface of the lake, users can use
               this boundary condition to specify a non-zero concentration 
               (default is zero) associated with the evaporation losses.

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
    >>> lkt = flopy.mt3d.Mt3dLkt(mt)

    """

    unitnumber = 45
    def __init__(self, model, nlkinit=0, mxlkbc=0, icbclk=0, ietlak=0, 
                 coldlak=0.0, lk_stress_period_data=None, dtype=None,
                 extension='lkt', unitnumber=None, **kwargs):
        #unit number
        if unitnumber is None:
            unitnumber = self.unitnumber
        Package.__init__(self, model, extension, 'LKT', self.unitnumber)

        # Set dimensions
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp

        # Set package specific parameters
        self.nlkinit = nlkinit
        self.mxlkbc = mxlkbc
        self.icbclk = icbclk
        self.ietlak = ietlak

        # Set initial lake concentrations
        if coldlak is not None:
            self.coldlak = Util2d(self.parent, (nlkinit,), np.float32, coldlak,
                                  name='coldlak', locat=self.unit_number[0])
        else:
            self.coldlak = Util2d(self.parent, (nlkinit,), np.float32, 0.0,
                                  name='coldlak', locat=self.unit_number[0])

        # Set transient data
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(ncomp)

        if lk_stress_period_data is None:
            self.lk_stress_period_data = None
        else:
            self.lk_stress_period_data = MfList(self, model=model,
                                                data=lk_stress_period_data)

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """

        # Open file for writing
        f_lkt = open(self.fn_path, 'w')

        # Item 1
        f_lkt.write('{0:10d}{1:10d}{2:10}{3:10}          '
              .format(self.nlkinit, self.mxlkbc, self.icbclk, self.ietlak) +
                    '# NLKINIT, MXLKBC, ICBCLK, IETLAK\n')

        # Item 2
        f_lkt.write(self.coldlak.get_file_entry())

        # Items 3-4
        # (Loop through each stress period and write LKT information)
        nper = self.parent.nper
        for kper in range(nper):
            if f_lkt.closed == True:
                f_lkt = open(f_lkt.name, 'a')

            # List of concentrations associated with fluxes in/out of lake
            # (Evap, precip, specified runoff into the lake, specified
            # withdrawl directly from the lake
            if self.lk_stress_period_data is not None:
                self.lk_stress_period_data.write_transient(f_lkt, single_per=kper)
            else:
                f_lkt.write('{}\n'.format(0))

        f_lkt.close()
        return

    @staticmethod
    def load(f, model, nlak=None, nper=None, ncomp=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        nlak : int
            number of lakes to be simulated 
        nper : int
            number of stress periods
        ncomp : int
            number of species to be simulated
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        lkt :  MT3D-USGS object
            MT3D-USGS object.

        Examples
        --------

        >>> import flopy
        >>> import os
        >>> os.chdir(r'C:\EDM_LT\GitHub\mt3d-usgs\autotest\temp\LKT')
        >>> mt = flopy.mt3d.Mt3dms.load('lkt_mt.nam', exe_name = 'mt3d-usgs_1.0.00.exe',
        >>>                            model_ws = r'.\LKT',
        >>>                            load_only='btn')
        >>> lkt = flopy.mt3d.Mt3dLkt.load('test.lkt', mt)

        """
        if model.verbose:
            sys.stdout.write('loading lkt package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set default nlay values
        nlay = None
        nrow = None
        ncol = None

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

        # Item 1 (NLKINIT,MXLKBC,ICBCLK,IETLAK)
        line = f.readline()
        if line[0] == '#':
            if model.verbose:
                print('   LKT package currently does not support comment lines...')
                sys.exit()

        if model.verbose:
            print('   loading nlkinit,mxlkbc,icbclk,ietlak   ')
        vals = line.strip().split()

        nlkinit = int(vals[0])
        mxlkbc = int(vals[1])
        icbclk = int(vals[2])
        ietlak = int(vals[3])

        if model.verbose:
            print('   NLKINIT {}'.format(nlkinit))
            print('   MXLKBC {}'.format(mxlkbc))
            print('   ICBCLK {}'.format(icbclk))
            print('   IETLAK {}'.format(ietlak))
            if ietlak == 0:
                print('   Mass does not exit the model via simulated lake evaporation   ')
            else:
                print('   Mass exits the lake via simulated lake evaporation   ')

        # Item 2 (COLDLAK - Initial concentration in this instance)
        if model.verbose:
            print('   loading initial concentration   ')
        if model.array_foramt == 'free':
            # ******************************
            # Need to fill this section out
            # ******************************
            pass
        else:
            # Read header line
            line = f.readline()
            
            # Next, read the values
            coldlak = np.empty((nlkinit), dtype=np.float)
            coldlak = read1d(f, coldlak)

        # dtype
        dtype = Mt3dLkt.get_default_dtype(ncomp)

        # Items 3-4
        lk_stress_period_data = {}

        for iper in range(nper):
            if model.verbose:
                print('   loading lkt boundary condition data for kper {0:5d}'
                      .format(iper + 1))

            # Item 3: NTMP: An integer value corresponding to the number of 
            #         specified lake boundary conditions to follow.  
            #         For the first stress period, this value must be greater 
            #         than or equal to zero, but may be less than zero in 
            #         subsequent stress periods.
            line = f.readline()
            vals = line.strip().split()
            ntmp = int(vals[0])
            if model.verbose:
                print("   {0:5d}".format(ntmp) + " lkt boundary conditions specified ")
                if (iper == 0) and (ntmp < 0):
                    print('   ntmp < 0 not allowed for first stress period   ')
                if (iper > 0) and (ntmp < 0):
                    print('   use lkt boundary conditions specified in last stress period   ')
            
            # Item 4: Read ntmp boundary conditions
            if ntmp > 0:
                current_lk = np.empty((ntmp), dtype=dtype)
                for ilkbnd in range(ntmp):
                    line = f.readline()
                    m_arr = line.strip().split()   # These items are free format
                    t = []
                    for ivar in range(2):
                        t.append(m_arr[ivar])
                    cbclk = len(current_sf.dtype.names) - 2
                    if cbcsf > 0:
                        for ilkvar in range(cbclk):
                            t.append(m_arr[ilkvar + 3])
                    current_lk[ilkbnd] = tuple(t[:len(current_lk.dtype.names)])
                # Convert ILKBC index to zero-based
                current_lk['ILKBC'] -= 1
                current_lk = current_lk.view(np.recarray)
                lk_stress_period_data[iper] = current_lk
            else:
                if model.verbose:
                    print('   No transient boundary conditions specified')
                pass

        if len(lk_stress_period_data) == 0:
            lk_stress_period_data = None

        # Construct and return LKT package
        lkt = Mt3dLkt(model, nlkinit=nlkinit, mxlkbc=mxlkbc, icbclk=icbclk,
                      ietlak=ietlak, coldlak=coldlak,
                      lk_stress_period_data=lk_stress_period_data)
        return lkt

    @staticmethod
    def get_default_dtype(ncomp=1):
        """
        Construct a dtype for the recarray containing the list of boundary 
        conditions interacting with the lake (i.e., pumps, specified runoff...)
        """
        type_list = [("ILKBC", np.int), ("ILKBCTYPE", np.int), ("CBCLK", np.float32)]
        if ncomp > 1:
            for comp in range(1,ncomp+1):
                comp_name = "clkt({0:02d})".format(comp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
        return dtype
