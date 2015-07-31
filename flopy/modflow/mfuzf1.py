import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d


class ModflowUzf1(Package):
    """
    MODFLOW Unsaturated Zone Flow 1 Boundary Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    nuztop : integer
        used to define which cell in a vertical column that recharge and discharge is simulated.
            1 Recharge to and discharge from only the top model layer. This option assumes land surface is defined
              as top of layer 1.
            2 Recharge to and discharge from the specified layer in variable IUZFBND. This option assumes land
              surface is defined as top of layer specified in IUZFBND.
            3 Recharge to and discharge from the highest active cell in each vertical column. Land surface is
              determined as top of layer specified in IUZFBND. A constant head node intercepts any recharge and
              prevents deeper percolation.
        (default is 1)
    iuzfopt : integer
        equal to 1 or 2. A value of 1 indicates that the vertical hydraulic conductivity will be
        specified within the UZF1 Package input file using array VKS. A value of 2 indicates that the vertical
        hydraulic conductivity will be specified within either the BCF or LPF Package input file.
        (default is 0)
    irunflg : integer
        specifies whether ground water that discharges to land surface will be routed to stream
        segments or lakes as specified in the IRUNBND array (IRUNFLG not equal to zero) or if ground-water
        discharge is removed from the model simulation and accounted for in the ground-water budget as a loss of
        water (IRUNFLG=0). The Streamflow-Routing (SFR2) and(or) the Lake (LAK3) Packages must be active if
        IRUNFLG is not zero.
        (default is 0
    ietflg : integer
        specifies whether or not evapotranspiration (ET) will be simulated. ET will not be
        simulated if IETFLG is zero, otherwise it will be simulated.
        (default is 0)
    iuzfcb1 : integer
        flag for writing ground-water recharge, ET, and ground-water discharge to land
        surface rates to a separate unformatted file using subroutine UBUDSV. If IUZFCB1>0, it is the unit number
        to which the cell-by-cell rates will be written when 'SAVE BUDGET' or a non-zero value for ICBCFL is
        specified in Output Control. If IUZFCB1 less than or equal to 0, cell-by-cell rates will not be written to a
        file.
        (default is 57)
    iuzfcb2 : integer
        flag for writing ground-water recharge, ET, and ground-water discharge to land
        surface rates to a separate unformatted file using module UBDSV3. If IUZFCB2>0, it is the unit number to
        which cell-by-cell rates will be written when 'SAVE BUDGET' or a non-zero value for ICBCFL is specified
        in Output Control. If IUZFCB2 less than or equal to 0, cell-by-cell rates will not be written to file.
        (default is 0)
    ntrail2 : integer
        equal to the number of trailing waves used to define the water-content profile following a
        decrease in the infiltration rate. The number of trailing waves varies depending on the problem, but a range
        between 10 and 20 is usually adequate. More trailing waves may decrease mass-balance error and will increase
        computational requirements and memory usage.
        (default is 10)
    nsets : integer
        equal to the number of wave sets used to simulate multiple infiltration periods. The number
        of wave sets should be set to 20 for most problems involving time varying infiltration. The total number of
        waves allowed within an unsaturated zone cell is equal to NTRAIL2*NSETS2. An error will occur if the
        number of waves in a cell exceeds this value.
        (default is 20)
    nuzgag : integer
        equal to the number of cells (one per vertical column) that will be specified for printing
        detailed information on the unsaturated zone water budget and water content. A gage also may be used to print
        the budget summed over all model cells.
        (default is 0)
    surfdep : float
        The average height of undulations, D (Figure 1 in UZF documentation), in the land surface altitude.
        (default is 1.0)
    iuzfbnd : integer
        used to define the aerial extent of the active model in which recharge and discharge
        will be simulated.
        (default is 1)
    irunbnd : integer
        used to define the stream segments within the Streamflow-Routing (SFR2) Package
        or lake numbers in the Lake (LAK3) Package that overland runoff from excess infiltration and ground-water
        discharge to land surface will be added. A positive integer value identifies the stream segment and a negative
        integer value identifies the lake number.
        (default is 0)
    vks : float
        used to define the saturated vertical hydraulic conductivity of the unsaturated
        zone (LT-1).
        (default is 1.0E-6)
    eps : float
        values for each model cell used to define the Brooks-Corey epsilon of the unsaturated
        zone. Epsilon is used in the relation of water content to hydraulic conductivity (Brooks and Corey, 1966).
        (default is 3.5)
    thts : float
        used to define the saturated water content of the unsaturated zone in units of
        volume of water to total volume (L3L-3).
        (default is 0.35)
    thtr : float
        used to define the residual water content for each vertical column of cells in units of volume of water to
        total volume (L3L-3). THTR is the irreducible water content and the unsaturated water content cannot drain
        to water contents less than THTR. This variable is not included unless the key word SPECIFYTHTR is specified.
        (default is 0.15)
    thti : float
        used to define the initial water content for each vertical column of cells in
        units of volume of water at start of simulation to total volume (L3L-3). THTI should not be specified for
        steady-state simulations.
       (default is 0.20)
    row_col_iftunit_iuzopt : list
        used to specify where information will be printed for each time step. IUZOPT specifies what that information
        will be. IUZOPT is
        1 Prints time, ground-water head, and thickness of unsaturated zone, and cumulative volumes of
          infiltration, recharge, storage, change in storage and ground-water discharge to land surface.
        2 Same as option 1 except rates of infiltration, recharge, change in storage, and ground-water discharge
          also are printed.
        3 Prints time, ground-water head, thickness of unsaturated zone, followed by a series of depths and
          water contents in the unsaturated zone.
        (default is [])
    specifythtr : boolean
        key word for specifying optional input variable THTR
        (default is 0)
    specifythti : boolean
        key word for specifying optional input variable THTI.
        (default is 0)
    nosurfleak : boolean
        key word for inactivating calculation of surface leakage.
        (default is 0)
    finf : float
        used to define the infiltration rates (LT-1) at land surface for each vertical column of cells.
        If FINF is specified as being greater than the vertical hydraulic conductivity then FINF is set equal to the
        vertical unsaturated hydraulic conductivity. Excess water is routed to streams or lakes when IRUNFLG is not
        zero, and if SFR2 or LAK3 is active.
        (default is 1.0E-8)
    pet : float
        used to define the ET demand rates (L1T-1) within the ET extinction depth interval for each vertical
        column of cells.
        (default is 5.0E-8)
    extdp : float
        used to define the ET extinction depths.
        The quantity of ET removed from a cell is limited by the volume of water stored in the unsaturated zone
        above the extinction depth. If ground water is within the ET extinction depth, then the rate removed is based
        on a linear decrease in the maximum rate at land surface and zero at the ET extinction depth. The linear
        decrease is the same method used in the Evapotranspiration Package (McDonald and Harbaugh, 1988, chap. 10).
        (default is 15.0)
    extwc : float
        used to define the extinction water content below which ET cannot be removed from the unsaturated zone.
        EXTWC must have a value between (THTS-Sy) and THTS, where Sy is the specific yield specified in either the
        LPF or BCF Package.
        (default is 0.1)
    uzfbud_ext : list
        appears to be used for sequential naming of budget output files
        (default is [])
    extension : string
        Filename extension (default is 'uzf')
    unitnumber : int
        File unit number (default is 19).

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
    >>> ml = flopy.modflow.Modflow()
    >>> uzf = flopy.modflow.ModflowUzf1(ml, ...)

    """
    def __init__(self, model, \
    nuztop = 1, iuzfopt = 0, irunflg = 0, ietflg = 0, iuzfcb1 = 57, iuzfcb2 = 0, ntrail2 = 10, nsets = 20, nuzgag = 0, surfdep = 1.0, \
    iuzfbnd = 1, irunbnd = 0, vks = 1.0E-6, eps = 3.5, thts = 0.35, thtr = 0.15, thti = 0.20, row_col_iftunit_iuzopt = [], \
    specifythtr = 0, specifythti = 0, nosurfleak = 0, \
    finf = 1.0E-8, pet = 5.0E-8, extdp = 15.0, extwc = 0.1, \
    uzfbud_ext = [], extension ='uzf', unitnumber = 19):
        Package.__init__(self, model, extension, ['UZF'], unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        if self.parent.get_package('RCH') != None or self.parent.get_package('EVT') != None:
            print('WARNING!\n The RCH and EVT packages should not be active when the UZF1 package is active!')
        if self.parent.version == 'mf2000':
            print('WARNING!\nThe UZF1 package is only compatible with MODFLOW-2005 and MODFLOW-NWT!')
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self.heading = '# UZF1 for MODFLOW, generated by Flopy.'
        self.url = 'uzf_unsaturated_zone_flow_pack.htm'
        # Data Set 1a
        self.specifythtr = specifythtr
        self.specifythti = specifythti
        self.nosurfleak  = nosurfleak
        # Data Set 1b
        # NUZTOP IUZFOPT IRUNFLG IETFLG IUZFCB1 IUZFCB2 [NTRAIL2 NSETS2] NUZGAG SURFDEP
        self.nuztop = nuztop
        self.iuzfopt = iuzfopt
        self.irunflg = irunflg
        self.ietflg = ietflg
        self.iuzfcb1 = iuzfcb1
        self.iuzfcb2 = iuzfcb2
        class_nam = ['UZF']
        if (not isinstance(unitnumber, list)):
            unitnumber = [unitnumber]
        if (not isinstance(extension, list)):
            extension = [extension]
        if iuzfcb1 > 0 and iuzfcb2 < 1:
            unitnumber.append(iuzfcb1)
            extension.append(extension[0]+'bt1')
            class_nam += ['DATA(BINARY)']
        elif iuzfcb1 < 1 and iuzfcb2 > 0:
            unitnumber.append(iuzfcb2)
            extension.append(extension[0]+'bt2')
            class_nam += ['DATA(BINARY)']
        elif iuzfcb1 > 0 and iuzfcb2 > 0:
            unitnumber.append(iuzfcb1)
            extension.append(extension[0]+'bt1')
            unitnumber.append(iuzfcb2)
            extension.append(extension[0]+'bt2')
            class_nam += ['DATA(BINARY)','DATA(BINARY)']
        if iuzfopt > 0:
            self.ntrail2 = ntrail2
            self.nsets = nsets
        self.nuzgag = nuzgag
        self.surfdep = surfdep
        #Data Set 2
        #IUZFBND (NCOL, NROW) -- U2DINT            
        self.iuzfbnd = util_2d(model,(nrow,ncol),np.int,iuzfbnd,name='iuzfbnd')
        #If IRUNFLG > 0: Read item 3
        #Data Set 3
        #[IRUNBND (NCOL, NROW)] -- U2DINT
        if irunflg > 0:            
            self.irunbnd = util_2d(model,(nrow,ncol),np.int,irunbnd,name='irunbnd')
        #IF the absolute value of IUZFOPT = 1: Read item 4.
        #Data Set 4
        #[VKS (NCOL, NROW)] -- U2DREL
        if abs(iuzfopt) == 1:            
            self.vks = util_2d(model,(nrow,ncol),np.float32,vks,name='vks')
        if iuzfopt > 0:
            #Data Set 5
            #EPS (NCOL, NROW) -- U2DREL            
            self.eps = util_2d(model,(nrow,ncol),np.float32,eps,name='eps')
            #Data Set 6a
            #THTS (NCOL, NROW) -- U2DREL            
            self.thts = util_2d(model,(nrow,ncol),np.float32,thts,name='thts')
            #Data Set 6b
            #THTS (NCOL, NROW) -- U2DREL
            if self.specifythtr > 0:                
                self.thtr = util_2d(model,(nrow,ncol),np.float32,thtr,name='thtr')
            #Data Set 7
            #[THTI (NCOL, NROW)] -- U2DREL            
            self.thti = util_2d(model,(nrow,ncol),np.float32,thti,name='thti')
        #Data Set 8
        #[IUZROW] [IUZCOL] IFTUNIT [IUZOPT]
        if len(row_col_iftunit_iuzopt) != nuzgag:
            print("WARNING!\nItem 8 doesn't correspond with NUZGAG.\nNUZGAG set to 0")
            self.nuzgag = 0
            self.row_col_iftunit_iuzopt = []
        else:
            self.row_col_iftunit_iuzopt = row_col_iftunit_iuzopt
            i = 0
            for l in row_col_iftunit_iuzopt:
                unitnumber.append(abs(l[0][2]))
                if uzfbud_ext ==[]:
                    extension.append(extension[0] + 'b' + str(i))
                else:
                    extension.append(uzfbud_ext[i])
                i += 1
            Package.__init__(self,model,extension,class_nam+nuzgag*['DATA'],unit_number=unitnumber)
        #Dataset 9, 11, 13 and 15 will be written automatically in the write_file function
        #Data Set 10
        #[FINF (NCOL, NROW)] – U2DREL
        self.finf = []
        if (not isinstance(finf, list)):
            finf = [finf]
        for i,a in enumerate(finf):          
            b = util_2d(model,(nrow,ncol),np.float32,a,name='finf_'+str(i+1))
            self.finf = self.finf + [b]
        if ietflg > 0:
            #Data Set 12
            #[PET (NCOL, NROW)] – U2DREL
            self.pet = []
            if (not isinstance(pet, list)):
                pet = [pet]
            for i,a in enumerate(pet):               
                b = util_2d(model,(nrow,ncol),np.float32,a,name='pet_'+str(i+1))
                self.pet = self.pet + [b]
            #Data Set 14
            #[EXTDP (NCOL, NROW)] – U2DREL
            self.extdp = []
            if (not isinstance(extdp, list)):
                extdp = [extdp]
            for i,a in enumerate(extdp):               
                b = util_2d(model,(nrow,ncol),np.float32,a,name='extdp_'+str(i+1))
                self.extdp = self.extdp + [b]
            #Data Set 16
            #[EXTWC (NCOL, NROW)] – U2DREL
            if iuzfopt> 0:
                self.extwc = []
                if (not isinstance(extwc, list)):
                    extwc = [extwc]
                for i,a in enumerate(extwc):                   
                    b = util_2d(model,(nrow,ncol),np.float32,a,name='extwc_'+str(i+1))
                    self.extwc = self.extwc + [b]
        self.parent.add_package(self)
    def __repr__( self ):
        return 'UZF1 class'
    def ncells( self):
        # Returns the  maximum number of cells that have recharge (developped for MT3DMS SSM package)
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        return (nrow * ncol)
    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        f_uzf = open(self.fn_path, 'w')
        f_uzf.write('%s\n' % self.heading)
        # Dataset 1a
        specify_temp = ''
        if self.specifythtr > 0:
            specify_temp += 'SPECIFYTHTR '
        if self.specifythti > 0:
            specify_temp += 'SPECIFYTHTI '
        if self.nosurfleak > 0:
            specify_temp += 'NOSURFLEAK'
        if (self.specifythtr + self.specifythti + self.nosurfleak) > 0:
            f_uzf.write('%s\n' % specify_temp)
        del specify_temp
        # Dataset 1b
        if self.iuzfopt > 0:
            comment = ' NUZTOP IUZFOPT IRUNFLG IETFLG IUZFCB1 IUZFCB2 NTRAIL NSETS NUZGAGES'            
            f_uzf.write('{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}{6:10d}{7:10d}{8:10d}{9:15.6E}{10:100s}\n'.\
                format(self.nuztop, self.iuzfopt, self.irunflg, self.ietflg, self.iuzfcb1, self.iuzfcb2, \
                self.ntrail2, self.nsets, self.nuzgag, self.surfdep, comment))
        else:
            comment = ' NUZTOP IUZFOPT IRUNFLG IETFLG IUZFCB1 IUZFCB2 NUZGAGES'            
            f_uzf.write('{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}{6:10d}{7:15.6E}{8:100s}\n'.\
                format(self.nuztop, self.iuzfopt, self.irunflg, self.ietflg, self.iuzfcb1, self.iuzfcb2, \
                self.nuzgag, self.surfdep, comment))       
        f_uzf.write(self.iuzfbnd.get_file_entry())
        if self.irunflg > 0:           
            f_uzf.write(self.irunbnd.get_file_entry())
        #IF the absolute value of IUZFOPT = 1: Read item 4.
        #Data Set 4
        #[VKS (NCOL, NROW)] -- U2DREL
        if abs(self.iuzfopt) == 1:            
            f_uzf.write(self.vks.get_file_entry())
        if self.iuzfopt > 0:
            #Data Set 5
            #EPS (NCOL, NROW) -- U2DREL           
            f_uzf.write(self.eps.get_file_entry())
            #Data Set 6a
            #THTS (NCOL, NROW) -- U2DREL            
            f_uzf.write(self.thts.get_file_entry())
            #Data Set 6b
            #THTR (NCOL, NROW) -- U2DREL
            if self.specifythtr > 0.0:               
                f_uzf.write(self.thtr.get_file_entry())
            #Data Set 7
            #[THTI (NCOL, NROW)] -- U2DREL
            if not self.parent.get_package('DIS').steady[0] or self.specifythti > 0.0:               
                f_uzf.write(self.thti.get_file_entry())
        #If NUZGAG>0: Item 8 is repeated NUZGAG times
        #Data Set 8
        #[IUZROW] [IUZCOL] IFTUNIT [IUZOPT]
        if self.nuzgag > 0:
            for n in range(self.nuzgag):
                if self.row_col_iftunit_iuzopt[n][0][2] > 0:
                    comment = ' IUZROW IUZCOL IFTUNIT IUZOPT'
                    f_uzf.write('%10i%10i%10i%10i%s\n' % (tuple(self.row_col_iftunit_iuzopt[n][0] + [comment])))
                    #f_uzf.write('{0:10d}{1:10d}{2:10d}{3:10d}{4:50s}\n'.\
                    #    format(tuple(self.row_col_iftunit_iuzopt[n][0] + [comment])))
                else:
                    comment = ' IFTUNIT'
                    f_uzf.write('%10i%s\n' % (tuple([self.row_col_iftunit_iuzopt[n][0][2]] + [comment])))
        for n in range(nper):
            comment = ' NUZF1 for stress period ' + str(n + 1)
            if (n < len(self.finf)):
                nuzf1 = 1
            else:
                nuzf1 = -1
            #f_uzf.write('%10i%s\n' % (nuzf1, comment))
            f_uzf.write('{0:10d}{1:20s}\n'.format(nuzf1,comment))
            comment = 'FINF for stress period ' + str(n + 1)
            if (n < len(self.finf)):               
                f_uzf.write(self.finf[n].get_file_entry())
            comment = ' NUZF2 for stress period ' + str(n + 1)
            if self.ietflg > 0:
                if (n < len(self.pet)):
                    nuzf2 = 1
                else:
                    nuzf2 = -1
                #f_uzf.write('%10i%s\n' % (nuzf2, comment))
                f_uzf.write('{0:10d}{1:20s}\n'.format(nuzf2,comment))
                comment = 'PET for stress period ' + str(n + 1)
                if (n < len(self.pet)):                   
                    f_uzf.write(self.pet[n].get_file_entry())
                comment = ' NUZF3 for stress period ' + str(n + 1)
                if (n < len(self.extdp)):
                    nuzf3 = 1
                else:
                    nuzf3 = -1               
                f_uzf.write('{0:10d}{1:20s}\n'.format(nuzf3,comment))
                comment = 'EXTDP for stress period ' + str(n + 1)
                if (n < len(self.extdp)):                    
                    f_uzf.write(self.extdp[n].get_file_entry())
                comment = ' NUZF4 for stress period ' + str(n + 1)
                if self.iuzfopt > 0:
                    if (n < len(self.extwc)):
                        nuzf4 = 1
                    else:
                        nuzf4 = -1                   
                    f_uzf.write('{0:10d}{1:20s}\n'.format(nuzf4,comment))
                    comment = 'EXTWC for stress period ' + str(n + 1)
                    if (n < len(self.extwc)):                       
                        f_uzf.write(self.extwc[n].get_file_entry())
        f_uzf.close()

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
        uzf : ModflowUZF1 object
            ModflowUZF1 object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> uzf = flopy.modflow.ModflowUZF1.load('test.uzf', m)

        """
        if model.verbose:
            sys.stdout.write('loading uzf package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        #dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        # dataset 1

        # todo: everything
        print('   Warning: load method not completed. default uzf object created.')

        # close the file
        f.close()

        # create uzf object
        uzf = ModflowUzf1(model)

        # return default uzf object
        return uzf
