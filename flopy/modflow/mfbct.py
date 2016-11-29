import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Util3d

class ModflowBct(Package):
    '''
    Block centered transport package class for MODFLOW-USG
    '''
    def __init__(self, model, itrnsp=1, ibctcb=0, mcomp=1, ic_ibound_flg=1,
                 itvd=1, iadsorb=0, ict=0, cinact=-999., ciclose=1.e-6,
                 idisp=1, ixdisp=0, diffnc=0., izod=0, ifod=0, icbund=1,
                 porosity=0.1, bulkd=1., arad=0., dlh=0., dlv=0., dth=0.,
                 dtv=0., sconc=0.,
                 extension='bct', unitnumber=None):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowBct.defaultunit()

        # Call ancestor's init to set self.parent, extension, name and unit
        # number
        Package.__init__(self, model, extension, ModflowBct.ftype(), unitnumber)

        self.url = 'bct.htm'
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self.itrnsp = itrnsp
        self.ibctcb = ibctcb
        self.mcomp = mcomp
        self.ic_ibound_flg = ic_ibound_flg
        self.itvd = itvd
        self.iadsorb = iadsorb
        self.ict = ict
        self.cinact = cinact
        self.ciclose = ciclose
        self.idisp = idisp
        self.ixdisp = ixdisp
        self.diffnc = diffnc
        self.izod = izod
        self.ifod = ifod
        self.icbund = Util3d(model, (nlay, nrow, ncol), np.float32, icbund,
                              'icbund',)
        self.porosity = Util3d(model, (nlay, nrow, ncol), np.float32,
                                porosity, 'porosity')
        #self.arad = Util2d(model, (1, nja), np.float32,
        #                        arad, 'arad')
        self.dlh = Util3d(model, (nlay, nrow, ncol), np.float32, dlh, 'dlh')
        self.dlv = Util3d(model, (nlay, nrow, ncol), np.float32, dlv, 'dlv')
        self.dth = Util3d(model, (nlay, nrow, ncol), np.float32, dth, 'dth')
        self.dtv = Util3d(model, (nlay, nrow, ncol), np.float32, dth, 'dtv')
        self.sconc = Util3d(model, (nlay, nrow, ncol), np.float32, sconc,
                             'sconc',)
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Open file for writing
        f_bct = open(self.fn_path, 'w')
        # Item 1: ITRNSP, IBCTCB, MCOMP, IC_IBOUND_FLG, ITVD, IADSORB,
        #         ICT, CINACT, CICLOSE, IDISP, IXDISP, DIFFNC, IZOD, IFOD
        s = '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}'
        s = s.format(self.itrnsp, self.ibctcb, self.mcomp, self.ic_ibound_flg,
                     self.itvd, self.iadsorb, self.ict, self.cinact,
                     self.ciclose, self.idisp, self.ixdisp, self.diffnc,
                     self.izod, self.ifod)
        f_bct.write(s + '\n')
        #
        #ibound
        if(self.ic_ibound_flg == 0):
            for k in range(nlay):
                f_bct.write(self.icbund[k].get_file_entry())
        #
        #porosity
        for k in range(nlay):
            f_bct.write(self.porosity[k].get_file_entry())
        #
        #bulkd
        if self.iadsorb != 0:
            for k in range(nlay):
                f_bct.write(self.bulkd[k].get_file_entry())
        #
        #arad
        if self.idisp != 0:
            f_bct.write('open/close arad.dat 1.0 (free) -1' + '\n')
        #
        #dlh
        if self.idisp == 1:
            for k in range(nlay):
                f_bct.write(self.dlh[k].get_file_entry())
        #
        #dlv
        if self.idisp == 2:
            for k in range(nlay):
                f_bct.write(self.dlv[k].get_file_entry())
        #
        #dth
        if self.idisp == 1:
            for k in range(nlay):
                f_bct.write(self.dth[k].get_file_entry())
        #
        #dtv
        if self.idisp == 2:
            for k in range(nlay):
                f_bct.write(self.dtv[k].get_file_entry())
        #
        #sconc
        for k in range(nlay):
            f_bct.write(self.sconc[k].get_file_entry())


        return


    @staticmethod
    def ftype():
        return 'BCT'


    @staticmethod
    def defaultunit():
        return 35

