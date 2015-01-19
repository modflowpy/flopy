import sys
import copy
import numpy as np
import warnings
from flopy.mbase import Package
from flopy.utils import util_2d
from flopy.utils.util_list import mflist

# Note: Order matters as first 6 need logical flag on line 1 of SSM file
SsmLabels = ['WEL', 'DRN', 'RCH', 'EVT', 'RIV', 'GHB', 'BAS6', 'CHD', 'PBC']
class SsmPackage(object):
    def __init__(self, label = '', instance = None, needTFstr = False):
        self.label = label
        self.instance = instance
        self.needTFstr = needTFstr
        self.TFstr = ' F'
        if (self.instance != None):
           self.TFstr = ' T'

class Mt3dSsm(Package):
    '''
    Sink & Source Mixing package class\n
    '''
    def __init__(self, model, crch = None, cevt = None, 
                 stress_period_data = None, dtype = None,
                 extension = 'ssm', 
                 **kwargs):
        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, 'SSM', 34)

        deprecated_kwargs = ['criv', 'cghb', 'cibd', 'cchd', 'cpbc', 'cwel'] 
        for key in kwargs:
            if (key in deprecated_kwargs):
                warnings.warn("Deprecation Warning: Keyword argument '" + key +
                              "' no longer supported. Use " +
                              "'stress_period_data' instead.")
                
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        
        self.__SsmPackages = []
        for i, label in enumerate(SsmLabels):
            self.__SsmPackages.append(SsmPackage(label, 
                               self.parent.mf.get_package(label),
                               (i < 6)))

        self.__maxssm = 0
        #if (self.parent.btn.icbund != None):
        #    self.maxssm = (self.parent.btn.icbund < 0).astype(int).sum()
        for p in self.__SsmPackages:
            if ((p.label == 'BAS6') and (p.instance != None)):
                self.__maxssm += (p.instance.ibound < 0).sum()
            elif p.instance != None:
                self.__maxssm += p.instance.ncells()
        
        if (crch != None):
            self.crch = []
            if (not isinstance(crch, list)):
                crch = [crch]
            for i, a in enumerate(crch):
                r = util_2d(model, (nrow, ncol), np.float32, a, 
                            name = 'crch_' + str(i + 1))
                self.crch.append(r)
        else:
            self.crch = None

        if (cevt != None):
            self.cevt = []
            if (not isinstance(cevt, list)):
                cevt = [cevt]
            for i, a in enumerate(cevt):
                r = util_2d(model,(nrow, ncol), np.float32, a, 
                            name = 'cevt_' + str(i + 1))
                self.cevt.append(r)
        else:
            self.cevt = None

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype()
        self.stress_period_data = mflist(self.parent.mf, self.dtype, 
                                         stress_period_data)

        #Add self to parent and return
        self.parent.add_package(self)
        return
        
    @staticmethod
    def get_default_dtype():
        dtype = np.dtype([("k", np.int), ("i", np.int), \
                          ("j", np.int), ("css", np.float32),\
                          ("itype", np.int)])
        return dtype

    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper

        # Open file for writing
        f_ssm = open(self.fn_path, 'w')
        for p in self.__SsmPackages:
            if p.needTFstr:
                f_ssm.write(p.TFstr)
        f_ssm.write(' F F F F\n')
        f_ssm.write('%10d\n' % (self.__maxssm))
        
        # Loop through each stress period and write ssm information
        for kper in range(nper):

            # Distributed sources and sinks (Recharge and Evapotranspiration)
            if (self.crch != None):
                
                if (kper < len(self.crch)):
                    incrch = 1
                else:
                    incrch = -1
                f_ssm.write('%10i\n' % (incrch))
                if (kper < len(self.crch)):
                    f_ssm.write(self.crch[kper].get_file_entry())

            if (self.cevt != None):
                if (kper < len(self.cevt)):
                    incevt = 1
                else:
                    incevt = -1
                f_ssm.write('%10i\n' % (incevt))
                if (kper < len(self.cevt)):
                    f_ssm.write(self.cevt[kper].get_file_entry())

            self.stress_period_data.write_transient(f_ssm, single_per = kper)

        f_ssm.close()
        return
        


