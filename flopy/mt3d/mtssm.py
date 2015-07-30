import sys
import copy
import numpy as np
import warnings
from flopy.mbase import Package
from flopy.utils import util_2d
from flopy.utils.util_list import mflist
from flopy.utils.util_array import transient_2d

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
    Sink & Source Mixing package class
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
        # ncomp > 1 support
        ncomp = self.parent.get_ncomp()

        self.__SsmPackages = []
        for i, label in enumerate(SsmLabels):
            self.__SsmPackages.append(SsmPackage(label, 
                               self.parent.mf.get_package(label),
                               (i < 6))) # First 6 need T/F flag in file line 1

        self.__maxssm = 0
        #if (self.parent.btn.icbund != None):
        if isinstance(self.parent.btn.icbund, np.ndarray):
            self.maxssm = (self.parent.btn.icbund < 0).sum()
        for p in self.__SsmPackages:
            if ((p.label == 'BAS6') and (p.instance != None)):
                self.__maxssm += (p.instance.ibound < 0).sum()
            elif p.instance != None:
                self.__maxssm += p.instance.ncells()
        
        # Note: list is used for multi-species, NOT for stress periods!        
        if (crch != None):
            self.crch = []
            t2d = transient_2d(model, (nrow, ncol), np.float32,
                               crch, name='crch1',
                               locat=self.unit_number[0])
            self.crch.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = "crch" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs[name]
                        kwargs.pop(name)
                    else:
                        print("SSM: setting crch for component " +\
                              str(icomp) + " to zero. kwarg name " +\
                              name)
                    t2d = transient_2d(model, (nrow, ncol), np.float32,
                                       val, name=name,
                                       locat=self.unit_number[0])
                    self.crch.append(t2d)
        else:
            self.crch = None

        if (cevt != None):
            self.cevt = []
            t2d = transient_2d(model, (nrow, ncol), np.float32,
                               cevt, name='cevt1',
                               locat=self.unit_number[0])
            self.cevt.append(t2d)
            if ncomp > 1:
                for icomp in range(2, ncomp+1):
                    val = 0.0
                    name = "cevt" + str(icomp)
                    if name in list(kwargs.keys()):
                        val = kwargs[name]
                        kwargs.pop(name)
                    else:
                        print("SSM: setting cevt for component " +\
                              str(icomp) + " to zero, kwarg name " +\
                              name)
                    t2d = transient_2d(model, (nrow, ncol), np.float32,
                                       val, name=name,
                                       locat=self.unit_number[0])
                    self.cevt.append(t2d)

        else:
            self.cevt = None

        if len(list(kwargs.keys())) > 0:
            raise Exception("SSM error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))

        if dtype is not None:
            self.dtype = dtype
        else:

            self.dtype = self.get_default_dtype(ncomp)
  
        self.stress_period_data = mflist(self, model=self.parent.mf,
                                         data=stress_period_data)

        #Add self to parent and return
        self.parent.add_package(self)
        return

    def from_package(self,package,ncomp_aux_names):
        """
        read the point source and sink info from a package
        ncomp_aux_names (list): the aux variable names in the package
        that are the component concentrations
        """
        raise NotImplementedError()

    @staticmethod
    def itype_dict():
        itype = {}
        itype["CHD"] = 1
        itype["BAS6"] = 1
        itype["PBC"] = 1
        itype["WEL"] = 2
        itype["DRN"] = 3
        itype["RIV"] = 4
        itype["GHB"] = 5
        itype["MAS"] = 15
        itype["CC"] = -1
        return itype


    @staticmethod
    def get_default_dtype(ncomp=1):
        type_list = [("k", np.int), ("i", np.int), ("j", np.int),
                     ("css", np.float32), ("itype", np.int)]
        if ncomp > 1:
            for comp in range(1,ncomp+1):
                comp_name = "cssm({0:02d})".format(comp)
                type_list.append((comp_name, np.float32))
        dtype = np.dtype(type_list)
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
            if f_ssm.closed == True:
                f_ssm = open(f_ssm.name,'a')
            # Distributed sources and sinks (Recharge and Evapotranspiration)
            if (self.crch != None):
                for c, t2d in enumerate(self.crch):
                    incrch, file_entry = t2d.get_kper_entry(kper)
                    if (c == 0):
                        f_ssm.write('%10i\n' % (incrch))
                    f_ssm.write(file_entry)

            if (self.cevt != None):
                for c, t2d in enumerate(self.cevt):
                    incevt, file_entry = t2d.get_kper_entry(kper)
                    if (c == 0):
                        f_ssm.write('%10i\n' % (incevt))
                    f_ssm.write(file_entry)
                
                '''
                if (kper < len(self.cevt)):
                    incevt = 1
                else:
                    incevt = -1
                f_ssm.write('%10i\n' % (incevt))
                if (kper < len(self.cevt)):
                    f_ssm.write(self.cevt[kper].get_file_entry())
                '''

            self.stress_period_data.write_transient(f_ssm, single_per=kper)

        f_ssm.close()
        return
        


