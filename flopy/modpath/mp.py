
from ..mbase import BaseModel
from ..pakbase import Package
from .mpsim import ModpathSim
from .mpbas import ModpathBas
import os

class ModpathList(Package):
    '''
    List package class
    '''
    def __init__(self, model, extension='list', listunit=7):
        """
        Package constructor.

        """
        #Call ancestor's init to set self.parent, extension, name and
        #unit number
        Package.__init__(self, model, extension, 'LIST', listunit)
        #self.parent.add_package(self) This package is not added to the base 
        #model so that it is not included in get_name_file_entries()
        return

    def write_file(self):
        # Not implemented for list class
        return


class Modpath(BaseModel):
    """
    Modpath base class

    """
    def __init__(self, modelname='modpathtest', simfile_ext='mpsim', namefile_ext='mpnam',
                 version='modpath', exe_name='mp6.exe', modflowmodel=None,
                 dis_file = None, dis_unit=87, head_file = None, budget_file = None, 
                 model_ws=None, external_path=None, verbose=False,
                 load=True, listunit=7):
        """
        Model constructor.

        """
        BaseModel.__init__(self, modelname, simfile_ext, exe_name, model_ws=model_ws)

        self.version_types = {'modpath': 'MODPATH'}
        self.set_version(version)

        self.__mf = modflowmodel
        self.lst = ModpathList(self, listunit=listunit)
        self.mpnamefile = '{}.{}'.format(self.name, namefile_ext)
        self.mpbas_file = '{}.mpbas'.format(modelname)
        self.dis_file = dis_file
        self.dis_unit = dis_unit
        self.head_file = head_file
        self.budget_file = budget_file
        self.__sim = None
        self.free_format = False
        self.array_format = 'modflow'
        self.external_path = external_path
        self.external = False
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.load = load
        self.__next_ext_unit = 500
        if external_path is not None:
            assert os.path.exists(external_path),'external_path does not exist'
            self.external = True         
        self.verbose = verbose            
        return

    def __repr__( self ):
        return 'Modpath model'

    # function to encapsulate next_ext_unit attribute
    def next_ext_unit(self):
        self.__next_ext_unit += 1
        return self.__next_ext_unit

    def getsim(self):
        if (self.__sim == None):
            for p in (self.packagelist):
                if isinstance(p, ModpathSim):
                    self.__sim = p
        return self.__sim

    def getmf(self):
        return self.__mf

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        fn_path = os.path.join(self.model_ws, self.mpnamefile)
        f_nam = open( fn_path, 'w' )
        f_nam.write('%s\n' % (self.heading) )
        if self.mpbas_file is not None:
            f_nam.write('%s %3i %s\n' % ('MPBAS', 86, self.mpbas_file))        
        if self.dis_file is not None:
            f_nam.write('%s %3i %s\n' % ('DIS', self.dis_unit, self.dis_file))        
        if self.head_file is not None:
            f_nam.write('%s %3i %s\n' % ('HEAD', 88, self.head_file))        
        if self.budget_file is not None:
            f_nam.write('%s %3i %s\n' % ('BUDGET', 89, self.budget_file))
        for u,f in zip(self.external_units,self.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u)+f+'\n'	)
        f_nam.close()

    sim = property(getsim) # Property has no setter, so read-only
    mf = property(getmf) # Property has no setter, so read-only


