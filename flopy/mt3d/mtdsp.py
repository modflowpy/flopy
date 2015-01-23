import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d,util_3d

class Mt3dDsp(Package):
    '''
    Dispersion package class\n
    '''
    def __init__(self, model, al=0.01, trpt=0.1, trpv=0.01, dmcoef=1e-9, 
                 extension='dsp', multiDiff=False):
        '''
        if dmcoef is passed as a list of (nlay, nrow, ncol) arrays,
        then the multicomponent diffusion is activated
        '''
        # Call ancestor's init to set self.parent, extension, name and 
        #unit number
        Package.__init__(self, model, extension, 'DSP', 33) 
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        ncomp = self.parent.get_ncomp()        
        if multiDiff:
            assert isinstance(dmcoef,list),('using multicomponent diffusion '
                                            'requires dmcoef is list of '
                                            'length ncomp')
            if len(dmcoef) != ncomp:
                raise TypeError,('using multicomponent diffusion requires '
                                 'dmcoef is list of length ncomp')
        self.multiDiff = multiDiff                                    
        #self.al = self.assignarray((nlay, nrow, ncol), np.float, al, 
        #                           name='al', load=model.load )
        self.al = util_3d(model,(nlay,nrow,ncol),np.float32,al,name='al',
                          locat=self.unit_number[0])
        #self.trpt = self.assignarray((nlay,), np.float, trpt, name='trpt', 
        #                             load=model.load)
        self.trpt = util_2d(model,(nlay,),np.float32,trpt,name='trpt',
                            locat=self.unit_number[0])
        #self.trpv = self.assignarray((nlay,), np.float, trpv, name='trpv', 
        #                             load=model.load)
        self.trpv = util_2d(model,(nlay,),np.float32,trpt,name='trpv',
                            locat=self.unit_number[0])
        if self.multiDiff:
            self.dmcoef = []
            for c in range(ncomp):                
                #self.dmcoef.append(self.assignarray( (nlay, nrow, ncol), 
                #                   np.float, dmcoef[c], 
                #                   name='dmcoef_sp_'+str(c), load=model.load))
                a = util_3d(model,(nlay,nrow,ncol),np.float32,dmcoef[c],\
                    name='dmcoef_sp_'+str(c),locat=self.unit_number[0])
                self.dmcoef.append(a)
        else:
            #self.dmcoef = self.assignarray((nlay,), np.float, dmcoef,
            #                               name='dmcoef', load=model.load)
            self.dmcoef = util_2d(model,(nlay,),np.float32,dmcoef,
                                  name='dmcoef', locat=self.unit_number[0])
        self.parent.add_package(self)
        return

    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        # Open file for writing
        f_dsp = open(self.fn_path, 'w')
        if self.multiDiff:
            f_dsp.write('$ MultiDiffusion\n')
        #self.parent.write_array(f_dsp, self.al, self.unit_number[0], True, 13,
        #                        ncol, 'Longitudinal dispersivity for Layer',
        #                        ext_base='al')
        f_dsp.write(self.al.get_file_entry())
        #self.parent.write_vector(f_dsp, self.trpt, self.unit_number[0], True, 
        #                         13, nlay, 
        #                         ('TRPT=(horizontal transverse dispersivity) /'
        #                         ' (Longitudinal dispersivity)'))
        f_dsp.write(self.trpt.get_file_entry())
        #self.parent.write_vector(f_dsp, self.trpv, self.unit_number[0], True, 
        #                         13, nlay, 
        #                         ('TRPV=(vertical transverse dispersivity) / '
        #                         '(Longitudinal dispersivity)'))
        f_dsp.write(self.trpv.get_file_entry())
        if self.multiDiff:
            for i,c in enumerate(self.dmcoef):
                #self.parent.write_array(f_dsp, c, self.unit_number[0], True, 
                #                        13, ncol, 'Effective dmcoef for comp '+
                #                        str(i+1)+' and for Layer ', 
                #                        ext_base='dmcoef')
                f_dsp.write(self.dmcoef[i].get_file_entry())
        else:
            #self.parent.write_vector(f_dsp, self.dmcoef, self.unit_number[0], 
            #                         True, 13, nlay, ('Effective molecular '
            #                         'diffusion coefficient'))
            f_dsp.write(self.dmcoef.get_file_entry())
        f_dsp.close()
        return
