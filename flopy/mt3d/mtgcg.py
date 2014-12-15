from flopy.mbase import Package

class Mt3dGcg(Package):
    '''
    Generalized Conjugate Gradient solver package class\n
    '''
    def __init__(self, model, mxiter=1, iter1=50, isolve=3, ncrs=0,
                 accl=1, cclose=1e-5, iprgcg=0, extension='gcg'):
        #Call ancestor's init to set self.parent, extension, name and 
        #unit number
        Package.__init__(self, model, extension, 'GCG', 35) 
        self.mxiter = mxiter
        self.iter1 = iter1
        self.isolve = isolve
        self.ncrs = ncrs
        self.accl = accl
        self.cclose = cclose
        self.iprgcg = iprgcg
        self.parent.add_package(self)
        return
        
    def write_file(self):
        # Open file for writing
        f_gcg = open(self.fn_path, 'w')
        f_gcg.write('%10d%10d%10d%10d\n' % 
                    (self.mxiter, self.iter1, self.isolve, self.ncrs))
        f_gcg.write('%10d%10.2e%10d\n' % 
                   (self.accl, self.cclose, self.iprgcg))
        f_gcg.close()
        return
