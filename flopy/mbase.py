import numpy as np
from numpy.lib.recfunctions import stack_arrays
import sys
import os
import subprocess as sp
import webbrowser as wb
import warnings
from modflow.mfparbc import ModflowParBc as mfparbc


# Global variables
iconst = 1 # Multiplier for individual array elements in integer and real arrays read by MODFLOW's U2DREL, U1DREL and U2DINT.
iprn = -1 # Printout flag. If >= 0 then array values read are printed in listing file.

class BaseModel(object):
    """
    MODFLOW based models base class
    """

    def __init__(self, modelname = 'modflowtest', namefile_ext = 'nam',
                 exe_name = 'mf2k.exe', model_ws = None):
        self.__name = modelname
        self.namefile_ext = namefile_ext
        self.namefile = self.__name + '.' + self.namefile_ext
        self.packagelist = []
        self.heading = ''
        self.exe_name = exe_name
        self.external_extension = 'ref'
        if model_ws is None: model_ws = os.getcwd()
        if not os.path.exists(model_ws):
            try:
                os.makedirs(model_ws)
            except:
                #print '\n%s not valid, workspace-folder was changed to %s\n' % (model_ws, os.getcwd())
                print '\n{0:s} not valid, workspace-folder was changed to {1:s}\n'.format(model_ws, os.getcwd())
                model_ws = os.getcwd()
        self.model_ws= model_ws
        self.cl_params = ''
    
    def set_exename(self, exe_name):
        self.exe_name = exe_name
        return
        
    def add_package(self, p):
        """
        Add a flopy package object to this model.
        """
        for pp in (self.packagelist):
            if pp.allowDuplicates:
                continue
            elif (isinstance(p, type(pp))):
                print '****Warning -- two packages of the same type: ',type(p),type(pp)                 
                print 'replacing existing Package...'                
                pp = p
                return        
        self.packagelist.append( p )       
    
    def remove_package(self, pname):
        """
        Remove a package from this model.
        """
        for i,pp in enumerate(self.packagelist):  
            if pname in pp.name:               
                print 'removing Package: ',pp.name
                self.packagelist.pop(i)
                return
        raise StopIteration , 'Package name '+pname+' not found in Package list'                
            
    def build_array_name(self,num,prefix):
       return self.external_path+prefix+'_'+str(num)+'.'+self.external_extension
       
    def assign_external(self,num,prefix):           
        fname = self.build_array_name(num,prefix)
        unit = (self.next_ext_unit())
        self.external_fnames.append(fname)
        self.external_units.append(unit)       
        self.external_binflag.append(False)        
        return fname,unit
    
    def add_external(self, fname, unit, binflag=False):
        """
        Supports SWR usage and non-loaded existing external arrays
        """
        self.external_fnames.append(fname)
        self.external_units.append(unit)        
        self.external_binflag.append(binflag)
        return
    
    def remove_external(self,fname=None,unit=None):                    
        if fname is not None:
            for i,e in enumerate(self.external_fnames):
                if fname in e:
                    self.external_fnames.pop(i)
                    self.external_units.pop(i)
                    self.external_binflag.pop(i)
        elif unit is not None:
            for i,u in enumerate(self.external_units):
                if u == unit:
                    self.external_fnames.pop(i)
                    self.external_units.pop(i)
                    self.external_binflag.pop(i)
        else:
            raise Exception,' either fname or unit must be passed to remove_external()'
        return            

    def get_name_file_entries(self):
        s = ''        
        for p in self.packagelist:
            for i in range(len(p.name)):
                #s = s + ('%s %3i %s %s\n' % (p.name[i], p.unit_number[i],
                #                             p.file_name[i],p.extra[i]))
                s = s + ('{0:s} {1:3d} {2:s} {3:s}\n'.format(p.name[i], p.unit_number[i],
                                             p.file_name[i],p.extra[i]))
        return s
                
    def get_package(self, name):
        for pp in (self.packagelist):
            if (pp.name[0].upper() == name.upper()):
                return pp
        return None
   
    def get_package_list(self):
        val = []
        for pp in (self.packagelist):
            val.append(pp.name[0].upper())
        return val
    
    def change_model_ws(self, new_pth=None):
        if new_pth is None: 
            new_pth = os.getcwd()
        if not os.path.exists(new_pth):
            try:
                os.makedirs(new_pth)
            except:
                #print '\n%s not valid, workspace-folder was changed to %s\n' % (new_pth, os.getcwd())
                print '\n{0:s} not valid, workspace-folder was changed to {1:s}\n'.format(new_pth, os.getcwd())
                new_pth = os.getcwd()
        #--reset the model workspace
        self.model_ws = new_pth
        #--reset the paths for each package
        for pp in (self.packagelist):
            pp.fn_path = os.path.join(self.model_ws,pp.file_name[0])
        return None
    
    def run_model(self, silent=False, pause=False, report=False):
        """
        This method will run the model using subprocess.Popen.

        Parameters
        ----------
        silent : boolean
            Echo run information to screen (default is True).
        pause : boolean, optional
            Pause upon completion (the default is False).
        report : string, optional
            Name of file to store stdout. (default is None).

        Returns
        -------
        (success, buff)
        success : boolean
        buff : list of lines of stdout

        """
        success = False
        buff = []
        proc = sp.Popen([self.exe_name,self.namefile], 
                        stdout=sp.PIPE, cwd=self.model_ws)
        while True:
          line = proc.stdout.readline()
          if line != '':
            if 'normal termination of simulation' in line.lower():
                success = True
            #c = line.split('\r')
            c = line.rstrip('\r\n')
            if not silent:
                print c
            if report == True:
                buff.append(c)
          else:
            break
        if pause == True:
            raw_input('Press Enter to continue...')
        return ([success,buff])
        
    def write_input(self, SelPackList=False):
        if self.verbose:
            print self # Same as calling self.__repr__()
            print 'Writing packages:'
        if SelPackList == False:
            for p in self.packagelist:            
                p.write_file()
                if self.verbose:
                    print p.__repr__()        
        else:
#            for i,p in enumerate(self.packagelist):  
#                for pon in SelPackList:
            for pon in SelPackList:
                for i,p in enumerate(self.packagelist):  
                    if pon in p.name:               
                        print 'writing Package: ',p.name
                        p.write_file()
                        if self.verbose:
                            print p.__repr__()        
                        break
        #--write name file
        self.write_name_file()
    
    def write_name_file(self):
        '''Every Package needs its own writenamefile function'''
        raise Exception, 'IMPLEMENTATION ERROR: writenamefile must be overloaded'

    def get_name(self):
        return self.__name

    def set_name(self, value):
        self.__name = value
        self.namefile = self.__name + '.' + self.namefile_ext
        for p in self.packagelist:
            for i in range(len(p.extension)):
                p.file_name[i] = self.__name + '.' + p.extension[i]
    name = property(get_name, set_name)

class Package(object):
    '''
    General Package class
      allowDuplicates allows more than one package of the same class to be added.
      This is needed for mfaddoutsidefile if used for more than one file.
    '''
    def __init__(self, parent, extension='glo', name='GLOBAL', unit_number=1, extra='', 
                 allowDuplicates=False):
        self.parent = parent # To be able to access the parent modflow object's attributes
        if (not isinstance(extension, list)):
            extension = [extension]
        self.extension = []
        self.file_name = []
        for e in extension:
            self.extension = self.extension + [e]
            self.file_name = self.file_name + [self.parent.name + '.' + e]
            self.fn_path = os.path.join(self.parent.model_ws,self.file_name[0])
        if (not isinstance(name, list)):
            name = [name]
        self.name = name
        if (not isinstance(unit_number, list)):
            unit_number = [unit_number]
        self.unit_number = unit_number
        if (not isinstance(extra, list)):
            self.extra = len(self.unit_number) * [extra]
        else:
            self.extra = extra
        self.url = 'index.html'
        self.allowDuplicates = allowDuplicates

        self.acceptable_dtypes = [int,np.float32,str]

    def __repr__( self ):
        s = self.__doc__
        exclude_attributes = ['extension', 'heading', 'name', 'parent', 'url']
        for attr, value in sorted(self.__dict__.iteritems()):
            if not (attr in exclude_attributes):
                if (isinstance(value, list)):
                    if (len(value) == 1):
                        #s = s + ' %s = %s (list)\n' % (attr, str(value[0]))
                        s = s + ' {0:s} = {1:s}\n'.format(attr,str(value[0]))
                    else:
                        #s = s + ' %s (list, items = %d)\n' % (attr, len(value))
                        s = s + ' {0:s} (list, items = {1:d}\n'.format(attr,len(value))
                elif (isinstance(value, np.ndarray)):
                    #s = s + ' %s (array, shape = %s)\n' % (attr, value.shape.__str__()[1:-1] )
                    s = s + ' {0:s} (array, shape = {1:s}\n'.fomrat(attr,value.shape__str__()[1:-1])
                else:
                    #s = s + ' %s = %s (%s)\n' % (attr, str(value), str(type(value))[7:-2])
                    s = s + ' {0:s} = {1:s} ({2:s}\n'.format(attr,str(value),str(type(value))[7:-2])
        return s

    def __getitem__(self, item):
        if not isinstance(item,list) and not isinstance(item,tuple):
            assert item in self.stress_period_data.data.keys(),"package.__getitem__() kper "+str(item)+" not in data.keys()"
            return self.stress_period_data[item]

        if item[1] not in self.dtype.names:
            raise Exception ("package.__getitem(): item \'"+item+"\' not in dtype names "+str(self.dtype.names))
        assert item[0] in self.stress_period_data.data.keys(),"package.__getitem__() kper "+str(item[0])+" not in data.keys()"
        if self.stress_period_data.vtype[item[0]] == np.recarray:
            return self.stress_period_data[item[0]][item[1]]

    def __setitem__(self, key, value):
        raise NotImplementedError("package.__setitem__() not implemented")

    @staticmethod
    def add_to_dtype(dtype,field_names,field_types):
        #assert field_type in self.acceptable_dtypes,"mbase.package.add_field_to_dtype() field_type "+\
        #                                            str(type(field_type))+" not a supported type:" +\
        #                                            str(self.acceptable_dtypes)
        if not isinstance(field_names,list):
            field_names = [field_names]
        if not isinstance(field_types,list):
            field_types = [field_types] * len(field_names)
        newdtypes = [dtype]
        for field_name,field_type in zip(field_names,field_types):
            tempdtype = np.dtype([(field_name,field_type)])
            newdtypes.append(tempdtype)
        newdtype = sum((dtype.descr for dtype in newdtypes), [])
        newdtype = np.dtype(newdtype)
        return newdtype

#    def assign_layer_row_column_data(self, layer_row_column_data, ncols, zerobase=True):
#        if (layer_row_column_data is not None):
#            new_layer_row_column_data = []
#            mxact = 0
#            for a in layer_row_column_data:
#                a = np.atleast_2d(a)                
#                nr, nc = a.shape                
#                assert nc == ncols, 'layer_row_column_Q must have {0:1d} columns'.format(ncols)+'\nentry: '+str(a.shape)                
#                mxact = max(mxact, nr)
#                if zerobase:
#                    new_layer_row_column_data.append(a)
#                else:
#                    warnings.warn('Deprecation Warning: One-based indexing will be deprecated in future FloPy versions. Use Zero-based indexing')
#                    #print 'Deprecation Warning: One-based indexing will be deprecated in future FloPy versions. Use Zero-based indexing'
#                    a[:,:3] -= 1  # one-base input data, subtract 1 from layers, rows, columns
#                    new_layer_row_column_data.append(a)
#            return mxact, new_layer_row_column_data
#        return

    def webdoc(self):
        if self.parent.version == 'mf2k':
            wb.open('http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/' + self.url)
        elif self.parent.version == 'mf2005':
            wb.open('http://water.usgs.gov/nrp/gwsoftware/modflow2005/Guide/' + self.url)
        elif self.parent.version == 'ModflowNwt':
            wb.open('http://water.usgs.gov/nrp/gwsoftware/modflow_nwt/Guide/' + self.url)

    def write_file(self):
        '''Every Package needs its own write_file function'''
        print 'IMPLEMENTATION ERROR: write_file must be overloaded'


#    def write_layer_row_column_data(self, f, layer_row_column_data):
#        for n in xrange(self.parent.get_package('DIS').nper):
#            if n < len(layer_row_column_data):
#                a = layer_row_column_data[n]
#                itmp = a.shape[0]
#                #f.write('%10i%10i\n' % (itmp, self.np))
#                f.write(' {0:9d} {1:9d}       STRESS PERIOD {2:6d}\n'.format(itmp, self.np, n+1))
#                for b in a:
#                    #f.write('%9i %9i %9i' % (b[0], b[1], b[2]) )
#                    f.write(' {0:9.0f} {1:9.0f} {2:9.0f}'.format(b[0]+1, b[1]+1, b[2]+1))  # write out layer+1, row+1, col+1
#                    for c in b[3:]:
#                        #f.write(' %13.6e' % c)
#                        f.write(' {:12.6g}'.format(c))
#                    f.write('\n')
#            else:
#                itmp = -1
#                #f.write('%10i%10i\n' % (itmp, self.np))
#                f.write(' {0:9d} {1:9d}\n'.format(itmp,self.np))

    @staticmethod
    def load(model, pack_type, f, nper=None):
        """
        The load method has not been implemented for this package.

        """

        bc_pack_types = []

        if type(f) is not file:
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        #--check for parameters
        nppak = 0
        if "parameter" in line.lower():
            t = line.strip().split()
            #assert int(t[1]) == 0,"Parameters are not supported"
            nppak = np.int(t[1])
            mxl = 0
            if nppak > 0:
                mxl = np.int(t[2])
                print 'Parameters detected. Number of parameters = ', nppak
            line = f.readline()
        #dataset 2a
        t = line.strip().split()
        ipakcb = 0
        try:
            if int(t[1]) != 0:
                ipakcb = 53
        except:
            pass
        options = []
        aux_names = []
        if len(t) > 2:
            it = 2
            while it < len(t):
                toption = t[it]
                print it,t[it]
                if toption.lower() is 'noprint':
                    options.append(toption)
                elif 'aux' in toption.lower():
                    options.append(' '.join(t[it:it+2]))
                    aux_names.append(t[it+1].lower())
                    it += 1
                it += 1
        
        #--set partype
        #  and read phiramp for modflow-nwt well package
        partype = ['cond']
        if 'flopy.modflow.mfwel.modflowwel'.lower() in str(pack_type).lower():
            partype = ['flux']
            specify = False
            ipos = f.tell()
            line = f.readline()
            #--test for specify keyword if a NWT well file - This is a temporary hack
            if 'specify' in line.lower():
                specify = True
                line = f.readline() #ditch line -- possibly save for NWT output
                t = line.strip().split()
                phiramp = np.float32(t[1])
                try:
                    phiramp_unit = np.int32(t[2])
                except:
                    phiramp_unit = 2
                options.append('specify {} {} '.format(phiramp, phiramp_unit))
            else:
                f.seek(ipos)
        elif 'flopy.modflow.mfchd.modflowchd'.lower() in str(pack_type).lower():
            partype = ['shead', 'ehead']

        #--read parameter data
        if nppak > 0:
            dt = pack_type.get_empty(1, aux_names=aux_names).dtype
            pak_parms = mfparbc.load(f, nppak, dt)
            #pak_parms = mfparbc.load(f, nppak, len(dt.names))

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
        
        
        #read data for every stress period
        bnd_output = None
        stress_period_data = {}
        for iper in xrange(nper):
            print "   loading "+str(pack_type)+" for kper {0:5d}".format(iper+1)
            line = f.readline()
            if line == '':
                break
            t = line.strip().split()
            itmp = int(t[0])
            itmpp = 0
            try:
                itmpp = int(t[1])
            except:
                pass
            
            if itmp == 0:
                bnd_output = None
            elif itmp > 0:
                current = pack_type.get_empty(itmp, aux_names=aux_names)
                for ibnd in xrange(itmp):
                    line = f.readline()
                    if "open/close" in line.lower():
                        raise NotImplementedError("load() method does not support \'open/close\'")
                    t = line.strip().split()
                    current[ibnd] = tuple(t[:len(current.dtype.names)])

                #--convert indices to zero-based
                current['k'] -= 1
                current['i'] -= 1
                current['j'] -= 1
                
                bnd_output = np.copy(current)
            else:
                bnd_output = np.copy(current)

            for iparm in xrange(itmpp):
                line = f.readline()
                t = line.strip().split()
                pname = t[0].lower()
                iname = 'static'
                try:
                    tn = t[1]
                    iname = tn
                except:
                    pass
                #print pname, iname
                par_dict, current_dict = pak_parms.get(pname)
                data_dict = current_dict[iname]
                #print par_dict
                #print data_dict
                
                par_current = pack_type.get_empty(par_dict['nlst'],aux_names=aux_names)
                
                #--
                #parval = np.float(par_dict['parval'])
                if model.mfpar.pval is None:
                    parval = np.float(par_dict['parval'])
                else:
                    try:
                        parval = np.float(model.mfpar.pval.pval_dict[par_dict['parval'].lower()])
                    except:
                        parval = np.float(par_dict['parval'])

                #--fill current parameter data (par_current)
                for ibnd, t in enumerate(data_dict):
                    par_current[ibnd] = tuple(t[:len(par_current.dtype.names)])
                    
                par_current['k'] -= 1
                par_current['i'] -= 1
                par_current['j'] -= 1

                for ptype in partype:
                    par_current[ptype] *= parval
                 
                if bnd_output is None:
                    bnd_output = np.copy(par_current)
                else:
                    bnd_output = stack_arrays((bnd_output, par_current), 
                                              asrecarray=True, usemask=False)
                     
            if bnd_output is None:
                stress_period_data[iper] = itmp
                #print 'crap'
            else: 
                stress_period_data[iper] = bnd_output
                #print bnd_output.shape
                #print bnd_output   
                
        pak = pack_type(model, ipakcb=ipakcb,
                        stress_period_data=stress_period_data,\
                        dtype=pack_type.get_empty(0,aux_names=aux_names).dtype,\
                        options=options)
        return pak

