import os
import shutil
import copy
import numpy as np
from flopy.utils.binaryfile import binaryheader
#
#
#
VERBOSE = False

def decode_fortran_descriptor(fd):    
    #--strip off '(' and ')'
    fd = fd.strip()[1:-1]
    if 'FREE' in fd.upper():
        return 'free',None,None,None
    elif 'BINARY' in fd.upper():
        return 'binary',None,None,None
    if '.' in fd:
        raw = fd.split('.')
        decimal = int(raw[1])
    else:
        raw = [fd]
        decimal=None
    fmts = ['I','G','E','F']
    raw = raw[0].upper()
    for fmt in fmts:
        if fmt in raw:
            raw = raw.split(fmt)
            npl = int(raw[0])
            width = int(raw[1])
            if fmt == 'G':
                fmt = 'E'
            return npl,fmt,width,decimal
    raise Exception('Unrecognized format type: '\
        +str(fd)+' looking for: '+str(fmts))
 
def build_fortran_desciptor(npl,fmt,width,decimal):
    fd = '('+str(npl)+fmt+str(width)
    if decimal != None:
        fd += '.'+str(decimal)+')'
    else:
        fd += ')'
    return fd   

def build_python_descriptor(npl,fmt,width,decimal):
    if fmt.upper() == 'I':
        fmt = 'd'
    pd = '{0:'+str(width)
    if decimal != None:
        pd += '.'+str(decimal)+fmt+'}'
    else:
        pd += fmt+'}'
    return pd

def array2string(a, fmt_tup):
        '''Converts a 1D or 2D array into a string
        Input:
            a: array
            fmt_tup = (npl,fmt_str)
            fmt_str: format string
            npl: number of numbers per line
        Output:
            s: string representation of the array'''

        aa = np.atleast_2d(a)
        nr, nc = np.shape(aa)[0:2]
        #print 'nr = %d, nc = %d\n' % (nr, nc)
        npl = fmt_tup[0]
        fmt_str = fmt_tup[1]
        s = ''
        for r in range(nr):
            for c in range(nc):
                #s = s + (fmt_str % aa[r, c])
                #s = s + (fmt_str.format(aa[r, c]))
                #--fix for numpy 1.6 bug
                if aa.dtype == 'float32':
                    s = s + (fmt_str.format(float(aa[r, c])))
                else:
                    s = s + (fmt_str.format(aa[r, c]))
                if (((c + 1) % npl == 0) or (c == (nc - 1))):
                    s = s + '\n'
        return s

def u3d_like(model,other):
    u3d = copy.deepcopy(other)
    u3d.model = model
    for i,u2d in enumerate(u3d.util_2ds):
        u3d.util_2ds[i].model = model

    return u3d

def u2d_like(model,other):
    u2d = copy.deepcopy(other)
    u2d.model = model
    return u2d


#def util_3d(model,shape,dtype,value,name,\
#        fmtin=None,cnstnt=1.0,iprn=-1,locat=None):
#    if isinstance(value,util_3d):
#        return value
#    else:
#        u3d = util_3d(model,shape,dtype,value,name,\
#        fmtin=fmtin,cnstnt=cnstnt,iprn=iprn,locat=locat)
#        return u3d

#def util_2d(model,shape,dtype,value,name,\
#        fmtin=None,cnstnt=1.0,iprn=-1,locat=None):
#    if isinstance(value,util_2d):
#        return value
#    else:
#        u2d = util_2d(model,shape,dtype,value,name,\
#        fmtin=fmtin,cnstnt=cnstnt,iprn=iprn,locat=locat)
#        return u2d

class meta_interceptor(type):
    '''this meta class is used to catch existing instances of util_2d and util_3d 
    to prevent re-creating them
    '''
    def __call__(cls,*args,**kwds):        
        for a in args:
            if isinstance(a,util_2d) or isinstance(a,util_3d):
                return a
        return type.__call__(cls,*args,**kwds)
        

class util_3d():
    """
    util_3d class for handling 3-D model arrays

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : lenght 3 tuple
        shape of the 3-D array
    dtype : [np.int,np.float32,np.bool]
        the type of the data
    value : variable
        the data to be assigned to the 3-D array.
        can be a scalar, list, or ndarray
    name : string
        name of the property
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None)


    Attributes
    ----------
    array : np.ndarray
        the array representation of the 3-D object


    Methods
    -------
    get_file_entry() : string
        get the model input file string including the control record

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """
    __metaclass__ = meta_interceptor
    def __init__(self,model,shape,dtype,value,name,\
        fmtin=None,cnstnt=1.0,iprn=-1,locat=None):
        '''3-D wrapper from util_2d - shape must be 3-D
        '''
        assert len(shape) == 3,'util_3d:shape attribute must be length 3'
        self.model = model
        self.shape = shape
        self.dtype = dtype
        self.__value = value
        self.name_base = name+' Layer '
        self.fmtin = fmtin
        self.cnstst = cnstnt
        self.iprn = iprn
        self.locat = locat
        if model.external_path != None:
            self.ext_filename_base = os.path.join(model.external_path,self.name_base.replace(' ','_'))
        self.util_2ds = self.build_2d_instances()
   
    
    def __getitem__(self,k):
        if isinstance(k, int):
            return self.util_2ds[k]
        elif len(k) == 3:
            return self.array[k[0], k[1], k[2]]
        # if np.isscalar(k):
        #     return self.util_2ds[k]
        # else:
        #     #--if a 3-d tuple was passed
        #     if len(k) == 3:
        #         #--if the util_2d instance for layer k[0]
        #         #--is a scalar then return the value of the scalar
        #         if np.isscalar(self.util_2ds[k[0]].get_value()):
        #             val = self.util_2ds[k[0]].get_value()
        #             return val
        #         #--otherwise, get 3-d the array position value
        #         else:
        #             val = self.util_2ds[k[0]].array[k[1:]]
        #             return val
                    
    def get_file_entry(self):
        s = ''
        for u2d in self.util_2ds:
            s += u2d.get_file_entry()
        return s

    def get_value(self):
        value = []
        for u2d in self.util_2ds:
            value.append(u2d.get_value())
        return value

    @property
    def array(self):
        a = np.empty((self.shape), dtype=self.dtype)
        #for i,u2d in self.uds:
        for i,u2d in enumerate(self.util_2ds):
            a[i] = u2d.array
        return a

    def build_2d_instances(self):
        u2ds = []        
        #--if value is not enumerable, then make a list of something
        if not isinstance(self.__value,list) \
            and not isinstance(self.__value,np.ndarray):
            self.__value = [self.__value] * self.shape[0]


        #--if this is a list or 1-D array with constant values per layer
        if isinstance(self.__value,list) \
            or (isinstance(self.__value,np.ndarray) \
            and (self.__value.ndim == 1)):
            
            assert len(self.__value) == self.shape[0],\
                'length of 3d enumerable:'+str(len(self.__value))+\
                ' != to shape[0]:'+str(self.shape[0])
            
            for i,item in enumerate(self.__value):  
                if isinstance(item,util_2d):
                    u2ds.append(item)
                else:
                    name = self.name_base+str(i+1)
                    ext_filename = None
                    if self.model.external_path != None:
                        ext_filename = self.ext_filename_base+str(i+1)+'.ref'
                    u2d = util_2d(self.model,self.shape[1:],self.dtype,item,\
                        fmtin=self.fmtin,name=name,ext_filename=ext_filename,\
                        locat=self.locat)
                    u2ds.append(u2d)
                                      
        elif isinstance(self.__value,np.ndarray):
            #--if an array of shape nrow,ncol was passed, tile it out for each layer
            if self.__value.shape[0] != self.shape[0]:
                if self.__value.shape == (self.shape[1],self.shape[2]):
                    self.__value = [self.__value] * self.shape[0]
                else:
                    raise Exception('value shape[0] != to self.shape[0] and' +\
                        'value.shape[[1,2]] != self.shape[[1,2]]'+\
                        str(self.__value.shape)+' '+str(self.shape))
            for i,a in enumerate(self.__value):
                a = np.atleast_2d(a)                
                ext_filename = None
                name = self.name_base+str(i+1)
                if self.model.external_path != None:
                    ext_filename = self.ext_filename_base+str(i+1)+'.ref'
                u2d = util_2d(self.model,self.shape[1:],self.dtype,a,\
                    fmtin=self.fmtin,name=name,ext_filename=ext_filename,\
                    locat=self.locat)
                u2ds.append(u2d)
                
        else:
            raise Exception('util_array_3d: value attribute must be list '+\
               ' or ndarray, not'+str(type(self.__value)))
        return u2ds

    @staticmethod
    def load(f_handle,model,shape,dtype,name,ext_unit_dict=None):
        assert len(shape) == 3,'util_3d:shape attribute must be length 3'
        nlay,nrow,ncol = shape
        u2ds = []
        for k in range(nlay):
            u2d = util_2d.load(f_handle,model,(nrow,ncol),dtype,name,ext_unit_dict=ext_unit_dict)
            u2ds.append(u2d)
        u3d = util_3d(model,shape,dtype,u2ds,name)
        return u3d



class util_2d():
    """
    util_2d class for handling 2-D model arrays

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : lenght 3 tuple
        shape of the 3-D array
    dtype : [np.int,np.float32,np.bool]
        the type of the data
    value : variable
        the data to be assigned to the 3-D array.
        can be a scalar, list, or ndarray
    name : string
        name of the property (optional). (the default is None
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None)


    Attributes
    ----------
    array : np.ndarray
        the array representation of the 2-D object


    Methods
    -------
    get_file_entry() : string
        get the model input file string including the control record

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """
    __metaclass__ = meta_interceptor  
    def __init__(self,model,shape,dtype,value,name=None,fmtin=None,\
        cnstnt=1.0,iprn=-1,ext_filename=None,locat=None,bin=False):
        '''1d or 2-d array support with minimum of mem footprint.  
        only creates arrays as needed, 
        otherwise functions with strings or constants
        shape = 1-d or 2-d tuple
        value =  an instance of string,list,np.int,np.float32,np.bool or np.ndarray
        vtype = str,np.int,np.float32,np.bool, or np.ndarray
        dtype = np.int, or np.float32
        if ext_filename is passed, scalars are written externally as arrays
        model instance bool attribute "free_format" used for generating control record
        model instance string attribute "external_path" 
        used to determine external array writing
        bin controls writing of binary external arrays
        '''
        self.model = model
        self.shape = shape
        self.dtype = dtype
        self.bin = bool(bin)
        self.name = name
        self.locat = locat
        self.__value = self.parse_value(value)
        self.__value_built = None        
        self.cnstnt = float(cnstnt)
        self.iprn = iprn
        self.ext_filename = None
        #--just for testing
        if hasattr(model,'use_existing'):
            self.use_existing = bool(model.use_existing)
        else:
            self.use_existing = False            
        #--set fmtin
        if fmtin != None:
            self.fmtin = fmtin
        else:
            if self.bin:
                self.fmtin = '(BINARY)'
            else:
                if len(shape) == 1:
                    npl = self.shape[0]
                else:
                    npl = self.shape[1]                        
                if self.dtype == np.int:
                    self.fmtin = '('+str(npl)+'I10) '
                else:
                    self.fmtin = '('+str(npl)+'G15.6) '
                    
        #--get (npl,python_format_descriptor) from fmtin
        self.py_desc = self.fort_2_py(self.fmtin)  

        #--some defense
        if dtype not in [np.int,np.float32,np.bool]:
            raise Exception('util_2d:unsupported dtype: '+str(dtype))
        if self.model.external_path != None and name == None \
            and ext_filename == None:
            raise Exception('util_2d: use external arrays requires either '+\
               'name or ext_filename attribute')
        elif self.model.external_path != None and ext_filename == None \
            and self.vtype not in [np.int,np.float32]:
            #self.ext_filename = self.model.external_path+name+'.ref'
            self.ext_filename = os.path.join(self.model.external_path,name+'.ref')
        elif self.vtype not in [np.int,np.float32]:
            self.ext_filename = ext_filename

        if self.bin and self.ext_filename is None:
            raise Exception('util_2d: binary flag requires ext_filename')

 

    def set_fmtin(self,fmtin):
        self.fmtin = fmtin
        self.py_desc = self.fort_2_py(self.fmtin)
        return

    def get_value(self):
        return self.__value
    
    #--overloads, tries to avoid creating arrays if possible
    def __add__(self,other):
        if self.vtype in [np.int,np.float32] and self.vtype == other.vtype:
            return self.__value + other.get_value()
        else:
            return self.array + other.array

    def __sub__(self,other):
        if self.vtype in [np.int,np.float32] and self.vtype == other.vtype:
            return self.__value - other.get_value()
        else:
            return self.array - other.array

    # def __getitem__(self,k):
    #     #--this explicit cast is to handle a bug in numpy versions < 1.6.2
    #     if self.dtype == np.float32:
    #         return float(self.array[k])
    #     else:
    #         return self.array[k]

    def __getitem__(self, k):
        if isinstance(k, int):
            #--this explicit cast is to handle a bug in numpy versions < 1.6.2
            if self.dtype == np.float32:
                return float(self.array[k])
            else:
                return self.array[k]
        else:
            if isinstance(k, tuple):
                if len(k) == 2:
                    return self.array[k[0], k[1]]
                if len(k) == 1:
                    return self.array[k]
            else:
                return self.array[(k,)]

    def __setitem__(self,k,value):
        '''this one is dangerous because it resets __value
        '''
        a = self.array
        a[k] = value
        a = a.astype(self.dtype)
        self.__value = a
        if self.__value_built is not None:
            self.__value_built = None
        
    def all(self):
        return self.array.all()
    
    def __len__(self):
        return self.shape[0]

    def sum(self):
        return self.array.sum()

    @property
    def vtype(self):
        return type(self.__value)
    
    def get_file_entry(self):
        '''this is the entry point for getting an 
        input file entry for this object
        '''
        #--call get_file_array first in case we need to
       #-- get a new external unit number and reset self.locat
        vstring = self.get_file_array()
        cr = self.get_control_record()
        return cr+vstring
    

    def get_file_array(self):
        '''increments locat and update model instance if needed.
        if the value is a constant, or a string, or external, 
        return an empty string
        '''       
        #--if the value is not a filename
        if self.vtype != str:
            
            #--if the ext_filename was passed, then we need 
            #-- to write an external array
            if self.ext_filename != None:
                #--if we need fixed format, reset self.locat and get a
               #--  new unit number                 
                if not self.model.free_format:
                    self.locat = self.model.next_ext_unit() 
                    if self.bin:
                        self.locat = -1 * np.abs(self.locat)
                        self.model.add_external(self.ext_filename,\
                            self.locat,binFlag=True)
                    else:
                        self.model.add_external(self.ext_filename,self.locat)
                #--write external formatted or unformatted array    
                if not self.use_existing:    
                    if not self.bin:
                        f = open(self.ext_filename,'w',0)
                        f.write(self.string)
                        f.close()
                    else:
                        a = self.array.tofile(self.ext_filename)                    
                return ''
                
            #--this internal array or constant
            else:
                if self.vtype is np.ndarray:
                    return self.string
                #--if this is a constant, return a null string
                else:
                    return ''
        else:         
            if os.path.exists(self.__value) and self.ext_filename != None:
                #--if this is a free format model, then we can use the same
                #-- ext file over and over - no need to copy
                #--also, loosen things up with FREE format
                if self.model.free_format:
                    self.ext_filename = self.__value
                    self.fmtin = '(FREE)'
                    self.py_desc =self.fort_2_py(self.fmtin)

                else:
                    if self.__value != self.ext_filename:
                        shutil.copy2(self.__value,self.ext_filename)
                    #--if fixed format, we need to get a new unit number 
                    #-- and reset locat
                    self.locat = self.model.next_ext_unit()
                    self.model.add_external(self.ext_filename,self.locat)
                    
                return '' 
            #--otherwise, we need to load the the value filename 
            #-- and return as a string
            else:
                return self.string

    @property
    def string(self):
        '''get the string represenation of value attribute
        '''
        a = self.array
        #--convert array to sting with specified format
        a_string = array2string(a,self.py_desc)
        return a_string
                                    
    @property
    def array(self):
        '''get the array representation of value attribute
           if value is a string or a constant, the array is loaded/built only once
        '''
        if self.vtype == str:
            if self.__value_built is None:
                file_in = open(self.__value,'r')
                self.__value_built = util_2d.load_txt(self.shape,file_in,self.dtype,self.fmtin).astype(self.dtype)
                file_in.close()
            return self.__value_built
        elif self.vtype != np.ndarray:
            if self.__value_built is None:
                self.__value_built = np.ones(self.shape,dtype=self.dtype) \
                    * self.__value
            return self.__value_built
        else:
            return self.__value
    
    @staticmethod
    def load_txt(shape,file_in,dtype,fmtin):
        '''load a (possibly wrapped format) array from a file 
        (self.__value) and casts to the proper type (self.dtype)
        made static to support the load functionality 
        this routine now supports fixed format arrays where the numbers
        may touch.
        '''
        #file_in = open(self.__value,'r')
        #file_in = open(filename,'r')
        #nrow,ncol = self.shape
        nrow,ncol = shape
        npl,fmt,width,decimal = decode_fortran_descriptor(fmtin)
        #data = np.zeros((nrow*ncol),dtype=self.dtype)-1.0E+10
        data = np.zeros((nrow*ncol),dtype=dtype)-1.0E+10
        d = 0
        while True:
            line = file_in.readline()
            if line is None or d == nrow*ncol:
                break
            if npl == 'free':
                raw = line.strip('\n').split()
            else:
                #split line using number of values in the line
                rawlist = []
                istart = 0
                istop = width
                for i in xrange(npl):
                    txtval = line[istart:istop]
                    if txtval.strip() != '':
                        rawlist.append(txtval)
                    else:
                        break
                    istart = istop
                    istop += width
                raw = rawlist

            for a in raw:
                try:
                    data[d] = dtype(a)
                except:
                    raise Exception ('util_2d:unable to cast value: '\
                        +str(a)+' to type:'+str(dtype))
                if d == (nrow*ncol)-1:
                    assert len(data) == (nrow*ncol)
                    data.resize(nrow,ncol)
                    return(data) 
                d += 1	
#        file_in.close()
        data.resize(nrow,ncol)
        return data

    @staticmethod
    def write_txt(shape,file_out,data,fortran_format='(FREE)',python_format=None):
        '''
        write a (possibly wrapped format) array from a file
        (self.__value) and casts to the proper type (self.dtype)
        made static to support the load functionality
        this routine now supports fixed format arrays where the numbers
        may touch.
        '''
        nrow,ncol = shape
        if python_format == None:
            column_length,fmt,width,decimal = decode_fortran_descriptor(fortran_format)
            output_fmt = '{0}0:{1}.{2}{3}{4}'.format('{', width, decimal, fmt, '}')
        else:
            try:
                column_length, output_fmt = int(python_format[0]), python_format[1]
            except:
                raise Exception ('util_2d.write_txt: \n'
                                 +'  unable to parse python_format:\n    {0}\n'.format(python_format)
                                 +'  python_format should be a list with\n'
                                 +'   [column_length, fmt]\n'
                                 +'    e.g., [10, {0:10.2e}]')
        if ncol%column_length == 0:
            lineReturnFlag = False
        else:
            lineReturnFlag = True
        #--write the array
        for i in xrange(nrow):
            icol = 0
            for j in xrange(ncol):
                try:
                    file_out.write(output_fmt.format(data[i,j]))
                except:
                    print 'Value {0} at row,col [{1},{2}] can not be written'.format(data[i,j],i,j)
                    sys.exit()
                if (j+1)%column_length == 0.0 and j != 0:
                    file_out.write('\n')
            if lineReturnFlag == True:
                file_out.write('\n')

    @staticmethod
    def load_bin(shape,file_in,dtype,bintype=None):
        nrow,ncol = shape
        if bintype is not None:
            if dtype not in [np.int]:
                header_dtype = binaryheader.set_dtype(bintype=bintype)
            header_data = np.fromfile(file_in,dtype=header_dtype,count=1)
        else:
            header_data = None
        data = np.fromfile(file_in,dtype=dtype,count=nrow*ncol)
        data.resize(nrow,ncol)
        return [header_data, data]

    @staticmethod
    def write_bin(shape,file_out,data,bintype=None,header_data=None):
        nrow,ncol = shape
        dtype = data.dtype
        if dtype.kind != 'i':
            if bintype is not None:
                if header_data is None:
                    header_data = binaryheader.create(bintype=bintype)
            if header_data is not None:
                header_data.tofile(file_out)
        data.tofile(file_out)
        return

    def get_control_record(self):
        '''get the modflow control record
        '''      
        lay_space = '{0:>27s}'.format( '' )
        if self.model.free_format:
            
            if self.ext_filename is None:
                if self.vtype in [np.int]:
                    lay_space = '{0:>32s}'.format( '' )
                if self.vtype in [np.int,np.float32]:
                    #--this explicit cast to float is to handle a bug in versions of nummpy < l.6.2
                    if self.dtype == np.float32:                    
                        cr = 'CONSTANT '+self.py_desc[1].format(float(self.__value))
                    else:
                        cr = 'CONSTANT '+self.py_desc[1].format(self.__value)
                    cr = '{0:s}{1:s}#{2:<30s}\n'.format( cr,lay_space,self.name )
                else:
                    cr = 'INTERNAL {0:15.6G} {1:>10s} {2:2.0f} #{3:<30s}\n'\
                        .format(self.cnstnt,self.fmtin,self.iprn,self.name)
            else:
                #--need to check if ext_filename exists, if not, need to 
                #-- write constant as array to file or array to file               
                cr = 'OPEN/CLOSE  {0:>30s} {1:15.6G} {2:>10s} {3:2.0f}  #{4:<30s}\n'\
                    .format(self.ext_filename,self.cnstnt,\
                    self.fmtin.strip(),self.iprn,self.name)
        else:                       
            #--if value is a scalar and we don't want external array
            if self.vtype in [np.int,np.float32] and self.ext_filename is None:
                locat = 0
                #--explicit cast for numpy bug in versions < 1.6.2
                if self.dtype == np.float32:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:10.0f} #{4}\n'\
                        .format(locat,float(self.__value),self.fmtin,self.iprn,self.name)
                else:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:10.0f} #{4}\n'\
                        .format(locat,self.__value,self.fmtin,self.iprn,self.name)
            else:
                if self.ext_filename is None:
                    assert self.locat != None,'util_2d:a non-constant value '+\
                       ' for an internal fixed-format requires LOCAT to be passed'                
                if self.dtype == np.int:
                    cr = '{0:>10.0f}{1:>10.0f}{2:>20s}{3:>10.0f} #{4}\n'\
                        .format(self.locat,self.cnstnt,self.fmtin,self.iprn,self.name)
                elif self.dtype == np.float32:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:>10.0f} #{4}\n'\
                        .format(self.locat,self.cnstnt,self.fmtin,self.iprn,self.name)
                else:
                    raise Exception('util_2d: error generating fixed-format '+\
                       ' control record,dtype must be np.int or np.float32')
        return cr                                 


    def fort_2_py(self,fd):
        '''converts the fortran format descriptor 
        into a tuple of npl and a python format specifier

        '''
        npl,fmt,width,decimal = decode_fortran_descriptor(fd)
        if npl == 'free':
            if self.vtype == np.int:
                return (self.shape[1],'{0:10.0f} ')
            else:
                return (self.shape[1],'{0:15.6G} ')
        elif npl == 'binary':
            return('binary',None)
        else:
            pd = build_python_descriptor(npl,fmt,width,decimal)
            return (npl,pd)    


    def parse_value(self,value):
        '''parses and casts the raw value into an acceptable format for __value
        lot of defense here, so we can make assumptions later
        '''
        if isinstance(value,list):
            if VERBOSE:
                print 'util_2d: casting list to array'
            value = np.array(value)
        if isinstance(value,bool):
            if self.dtype == np.bool:
                try:
                    value = np.bool(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast '+\
                        'boolean value to type "np.bool": '+str(value))
            else:
                raise Exeception('util_2d:value type is bool, '+\
                   ' but dtype not set as np.bool') 
        if isinstance(value,str):
            if self.dtype == np.int:
                try:
                    value = int(value)
                except:
                    value = os.path.abspath(os.path.join(self.model.model_ws, value))
                    assert os.path.exists(value),'could not find file: '+str(value)
                    return value
            else:
                try:
                    value = float(value)
                except:
                    value = os.path.abspath(os.path.join(self.model.model_ws, value))
                    assert os.path.exists(value),'could not find file: '+str(value)
                    return value
        if np.isscalar(value):
            if self.dtype == np.int:
                try:
                    value = np.int(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast scalar '+\
                        'value to type "int": '+str(value))
            elif self.dtype == np.float32:
                try:
                    value = np.float32(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast '+\
                        'scalar value to type "float": '+str(value))
            
        if isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise Exception('util_2d:self.shape: '+str(self.shape)+\
                    ' does not match value.shape: '+str(value.shape))
            if self.dtype != value.dtype:
                if VERBOSE:
                    print 'util_2d:warning - casting array of type: '+\
                    str(value.dtype)+' to type: '+str(self.dtype)
            return value.astype(self.dtype)
        
        else:
            raise Exception('util_2d:unsupported type in util_array: '\
                +str(type(value))) 


    @staticmethod
    def load(f_handle,model,shape,dtype,name,ext_unit_dict=None):
        '''functionality to load util_2d instance from an existing
        model input file.
        external and internal record types must be fully loaded
        if you are using fixed format record types,make sure 
        ext_unit_dict has been initialized from the NAM file
        '''     

        curr_unit = None
        if ext_unit_dict is not None:
            # determine the current file's unit number
            cfile = f_handle.name
            for cunit in ext_unit_dict:
                if cfile == ext_unit_dict[cunit].filename:
                    curr_unit = cunit
                    break

        cr_dict = util_2d.parse_control_record(f_handle.readline(), current_unit=curr_unit, dtype=dtype)
        
        if cr_dict['type'] == 'constant':
            u2d = util_2d(model,shape,dtype,cr_dict['cnstnt'],name=name,\
                iprn=cr_dict['iprn'],fmtin=cr_dict['fmtin'])
        
        elif cr_dict['type'] == 'open/close':
            #--clean up the filename a little
            fname = cr_dict['fname']
            fname = fname.replace('\'','')
            fname = fname.replace('\"','')
            fname = fname.replace('\\', os.path.sep)
            u2d = util_2d(model,shape,dtype,fname,name=name,\
                iprn=cr_dict['iprn'],fmtin=cr_dict['fmtin'],\
                ext_filename=fname)
        elif cr_dict['type'] == 'internal':
            data = util_2d.load_txt(shape,f_handle,dtype,cr_dict['fmtin'])
            u2d = util_2d(model,shape,dtype,data,name=name,\
                iprn=cr_dict['iprn'],fmtin=cr_dict['fmtin'])
        elif cr_dict['type'] == 'external':
            assert cr_dict['nunit'] in ext_unit_dict.keys()
            if 'binary' not in cr_dict['fmtin'].lower():
                data = util_2d.load_txt(shape,ext_unit_dict[cr_dict['nunit']].filehandle,dtype,cr_dict['fmtin'])
            else:
                header_data, data = util_2d.load_bin(shape,ext_unit_dict[cr_dict['nunit']].filehandle,dtype)
            u2d = util_2d(model,shape,dtype,data,name=name,\
                 iprn=cr_dict['iprn'],fmtin=cr_dict['fmtin'])            
        return u2d             
            

    @staticmethod
    def parse_control_record(line, current_unit=None, dtype=np.float32):
        '''parses a control record when reading an existing file
        rectifies fixed to free format
        current_unit (optional) indicates the unit number of the file being parsed
        '''
        free_fmt = ['open/close','internal','external','constant']
        raw = line.lower().strip().split()
        freefmt,cnstnt,fmtin,iprn,nunit = None,None,None,-1,None
        fname = None 
        isFloat = False
        if dtype == np.float or dtype==np.float32:
            isFloat = True       
        #--if free format keywords
        if raw[0] in free_fmt:
            freefmt = raw[0]
            if raw[0] == 'constant':
                if isFloat:                
                    cnstnt = np.float(raw[1].lower().replace('d', 'e'))
                else:
                    cnstnt = np.int(raw[1].lower())                   
            if raw[0] == 'internal':
                fmtin = raw[2].strip()
                iprn = int(raw[3])
            elif raw[0] == 'external':
                nunit = int(raw[1])
                if isFloat:                
                    cnstnt = np.float(raw[2].lower().replace('d', 'e'))
                else:
                    cnstnt = np.int(raw[2].lower())                   
                fmtin = raw[3].strip()
                iprn = int(raw[4])
            elif raw[0] == 'open/close':
                fname = raw[1].strip()
                if isFloat:                
                    cnstnt = np.float(raw[2].lower().replace('d', 'e'))
                else:
                    cnstnt = np.int(raw[2].lower())                   
                fmtin = raw[3].strip()
                iprn = int(raw[4])
                npl,fmt,width,decimal = None,None,None,None
        else:
            locat = np.int(line[0:10].strip())
            if isFloat:                
                cnstnt = np.float(line[10:20].strip().lower().replace('d', 'e'))
            else:
                cnstnt = np.int(line[10:20].strip())
            if locat != 0:
                fmtin = line[20:40].strip()
                iprn = np.int(line[40:50].strip())
            #locat = int(raw[0])        
            #cnstnt = float(raw[1])
            #fmtin = raw[2].strip()
            #iprn = int(raw[3])

            if locat == 0:
                freefmt = 'constant'
            elif locat < 0:
                freefmt = 'external'
                nunit = np.int(locat) * -1    
                fmtin = '(binary)'
            elif locat > 0:
                # if the unit number matches the current file, it's internal
                if locat == current_unit:
                    freefmt = 'internal'
                else:
                    freefmt = 'external'
                nunit = np.int(locat)                                                    
        cr_dict = {}                                                 
        cr_dict['type'] = freefmt
        cr_dict['cnstnt'] = cnstnt
        cr_dict['nunit'] = nunit        
        cr_dict['iprn'] = iprn
        cr_dict['fmtin'] = fmtin
        cr_dict['fname'] = fname           
        return cr_dict