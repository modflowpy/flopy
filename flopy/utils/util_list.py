import os
import warnings
import numpy as np
import flopy



class mflist(object):

    def __init__(self,model,dtype,data=None):
        assert isinstance(model,flopy.mbase.BaseModel),"mflist error: model type incorrect:"+str(type(model))
        self.model = model
        assert isinstance(dtype,np.dtype)
        self.__dtype = dtype
        self.__vtype = {}
        self.__data = {}
        if data is not None:
            self.__cast_data(data)


    @property
    def data(self):
        return self.__data


    @property
    def vtype(self):
        return self.__vtype


    @property
    def dtype(self):
        return self.__dtype

    @property
    def mxact(self):
        mxact = -1.0E+10
        for kper,data in self.__data.iteritems():
            if self.vtype[kper] == str:
                self.__data[kper] = self.__fromfile(data)
                self.__vtype[kper] = np.recarray
            if self.vtype[kper] == np.recarray:
                mxact = max(mxact,self.data[kper].shape[0])
        return mxact


    @property
    def fmt_string(self):
        fmt_string = ''
        for field in self.dtype.descr:
            vtype = field[1][1]
            if vtype == 'i':
                fmt_string += ' %9d'
            elif vtype == 'f':
                fmt_string += ' %9f'
            elif vtype == 's':
                fmt_string += '%s'
            else:
                raise Exception("mflist.fmt_string error: unknown vtype in dtype:"+vtype)
        return fmt_string


    def __cast_data(self,data):
        if isinstance(data,list):
            #print "mflist warning: casting list to array at kper {0:d}".format(kper)
            warnings.warn("mflist casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("mflist error: casting list to ndarray: "+str(e))

        if isinstance(data,dict):
            for kper,d in data.iteritems():
                assert isinstance(kper,int), "mflist error: data dict key \'{0:s}\' "+\
                        "not integer: ".format(kper)+str(type(kper))
                if isinstance(d,list):
                    #print "mflist warning: casting list to array at kper {0:d}".format(kper)
                    warnings.warn("mflist: casting list to array at kper {0:d}".format(kper))
                    try:
                        d = np.array(d)
                    except Exception as e:
                        raise Exception("mflist error: casting list to ndarray")

                if isinstance(d,np.recarray):
                    self.__cast_recarray(kper,d)
                #cast ndarray to recarray
                elif isinstance(d,np.ndarray):
                    self.__cast_ndarray(kper,d)
                elif isinstance(d,int):
                    self.__cast_int(kper,d)
                elif isinstance(d,str):
                    self.__cast_str(kper,d)
                else:
                    raise Exception("mflist error: unsupported data type: "\
                                    +str(type(d))+" at kper {0:d}".format(kper))

        #a single recarray - same mflist for all stress periods
        elif isinstance(data,np.recarray):
            self.__cast_recarray(0,data)
        elif isinstance(data,np.ndarray):
            self.__cast_ndarray(0,data)
        #a single filename
        elif isinstance(data,str):
            self.__cast_str(0,data)
        else:
            raise Exception("mflist error: unsupported data type: "+str(type(data)))


    def __cast_str(self,kper,d):
        assert os.path.exists(d),"mflist error: dict filename (string) \'"+d+"\'"+\
                                    "value for kper {0:d} not found".format(kper)
        self.__data[kper] = d
        self.__vtype[kper] = str


    def __cast_int(self,kper,d):
        if d > 0:
            raise Exception("mflist error: dict integer value for "+\
                    "kper {0:10d} must be 0 or -1, not {1:10d}".format(kper,d))
        if d == 0:
            #fill the previous stress periods with 0
            if kper != 0 and len(self.__data.keys()) == 0:
                for kp in xrange(kper):
                    self.__data[kp] = 0
                    self.__vtype[kp] = None
            self.__data[kper] = 0
            self.__vtype[kper] = None
        else:
            if kper == 0:
                raise Exception("mflist error: dict integer value for "+\
                                "kper 0 for cannot be negative")
            self.__data[kper] = -1
            self.__vtype[kper]= None


    def __cast_recarray(self,kper,d):
        assert d.dtype == self.__dtype,"mflist error: recarray dtype: "+\
                                   str(d.dtype)+" doesn't match self dtype: "+\
                                   str(self.dtype)
        self.__data[kper] = d
        self.__vtype[kper] = np.recarray


    def __cast_ndarray(self,kper,d):
        d = np.atleast_2d(d)
        if d.dtype != self.__dtype:

            assert d.shape[1] == len(self.dtype),"mflist error: ndarray shape "+\
                                                       str(d.shape)+" doesn't match dtype len: "+\
                                                       str(len(self.dtype))
            warnings.warn("mflist: ndarray dtype does not match self dtype, trying to cast")
        try:
            self.__data[kper] = np.core.records.fromarrays(d.transpose(),dtype=self.dtype)
        except Exception as e:
            raise Exception("mflist error: casting ndarray to recarray: "+str(e))
        self.__vtype[kper] = np.recarray


    def add_record(self,kper,index,values):
        assert len(index) + len(values) == len(self.dtype),"mflist.add_record() error: length of index arg +"\
                "length of value arg != length of self dtype"

        if kper in self.__data.keys():
            #if a 0 or -1, reset
            if self.vtype[kper] == int:
                self.__data[kper] = self.get_empty(1)
                self.__vtype[kper] = np.recarray
            #if filename, load into recarray
            if self.vtype[kper] == str:
                d = self.__fromfile(self.data[kper])
                d.resize(d.shape[0]+1,d.shape[1])
                self.__data[kper] = d
                self.__vtype[kper] = np.recarray
            #extend the recarray
            if self.vtype[kper] == np.recarray:
                shape = self.__data[kper].shape
                self.__data[kper].resize(shape[0]+1,shape[1])
        else:
            self.__data[kper] = self.get_empty(1)
            self.__vtype[kper] = np.recarray
        rec = list(index)
        rec.extend(list(values))
        try:
            self.__data[kper][-1] = tuple(rec)
        except Exception as e:
            raise Exception("mflist.add_record() error: adding record to recarray: "+str(e))


    def get_empty(self,nrow=0):
        d = np.zeros((nrow,len(self.dtype)),dtype=self.dtype)
        d[:,:] = -1.0E+10
        return d


    def __fromfile(self,f,count=-1):
        d = np.fromfile(f,dtype=self.dtype,count=count)
        if d.shape[0] == 0 or d.shape[1] != len(self.dtype):
            raise Exception("mflist.__fromfile() error: reading list from file: ")
        return d

    def write_transient(self,f):
        nl,nr,nc,nper = self.model.get_nrow_ncol_nlay_nper()
        nper = 10
        assert isinstance(f,file),"mflist.write() error: f argument must be a file handle"
        kpers = self.data.keys()
        assert 0 in kpers,"mflist.write() error: kper 0 not defined"

        for kper in range(0,nper):
            if kper in kpers:
                kper_data = self.__data[kper]
                kper_vtype = self.__vtype[kper]
                if self.vtype == str:
                    kper_data = self.__fromfile(kper_data)
                    kper_vtype = np.recarray
                if kper_vtype == np.recarray:
                    itmp = kper_data.shape[0]
                else:
                    itmp = kper_data
            else:
                itmp = -1
                kper_vtype = int
            f.write(" {0:9d} {1:9d} # stress period {2:d}\n".format(itmp,0,kper))
            if kper_vtype == np.recarray:
                if True:
                    self.check_ijk()
                np.savetxt(f,kper_data,fmt=self.fmt_string,delimiter='')
