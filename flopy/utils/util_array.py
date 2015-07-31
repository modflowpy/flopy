"""
util_array module.  Contains the util_2d, util_3d and transient_2d classes.
 These classes encapsulate modflow-style array inputs away
 from the individual packages.  The end-user should not need to
 instantiate these classes directly.

"""
from __future__ import division, print_function
#from future.utils import with_metaclass

import os
import shutil
import copy
import numpy as np
import flopy.utils
VERBOSE = False



def decode_fortran_descriptor(fd):    
    # strip off any quotes around format string
    fd = fd.replace("'", "")
    fd = fd.replace('"', '')
    # strip off '(' and ')'
    fd = fd.strip()[1:-1]
    if str('FREE') in str(fd.upper()):
        return 'free', None, None, None
    elif str('BINARY') in str(fd.upper()):
        return 'binary', None, None, None
    if str('.') in str(fd):
        raw = fd.split('.')
        decimal = int(raw[1])
    else:
        raw = [fd]
        decimal=None
    fmts = ['I', 'G', 'E', 'F']
    raw = raw[0].upper()
    for fmt in fmts:
        if fmt in raw:
            raw = raw.split(fmt)
            # '(F9.0)' will return raw = ['', '9']
            #  try and except will catch this
            try:
                npl = int(raw[0])
                width = int(raw[1])
            except:
                npl = 1
                width = int(raw[1])
            if fmt == 'G':
                fmt = 'E'
            return npl, fmt, width, decimal
    raise Exception('Unrecognized format type: ' +
                    str(fd) + ' looking for: ' + str(fmts))
 
def build_fortran_desciptor(npl, fmt, width, decimal):
    fd = '(' + str(npl) + fmt + str(width)
    if decimal is not None:
        fd += '.' + str(decimal) + ')'
    else:
        fd += ')'
    return fd   

def build_python_descriptor(npl, fmt, width, decimal):
    if fmt.upper() == 'I':
        fmt = 'd'
    pd = '{0:' + str(width)
    if decimal is not None:
        pd += '.' + str(decimal) + fmt+'}'
    else:
        pd += fmt+'}'
    return pd

def read1d(f, a):
    """
    Fill the 1d array, a, with the correct number of values.  Required in
    case lpf 1d arrays (chani, layvka, etc) extend over more than one line

    """
    values = []
    while True:
        line = f.readline()
        t = line.strip().split()
        values = values + t
        if len(values) >= a.shape[0]:
            break
    a[:] = np.array(values[0:a.shape[0]], dtype=a.dtype)
    return a

def array2string(a, fmt_tup):
        """
        Converts a 1D or 2D array into a string
        Input:
            a: array
            fmt_tup = (npl,fmt_str)
            fmt_str: format string
            npl: number of numbers per line
        Output:
            s: string representation of the array
        """

        aa = np.atleast_2d(a)
        nr, nc = np.shape(aa)[0:2]
        #print 'nr = %d, nc = %d\n' % (nr, nc)
        npl = fmt_tup[0]
        fmt_str = fmt_tup[1]
        s = ''
        for r in range(nr):
            for c in range(nc):
                # fix for numpy 1.6 bug
                if aa.dtype == 'float32':
                    s = s + (fmt_str.format(float(aa[r, c])))
                else:
                    s = s + (fmt_str.format(aa[r, c]))
                if ((c + 1) % npl == 0) or (c == (nc - 1)):
                    s += '\n'
        return s

def cast2dict(value):
    """
    Converts scalar or list to dictionary keyed on kper that
    can be used with the transient_2d class.
    """
    # If already a dictionary return
    if (isinstance(value, dict)):
        return value

    # If value is a scalar or a single numpy array convert to a list
    if (np.isscalar(value) or isinstance(value, np.ndarray)):
        value = [value]

    rv = {}
    # Convert list to dictionary
    if (isinstance(value, list)):
        for kper, a in enumerate(value):
            rv[kper] = a
    else:
        raise Exception("cast2dict error: value type not " +
                        " recognized: " + str(type(value)))

    return rv

def u3d_like(model, other):
    u3d = copy.deepcopy(other)
    u3d.model = model
    for i, u2d in enumerate(u3d.util_2ds):
        u3d.util_2ds[i].model = model

    return u3d

def u2d_like(model, other):
    u2d = copy.deepcopy(other)
    u2d.model = model
    return u2d


# class meta_interceptor(type):
#     """
#     meta class to catch existing instances of util_2d,
#     transient_2d and util_3d to prevent re-instantiating them.
#     a lot of python trickery here...
#     """
#     def __call__(cls, *args, **kwds):
#         for a in args:
#             if isinstance(a, util_2d) or isinstance(a, util_3d):
#                 return a
#         return type.__call__(cls, *args, **kwds)


#class util_3d((with_metaclass(meta_interceptor, object))):
class util_3d(object):
    """
    util_3d class for handling 3-D model arrays.  just a thin wrapper around
        util_2d

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : length 3 tuple
        shape of the 3-D array, typically (nlay,nrow,ncol)
    dtype : [np.int,np.float32,np.bool]
        the type of the data
    value : variable
        the data to be assigned to the 3-D array.
        can be a scalar, list, or ndarray
    name : string
        name of the property, used for writing comments to input files
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
        to read the array from
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename, the
        ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the defaut is False)

    Attributes
    ----------
    array : np.ndarray
        the array representation of the 3-D object


    Methods
    -------
    get_file_entry : string
        get the model input file string including the control record for the
        entire 3-D property

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, model, shape, dtype, value, name,
        fmtin=None, cnstnt=1.0, iprn=-1, locat=None, ext_unit_dict=None):
        """
        3-D wrapper from util_2d - shape must be 3-D
        """
        if isinstance(value,util_3d):
            for attr in value.__dict__.items():
                setattr(self,attr[0],attr[1])
            return
        assert len(shape) == 3, 'util_3d:shape attribute must be length 3'
        self.model = model
        self.shape = shape
        self.dtype = dtype
        self.__value = value
        self.name = name
        self.name_base = name + ' Layer '
        self.fmtin = fmtin
        self.cnstnt = cnstnt
        self.iprn = iprn
        self.locat = locat
        if model.external_path is not None:
            self.ext_filename_base = os.path.join(
                model.external_path, self.name_base.replace(' ', '_'))
        self.util_2ds = self.build_2d_instances()


    def to_shapefile(self, filename):
        """
        Export 3-D model data to shapefile (polygons).  Adds an
            attribute for each util_2d in self.u2ds

        Parameters
        ----------
        filename : str
            Shapefile name to write

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.lpf.hk.to_shapefile('test_hk.shp')
        """

        from flopy.utils.flopy_io import write_grid_shapefile
        array_dict = {}
        for ilay in range(self.model.nlay):
            u2d = self[ilay]
            array_dict[u2d.name] = u2d.array
        write_grid_shapefile(filename, self.model.dis.sr,
                             array_dict)


    def plot(self, filename_base=None, file_extension=None, mflay=None,
             fignum=None, **kwargs):
        """
        Plot 3-D model input data

        Parameters
        ----------
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        mflay : int
            MODFLOW zero-based layer number to return.  If None, then all
            all layers will be included. (default is None)
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        ----------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.lpf.hk.plot()
        
        """
        import flopy.plot.plotutil as pu
        

        if file_extension is not None:
            fext = file_extension
        else:
            fext = 'png'
        
        names = ['{} layer {}'.format(self.name, k+1) for k in range(self.shape[0])]
        
        filenames = None
        if filename_base is not None:
            if mflay is not None:
                i0 = int(mflay)
                if i0+1 >= self.shape[0]:
                    i0 = self.shape[0] - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = self.shape[0]
            # build filenames
            filenames = ['{}_{}_Layer{}.{}'.format(filename_base, self.name,
                                                   k+1, fext) for k in range(i0, i1)]

        return pu._plot_array_helper(self.array, self.model,
                                     names=names, filenames=filenames, 
                                     mflay=mflay, fignum=fignum, **kwargs)


    def __getitem__(self, k):
        if isinstance(k, int):
            return self.util_2ds[k]
        elif len(k) == 3:
            return self.array[k[0], k[1], k[2]]
                    
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
        # if value is not enumerable, then make a list of something
        if not isinstance(self.__value, list) \
            and not isinstance(self.__value, np.ndarray):
            self.__value = [self.__value] * self.shape[0]

        # if this is a list or 1-D array with constant values per layer
        if isinstance(self.__value, list) \
            or (isinstance(self.__value, np.ndarray)
                and (self.__value.ndim == 1)):
            
            assert len(self.__value) == self.shape[0],\
                'length of 3d enumerable:' + str(len(self.__value)) +\
                ' != to shape[0]:' + str(self.shape[0])
            
            for i,item in enumerate(self.__value):  
                if isinstance(item, util_2d):
                    u2ds.append(item)
                else:
                    name = self.name_base + str(i + 1)
                    ext_filename = None
                    if self.model.external_path is not None:
                        ext_filename = self.ext_filename_base+str(i + 1) +\
                                    '.ref'
                    u2d = util_2d(self.model, self.shape[1:], self.dtype, item,
                        fmtin=self.fmtin, name=name, ext_filename=ext_filename,
                        locat=self.locat)
                    u2ds.append(u2d)
                                      
        elif isinstance(self.__value, np.ndarray):
            # if an array of shape nrow,ncol was passed, tile it out for each layer
            if self.__value.shape[0] != self.shape[0]:
                if self.__value.shape == (self.shape[1], self.shape[2]):
                    self.__value = [self.__value] * self.shape[0]
                else:
                    raise Exception('value shape[0] != to self.shape[0] and' +
                        'value.shape[[1,2]] != self.shape[[1,2]]' +
                        str(self.__value.shape) + ' '+str(self.shape))
            for i, a in enumerate(self.__value):
                a = np.atleast_2d(a)                
                ext_filename = None
                name = self.name_base+str(i + 1)
                if self.model.external_path is not None:
                    ext_filename = self.ext_filename_base+str(i+1) + '.ref'
                u2d = util_2d(self.model, self.shape[1:], self.dtype, a,
                    fmtin=self.fmtin, name=name, ext_filename=ext_filename,
                    locat=self.locat)
                u2ds.append(u2d)
                
        else:
            raise Exception('util_array_3d: value attribute must be list ' +
                            ' or ndarray, not' + str(type(self.__value)))
        return u2ds

    @staticmethod
    def load(f_handle, model, shape, dtype, name, ext_unit_dict=None):
        assert len(shape) == 3, 'util_3d:shape attribute must be length 3'
        nlay, nrow, ncol = shape
        u2ds = []
        for k in range(nlay):
            u2d = util_2d.load(f_handle, model, (nrow, ncol), dtype, name,
                               ext_unit_dict=ext_unit_dict)
            u2ds.append(u2d)
        u3d = util_3d(model, shape, dtype, u2ds, name)
        return u3d


    def __mul__(self, other):
        if np.isscalar(other):
            new_u2ds = []
            for u2d in self.util_2ds:
                new_u2ds.append(u2d * other)
            return util_3d(self.model,self.shape,self.dtype,new_u2ds,
                           self.name,self.fmtin,self.cnstnt,self.iprn,
                           self.locat)
        elif isinstance(other,list):
            assert len(other) == self.shape[0]
            new_u2ds = []
            for i in range(self.shape[0]):
                new_u2ds.append(u2d * other)
            return util_3d(self.model,self.shape,self.dtype,new_u2ds,
                           self.name,self.fmtin,self.cnstnt,self.iprn,
                           self.locat)

#class transient_2d((with_metaclass(meta_interceptor, object))):
class transient_2d(object):
    """
    transient_2d class for handling time-dependent 2-D model arrays.
    just a thin wrapper around util_2d

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : length 2 tuple
        shape of the 2-D transient arrays, typically (nrow,ncol)
    dtype : [np.int,np.float32,np.bool]
        the type of the data
    value : variable
        the data to be assigned to the 2-D arrays. Typically a dict
        of {kper:value}, where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to util_2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
    name : string
        name of the property, used for writing comments to input files
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
         to read the array from
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename,
        the ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the default is False)

    Attributes
    ----------
    transient_2ds : dict{kper:util_2d}
        the transient sequence of util_2d objects

    Methods
    -------
    get_kper_entry : (itmp,string)
        get the itmp value and the util_2d file entry of the value in
        transient_2ds in bin kper.  if kper < min(transient_2d.keys()),
        return (1,zero_entry<util_2d>).  If kper > < min(transient_2d.keys()),
        but is not found in transient_2d.keys(), return (-1,'')

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, model, shape, dtype, value, name=None, fmtin=None,
        cnstnt=1.0, iprn=-1, ext_filename=None, locat=None, bin=False):

        if isinstance(value,transient_2d):
            for attr in value.__dict__.items():
                setattr(self,attr[0],attr[1])
            return

        self.model = model
        assert len(shape) == 2, "transient_2d error: shape arg must be " +\
                                "length two (nrow, ncol), not " +\
                                str(shape)
        self.shape = shape
        self.dtype = dtype
        self.__value = value
        self.name_base = name
        self.fmtin = fmtin
        self.cnstst = cnstnt
        self.iprn = iprn
        self.locat = locat
        if model.external_path != None:
            self.ext_filename_base = \
                os.path.join(model.external_path,
                             self.name_base.replace(' ', '_'))
        self.transient_2ds = self.build_transient_sequence()

    def get_zero_2d(self, kper):
        name = self.name_base + str(kper + 1) + '(filled zero)'
        return util_2d(self.model, self.shape,
                       self.dtype, 0.0, name=name).get_file_entry()


    def to_shapefile(self, filename):
        """
        Export transient 2D data to a shapefile (as polygons). Adds an 
            attribute for each unique util_2d instance in self.data

        Parameters
        ----------
        filename : str
            Shapefile name to write

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.rch.rech.as_shapefile('test_rech.shp')
        """
        from flopy.utils.flopy_io import write_grid_shapefile
        array_dict = {}
        for kper in range(self.model.nper):
            u2d = self[kper]
            array_dict[u2d.name] = u2d.array
        write_grid_shapefile(filename, self.model.dis.sr, array_dict)


    def plot(self, filename_base=None, file_extension=None, **kwargs):
        """
        Plot transient 2-D model input data

        Parameters
        ----------
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.
            kper : str
                MODFLOW zero-based stress period number to return. If
                kper='all' then data for all stress period will be
                extracted. (default is zero).

        Returns
        ----------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.rch.rech.plot()
        
        """
        import flopy.plot.plotutil as pu
        
        if file_extension is not None:
            fext = file_extension
        else:
            fext = 'png'
        
        if 'kper' in kwargs:
            kk = kwargs['kper']
            kwargs.pop('kper')
            try:
                kk = kk.lower()
                if kk == 'all':
                    k0 = 0
                    k1 = self.model.nper
                else:
                    k0 = 0
                    k1 = 1
            except:
                k0 = int(kk)
                k1 = k0 + 1
            # if kwargs['kper'] == 'all':
            #     kwargs.pop('kper')
            #     k0 = 0
            #     k1 = self.model.nper
            # else:
            #     k0 = int(kwargs.pop('kper'))
            #     k1 = k0 + 1
        else:
            k0 = 0
            k1 = 1

        if 'fignum' in kwargs:
            fignum = kwargs.pop('fignum')
        else:
            fignum = list(range(k0, k1))

        if 'mflay' in kwargs:
            kwargs.pop('mflay')

        axes = []
        for idx, kper in enumerate(range(k0, k1)):
            title = '{} stress period {:d}'.\
                     format(self.name_base.replace('_', '').upper(),
                            kper+1)
            if filename_base is not None:
                filename = filename_base + '_{:05d}.{}'.format(kper+1, fext)
            else:
                filename = None
            axes.append(pu._plot_array_helper(self[kper].array, self.model,
                                              names=title, filenames=filename,
                                              fignum=fignum[idx], **kwargs))
        return axes


    def __getitem__(self, kper):
        if kper in list(self.transient_2ds.keys()):
            return self.transient_2ds[kper]
        elif kper < min(self.transient_2ds.keys()):
            return self.get_zero_2d(kper)
        else:
            for i in range(kper,0,-1):
                if i in list(self.transient_2ds.keys()):
                    return self.transient_2ds[i]
            raise Exception("transient_2d.__getitem__(): error:" +\
                            " could find an entry before kper {0:d}".format(kper))

    def get_kper_entry(self, kper):
        """
        get the file entry info for a given kper
        returns (itmp,file entry string from util_2d)
        """
        if kper in list(self.transient_2ds.keys()):
            return (1, self.transient_2ds[kper].get_file_entry())
        elif kper < min(self.transient_2ds.keys()):
            return (1, self.get_zero_2d(kper))
        else:
            return (-1, '')

    def build_transient_sequence(self):
        """
        parse self.__value into a dict{kper:util_2d}
        """

        # a dict keyed on kper (zero-based)
        if isinstance(self.__value, dict):
            tran_seq = {}
            for key, val in self.__value.items():
                try:
                    key = int(key)
                except:
                    raise Exception("transient_2d error: can't cast key: " +
                                    str(key) + " to kper integer")
                if key < 0:
                    raise Exception("transient_2d error: key can't be " +
                                    " negative: " + str(key))
                try:
                    u2d = self.__get_2d_instance(key, val)
                except Exception as e:
                    raise Exception("transient_2d error building util_2d " +
                                    " instance from value at kper: " +
                                    str(key) + "\n" + str(e))
                tran_seq[key] = u2d
            return tran_seq

        # these are all for single entries - use the same util_2d for all kper
        # an array of shape (nrow,ncol)
        elif isinstance(self.__value,np.ndarray):
            return {0: self.__get_2d_instance(0, self.__value)}

        # a filename
        elif isinstance(self.__value,str):
            return {0: self.__get_2d_instance(0, self.__value)}

        # a scalar
        elif np.isscalar(self.__value):
            return {0: self.__get_2d_instance(0, self.__value)}

        # lists aren't allowed
        elif isinstance(self.__value,list):
            raise Exception("transient_2d error: value cannot be a list " +
                            "anymore.  try a dict{kper,value}")
        else:
            raise Exception("transient_2d error: value type not " +
                            " recognized: " + str(type(self.__value)))


    def __get_2d_instance(self,kper,arg):
        """
        parse an argument into a util_2d instance
        """
        ext_filename = None
        name = self.name_base+str(kper + 1)
        if self.model.external_path != None:
            ext_filename = self.ext_filename_base+str(kper)+'.ref'
        u2d = util_2d(self.model, self.shape, self.dtype, arg,
                      fmtin=self.fmtin, name=name,
                      ext_filename=ext_filename,
                      locat=self.locat)
        return u2d


#class util_2d((with_metaclass(meta_interceptor, object))):
class util_2d(object):
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
        the data to be assigned to the 2-D array.
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
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
         to read the array from)
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename,
        the ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the default is False)

    Attributes
    ----------
    array : np.ndarray
        the array representation of the 2-D object

    Methods
    -------
    get_file_entry : string
        get the model input file string including the control record

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, model, shape, dtype, value, name=None, fmtin=None,
        cnstnt=1.0, iprn=-1, ext_filename=None, locat=None, bin=False,
        ext_unit_dict=None):
        """
        1d or 2-d array support with minimum of mem footprint.
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
        """
        if isinstance(value, util_2d):
            for attr in value.__dict__.items():
                setattr(self, attr[0], attr[1])
            return
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
        # just for testing
        if hasattr(model, 'use_existing'):
            self.use_existing = bool(model.use_existing)
        else:
            self.use_existing = False            
        # set fmtin
        if fmtin is not None:
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
                    self.fmtin = '(' + str(npl) + 'I10) '
                else:
                    self.fmtin = '(' + str(npl) + 'G15.6) '
                    
        # get (npl,python_format_descriptor) from fmtin
        self.py_desc = self.fort_2_py(self.fmtin)  

        # some defense
        if dtype not in [np.int, np.float32, np.bool]:
            raise Exception('util_2d:unsupported dtype: ' + str(dtype))
        if self.model.external_path != None and name == None \
            and ext_filename == None:
            raise Exception('util_2d: use external arrays requires either ' +
               'name or ext_filename attribute')
        elif self.model.external_path != None and ext_filename == None \
            and self.vtype not in [np.int, np.float32]:
            #self.ext_filename = self.model.external_path+name+'.ref'
            self.ext_filename = os.path.join(self.model.external_path,
                                             name + '.ref')
        elif self.vtype not in [np.int, np.float32]:
            self.ext_filename = ext_filename

        if self.bin and self.ext_filename is None:
            raise Exception('util_2d: binary flag requires ext_filename')

    def plot(self, title=None, filename_base=None, file_extension=None,
             fignum=None, **kwargs):
        """
        Plot 2-D model input data

        Parameters
        ----------
        title : str
            Plot title. If a plot title is not provide one will be
            created based on data name (self.name). (default is None)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        ----------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.dis.top.plot()
        
        """       
        import flopy.plot.plotutil as pu
        if title is None:
            title = self.name
        
        if file_extension is not None:
            fext = file_extension
        else:
            fext = 'png'
            
        filename = None
        if filename_base is not None:
            filename = '{}_{}.{}'.format(filename_base, self.name, fext)
        
        return pu._plot_array_helper(self.array, self.model,
                                     names=title, filenames=filename,
                                     fignum=fignum, **kwargs)


    def to_shapefile(self, filename):
        """
        Export 2-D model data to a shapefile (as polygons) of self.array

        Parameters
        ----------
        filename : str
            Shapefile name to write

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.dis.top.as_shapefile('test_top.shp')
        """
        from flopy.utils.flopy_io import write_grid_shapefile
        write_grid_shapefile(filename, self.model.dis.sr, {self.name: self.array})

    @staticmethod
    def get_default_numpy_fmt(dtype):
        if dtype == np.int:
            return "%6d"
        elif dtype == np.float32:
            return "%15.6E"
        else:
            raise Exception("util_2d.get_default_numpy_fmt(): unrecognized " +\
                            "dtype, must be np.int or np.float32")

    def set_fmtin(self, fmtin):
        self.fmtin = fmtin
        self.py_desc = self.fort_2_py(self.fmtin)
        return

    def get_value(self):
        return self.__value
    
    # overloads, tries to avoid creating arrays if possible
    def __add__(self, other):
        if self.vtype in [np.int, np.float32] and self.vtype == other.vtype:
            return self.__value + other.get_value()
        else:
            return self.array + other.array

    def __sub__(self, other):
        if self.vtype in [np.int, np.float32] and self.vtype == other.vtype:
            return self.__value - other.get_value()
        else:
            return self.array - other.array

    def __getitem__(self, k):
        if isinstance(k, int):
            # this explicit cast is to handle a bug in numpy versions < 1.6.2
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

    def __setitem__(self, k, value):
        """
        his one is dangerous because it resets __value
        """
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
        """
        this is the entry point for getting an
        input file entry for this object
        """
        # call get_file_array first in case we need to
       #  get a new external unit number and reset self.locat
        vstring = self.get_file_array()
        cr = self.get_control_record()
        return cr+vstring
    

    def get_file_array(self):
        """
        increments locat and update model instance if needed.
        if the value is a constant, or a string, or external, 
        return an empty string
        """       
        # if the value is not a filename
        if self.vtype != str:
            
            # if the ext_filename was passed, then we need
            #  to write an external array
            if self.ext_filename != None:
                # if we need fixed format, reset self.locat and get a
               #   new unit number
                if not self.model.free_format:
                    self.locat = self.model.next_ext_unit() 
                    if self.bin:
                        self.locat = -1 * np.abs(self.locat)
                        self.model.add_external(self.ext_filename,
                            self.locat, binFlag=True)
                    else:
                        self.model.add_external(self.ext_filename, self.locat)
                # write external formatted or unformatted array
                if not self.use_existing:    
                    if not self.bin:
                        f = open(self.ext_filename, 'w')
                        f.write(self.string)
                        f.close()
                    else:
                        a = self.array.tofile(self.ext_filename)                    
                return ''
                
            # this internal array or constant
            else:
                if self.vtype is np.ndarray:
                    return self.string
                # if this is a constant, return a null string
                else:
                    return ''
        else:         
            if os.path.exists(self.__value) and self.ext_filename is not None:
                # if this is a free format model, then we can use the same
                #  ext file over and over - no need to copy
                # also, loosen things up with FREE format
                if self.model.free_format:
                    self.ext_filename = self.__value
                    self.fmtin = '(FREE)'
                    self.py_desc = self.fort_2_py(self.fmtin)

                else:
                    if self.__value != self.ext_filename:
                        shutil.copy2(self.__value, self.ext_filename)
                    # if fixed format, we need to get a new unit number
                    #  and reset locat
                    self.locat = self.model.next_ext_unit()
                    self.model.add_external(self.ext_filename, self.locat)
                    
                return '' 
            # otherwise, we need to load the the value filename
            #  and return as a string
            else:
                return self.string

    @property
    def string(self):
        """
        get the string represenation of value attribute
        """
        a = self.array
        # convert array to sting with specified format
        a_string = array2string(a,self.py_desc)
        return a_string
                                    
    @property
    def array(self):
        """
        get the array representation of value attribute
        if value is a string or a constant, the array is loaded/built only once
        """
        if self.vtype == str:
            if self.__value_built is None:
                file_in = open(self.__value,'r')
                self.__value_built = \
                    util_2d.load_txt(self.shape, file_in, self.dtype,
                                     self.fmtin).astype(self.dtype)
                file_in.close()
            return self.__value_built
        elif self.vtype != np.ndarray:
            if self.__value_built is None:
                self.__value_built = np.ones(self.shape, dtype=self.dtype) \
                    * self.__value
            return self.__value_built
        else:
            return self.__value
    
    @staticmethod
    def load_txt(shape, file_in, dtype, fmtin):
        """
        load a (possibly wrapped format) array from a file
        (self.__value) and casts to the proper type (self.dtype)
        made static to support the load functionality 
        this routine now supports fixed format arrays where the numbers
        may touch.
        """
        #file_in = open(self.__value,'r')
        #file_in = open(filename,'r')
        #nrow,ncol = self.shape
        nrow, ncol = shape
        npl, fmt, width, decimal = decode_fortran_descriptor(fmtin)
        data = np.zeros((nrow * ncol), dtype=dtype) + np.NaN
        d = 0
        if not hasattr(file_in, 'read'):
            file_in = open(file_in,'r')
        while True:
            line = file_in.readline()
            if line in [None,''] or d == nrow*ncol:
                break
            if npl == 'free':
                raw = line.strip('\n').split()
            else:
                #split line using number of values in the line
                rawlist = []
                istart = 0
                istop = width
                for i in range(npl):
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
                    raise Exception('util_2d:unable to cast value: ' +
                        str(a) + ' to type:' + str(dtype))
                if d == (nrow*ncol)-1:
                    assert len(data) == (nrow * ncol)
                    data.resize(nrow, ncol)
                    return(data) 
                d += 1	
#        file_in.close()
        if np.isnan(np.sum(data)):
            raise Exception("util_2d.load_txt() error: np.NaN in data array")
        data.resize(nrow, ncol)
        return data * mult

    @staticmethod
    def write_txt(shape, file_out, data, fortran_format="(FREE)",
                  python_format=None):
        """
        write a (possibly wrapped format) array from a file
        (self.__value) and casts to the proper type (self.dtype)
        made static to support the load functionality
        this routine now supports fixed format arrays where the numbers
        may touch.
        """

        if fortran_format.upper() == '(FREE)' and python_format is None:
            np.savetxt(file_out,data,util_2d.get_default_numpy_fmt(data.dtype),
                       delimiter='')
            return

        nrow, ncol = shape
        if python_format is None:
            column_length, fmt, width, decimal = \
                decode_fortran_descriptor(fortran_format)
            output_fmt = '{0}0:{1}.{2}{3}{4}'.format('{', width, decimal, fmt,
                                                     '}')
        else:
            try:
                column_length, output_fmt = int(python_format[0]), \
                                            python_format[1]
            except:
                raise Exception('util_2d.write_txt: \nunable to parse'
                                + 'python_format:\n    {0}\n'.
                                format(python_format)
                                + '  python_format should be a list with\n'
                                + '   [column_length, fmt]\n'
                                + '    e.g., [10, {0:10.2e}]')
        if ncol%column_length == 0:
            lineReturnFlag = False
        else:
            lineReturnFlag = True
        # write the array
        for i in range(nrow):
            icol = 0
            for j in range(ncol):
                try:
                    file_out.write(output_fmt.format(data[i,j]))
                except:
                    print('Value {0} at row,col [{1},{2}] can not be written'\
                        .format(data[i, j], i, j))
                    raise Exception
                if (j + 1) % column_length == 0.0 and j != 0:
                    file_out.write('\n')
            if lineReturnFlag == True:
                file_out.write('\n')

    @staticmethod
    def load_bin(shape, file_in, dtype, bintype=None):
        nrow,ncol = shape
        if bintype is not None:
            if dtype not in [np.int]:
                header_dtype = BinaryHeader.set_dtype(bintype=bintype)
            header_data = np.fromfile(file_in, dtype=header_dtype, count=1)
        else:
            header_data = None
        data = np.fromfile(file_in,dtype=dtype,count=nrow*ncol)
        data.resize(nrow, ncol)
        return [header_data, data]

    @staticmethod
    def write_bin(shape, file_out, data, bintype=None, header_data=None):
        dtype = data.dtype
        if dtype.kind != 'i':
            if bintype is not None:
                if header_data is None:
                    header_data = BinaryHeader.create(bintype=bintype)
            if header_data is not None:
                header_data.tofile(file_out)
        data.tofile(file_out)
        return

    def get_control_record(self):
        """
        get the modflow control record
        """      
        lay_space = '{0:>27s}'.format('')
        if self.model.free_format:
            if self.ext_filename is None:
                if self.vtype in [np.int]:
                    lay_space = '{0:>32s}'.format('')
                if self.vtype in [np.int, np.float32]:
                    # this explicit cast to float is to handle a
                    #- bug in versions of numpy < l.6.2
                    if self.dtype == np.float32:                    
                        cr = 'CONSTANT ' + \
                             self.py_desc[1].format(float(self.__value))
                    else:
                        cr = 'CONSTANT '+self.py_desc[1].format(self.__value)
                    cr = '{0:s}{1:s}#{2:<30s}\n'.format(cr, lay_space,
                                                        self.name)
                else:
                    cr = 'INTERNAL {0:15.6G} {1:>10s} {2:2.0f} #{3:<30s}\n'\
                        .format(self.cnstnt, self.fmtin, self.iprn, self.name)
            else:
                # need to check if ext_filename exists, if not, need to
                #  write constant as array to file or array to file
                f = self.ext_filename
                fr = os.path.relpath(f, self.model.model_ws)
                #if self.locat is None:
                cr = 'OPEN/CLOSE  {0:>30s} {1:15.6G} {2:>10s} {3:2.0f} {4:<30s}\n'.format(fr, self.cnstnt,
                     self.fmtin.strip(), self.iprn, self.name)
                # else:
                #     cr = 'EXTERNAL  {0:5d} {1:15.6G} {2:>10s} {3:2.0f} {4:<30s}\n'.format(self.locat, self.cnstnt,
                #          self.fmtin.strip(), self.iprn, self.name)

        else:                       
            # if value is a scalar and we don't want external array
            if self.vtype in [np.int, np.float32] and self.ext_filename is None:
                locat = 0
                # explicit cast for numpy bug in versions < 1.6.2
                if self.dtype == np.float32:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:10.0f} #{4}\n'\
                        .format(locat, float(self.__value),
                                self.fmtin, self.iprn, self.name)
                else:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:10.0f} #{4}\n'\
                        .format(locat, self.__value, self.fmtin, self.iprn,
                                self.name)
            else:
                if self.ext_filename is None:
                    assert self.locat != None,'util_2d:a non-constant value ' +\
                    ' for an internal fixed-format requires LOCAT to be passed'
                if self.dtype == np.int:
                    cr = '{0:>10.0f}{1:>10.0f}{2:>20s}{3:>10.0f} #{4}\n'\
                        .format(self.locat, self.cnstnt, self.fmtin,
                                self.iprn, self.name)
                elif self.dtype == np.float32:
                    cr = '{0:>10.0f}{1:>10.5G}{2:>20s}{3:>10.0f} #{4}\n'\
                        .format(self.locat, self.cnstnt, self.fmtin,
                                self.iprn, self.name)
                else:
                    raise Exception('util_2d: error generating fixed-format ' +
                       ' control record,dtype must be np.int or np.float32')
        return cr                                 


    def fort_2_py(self,fd):
        """
        converts the fortran format descriptor
        into a tuple of npl and a python format specifier

        """
        npl,fmt,width,decimal = decode_fortran_descriptor(fd)
        if npl == 'free':
            if self.vtype == np.int:
                return (self.shape[1], '{0:10.0f} ')
            else:
                return (self.shape[1], '{0:15.6G} ')
        elif npl == 'binary':
            return('binary', None)
        else:
            pd = build_python_descriptor(npl, fmt, width, decimal)
            return (npl, pd)


    def parse_value(self, value):
        """
        parses and casts the raw value into an acceptable format for __value
        lot of defense here, so we can make assumptions later
        """
        if isinstance(value, list):
            if VERBOSE:
                print('util_2d: casting list to array')
            value = np.array(value)
        if isinstance(value, bool):
            if self.dtype == np.bool:
                try:
                    value = np.bool(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast ' +
                                    'boolean value to type "np.bool": ' +
                                    str(value))
            else:
                raise Exception('util_2d:value type is bool, ' +
                                ' but dtype not set as np.bool')
        if isinstance(value, str):
            if self.dtype == np.int:
                try:
                    value = int(value)
                except:
                    # print value
                    # print os.path.join(self.model.model_ws,value)
                    # JDH Note: value should be the filename with
                    #            the relative path. Trace through code to
                    #            determine why it isn't
                    if os.path.basename(value) == value:
                        value = os.path.join(self.model.model_ws, value)
                    assert os.path.exists(value), 'could not find file: ' + str(value)
                    #assert os.path.exists(value), \
                    #    'could not find file: ' + str(value)
                    return value
            else:
                try:
                    value = float(value)
                except:
                    assert os.path.exists(
                        os.path.join(self.model.model_ws,value)), \
                        'could not find file: ' + str(value)
                    return value
        if np.isscalar(value):
            if self.dtype == np.int:
                try:
                    value = np.int(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast scalar ' +
                                    'value to type "int": ' + str(value))
            elif self.dtype == np.float32:
                try:
                    value = np.float32(value)
                    return value
                except:
                    raise Exception('util_2d:could not cast ' +
                                    'scalar value to type "float": ' +
                                    str(value))
            
        if isinstance(value, np.ndarray):
            if self.shape != value.shape:
                raise Exception('util_2d:self.shape: ' + str(self.shape) +
                                ' does not match value.shape: ' +
                                str(value.shape))
            if self.dtype != value.dtype:
                if VERBOSE:
                    print('util_2d:warning - casting array of type: ' +\
                          str(value.dtype)+' to type: '+str(self.dtype))
            return value.astype(self.dtype)
        
        else:
            raise Exception('util_2d:unsupported type in util_array: ' +
                            str(type(value)))


    @staticmethod
    def load(f_handle, model, shape, dtype, name, ext_unit_dict=None):
        """
        functionality to load util_2d instance from an existing
        model input file.
        external and internal record types must be fully loaded
        if you are using fixed format record types,make sure 
        ext_unit_dict has been initialized from the NAM file
        """     

        curr_unit = None
        if ext_unit_dict is not None:
            # determine the current file's unit number
            cfile = f_handle.name
            for cunit in ext_unit_dict:
                if cfile == ext_unit_dict[cunit].filename:
                    curr_unit = cunit
                    break

        cr_dict = util_2d.parse_control_record(f_handle.readline(),
                                               current_unit=curr_unit,
                                               dtype=dtype,
                                               ext_unit_dict=ext_unit_dict)
        
        if cr_dict['type'] == 'constant':
            u2d = util_2d(model, shape, dtype, cr_dict['cnstnt'], name=name,
                iprn=cr_dict['iprn'], fmtin=cr_dict['fmtin'])
        
        elif cr_dict['type'] == 'open/close':
            # clean up the filename a little
            fname = cr_dict['fname']
            fname = fname.replace("'", "")
            fname = fname.replace('"', '')
            fname = fname.replace('\'', '')
            fname = fname.replace('\"', '')
            fname = fname.replace('\\', os.path.sep)
            fname = os.path.join(model.model_ws,fname)
            #load_txt(shape, file_in, dtype, fmtin):
            assert os.path.exists(fname),"util_2d.load() error: open/close " +\
                "file " + str(fname) + " not found"
            f = open(fname, 'r')
            data = util_2d.load_txt(shape=shape,
                                    file_in=f,
                                    dtype=dtype, fmtin=cr_dict['fmtin'])
            f.close()
            # u2d = util_2d(model, shape, dtype, fname, name=name,
            #               iprn=cr_dict['iprn'], fmtin=cr_dict['fmtin'],
            #               ext_filename=fname)
            u2d = util_2d(model, shape, dtype, data, name=name,
                          iprn=cr_dict['iprn'], fmtin=cr_dict['fmtin'],
                          ext_filename=fname)

        elif cr_dict['type'] == 'internal':
            data = util_2d.load_txt(shape, f_handle, dtype, cr_dict['fmtin'])
            data *= cr_dict['cnstnt']
            u2d = util_2d(model, shape, dtype, data, name=name,
                          iprn=cr_dict['iprn'], fmtin=cr_dict['fmtin'])

        elif cr_dict['type'] == 'external':
            assert cr_dict['nunit'] in list(ext_unit_dict.keys())
            if str('binary') not in str(cr_dict['fmtin'].lower()):
                data = util_2d.load_txt(shape,
                                    ext_unit_dict[cr_dict['nunit']].filehandle,
                                        dtype, cr_dict['fmtin'])
            else:
                header_data, data = util_2d.load_bin(
                    shape, ext_unit_dict[cr_dict['nunit']].filehandle, dtype)
            data *= cr_dict['cnstnt']
            u2d = util_2d(model, shape, dtype, data, name=name,
                          iprn=cr_dict['iprn'], fmtin=cr_dict['fmtin'])
            # track this unit number so we can remove it from the external
            # file list later
            model.pop_key_list.append(cr_dict['nunit'])
        return u2d
            

    @staticmethod
    def parse_control_record(line, current_unit=None, dtype=np.float32,
                             ext_unit_dict=None):
        """
        parses a control record when reading an existing file
        rectifies fixed to free format
        current_unit (optional) indicates the unit number of the file being parsed
        """
        free_fmt = ['open/close', 'internal', 'external', 'constant']
        raw = line.lower().strip().split()
        freefmt, cnstnt, fmtin, iprn, nunit = None, None, None, -1, None
        fname = None 
        isFloat = False
        if dtype == np.float or dtype == np.float32:
            isFloat = True       
        # if free format keywords
        if str(raw[0]) in str(free_fmt):
            freefmt = raw[0]
            if raw[0] == 'constant':
                if isFloat:                
                    cnstnt = np.float(raw[1].lower().replace('d', 'e'))
                else:
                    cnstnt = np.int(raw[1].lower())                   
            if raw[0] == 'internal':
                if isFloat:
                    cnstnt = np.float(raw[1].lower().replace('d', 'e'))
                else:
                    cnstnt = np.int(raw[1].lower())
                fmtin = raw[2].strip()
                iprn = int(raw[3])
            elif raw[0] == 'external':
                if ext_unit_dict is not None:
                    try:
                        #td = ext_unit_dict[int(raw[1])]
                        fname = ext_unit_dict[int(raw[1])].filename.strip()
                    except:
                        pass
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
                npl, fmt, width, decimal = None, None, None, None
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

    def __mul__(self, other):
        if np.isscalar(other):
            self.array
            return util_2d(self.model,self.shape,self.dtype,
                           self.__value_built * other,self.name,
                           self.fmtin,self.cnstnt,self.iprn,self.ext_filename,
                           self.locat,self.bin)
        else:
            raise NotImplementedError(
                "util_2d.__mul__() not implemented for non-scalars")
