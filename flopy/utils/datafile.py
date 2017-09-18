"""
Module to read MODFLOW output files.  The module contains shared
abstract classes that should not be directly accessed.

"""
from __future__ import print_function
import os
import numpy as np
import flopy.utils


class Header(object):
    """
    The header class is an abstract base class to create headers for MODFLOW files
    """

    def __init__(self, filetype=None, precision='single'):
        floattype = 'f4'
        if precision == 'double':
            floattype = 'f8'
        self.header_types = ['head', 'drawdown', 'ucn']
        if filetype is None:
            self.header_type = None
        else:
            if isinstance(filetype, bytes):
                filetype = filetype.decode()
            self.header_type = filetype.lower()
        if self.header_type in self.header_types:
            if self.header_type == 'head':
                self.dtype = np.dtype([('kstp', 'i4'), ('kper', 'i4'),
                                       ('pertim', floattype),
                                       ('totim', floattype),
                                       ('text', 'a16'),
                                       ('ncol', 'i4'), ('nrow', 'i4'),
                                       ('ilay', 'i4')])
            elif self.header_type == 'drawdown':
                self.dtype = np.dtype([('kstp', 'i4'), ('kper', 'i4'),
                                       ('pertim', floattype),
                                       ('totim', floattype),
                                       ('text', 'a16'),
                                       ('ncol', 'i4'), ('nrow', 'i4'),
                                       ('ilay', 'i4')])
            elif self.header_type == 'ucn':
                self.dtype = np.dtype(
                    [('ntrans', 'i4'), ('kstp', 'i4'), ('kper', 'i4'),
                     ('totim', floattype), ('text', 'a16'),
                     ('ncol', 'i4'), ('nrow', 'i4'), ('ilay', 'i4')])
            self.header = np.ones(1, self.dtype)
        else:
            self.dtype = None
            self.header = None
            print(
                'Specified {0} type is not available. Available types are:'.format(
                    self.header_type))
            for idx, t in enumerate(self.header_types):
                print('  {0} {1}'.format(idx + 1, t))
        return

    def get_dtype(self):
        """
        Return the dtype
        """
        return self.dtype

    def get_names(self):
        """
        Return the dtype names
        """
        return self.dtype.names

    def get_values(self):
        """
        Return the header values
        """
        if self.header is None:
            return None
        else:
            return self.header[0]


class LayerFile(object):
    """
    The LayerFile class is the abstract base class from which specific derived
    classes are formed.  LayerFile This class should not be instantiated directly.

    """

    def __init__(self, filename, precision, verbose, kwargs):
        assert os.path.exists(
            filename), "datafile error: datafile not found:" + str(filename)
        self.filename = filename
        self.precision = precision
        self.verbose = verbose
        self.file = open(self.filename, 'rb')
        self.nrow = 0
        self.ncol = 0
        self.nlay = 0
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposarray = []

        if precision == 'single':
            self.realtype = np.float32
        elif precision == 'double':
            self.realtype = np.float64
        else:
            raise Exception('Unknown precision specified: ' + precision)

        self.model = None
        self.dis = None
        self.sr = None
        if 'model' in kwargs.keys():
            self.model = kwargs.pop('model')
            self.sr = self.model.sr
            self.dis = self.model.dis
        if 'dis' in kwargs.keys():
            self.dis = kwargs.pop('dis')
            self.sr = self.dis.parent.sr
        if 'sr' in kwargs.keys():
            self.sr = kwargs.pop('sr')
        if len(kwargs.keys()) > 0:
            args = ','.join(kwargs.keys())
            raise Exception('LayerFile error: unrecognized kwargs: ' + args)

        # read through the file and build the pointer index
        self._build_index()

        # now that we read the data and know nrow and ncol,
        # we can make a generic sr if needed
        if self.sr is None:
            self.sr = flopy.utils.SpatialReference(np.ones(self.ncol),
                                                   np.ones(self.nrow), 0)
        return

    def to_shapefile(self, filename, kstpkper=None, totim=None, mflay=None,
                     attrib_name='lf_data'):
        """
         Export model output data to a shapefile at a specific location
          in LayerFile instance.

         Parameters
         ----------
         filename : str
             Shapefile name to write
         kstpkper : tuple of ints
             A tuple containing the time step and stress period (kstp, kper).
             These are zero-based kstp and kper values.
         totim : float
             The simulation time.
         mflay : integer
            MODFLOW zero-based layer number to return.  If None, then layer 1
            will be written
         attrib_name : str
             Base name of attribute columns. (default is 'lf_data')

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
         >>> hdobj = flopy.utils.HeadFile('test.hds')
         >>> times = hdobj.get_times()
         >>> hdobj.to_shapefile('test_heads_sp6.shp', totim=times[-1])
         """

        plotarray = np.atleast_3d(self.get_data(kstpkper=kstpkper,
                                                totim=totim, mflay=mflay)
                                  .transpose()).transpose()
        if mflay != None:
            attrib_dict = {
                attrib_name + '{0:03d}'.format(mflay): plotarray[0, :, :]}
        else:
            attrib_dict = {}
            for k in range(plotarray.shape[0]):
                name = attrib_name + '{0:03d}'.format(k)
                attrib_dict[name] = plotarray[k]

        from ..export.shapefile_utils import write_grid_shapefile
        write_grid_shapefile(filename, self.sr, attrib_dict)

    def plot(self, axes=None, kstpkper=None, totim=None, mflay=None,
             filename_base=None, **kwargs):
        '''
        Plot 3-D model output data in a specific location
        in LayerFile instance

        Parameters
        ----------
        axes : list of matplotlib.pyplot.axis
            List of matplotlib.pyplot.axis that will be used to plot 
            data for each layer. If axes=None axes will be generated.
            (default is None)
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            These are zero-based kstp and kper values.
        totim : float
            The simulation time.
        mflay : int
            MODFLOW zero-based layer number to return.  If None, then all
            all layers will be included. (default is None)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        **kwargs : dict
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
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
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')

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
        >>> hdobj = flopy.utils.HeadFile('test.hds')
        >>> times = hdobj.get_times()
        >>> hdobj.plot(totim=times[-1])
        
        '''

        if 'file_extension' in kwargs:
            fext = kwargs.pop('file_extension')
            fext = fext.replace('.', '')
        else:
            fext = 'png'

        masked_values = kwargs.pop("masked_values", [])
        if self.model is not None:
            if self.model.bas6 is not None:
                masked_values.append(self.model.bas6.hnoflo)
        kwargs["masked_values"] = masked_values

        filenames = None
        if filename_base is not None:
            if mflay is not None:
                i0 = int(mflay)
                if i0 + 1 >= self.nlay:
                    i0 = self.nlay - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = self.nlay
            filenames = []
            [filenames.append(
                '{}_Layer{}.{}'.format(filename_base, k + 1, fext)) for k in
             range(i0, i1)]

        # make sure we have a (lay,row,col) shape plotarray
        plotarray = np.atleast_3d(self.get_data(kstpkper=kstpkper,
                                                totim=totim, mflay=mflay)
                                  .transpose()).transpose()
        import flopy.plot.plotutil as pu
        return pu._plot_array_helper(plotarray, model=self.model, sr=self.sr,
                                     axes=axes,
                                     filenames=filenames,
                                     mflay=mflay, **kwargs)

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        raise Exception(
            'Abstract method _build_index called in LayerFile.  This method needs to be overridden.')

    def list_records(self):
        """
        Print a list of all of the records in the file
        obj.list_records()

        """
        for header in self.recordarray:
            print(header)
        return

    def _get_data_array(self, totim=0):
        """
        Get the three dimensional data array for the
        specified kstp and kper value or totim value.

        """

        if totim >= 0.:
            keyindices = np.where((self.recordarray['totim'] == totim))[0]
            if len(keyindices) == 0:
                msg = 'totim value ({}) not found in file...'.format(totim)
                raise Exception(msg)
        else:
            raise Exception('Data not found...')

        # initialize head with nan and then fill it
        idx = keyindices[0]
        nrow = self.recordarray['nrow'][idx]
        ncol = self.recordarray['ncol'][idx]
        data = np.empty((self.nlay, nrow, ncol), dtype=self.realtype)
        data[:, :, :] = np.nan
        for idx in keyindices:
            ipos = self.iposarray[idx]
            ilay = self.recordarray['ilay'][idx]
            if self.verbose:
                msg = 'Byte position in file: {} for '.format(ipos) + \
                      'layer {}'.format(ilay)
                print(msg)
            self.file.seek(ipos, 0)
            nrow = self.recordarray['nrow'][idx]
            ncol = self.recordarray['ncol'][idx]
            shp = (nrow, ncol)
            data[ilay - 1] = self._read_data(shp)
        return data

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the file

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in binary file.  kstp and
            kper values are presently zero-based.

        """
        kstpkper = []
        for kstp, kper in self.kstpkper:
            kstpkper.append((kstp - 1, kper - 1))
        return kstpkper

    def get_data(self, kstpkper=None, idx=None, totim=None, mflay=None):
        """
        Get data from the file for the specified conditions.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            These are zero-based kstp and kper values.
        totim : float
            The simulation time.
        mflay : integer
           MODFLOW zero-based layer number to return.  If None, then all
           all layers will be included. (Default is None.)

        Returns
        ----------
        data : numpy array
            Array has size (nlay, nrow, ncol) if mflay is None or it has size
            (nrow, ncol) if mlay is specified.

        See Also
        --------

        Notes
        -----
        if both kstpkper and totim are None, will return the last entry
        Examples
        --------

        """
        # One-based kstp and kper for pulling out of recarray
        if kstpkper is not None:
            kstp1 = kstpkper[0] + 1
            kper1 = kstpkper[1] + 1
            idx = np.where(
                (self.recordarray['kstp'] == kstp1) &
                (self.recordarray['kper'] == kper1))
            if idx[0].shape[0] == 0:
                raise Exception("get_data() error: kstpkper not found:{0}".
                                format(kstpkper))
            totim1 = self.recordarray[idx]["totim"][0]
        elif totim is not None:
            totim1 = totim
        elif idx is not None:
            totim1 = self.recordarray['totim'][idx]
        else:
            totim1 = self.times[-1]

        data = self._get_data_array(totim1)
        if mflay is None:
            return data
        else:
            return data[mflay, :, :]

    def get_alldata(self, mflay=None, nodata=-9999):
        """
        Get all of the data from the file.

        Parameters
        ----------
        mflay : integer
           MODFLOW zero-based layer number to return.  If None, then all
           all layers will be included. (Default is None.)

        nodata : float
           The nodata value in the data array.  All array values that have the
           nodata value will be assigned np.nan.

        Returns
        ----------
        data : numpy array
            Array has size (ntimes, nlay, nrow, ncol) if mflay is None or it
            has size (ntimes, nrow, ncol) if mlay is specified.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        rv = []
        for totim in self.times:
            h = self.get_data(totim=totim, mflay=mflay)
            rv.append(h)
        rv = np.array(rv)
        rv[rv == nodata] = np.nan
        return rv

    def _read_data(self, shp):
        """
        Read data from file

        """
        raise Exception(
            'Abstract method _read_data called in LayerFile.  This method needs to be overridden.')

    def _build_kijlist(self, idx):
        if isinstance(idx, list):
            kijlist = idx
        elif isinstance(idx, tuple):
            kijlist = [idx]
        else:
            raise Exception('Could not build kijlist from ', idx)

        # Check to make sure that k, i, j are within range, otherwise
        # the seek approach won't work.  Can't use k = -1, for example.
        for k, i, j in kijlist:
            fail = False
            errmsg = 'Invalid cell index. Cell ' + str(
                (k, i, j)) + ' not within model grid: ' + \
                     str((self.nlay, self.nrow, self.ncol))
            if k < 0 or k > self.nlay - 1:
                fail = True
            if i < 0 or i > self.nrow - 1:
                fail = True
            if j < 0 or j > self.ncol - 1:
                fail = True
            if fail:
                raise Exception(errmsg)
        return kijlist

    def _get_nstation(self, idx, kijlist):
        if isinstance(idx, list):
            return len(kijlist)
        elif isinstance(idx, tuple):
            return 1

    def _init_result(self, nstation):
        # Initialize result array and put times in first column
        result = np.empty((len(self.times), nstation + 1),
                          dtype=self.realtype)
        result[:, :] = np.nan
        result[:, 0] = np.array(self.times)
        return result

    def close(self):
        """
        Close the file handle.

        """
        self.file.close()
        return
