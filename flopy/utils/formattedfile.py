"""
Module to read MODFLOW formatted output files.  The module contains one
important classes that can be accessed by the user.

*  FormattedHeadFile (Formatted head file.  Can also be used for drawdown)

"""

import numpy as np
from ..utils.datafile import Header, LayerFile


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class FormattedHeader(Header):
    """
    The TextHeader class is a class to read in headers from MODFLOW
    formatted files.

    Parameters
    ----------
        text_ident is the text string in the header that identifies the type of data (eg. 'head')
        precision is the precision of the floating point data in the file
    """

    def __init__(self, text_ident, precision='single'):
        Header.__init__(self, text_ident, precision)
        self.format_string = ''
        self.text_ident = text_ident

    def read_header(self, text_file):
        """
        Read header information from a formatted file

        Parameters
        ----------
            text_file is an open file object currently at the beginning of the header

        Returns
        ----------
        out : numpy array of header information
        also stores the header's format string as self.format_string
        """

        header_text = text_file.readline().decode('ascii')
        arrheader = header_text.split()

        # Verify header exists and is in the expected format
        if len(arrheader) >= 5 and arrheader[
            4].upper() != self.text_ident.upper():
            raise Exception(
                'Expected header not found.  Make sure the file being processed includes headers ' +
                '(LABEL output control option): ' + header_text)
        if len(arrheader) != 9 or not is_int(arrheader[0]) or not is_int(
                arrheader[1]) or not is_float(arrheader[2]) \
                or not is_float(arrheader[3]) or not is_int(
            arrheader[5]) or not is_int(arrheader[6]) or not is_int(
            arrheader[7]):
            raise Exception(
                'Unexpected format for FHDTextHeader: ' + header_text)

        headerinfo = np.empty([8], dtype=self.dtype)
        headerinfo['kstp'] = int(arrheader[0])
        headerinfo['kper'] = int(arrheader[1])
        headerinfo['pertim'] = float(arrheader[2])
        headerinfo['totim'] = float(arrheader[3])
        headerinfo['text'] = arrheader[4]
        headerinfo['ncol'] = int(arrheader[5])
        headerinfo['nrow'] = int(arrheader[6])
        headerinfo['ilay'] = int(arrheader[7])

        self.format_string = arrheader[8]

        return headerinfo


class FormattedLayerFile(LayerFile):
    """
    The FormattedLayerFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly

    """

    def __init__(self, filename, precision, verbose, kwargs):
        super(FormattedLayerFile, self).__init__(filename, precision, verbose,
                                                 kwargs)
        return

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        self.kstpkper  # array of time step/stress periods with data available
        self.recordarray  # array of data headers
        self.iposarray  # array of seek positions for each record
        self.nlay  # Number of model layers

        # Get total file size
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)

        # Process first header
        self.header = self._get_text_header()
        header_info = self.header.read_header(self.file)[0]

        self.nrow = header_info['nrow']
        self.ncol = header_info['ncol']

        ipos = self.file.tell()
        self._store_record(header_info, ipos)

        # Process enough data to calculate seek distance between headers
        self._col_data_size = self._get_data_size(header_info)
        self._data_size = self._col_data_size * self.nrow

        # While more data in file
        while ipos + self._data_size < self.totalbytes:
            # Seek and get next header
            self.file.seek(ipos + self._data_size)
            header_info = self.header.read_header(self.file)[0]
            ipos = self.file.tell()
            self._store_record(header_info, ipos)

        # self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, self.header.get_dtype())
        self.iposarray = np.array(self.iposarray)
        self.nlay = np.max(self.recordarray['ilay'])
        return

    def _store_record(self, header, ipos):
        """
        Store file header information in various formats for quick retreival

        """
        self.recordarray.append(header)
        self.iposarray.append(ipos)  # store the position right after header2
        totim = header['totim']
        if totim > 0 and totim not in self.times:
            self.times.append(totim)
        kstpkper = (header['kstp'], header['kper'])
        if kstpkper not in self.kstpkper:
            self.kstpkper.append(kstpkper)

    def _get_text_header(self):
        """
        Return a text header object containing header formatting information

        """
        raise Exception(
            'Abstract method _get_text_header called in FormattedLayerFile. ' +
            'This method needs to be overridden.')

    def _read_data(self, shp):
        """
        Read 2-D data from file

        """

        nrow, ncol = shp
        current_row = 0
        current_col = 0
        result = np.empty((nrow, ncol), self.realtype)
        # Loop until all data retreived or eof
        while (
                current_row < nrow or current_col < ncol) and self.file.tell() != self.totalbytes:
            line = self.file.readline()

            # Read data into 2-D array
            arrline = line.split()
            for val in arrline:
                if not is_float(val):
                    raise Exception(
                        'Invalid data encountered while reading data file.' +
                        ' Unable to convert data to float.')
                result[current_row, current_col] = float(val)
                current_col += 1
                if current_col >= ncol:
                    current_row += 1
                    if current_row < nrow:
                        current_col = 0

        if current_row < nrow - 1 or current_col < ncol - 1:
            raise Exception('Unexpected end of file while reading data.')

        return result

    def _read_val(self, i):
        """
        Read ith data value from file

        """
        current_col = 0
        result = None
        # Loop until data retreived or eof
        while (
                current_col < self.ncol - 1 or self.file.tell() == self.totalbytes) and current_col <= i:
            line = self.file.readline()
            arrline = line.split()
            for val in arrline:
                if not is_float(val):
                    raise Exception(
                        'Invalid data encountered while reading data file.' +
                        ' Unable to convert data to float.')
                result = float(val)
                current_col = current_col + 1
                if current_col > i:
                    break

        if (current_col < self.ncol - 1) and (current_col < i):
            raise Exception('Unexpected end of file while reading data.')

        return result

    def get_ts(self, idx):
        """
        Get a time series from the formatted file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        istat = 1
        for k, i, j in kijlist:
            recordlist = []
            ioffset_col = (i * self._col_data_size)
            for irec, header in enumerate(self.recordarray):
                ilay = header[
                           'ilay'] - 1  # change ilay from header to zero-based
                if ilay != k:
                    continue
                ipos = self.iposarray[irec]

                # Calculate offset necessary to reach intended column
                self.file.seek(ipos + ioffset_col, 0)

                # Find the time index and then put value into result in the
                # correct location.
                itim = np.where(result[:, 0] == header['totim'])[0]
                result[itim, istat] = self._read_val(j)
            istat += 1
        return result

    def close(self):
        """
        Close the file handle.

        """
        self.file.close()
        return


class FormattedHeadFile(FormattedLayerFile):
    """
    FormattedHeadFile Class.

    Parameters
    ----------
    filename : string
        Name of the formatted head file
    text : string
        Name of the text string in the formatted head file.  Default is 'head'
    precision : string
        'single' or 'double'.  Default is 'single'.
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The FormattedHeadFile class provides simple ways to retrieve 2d and 3d
    head arrays from a MODFLOW formatted head file and time series
    arrays for one or more cells.

    The FormattedHeadFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    FormattedHeadFile can only read formatted head files containing headers.
    Use the LABEL option in the output control file to generate head files
    with headers.

    Examples
    --------

    >>> import flopy.utils.formattedfile as ff
    >>> hdobj = ff.FormattedHeadFile('model.fhd', precision='single')
    >>> hdobj.list_records()
    >>> rec = hdobj.get_data(kstpkper=(1, 50))
    >>> rec2 = ddnobj.get_data(totim=100.)


    """

    def __init__(self, filename, text='head', precision='single',
                 verbose=False, **kwargs):
        self.text = text
        super(FormattedHeadFile, self).__init__(filename, precision, verbose,
                                                kwargs)
        return

    def _get_text_header(self):
        """
        Return a text header object containing header formatting information

        """
        return FormattedHeader(self.text, self.precision)

    def _get_data_size(self, header):
        """
        Calculate the size of the data set in terms of a seek distance

        """
        start_pos = self.file.tell()
        data_count = 0
        # Loop through data until at end of column
        while data_count < header['ncol']:
            column_data = self.file.readline()
            arr_column_data = column_data.split()
            data_count += len(arr_column_data)

        if data_count != header['ncol']:
            raise Exception(
                'Unexpected data formatting in head file.  Expected %d columns, but found %d.' %
                header['ncol'], data_count)

        # Calculate seek distance based on data size
        stop_pos = self.file.tell()
        data_seek_distance = stop_pos - start_pos

        # Return to last file position
        self.file.seek(start_pos)

        return data_seek_distance
