import os
import numpy as np
from ...utils import binaryfile as bf


class MFOutput:
    """
    Wrapper class for Binary Arrays. This class enables directly getting slices
    from the binary output. It is intended to be called from the __getitem__
    method of the  SimulationDict() class.  Implemented to conserve memory.

    Parameters
    ----------
    path: binary file path location
    mfdict: SimulationDict() object
    key: OrderedDictionary key ex. ('flow15','CBC','FLOW RIGHT FACE')

    Returns
    -------
    Xarray of [n,n,n,n] dimension

    Usage:
    -----
    >>> val = MFOutput(mfdict, path, key)
    >>> return val.data

    User interaction:
    -----------------
    >>> data[('flow15','CBC','FLOW RIGHT FACE')][:,0,1,:]
    or
    >>> data[('flow15','CBC','FLOW RIGHT FACE')]
    """

    def __init__(self, mfdict, path, key):
        self.mfdict = mfdict
        data = MFOutputRequester(mfdict, path, key)
        try:
            self.data = data.querybinarydata
        except AttributeError:
            self.data = np.array([[[[]]]])

    def __iter__(self):
        yield self.data

    def __getitem__(self, index):
        self.data = self.data[index]
        return self.data


class MFOutputRequester:
    """
    MFOutputRequest class is a helper function to enable the user to query
    binary data from the SimulationDict() object on the fly without
    actually storing it in the SimulationDict() object.

    Parameters:
    ----------
    mfdict: OrderedDict
        local instance of the SimulationDict() object
    path:
        pointer to the MFSimulationPath object
    key: tuple
        user requested data key

    Methods:
    -------
    MFOutputRequester.querybinarydata
        returns: Xarray object

    Examples:
    --------
    >>> data = MFOutputRequester(mfdict, path, key)
    >>> data.querybinarydata
    """

    def __init__(self, mfdict, path, key):
        self.path = path
        self.mfdict = mfdict
        self.dataDict = {}
        # get the binary file locations, create a dictionary key to look them
        # up from, store in self.dataDict
        self._getbinaryfilepaths()

        # check if supplied key exists, and model grid type
        if key in self.dataDict:
            if (key[0], "disv", "dimensions", "nvert") in self.mfdict:
                self.querybinarydata = self._querybinarydata_vertices(
                    self.mfdict, key
                )
            elif (key[0], "disu", "connectiondata", "iac") in self.mfdict:
                self.querybinarydata = self._querybinarydata_unstructured(key)
            else:
                self.querybinarydata = self._querybinarydata(key)
        elif key == ("model", "HDS", "IamAdummy"):
            pass
        else:
            print("\nValid Keys Are:\n")
            for valid_key in self.dataDict:
                print(valid_key)
            raise KeyError("Invalid key {}".format(key))

    def _querybinarydata(self, key):
        # Basic definition to get output from modflow binary files for
        # simulations using a structured grid
        path = self.dataDict[key]
        bintype = key[1]

        bindata = self._get_binary_file_object(path, bintype, key)

        if bintype == "CBC":
            try:
                return np.array(bindata.get_data(text=key[-1], full3D=True))
            except ValueError:
                # imeth == 6
                return np.array(bindata.get_data(text=key[-1], full3D=False))
        else:
            return np.array(bindata.get_alldata())

    def _querybinarydata_vertices(self, mfdict, key):
        # Basic definition to get output data from binary output files for
        # simulations that define grid by vertices
        path = self.dataDict[key]
        bintype = key[1]

        bindata = self._get_binary_file_object(path, bintype, key)

        if bintype == "CBC":
            if key[-1] == "FLOW-JA-FACE":
                data = np.array(bindata.get_data(text=key[-1]))
                # uncomment line to remove extra dimensions from data
                # data data.shape = (len(times), -1)
                return data

            else:
                try:
                    data = np.array(
                        bindata.get_data(text=key[-1], full3D=True)
                    )
                except ValueError:
                    # imeth == 6
                    data = np.array(
                        bindata.get_data(text=key[-1], full3D=False)
                    )
        else:
            data = np.array(bindata.get_alldata())

        # uncomment line to remove extra dimensions from data
        # data = _reshape_binary_data(data, 'V')
        return data

    def _querybinarydata_unstructured(self, key):
        # get unstructured binary data in numpy array format.
        path = self.dataDict[key]
        bintype = key[1]

        bindata = self._get_binary_file_object(path, bintype, key)

        if bintype == "CBC":
            try:
                data = np.array(bindata.get_data(text=key[-1], full3D=True))
            except ValueError:
                data = np.array(bindata.get_data(text=key[-1], full3D=False))
        else:
            data = bindata.get_alldata()

        # remove un-needed dimensions
        data = _reshape_binary_data(data, "U")

        if key[-1] == "FLOW-JA-FACE":
            return data

        else:
            return data

    def _get_binary_file_object(self, path, bintype, key):
        # simple method that trys to open the binary file object using Flopy
        if bintype == "CBC":
            try:
                return bf.CellBudgetFile(path, precision="double")
            except AssertionError:
                raise AssertionError(
                    "{} does not " "exist".format(self.dataDict[key])
                )

        elif bintype == "HDS":
            try:
                return bf.HeadFile(path, precision="double")
            except AssertionError:
                raise AssertionError(
                    "{} does not " "exist".format(self.dataDict[key])
                )

        elif bintype == "DDN":
            try:
                return bf.HeadFile(path, text="drawdown", precision="double")
            except AssertionError:
                raise AssertionError(
                    "{} does not " "exist".format(self.dataDict[key])
                )

        elif bintype == "UCN":
            try:
                return bf.UcnFile(path, precision="single")
            except AssertionError:
                raise AssertionError(
                    "{} does not " "exist".format(self.dataDict[key])
                )

        else:
            raise AssertionError()

    @staticmethod
    def _get_vertices(mfdict, key):
        """
        Depreciated! Consider removing from code.

        Parameters
        ----------
        key: binary query dictionary key

        Returns
        -------
        information defining specified vertices for all model cells to be added
        to xarray as coordinates.
        cellid: (list) corresponds to the modflow CELL2d cell number
        xcyc: (n x 2) dimensional Pandas object of tuples defining the CELL2d
        center coordinates
        nverts: (list) number of xy vertices corresponding to a cell
        xv: (n x nverts) dimensional Pandas object of tuples. Contains x
        vertices for a cell
        yv: (n x nverts) dimensional Pandas object of tuples. Contains y
        vertices for a cell
        topv: (n x nlayers) dimensional Pandas object of cell top elevations
        corresponding to a row column location
        botmv: (n x nlayers) dimensional Pandas object of cell bottom
        elevations corresponding to a row column location
        """

        try:
            import pandas as pd
        except Exception as e:
            msg = "MFOutputRequester._get_vertices(): requires pandas"
            raise ImportError(msg)

        mname = key[0]
        cellid = mfdict[(mname, "DISV8", "CELL2D", "cell2d_num")]

        cellxc = mfdict[(mname, "DISV8", "CELL2D", "xc")]
        cellyc = mfdict[(mname, "DISV8", "CELL2D", "yc")]
        xcyc = [(cellxc[i], cellyc[i]) for i in range(len(cellxc))]
        xcyc = pd.Series(xcyc, dtype="object")

        nverts = mfdict[(mname, "DISV8", "CELL2D", "nvert")]
        vertnums = mfdict[(mname, "DISV8", "CELL2D", "iv")]
        vertid = mfdict[(mname, "DISV8", "VERTICES", "vert_num")]
        vertx = mfdict[(mname, "DISV8", "VERTICES", "x")]
        verty = mfdict[(mname, "DISV8", "VERTICES", "y")]
        # get vertices that correspond to CellID list
        xv = []
        yv = []
        for line in vertnums:
            tempx = []
            tempy = []
            for vert in line:
                idx = vertid.index(vert)
                tempx.append(vertx[idx])
                tempy.append(verty[idx])
            xv.append(tempx)
            yv.append(tempy)
        xv = pd.Series(xv, dtype="object")
        yv = pd.Series(yv, dtype="object")

        top = np.array(mfdict[(mname, "DISV8", "CELLDATA", "top")])
        botm = np.array(mfdict[(mname, "DISV8", "CELLDATA", "botm")])
        top = top.tolist()
        botm = botm.tolist()
        # get cell top and bottom by layer
        topv = list(zip(top, *botm[:-1]))
        botmv = list(zip(*botm))
        topv = pd.Series(topv, dtype="object")
        botmv = pd.Series(botmv, dtype="object")

        return cellid, xcyc, nverts, xv, yv, topv, botmv

    def _getbinaryfilepaths(self):
        # model paths
        self.modelpathdict = {}
        for i in self.path.model_relative_path:
            self.modelpathdict[i] = self.path.get_model_path(i)
        sim_path = self.path.get_sim_path()
        self.binarypathdict = {}
        # check output control to see if a binary file is supposed to exist.
        # Get path to that file
        for i in self.modelpathdict:
            if (i, "oc", "options", "budget_filerecord") in self.mfdict:
                cbc = self.mfdict[(i, "oc", "options", "budget_filerecord")]
                if cbc.get_data() is not None:
                    self.binarypathdict[(i, "CBC")] = os.path.join(
                        sim_path, cbc.get_data()[0][0]
                    )

            if (i, "oc", "options", "head_filerecord") in self.mfdict:
                hds = self.mfdict[(i, "oc", "options", "head_filerecord")]
                if hds.get_data() is not None:
                    self.binarypathdict[(i, "HDS")] = os.path.join(
                        sim_path, hds.get_data()[0][0]
                    )

            if (i, "oc", "options", "drawdown_filerecord") in self.mfdict:
                ddn = self.mfdict[(i, "oc", "options", "drawdown_filerecord")]
                if ddn.get_data() is not None:
                    self.binarypathdict[(i, "DDN")] = os.path.join(
                        sim_path, ddn.get_data()[0][0]
                    )

        self._setbinarykeys(self.binarypathdict)

    def _setbinarykeys(self, binarypathdict):
        # check that if a binary file is supposed to exist, it does, and create
        # a dictionary key to access that data
        for key in binarypathdict:
            path = binarypathdict[key]
            if key[1] == "CBC":
                try:
                    readcbc = bf.CellBudgetFile(path, precision="double")
                    for record in readcbc.get_unique_record_names():
                        name = record.decode("utf-8").strip(" ")
                        # store keys along with model name in ordered dict?
                        self.dataDict[(key[0], key[1], name)] = path
                    readcbc.close()

                except:
                    pass

            elif key[1] == "HDS":
                try:
                    readhead = bf.HeadFile(path, precision="double")
                    self.dataDict[(key[0], key[1], "HEAD")] = path
                    readhead.close()

                except:
                    pass

            elif key[1] == "DDN":
                try:
                    readddn = bf.HeadFile(
                        path, text="drawdown", precision="double"
                    )
                    self.dataDict[(key[0], key[1], "DRAWDOWN")] = path
                    readddn.close()

                except:
                    pass

            elif key[1] == "UCN":
                try:
                    readucn = bf.UcnFile(path, precision="single")
                    self.dataDict[(key[0], key[1], "CONCENTRATION")] = path
                    readucn.close()

                except:
                    pass

            else:
                pass

    @staticmethod
    def getkeys(mfdict, path, print_keys=True):
        # use a dummy key to get valid binary output keys
        dummy_key = ("model", "HDS", "IamAdummy")
        x = MFOutputRequester(mfdict, path, dummy_key)
        keys = [i for i in x.dataDict]
        if print_keys is True:
            for key in keys:
                print(key)
        return x


def _reshape_binary_data(data, dtype=None):
    # removes unnecessary dimensions from data returned by
    # flopy.utils.binaryfile
    time = len(data)
    data = np.array(data)
    if dtype is None:
        return data
    elif dtype == "V":
        nodes = len(data[0][0][0])
        data.shape = (time, -1, nodes)
    elif dtype == "U":
        data.shape = (time, -1)
    else:
        err = "Invalid dtype flag supplied, valid are dtype='U', dtype='V'"
        raise Exception(err)
    return data
