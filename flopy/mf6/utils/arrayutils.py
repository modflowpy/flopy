# package containing array reshaping utilities to assist in plotting and creating shapefiles.
import numpy as np
# from flopy6.modflow.mfdata import MFArray, MFList, MFScalar
# todo: set a single import statement for MFArray, etc. after input is finished
# from flopy6.utils.StructTestData.StructDataTest import MFArray, MFScalar, MFList, MFTransientArray, MFTransientList
from ..utils.VertexTestData.VertexDataTest import MFArray, MFScalar, MFList, MFTransientArray, MFTransientList


class StructuredArray(object):
    """
    Array utility class that reshapes data into three dimensional array format that can be passed to
    ModelMap for plotting. For Structured grids [DIS8]

    Parameters
    ----------
        dataset: (object) MFArray, MFList, or MFScalar type
        key: (str) key value of data requested for to shape into 3d array
        sr: (SpatialReference) spatial reference class of modflow model

    """
    def __init__(self, dataset, key, sr):
        # todo: replace local imports with a single global import after input code is working
        from ..utils.StructTestData.StructDataTest import MFArray, \
            MFScalar, MFList, MFTransientArray, MFTransientList
        self.sr = sr
        self.nlay = sr.nlay
        self.nrow = sr.nrow
        self.ncol = sr.ncol
        self.array3d = np.zeros((sr.nlay, sr.nrow, sr.ncol))

        try:
            self.key = key.lower()
        except AttributeError:
            self.key = key

        if isinstance(dataset, AdvancedPackageUtil):
            if type(dataset.data) != dict:
                self.mflist_to_numpy(dataset)
            else:
                self.mftransientlist_to_numpy(dataset)

        elif isinstance(dataset, MFList):
            self.mflist_to_numpy(dataset)

        elif isinstance(dataset, MFTransientList):
            self.mftransientlist_to_numpy(dataset)

        elif isinstance(dataset, MFScalar):
            self.mfscalar_to_numpy(dataset)

        elif isinstance(dataset, MFArray):
            self.mfarray_to_numpy(dataset)

        elif isinstance(dataset, MFTransientArray):
            self.mfarray_to_numpy(dataset)

    def mflist_to_numpy(self, mflist):
        """
        Method to create a 3d array for plotting from a MFArray object
        """
        # todo: this call will likely change from mfarray.data to something else.
        mflist = mflist.data
        try:
            # todo: check if mflist objects return zero based indicies or raw data
            # todo: currently set up for raw data, remove <-1> for zero based indicies
            k = mflist['layer'] - 1
            i = mflist['row'] - 1
            j = mflist['column'] - 1

        except ValueError:
            try:
                k = mflist['cellid'][:, 0] - 1
                i = mflist['cellid'][:, 1] - 1
                j = mflist['cellid'][:, 2] - 1
            except ValueError:
                raise KeyError('cellid information not provided for this transient array')

        self.array3d[k, i, j] = np.array(mflist[self.key])

    def mftransientlist_to_numpy(self, mftransientlist):
        """
        Method to create a 3d array for plotting from MFList[kper] object
        """
        # todo: this call will likely change from transientlist.data to something else.
        # todo: this call will likely change from mfarray.data to something else.
        mftransientlist = mftransientlist.data
        arr3d = []

        for ts, mflist in mftransientlist.items():
            # todo: check if mflist objects return zero based indicies or raw data
            # todo: currently set up for raw data, remove <-1> for zero based indicies
            try:
                arr = np.copy(self.array3d)
                k = mflist['layer'] - 1
                i = mflist['row'] - 1
                j = mflist['column'] - 1
                arr[k, i, j] = np.array(mflist[self.key])
                arr3d.append(arr)
            except ValueError:
                try:
                    arr = np.copy(self.array3d)
                    k = mflist['cellid'][:,0] - 1
                    i = mflist['cellid'][:,1] - 1
                    j = mflist['cellid'][:,2] - 1
                    arr[k, i, j] = np.array(mflist[self.key])
                    arr3d.append(arr)
                except ValueError:
                    raise KeyError('cellid information not provided for this transient array')

        self.array3d = np.array(arr3d)

    def mfscalar_to_numpy(self, mfscalar):
        """
        Method to create a 3d array from a MFScalar object
        """
        # todo: this call will likely change from mfscalar.data to something else.
        for i, j in enumerate(self.array3d):
            self.array3d[i] += mfscalar.data[i]

    def mfarray_to_numpy(self, mfarray):
        """
        Method to create a 3d array from a MFArray readarray object
        """
        # todo: this call will likely change from mfarray.data to something else.
        self.array3d = np.array(mfarray.data)


class VertexArray(object):
    """
    Array utility class that reshapes data into two dimensional array format that can be passed to
    ModelMap for plotting. For Vertex grids [DISV8]

    Parameters
    ----------
        dataset: (object) MFArray, MFList, or MFScalar type
        key: (str) key value of data requested for to shape into 3d array
        sr: (SpatialReference) spatial reference class of modflow model

    """
    def __init__(self, dataset, key, sr):
        # todo: replace local imports with a single global import after input code is working
        from flopy6.utils.VertexTestData.VertexDataTest import MFArray, \
            MFScalar, MFList, MFTransientArray, MFTransientList
        self.sr = sr
        self.nlay = sr.nlay
        self.ncpl = sr.ncpl
        self.key = key
        self.array2d = np.zeros((sr.nlay, sr.ncpl))

        try:
            self.key = key.lower()
        except AttributeError:
            self.key = key

        if isinstance(dataset, AdvancedPackageUtil):
            if type(dataset.data) != dict:
                self.mflist_to_numpy(dataset)
            else:
                self.mftransientlist_to_numpy(dataset)

        elif isinstance(dataset, MFList):
            self.mflist_to_numpy(dataset)

        elif isinstance(dataset, MFTransientList):
            self.mftransientlist_to_numpy(dataset)

        elif isinstance(dataset, MFScalar):
            self.mfscalar_to_numpy(dataset)

        elif isinstance(dataset, MFArray):
            self.mfarray_to_numpy(dataset)

        elif isinstance(dataset, MFTransientArray):
            self.mfarray_to_numpy(dataset)

    def mflist_to_numpy(self, mflist):
        """
        Method to create a 2d array from a MFArray object
        """
        mflist = mflist.data
        # todo: check if mflist objects return zero based indicies or raw data
        # todo: currently set up for raw data, remove <-1> for zero based indicies
        try:
            k = mflist['layer'] - 1
            j = mflist['ncpl'] - 1

        except ValueError:
            try:
                k = mflist['cellid'][:, 0] - 1
                j = mflist['cellid'][:, 1] - 1

            except ValueError:
                raise KeyError('cellid information not provided for this transient array')

        self.array2d[k, j] = np.array(mflist[self.key])

    def mftransientlist_to_numpy(self, mftransientlist):
        """
        Method to create a 2d array for plotting from MFList[kper] object
        """
        # todo: this call will likely change from transientlist.data to something else.
        # todo: this call will likely change from mfarray.data to something else.
        mftransientlist = mftransientlist.data
        arr2d = []

        for ts, mflist in mftransientlist.items():
            # todo: check if mflist objects return zero based indicies or raw data
            # todo: currently set up for raw data, remove <-1> for zero based indicies
            try:
                arr = np.copy(self.array2d)
                k = mflist['layer'] - 1
                j = mflist['ncpl'] - 1
                arr[k, j] = np.array(mflist[self.key])
                arr2d.append(arr)
            except ValueError:
                try:
                    arr = np.copy(self.array2d)
                    k = mflist['cellid'][:, 0] - 1
                    j = mflist['cellid'][:, 1] - 1
                    arr[k, j] = np.array(mflist[self.key])
                    arr2d.append(arr)
                except ValueError:
                    raise KeyError('cellid information not provided for this transient array')

        self.array2d = np.array(arr2d)

    def mfscalar_to_numpy(self, mfscalar):
        """
        Method to create a 2d array for plotting from MFScalar object
        """
        for i, j in enumerate(self.array2d):
            self.array2d[i] += mfscalar.data[i]

    def mfarray_to_numpy(self, mfarray):
        """
        Method to create a 2d array from a MFArray readarray object
        """
        self.array2d = np.array(mfarray.data)


class AdvancedPackageUtil(object):
    """
    Array utility class for advanced package data that converts data to a standard format compatable
    with the rest of the Modflow Arrays.

    Parameters:
    -----------
        package_name: (str) Advanced package name
        mftransientlist: (mflist) MFTransientList type of stress period data for model
        MFList: (array) MFList object containing cellid information from adv. package
        nper: (int) total number of stress periods in model/simulation

    Attributes:
    -----------
        mflist: an mflist object that explicitly lists data for each stress period
    """
    def __init__(self, package_name, mftransientlist, mflist, nper):
        self.mflist = mflist.data
        self.stress_period_data = mftransientlist.data
        self.package_name = package_name
        self.nper = nper
        self.MFList = False
        if isinstance(mftransientlist, MFList):
            self.MFList = True
        self.data = self.check_package_type()

    def check_package_type(self):
        """
        Method to check package type and send it to the proper utility to return a mflist like
        pandas object

        Returns
        -------
            mflist: (pd object) mflist like object
        """
        if self.package_name == 'UZF8':
            mflist = self.uzf_package()
        elif self.package_name == 'LAK8':
            mflist = self.lak_package()
        elif self.package_name == 'MAW8':
            if self.MFList is False:
                mflist = self.maw_package()
            else:
                mflist = self.maw_list()
        elif self.package_name == 'SFR8':
            mflist = self.sfr_package()
        else:
            raise TypeError('Package is not an advanced type package')

        return mflist

    def uzf_package(self):
        """
        Method to associate stress period data from UZF advanced package with its specific cellid.
        Explicitly shows all transient data

        Returns
        -------
            MFTransientList type object
        """

        #  local definition
        def add_cellid(stress_period_data, mflist):
            """
            adds cellid to the mflist containing stress period data based on uzfno

            Inputs:
                stress_period_data: mflist, single timestep of the mftransient list
                mflist: mflist corresponding to cellid and other non-transient package data
            """
            cellid = []
            for record in stress_period_data:
                idx = record['uzfno'] - 1
                try:
                    cellid.append((mflist['layer'][idx], mflist['row'][idx], mflist['column'][idx]))
                    dtype = ('cellid', 'i4', (3))
                except ValueError:
                    try:
                        cellid.append((mflist['layer'][idx], mflist['ncpl'][idx]))
                        dtype = ('cellid', 'i4', (2))
                    except ValueError:
                        try:
                            cellid.append((mflist['ncpl'][idx]))
                            dtype = ('cellid', 'i4')
                        except ValueError:
                            raise KeyError('no cellid information associated with mflist')

            dtypes = np.dtype(stress_period_data.dtype.descr + [dtype])
            nrecords = len(stress_period_data)
            stress_period_data_with_cellid = np.zeros((nrecords,), dtype=dtypes)
            for i, j in enumerate(stress_period_data):
                stress_period_data_with_cellid[i] = tuple(list(j) + [cellid[i]])

            return stress_period_data_with_cellid
        #  End local definition

        data = {}
        mflist = self.mflist
        for iper in range(self.nper):
            if iper == 0:
                try:
                    data[iper] = add_cellid(self.stress_period_data[iper], mflist)
                except KeyError:
                    data[iper] = None

            else:
                try:
                    stress_period_data2 = add_cellid(self.stress_period_data[iper], mflist)
                    stress_period_data = data[iper - 1]

                    stress_period_data2 = self.update_stress_period(stress_period_data,\
                                                                    stress_period_data2)

                    data[iper] = stress_period_data2

                except KeyError:
                    data[iper] = data[iper - 1]
        return data

    def sfr_package(self):
        """
        Method to associate stress period data from SFR advanced package with its specific cellid.
        Explicitly lists all transient data

        Returns
        -------
            MFTransientList like object
        """
        data = {}
        mflist = self.mflist
        for iper in range(self.nper):
            if iper == 0:
                try:
                    stress_period_data = self.pivot_keyarray('rno', self.stress_period_data[iper])
                    data[iper] = self.add_cellid(stress_period_data, mflist, 'rno')

                except KeyError:
                    data[iper] = None

            else:
                try:
                    stress_period_data2 = self.pivot_keyarray('rno', self.stress_period_data[iper])
                    stress_period_data2 = self.add_cellid(stress_period_data2, mflist, 'rno')
                    stress_period_data = data[iper - 1]

                    stress_period_data2 = self.update_stress_period(stress_period_data,\
                                                                    stress_period_data2)

                    data[iper] = stress_period_data2
                except KeyError:
                    data[iper] = data[iper - 1]
        return data

    def maw_package(self):
        """
        Method to associate stress period data from MAW advanced package with its specific cellid.
        Explicitly lists all transient data.

        Returns
        -------
            MFTransientList like object
        """
        data = {}
        mflist = self.mflist
        for iper in range(self.nper):
            if iper == 0:
                try:
                    stress_period_data = self.pivot_keyarray('wellno', self.stress_period_data[iper])
                    data[iper] = self.add_cellid(stress_period_data, mflist, 'wellno')

                except KeyError:
                    data[iper] = None
            else:
                try:
                    stress_period_data2 = self.pivot_keyarray('wellno', self.stress_period_data[iper])
                    stress_period_data2 = self.add_cellid(stress_period_data2, mflist, 'wellno')
                    stress_period_data = data[iper - 1]

                    stress_period_data2 = self.update_stress_period(stress_period_data, \
                                                                    stress_period_data2)

                    data[iper] = stress_period_data2

                except KeyError:
                    data[iper] = data[iper - 1]

        return data

    def lak_package(self):
        """
        Method to associate stress period data from LAK advanced package with its specific cellid.
        Explicity lists all transient data

        Returns
        -------
            MFTransientList like object
        """
        data = {}
        mflist = self.mflist
        for iper in range(self.nper):
            if iper == 0:
                try:
                    stress_period_data = self.pivot_keyarray('lakeno', self.stress_period_data[iper])
                    data[iper] = self.add_cellid(stress_period_data, mflist, 'lakeno')
                except KeyError:
                    data[iper] = None
            else:
                try:
                    stress_period_data2 = self.pivot_keyarray('lakeno', self.stress_period_data[iper])
                    stress_period_data2 = self.add_cellid(stress_period_data2, mflist, 'lakeno')
                    stress_period_data = data[iper - 1]

                    stress_period_data2 = self.update_stress_period(stress_period_data, \
                                                                    stress_period_data2)

                    data[iper] = stress_period_data2
                except KeyError:
                    data[iper] = data[iper - 1]
        return data

    def maw_list(self):
        """
        Method to associate cellid with another mflist object that is not stress period
        data. Developed for the well infromation block of MAW package

        Returns
        -------
            MFTransientList like object
        """

        mflist = self.mflist
        welldata = self.stress_period_data

        return self.add_cellid(welldata, mflist, 'wellno')


    def pivot_keyarray(self, idname, keyarray):
        """
        Method to pivot keyword value data and align all keyword value pairs with its id number.

        Parameters
        ----------
            idname: (str) specific id name associated with package data ex. 'wellid' or 'lakeno'
            keyarray: (pd object) pandas array

        Returns
        -------
            x: (pd object) pivoted pandas object that is oriented in the same fashion as other advanced
                           packages

        """
        try:
            import pandas as pd
        except Exception as e:
            print("this feature requires pandas")
            return None
        keyarray = pd.DataFrame.from_records(keyarray)
        x = keyarray.pivot(index=idname, columns='keyword', values='value')
        x = x.rename(columns=lambda x: x.lower())
        return x.to_records()

    def add_cellid(self, stress_period_data, mflist, idname):
        """
        adds cellid to the mflist containing stress period data based on the idname
        Method compatible with SFR, LAK, and MAW advanced packages
        method not compatible with the UZF package.

        Inputs:
            stress_period_data: mflist, single timestep of the mftransient list
            mflist: mflist corresponding to cellid and other non-transient package data
            idname: string: 'rno', 'lakeno', or 'wellno'
        """
        cellid = []
        stress_period_data_temp = []
        for record in stress_period_data:
            for rec in mflist:
                if record[idname] == rec[idname]:
                    try:
                        cellid.append((rec['layer'], rec['row'], rec['column']))
                        stress_period_data_temp.append(record)
                        dtype = ('cellid', 'i4', (3))
                    except ValueError:
                        try:
                            cellid.append((rec['layer'], rec['ncpl']))
                            stress_period_data_temp.append(record)
                            dtype = ('cellid', 'i4', (2))
                        except ValueError:
                            try:
                                cellid.append(rec['ncpl'])
                                stress_period_data_temp.append(record)
                                dtype = ('cellid', 'i4')
                            except ValueError:
                                raise KeyError('no cellid information associated with mflist')

        dtypes = np.dtype(stress_period_data.dtype.descr + [dtype])
        nrecords = len(stress_period_data_temp)
        stress_period_data_with_cellid = np.zeros((nrecords,), dtype=dtypes)
        for i, j in enumerate(stress_period_data_temp):
            stress_period_data_with_cellid[i] = tuple(list(j) + [cellid[i]])

        return stress_period_data_with_cellid

    def update_stress_period(self, stress_period_data, stress_period_data2):
        """
        Method to create a recarray that explicitly lisst all fluxes within a given
        stress period

        Parameters
        ----------
        stress_period_data: mflist of stress period data from iper - 1
        stress_period_data2: mflist of stress period data from iper

        Returns
        -------
        mflist of explicit stress period data
        """
        if stress_period_data is None:
            return stress_period_data2
        else:
            for record in stress_period_data:
                flag = 0
                for i, row in enumerate(stress_period_data2):
                    if (record['cellid'] == row['cellid']).all():
                        flag = 1
                    else:
                        pass
                if flag == 0:
                    stress_period_data2. \
                        resize(stress_period_data2.size + 1, refcheck=False)
                    stress_period_data2[-1] = record

            return stress_period_data2

