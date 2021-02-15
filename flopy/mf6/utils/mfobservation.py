import numpy as np
import csv


def try_float(data):
    try:
        data = float(data)
    except ValueError:
        pass
    return data


class MFObservation:
    """
    Wrapper class to request the MFObservation object:
    Class is called by the MFSimulation.SimulationDict() class and is not
    called by the user

    Inputs:
    -------
    mfdict: (dict) the sim.simulation_dict.mfdict object for the flopy project
    path: (object) the path object detailing model names and paths
    key: (tuple, stings) user supplied dictionary key to request observation
    utility data

    Returns:
    --------\
    self.data: (xarray) array of observations
    """

    def __init__(self, mfdict, path, key):
        self.mfdict = mfdict
        data = MFObservationRequester(mfdict, path, key)
        try:
            self.data = data.query_observation_data
        except AttributeError:
            self.data = np.array([[]])

    def __iter__(self):
        yield self.data

    def __getitem__(self, index):
        self.data = self.data[index]
        return self.data


class Observations:
    """
    Simple class to extract and view Observation files for Uzf models
    (possibly all obs/hobs)?

    Input:
    ------
    fi = (sting) name of the observation binary output file

    Methods:
    --------
    get_data(): (np.array) returns array of observation data
        parameters:
        -----------
        text = (str) specific modflow record name contained in Obs.out file
        idx = (int), (slice(start, stop)) integer or slice of data to be
        returned. corresponds to kstp*kper - 1
        totim = (float) model time value to return data from

    list_records(): prints a list of all valid record names contained within
    the Obs.out file
    get_times(): (list) returns list of time values contained in Obs.out
    get_nrecords(): (int) returns number of records
    get_ntimes(): (int) returns number of times
    get_nobs(): (int) returns total number of observations (ntimes * nrecords)

    """

    def __init__(self, fi):
        self.Obsname = fi

    def _reader(self, fi):
        # observation file reader is a standard csv reader that we try to
        # convert each entry to floating point
        with open(fi) as f:
            reader = csv.reader(f)
            data = [[try_float(point) for point in line] for line in reader]
        return np.array(data)

    def _array_to_dict(self, data, key=None):
        # convert np.array to dictionary of observation names and data
        data = data.T
        data = {
            line[0]: [try_float(point) for point in line[1:]] for line in data
        }
        if key is None:
            return data
        else:
            return data[key]

    def list_records(self):
        # requester option to list all records (observation names) within an
        # observation file
        data_str = self._reader(self.Obsname)
        data = self._array_to_dict(data_str)
        for key in data:
            print(key)

    def get_data(self, key=None, idx=None, totim=None):
        """
        Method to request and return array of data from an Observation
        output file

        Parameters
        ----------
        key: (str) dictionary key for a specific observation contained within
                   the observation file (optional)
        idx: (int) time index (optional)
        totim: (float) simulation time (optional)

        Returns
        -------
        data: (list) observation file data in list
        """
        data = self._reader(self.Obsname)

        # check if user supplied observation key, default is to return
        # all observations
        if key is None:
            header = data[0]
            if idx is not None:
                data = data[idx, :]
            elif totim is not None:
                try:
                    times = self.get_times()
                    idx = times.index(totim)
                    data = data[idx, :]
                except ValueError:
                    err = (
                        "Invalid totim value provided: obs.get_times() "
                        "returns a list of valid times for totim = <>"
                    )
                    raise ValueError(err)
            else:
                pass

        else:
            data = self._array_to_dict(data, key)
            if idx is not None:
                data = data[idx]
            elif totim is not None:
                try:
                    times = self.get_times()
                    idx = times.index(totim)
                    data = data[idx]
                except ValueError:
                    err = (
                        "Invalid totim value provided: obs.get_times() "
                        "returns a list of valid times for totim = <>"
                    )
                    raise ValueError(err)
            else:
                pass
        return data

    def get_times(self):
        return self.get_data(key="time")

    def get_nrecords(self):
        data_str = self._reader(self.Obsname)
        return len(self._array_to_dict(data_str))

    def get_ntimes(self):
        return len(self.get_times())

    def get_nobs(self):
        x = self.get_data().shape
        prod = 1
        for i in x:
            prod *= i
        nrecords = self.get_nrecords()
        ntimes = self.get_ntimes()
        nobs = prod - ntimes - nrecords
        return nobs

    def get_dataframe(
        self,
        keys=None,
        idx=None,
        totim=None,
        start_datetime=None,
        timeunit="D",
    ):
        """
        Creates a pandas dataframe object from the observation data, useful
        backend if the user does not like the x-array format!

        Parameters
        ----------
        keys: (string) sting of dictionary/observation keys separated by comma.
              (optional)
        idx: (int) time index location (optional)
        totim: (float) simulation time (optional)
        start_datetime: (string) format is 'dd/mm/yyyy' or
                        'dd/mm/yyyy hh:mm:ss' (optional)
        timeunit: (string) specifies the time unit associated with totim when
                           setting a datetime

        Returns
        -------
        pd.DataFrame

        """
        try:
            import pandas as pd
        except Exception as e:
            print("this feature requires pandas")
            return None

        data_str = self._reader(self.Obsname)
        data = self._array_to_dict(data_str)
        time = data["time"]

        if start_datetime is not None:
            time = self._get_datetime(time, start_datetime, timeunit)
        else:
            pass

        # check to see if user supplied keys, if not get all observations,
        # adjust for time if necessary.
        if keys is None:
            if idx is not None or totim is not None:
                if totim is not None:
                    try:
                        times = self.get_times()
                        idx = times.index(totim)
                    except ValueError:
                        err = (
                            "Invalid totim value provided: obs.get_times() "
                            "returns a list of valid times for totim = <>"
                        )
                        raise ValueError(err)

                # use dictionary comprehension to create a set of pandas series
                # that can be added to a pd.DataFrame
                d = {
                    key: pd.Series(data[key][idx], index=[time[idx]])
                    for key in data
                    if key != "time"
                }
            else:
                d = {
                    key: pd.Series(data[key], index=time)
                    for key in data
                    if key != "time"
                }

        else:
            keys = self._key_list(keys)
            for key in keys:
                if key not in data:
                    raise KeyError(
                        "Supplied data key: {} is not " "valid".format(key)
                    )
                else:
                    pass

            if idx is not None or totim is not None:
                if totim is not None:
                    try:
                        times = self.get_times()
                        idx = times.index(totim)
                    except ValueError:
                        err = (
                            "Invalid totim value provided: obs.get_times() "
                            "returns a list of valid times for totim\
                         = <>"
                        )
                        raise ValueError(err)

                d = {
                    key: pd.Series(data[key][idx], index=[time[idx]])
                    for key in data
                    if key != "time" and key in keys
                }
            else:
                d = {
                    key: pd.Series(data[key], index=time)
                    for key in data
                    if key != "time" and key in keys
                }

        # create dataframe from pd.Series dictionary
        df = pd.DataFrame(d)

        return df

    def _key_list(self, keys):
        # check if user supplied keys is single or multiple, string or list.
        # Return a list of keys.
        key_type = type(keys)
        if key_type is str:
            keys = keys.split(",")
            keys = [key.strip(" ") for key in keys]
        elif key_type is list:
            pass
        else:
            err = (
                "Invalid key type: supply a string of keys separated by , "
                "or a list of keys"
            )
            raise TypeError(err)
        return keys

    def _get_datetime(self, times, start_dt, unit):
        # use to create datetime objects for time in pandas dataFrames
        import datetime as dt

        # check user supplied format of datetime, is it dd/mm/yyyy or
        # dd/mm/yyyy hh:mm:ss?
        if ":" in start_dt:
            date, time = start_dt.split(" ")
            dlist = date.split("/")
            tlist = time.split(":")
        else:
            dlist = start_dt.split("/")
            tlist = [0, 0, 0]

        # parse data from the datetime lists
        try:
            month = int(dlist[0])
            day = int(dlist[1])
            year = int(dlist[2])
            hour = int(tlist[0])
            minute = int(tlist[1])
            second = int(tlist[2])
        except IndexError:
            err = (
                'please supply start_datetime in the format "dd/mm/yyyy '
                'hh:mm:ss" or "dd/mm/yyyy"'
            )
            raise AssertionError(err)

        # create list of datetimes
        t0 = dt.datetime(year, month, day, hour, minute, second)
        if unit == "Y":
            dtlist = [
                dt.datetime(int(year + time), month, day, hour, minute, second)
                for time in times
            ]
        elif unit == "D":
            dtlist = [t0 + dt.timedelta(days=time) for time in times]
        elif unit == "H":
            dtlist = [t0 + dt.timedelta(hours=time) for time in times]
        elif unit == "M":
            dtlist = [t0 + dt.timedelta(minutes=time) for time in times]
        elif unit == "S":
            dtlist = [t0 + dt.timedelta(seconds=time) for time in times]
        else:
            raise TypeError("invalid time unit supplied")

        return dtlist

    def get_obs_data(self, key=None, idx=None, totim=None):
        """
        Method to request observation output data as an x-array
        Parameters
        ----------
        key: (string) dictionary key for a specific observation contained
                      within the observation file (optional)
        idx: (int) time index (optional)
        totim: (float) simulation time (optional)

        Returns
        -------
        xarray.DataArray: (NxN) dimensions are totim, header == keys*
        """
        data = self.get_data(key=key, idx=idx, totim=totim)
        # create x-array coordinates from time and header
        totim = data.T[0][1:].astype(float)
        header = data[0][1:].astype(str)

        # strip time and header off of data
        data = data[1:, 1:].astype(float)

        return data


class MFObservationRequester:
    """
    Wrapper class for MFObservation.Observations. Class checks which
    observation data is available, and creates a dictionary key to access
    the set of observation data from the SimulationDict()
    """

    def __init__(self, mfdict, path, key, **kwargs):
        self.mfdict = mfdict
        self.path = path
        self.obs_dataDict = {}
        # check that observation files exist, create a key and path to them and
        # set to self.obs_dataDict
        self._check_for_observations()

        # check if user supplied dictionary key is valid, or if it is a dummy
        # key for a key request.
        if key in self.obs_dataDict:
            modelpath = path.get_model_path(key[0])
            self.query_observation_data = self._query_observation_data(
                modelpath, key
            )
            return

        elif key == ("model", "OBS8", "IamAdummy"):
            pass

        else:
            err = "{} is not a valid dictionary key\n".format(str(key))
            raise KeyError(err)

    def _query_observation_data(self, modelpath, key):
        # get absolute path for observation data files
        fi = modelpath + self.obs_dataDict[key]
        # request observation data
        Obs = Observations(fi)
        data = Obs.get_obs_data()
        return data

    def _check_for_observations(self):
        """
        Checks all entries of mfdict for the string
        'observation-input-filenames', finds path to file, creates
        dictionary key to access observation output data.

        Returns
        -------
        sets key: path to self.Obs_dataDict{}

        """
        possible_observations = [
            k
            for k in self.mfdict
            if "observation-input-filename" in k and "FORMAT" not in k
        ]
        partial_key = []
        for k in possible_observations:
            if self.mfdict[k] is not None:
                partial_key.append([k[0], k[1]])

        # check if there are multiple OBS8 files associated with this project
        for line in partial_key:
            check = partial_key.count(line)
            if check > 1:
                multi_observations = [i for i in partial_key if i == line]
                for i in range(len(multi_observations)):
                    obs8_file = "OBS8_{}".format(i + 1)
                    # check for single observations, continuous observations
                    self._get_obsfile_names(
                        multi_observations[i], obs8_file, "SINGLE"
                    )
                    self._get_obsfile_names(
                        multi_observations[i], obs8_file, "CONTINUOUS"
                    )

            elif check <= 1:
                for i in range(len(partial_key)):
                    self._get_obsfile_names(partial_key[i], "OBS8", "SINGLE")
                    self._get_obsfile_names(
                        partial_key[i], "OBS8", "CONTINUOUS"
                    )

            else:
                raise KeyError(
                    "There are no observation files associated "
                    "with this project"
                )

    def _get_obsfile_names(self, partial_key, OBS8, obstype):
        """
        Creates a data dictionary key for user to request data. This key holds
        the path to the observation file

        Parameters
        ----------
        partial_key: (list) partial dictionary key
        OBS8: (string) OBS8 mfdict key name
        obstype: (string) SINGLE or CONTINUOUS

        Returns:
        --------
         sets key: path to self.obs_dataDict

        """
        try:
            obstypes = self.mfdict[
                (partial_key[0], partial_key[1], OBS8, obstype, "obstype")
            ]
            obspackage = self._get_package_type(obstypes)
            obs_fname = self.mfdict[
                (
                    partial_key[0],
                    partial_key[1],
                    OBS8,
                    obstype,
                    "obs_output_file_name",
                )
            ]
            self.obs_dataDict[
                (partial_key[0], obspackage, obstype, "Observations")
            ] = obs_fname
        except KeyError:
            pass

    def _get_package_type(self, obstypes):
        # check the observation name in the OBS8 dictionary to get the
        # package type
        valid_packages = (
            "CHD",
            "DRN",
            "GHB",
            "GWF",
            "LAK",
            "MAW",
            "RIV",
            "SFR",
            "UZF",
            "WEL",
        )
        valid_gwf = ("head", "drawdown", "intercell-flow")
        package = obstypes[0][:3].upper()
        model = obstypes[0]

        if package in valid_packages:
            return package

        elif model in valid_gwf:
            return "GWF"

        else:
            raise KeyError(
                "{} is not a valid observation " "type".format(package)
            )

    @staticmethod
    def getkeys(mfdict, path):
        # staticmethod to return a valid set of mfdict keys to the user to
        # access this data
        key = ("model", "OBS8", "IamAdummy")
        x = MFObservationRequester(mfdict, path, key)
        for key in x.obs_dataDict:
            print(key)
