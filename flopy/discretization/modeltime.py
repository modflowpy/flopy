import calendar
import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


class ModelTime:
    """
    Class for MODFLOW simulation time

    Parameters
    ----------
    perlen : list, np.ndarray
        list or numpy array of stress period lengths
    nstp : list, np.ndarray
        list or numpy array of number of time steps per stress period
    tsmult : list, np.ndarray
        list or numpy array of time-step mult infomation
    time_units : int or str
        string or pre-mf6 integer representation (ITMUNI) of time units
    start_datetime : various objects
        user-supplied starting datetime representation. Please see the
        ModelTime.parse_datetime documentation for a list
        of the supported representation types
    steady_state : list, np.ndarray
        optional list or numpy array of boolean flags that indicate if
        stress periods are steady-state or transient
    """

    def __init__(
        self,
        perlen,
        nstp,
        tsmult=None,
        time_units=None,
        start_datetime=None,
        steady_state=None,
    ):
        nrecs = len(perlen)
        if tsmult is None:
            tsmult = np.full((nrecs,), 1)

        if nrecs != len(nstp):
            raise AssertionError()

        if len(tsmult) != len(nstp):
            raise AssertionError

        self._perlen = perlen
        self._nstp = nstp
        self._tsmult = tsmult
        self._time_units = self.timeunits_from_user_input(time_units)
        self._start_datetime = self.parse_datetime(start_datetime)
        self._steady_state = steady_state
        self._totim_dict = {}
        self.__str_format = "%Y-%m-%dt%H:%M:%S"

    @property
    def time_units(self):
        """
        Returns a normalized string representation of the time units
        """
        return self._time_units

    @property
    def start_datetime(self):
        """
        Returns a datetime.datetime object of the model start time
        """
        return self._start_datetime

    @property
    def perlen(self):
        """
        Returns a list or array of stress period lengths

        """
        return self._perlen.copy()

    @property
    def nper(self):
        """
        Returns the number of stress periods
        """
        return len(self._perlen)

    @property
    def nstp(self):
        """
        Returns a list or array of number of time steps per stress period
        """
        return self._nstp.copy()

    @property
    def tsmult(self):
        """
        Returns the time step multiplier value for each stress period

        """
        return self._tsmult.copy()

    @property
    def perioddata(self):
        """
        Returns a tuple of period data for the MF6 TDIS package containing records
        of [(perlen, nstp, tsmult), ....] for each stress period
        """
        return [(per, self.nstp[ix], self.tsmult[ix]) for ix, per in self.perlen]

    @property
    def steady_state(self):
        """
        Returns a boolean that indicates either steady-state or transient stress period

        """
        return self._steady_state

    @property
    def totim(self):
        """
        Returns a list of totim values at the end of each time step

        """
        if not self._totim_dict:
            self._set_totim_dict()

        return list(self._totim_dict.values())

    @property
    def kper_kstp(self):
        """
        Returns a list of kper, kstp tuples for all time steps

        """
        if not self._totim_dict:
            self._set_totim_dict()
        return list(self._totim_dict.keys())

    @property
    def tslen(self):
        """
        Method to get a list of time step lengths for all time steps

        """
        n = 0
        tslen = []
        totim = self.totim
        for ix, stp in enumerate(self.nstp):
            for i in range(stp):
                if not tslen:
                    tslen = [totim[n]]
                else:
                    tslen.append(totim[n] - totim[n - 1])
                n += 1

        return np.array(tslen)

    @staticmethod
    def get_datetime_string(datetime_obj):
        """
        Method to get a standarized ISO 8601 compliant datetime string

        Parameters
        ----------
        datetime_obj : various objects
            user-supplied datetime representation. Please see the
            ModelTime.parse_datetime documentation for a list
            of the supported representation types

        """
        dt = ModelTime.parse_datetime(datetime_obj)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def set_start_datetime(self, datetime_obj):
        """
        Method to reset the start datetime of the ModelTime class

        Parameters
        ----------
        datetime_obj : various objects
            user-supplied datetime representation. Please see the
            ModelTime.parse_datetime documentation for a list
            of the supported representation types

        """
        start_dt = self.parse_datetime(datetime_obj)
        self._start_datetime = start_dt

    def set_units(self, units):
        """
        Method to reset the time units of the ModelTime class

        Parameters
        ----------
        units : str or int
            string or pre-mf6 integer representation (ITMUNI) of time units

        """
        units = self.timeunits_from_user_input(units)
        self._time_units = units

    def _set_totim_dict(self):
        """
        Method to setup a dictionary of (kper, kstp): totim that is used
        by multiple methods

        Returns
        -------
            None
        """
        delt = []
        per_stp = []
        perlen_array = self.perlen
        nstp_array = self.nstp
        tsmult_array = self.tsmult
        for per, nstp in enumerate(nstp_array):
            perlen = perlen_array[per]
            tsmult = tsmult_array[per]
            for stp in range(nstp):
                if stp == 0:
                    if tsmult != 1.0:
                        dt = perlen * (tsmult - 1) / ((tsmult**nstp) - 1)
                    else:
                        dt = perlen / nstp
                else:
                    dt = delt[-1] * tsmult
                delt.append(dt)
                per_stp.append((per, stp))

        totim = np.add.accumulate(delt)
        self._totim_dict = {ps: totim[i] for i, ps in enumerate(per_stp)}

    @staticmethod
    def timeunits_from_user_input(units):
        """
        Method to get a normalized time unit string from user input. User
        input can be either a string representation or ITMUNI integer. String
        representations use "sequence scoring" to fuzzy match to the normalized
        time unit.

        Parameters
        ----------
        units: str or int
            string or pre-mf6 integer representation (ITMUNI) of time units

        Returns
        -------
            str: standardized unit string
        """
        if units is None:
            units = 0

        valid_units = {
            0: "unknown",
            1: "seconds",
            2: "minutes",
            3: "hours",
            4: "days",
            5: "years",
        }
        valid_units_list = list(valid_units.values())
        valid_unit = None

        if isinstance(units, int):
            # map to pre-mf6 conventions
            if 0 <= units <= 5:
                valid_unit = valid_units[units]
            else:
                raise ValueError("Integer units should be between 0 - 5")
        else:
            units = units.lower()
            if len(units) == 1:
                for vu in valid_units_list:
                    if vu.startswith(units):
                        valid_unit = vu
                        break
            else:
                scores = []
                for vu in valid_units_list:
                    score = SequenceMatcher(None, vu, units).ratio()
                    scores.append(score)

                uidx = scores.index(max(scores))
                valid_unit = valid_units_list[uidx]

        if valid_unit is None:
            raise ValueError(f"Could not determine time units from user input {units}")

        return valid_unit

    @staticmethod
    def _get_datetime_string_format(str_datetime):
        """
        Method to parse a limited number string representations of datetime
        formats. Currently supported string formats for date time combinations
        are....

        Parameters
        ----------
        str_datetime : str
            string representation of date time. See the
            ModelTime.parse_datetime documentation for supported
            formats

        Returns
        -------
            datetime.datetime object
        """
        str_datetime = str_datetime.strip().lower()
        if "/" in str_datetime:
            dsep = "/"
        elif "-" in str_datetime:
            dsep = "-"
        else:
            raise ValueError(
                "Seperator type for date part of date time representation "
                "not recognized, supported date seperator types include '/' "
                "and '-'"
            )

        # check for time component
        if "t" in str_datetime:
            dtsep = "t"
        elif " " in str_datetime:
            dtsep = " "
        else:
            dtsep = None

        # check if year first (yr, month, day) combo...
        year_first = False
        tmp = str_datetime.split(dsep)[0]
        if len(tmp) == 4:
            year_first = True

        if dtsep is not None:
            if year_first:
                str_rep = f"%Y{dsep}%m{dsep}%d{dtsep}%H:%M:%S"
            else:
                str_rep = f"%m{dsep}%d{dsep}%Y{dtsep}%H:%M:%S"

        else:
            if year_first:
                str_rep = f"%Y{dsep}%m{dsep}%d"
            else:
                str_rep = f"%m{dsep}%d{dsep}%Y"

        return str_rep

    @staticmethod
    def parse_datetime(datetime_obj):
        """
        Method to create a datetime.datetime object from a variety of user
        inputs including the following:

        datetime.datetime objects
        numpy.datetime64 objects
        pandas.Timestamp objects
        string objects

        Supported formats for string objects representing November 12th, 2024
        are as follows:

        '11/12/2024'
        '11-12-2024'
        '2024/11/12'
        '2024-11-12'

        Time can also be represented in the string object. Example formats
        representing 2:31 pm on November 12th, 2024 are as follows:

        '2024-11-12T14:31:00'
        '2024/11/12T14:31:00'
        '11-12-2024t14:31:00'
        '11/12/2024t14:31:00'
        '2024-11-12 14:31:00'
        '2024/11/12 14:31:00'
        '11-12-2024 14:31:00'
        '11/12/2024 14:31:00'

        Parameters
        ----------
        datetime_obj : various formats
            a user-supplied representation of date or datetime

        Returns
        -------
            datetime.datetime object
        """
        if datetime_obj is None:
            datetime_obj = datetime.datetime(1970, 1, 1)  # unix time zero
        elif isinstance(datetime_obj, np.datetime64):
            unix_time_0 = datetime.datetime(1970, 1, 1)
            ts = (datetime_obj - np.datetime64(unix_time_0)) / np.timedelta64(1, "s")
            datetime_obj = datetime.datetime.utcfromtimestamp(ts)
        elif isinstance(datetime_obj, pd.Timestamp):
            datetime_obj = datetime_obj.to_pydatetime()
        elif isinstance(datetime_obj, datetime.datetime):
            pass
        elif isinstance(datetime_obj, str):
            str_rep = ModelTime._get_datetime_string_format(datetime_obj)
            datetime_obj = datetime.datetime.strptime(datetime_obj, str_rep)

        else:
            raise NotImplementedError(
                f"{type(datetime_obj)} date representations "
                f"are not currently supported"
            )

        return datetime_obj

    def get_totim(self, kper, kstp=None):
        """
        Method to get the total simulation time at the end of a given
        stress period or stress period and time step combination

        Parameters
        ----------
        kper : int
            zero based stress period number
        kstp : int or None
            optional zero based time-step number

        Returns
        -------
            totim : float
        """
        if kstp is None:
            kstp = self.nstp[kper] - 1

        if not self._totim_dict:
            self._set_totim_dict()

        if (kper, kstp) not in self._totim_dict:
            raise KeyError(
                f"(kper, kstp): ({kper} {kstp}) not a valid combination of "
                f"stress period and time step"
            )

        return self._totim_dict[(kper, kstp)]

    def get_datetime(self, kper, kstp=None):
        """
        Method to get the datetime at the end of a given stress period or
        stress period and time step combination

        Parameters
        ----------
        kper : int
            zero based modflow stress period number
        kstp : int
            zero based time-step number

        Returns
        -------
            datetime.datetime object
        """
        if self.time_units == "unknown":
            raise AssertionError(
                "time units must be set in order to calculate datetime"
            )

        totim = self.get_totim(kper=kper, kstp=kstp)

        if self.time_units == "years":
            ndays = 365
            years = np.floor(totim)
            year = self.start_datetime.year + years
            if self.start_datetime.month > 2:
                isleap = calendar.isleap(year + 1)
            else:
                isleap = calendar.isleap(year)

            if isleap:
                ndays = 366

            days = ndays * (totim - years)
            day_td = datetime.timedelta(days=days)

            dt = datetime.datetime(
                year,
                self.start_datetime.month,
                self.start_datetime.day,
                self.start_datetime.hour,
                self.start_datetime.minute,
                self.start_datetime.second,
            )

            dt += day_td

        else:
            kwargs = {self.time_units: totim}
            dt = self.start_datetime + datetime.timedelta(**kwargs)

        return dt

    def intersect(self, datetime_obj=None, totim=None, forgive=False):
        """
        Method to intersect a datetime or totim value with the model and
        get the model stress period and optional time-step associated with that
        time.

        Parameters
        ----------
        datetime_obj : various objects
            user-supplied starting datetime representation. Please see the
            ModelTime.parse_datetime documentation for a list
            of the supported representation types
        totim : float
            optional total time elapsed from the beginning of the model
        forgive : bool
            optional flag to forgive time intersections that are outside of
            the model time domain. Default is False

        Returns
        -------
            tuple: (kper, kstp)
        """
        if datetime_obj is not None:
            datetime_obj = self.parse_datetime(datetime_obj)
            timedelta = datetime_obj - self.start_datetime

            if self.time_units == "unknown":
                raise AssertionError(
                    "time units must be set in order to intersect datetime "
                    "objects, set time units or use totim for intersection"
                )

            elif self.time_units == "days":
                totim = timedelta.days

            elif self.time_units in ("hours", "minutes", "seconds"):
                totim = timedelta.total_seconds()
                if self.time_units == "minutes":
                    totim /= 60
                elif self.time_units == "hours":
                    totim /= 3600

            else:
                # years condition
                totim = datetime_obj.year - self.start_datetime.year

                # get the remainder for the current year
                ndays = 365
                if calendar.isleap(datetime_obj.year):
                    ndays = 365

                dt_iyear = datetime.datetime(
                    datetime_obj.year,
                    self.start_datetime.month,
                    self.start_datetime.day,
                    self.start_datetime.hour,
                    self.start_datetime.minute,
                    self.start_datetime.second,
                )

                timedelta = datetime_obj - dt_iyear
                days = timedelta.days
                yr_frac = days / ndays
                totim += yr_frac

        elif totim is not None:
            pass

        else:
            raise AssertionError(
                "A date-time representation or totim needs to be provided"
            )

        if totim > self.totim[-1] or totim <= 0:
            if forgive:
                return None
            if datetime_obj is None:
                msg = (
                    f"supplied totim {totim} is outside of model's "
                    f"time domain 0 - {self.totim[-1]}"
                )
            else:
                end_dt = self.get_datetime(self.nper - 1, self.nstp[-1] - 1)
                msg = (
                    f"supplied datetime"
                    f" {datetime_obj.strftime(self.__str_format)} is "
                    f"outside of the model's time domain "
                    f"{self.start_datetime.strftime(self.__str_format)} - "
                    f"{end_dt}"
                )
            raise ValueError(msg)

        idx = sorted(np.where(np.array(self.totim) >= totim)[0])[0]
        per, stp = self.kper_kstp[idx]

        return per, stp
