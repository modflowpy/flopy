import calendar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas.api.types import is_scalar


def _maybe_collect(l):
    return l if is_scalar(l) else list(l)


@dataclass
class ModelTime:
    """
    Simulation time discretization.
    """

    _perlen: NDArray[np.floating] = field(init=False)
    _nstp: NDArray[np.integer] = field(init=False)
    _tsmult: NDArray[np.floating] = field(init=False)
    _time_units: int | str = field(init=False)
    _start_datetime: datetime = field(init=False)
    _steady_state: NDArray[np.bool_] | None = field(init=False)

    _totim_dict: dict = field(init=False, repr=False, compare=False)
    _pertim_dict: dict = field(init=False, repr=False, compare=False)
    _datetime_dict: dict = field(init=False, repr=False, compare=False)
    _str_format: str = field(init=False, repr=False, compare=False)

    """

    Parameters
    ----------
    perlen : ArrayLike[float]
        list or numpy array of stress-period lengths
    nstp : ArrayLike[int]
        list or numpy array of number of time-steps per stress period
    tsmult : ArrayLike[float] or None
        list or numpy array of timestep mult information
    time_units : ArrayLike[float] or None
        string or pre-mf6 integer representation (ITMUNI) of time units
    start_datetime : various objects
        user supplied datetime representation. Please see the
        ModelTime.parse_datetime documentation for a list
        of the supported representation types
    steady_state : list, np.ndarray
        optional list or numpy array of boolean flags that identify whether a stress
        period is steady-state or transient
    """

    def __init__(
        self,
        perlen: ArrayLike,
        nstp: ArrayLike,
        tsmult: ArrayLike = None,
        time_units: int | str | None = None,
        start_datetime: str | datetime | np.datetime64 | pd.Timestamp | None = None,
        steady_state: ArrayLike | None = None,
    ):
        self._totim_dict = {}
        self._pertim_dict = {}
        self._datetime_dict = {}
        self._str_format = "%Y-%m-%dT%H:%M:%S"
        self._perlen = np.atleast_1d(_maybe_collect(perlen)).astype(float)
        self._nstp = np.atleast_1d(_maybe_collect(nstp)).astype(int)
        if tsmult is None:
            tsmult = np.full((self._perlen.shape[0],), 1.0)
        self._tsmult = np.atleast_1d(_maybe_collect(tsmult)).astype(float)
        if len(self._perlen) != len(self._nstp) or len(self._perlen) != len(
            self._tsmult
        ):
            raise ValueError("perlen, nstp and tsmult must have the same length")
        if steady_state is None:
            self._steady_state = None
        else:
            self._steady_state = np.atleast_1d(_maybe_collect(steady_state)).astype(
                bool
            )
            if len(self._steady_state) != len(self._perlen):
                raise ValueError(
                    "perlen, nstp, tsmult and steady_state must have the same length"
                )
        self.time_units = time_units
        self.start_datetime = start_datetime

    def __eq__(self, other):
        if not isinstance(other, ModelTime):
            return False

        return (
            np.array_equal(self.perlen, other.perlen)
            and np.array_equal(self.nstp, other.nstp)
            and np.array_equal(self.tsmult, other.tsmult)
            and self.time_units == other.time_units
            and self.start_datetime == other.start_datetime
            and np.array_equal(self.steady_state, other.steady_state)
        )

    @property
    def perlen(self) -> NDArray[np.floating]:
        """Stress period lengths."""
        return self._perlen

    @property
    def nstp(self) -> NDArray[np.integer]:
        """Number of time steps per stress period."""
        return self._nstp

    @property
    def tsmult(self) -> NDArray[np.floating]:
        """ModelTime step multipliers."""
        return self._tsmult

    @property
    def time_units(self) -> str:
        """ModelTime units."""
        return self._time_units

    @time_units.setter
    def time_units(self, value: int | str | None):
        self._time_units = self.parse_timeunits(value)
        self._datetime_dict = {}  # clear cache when time units change

    @property
    def start_datetime(self) -> datetime:
        """Start datetime."""
        return self._start_datetime

    @start_datetime.setter
    def start_datetime(
        self, value: str | datetime | np.datetime64 | pd.Timestamp | None
    ):
        self._start_datetime = ModelTime.parse_datetime(value)
        self._datetime_dict = {}  # clear cache when start datetime changes

    @property
    def steady_state(self) -> NDArray[np.bool_] | None:
        """Steady state flags."""
        return self._steady_state

    @property
    def nper(self) -> int:
        """
        The number of stress periods in the simulation.
        """
        return len(self.perlen)

    @property
    def perioddata(self) -> list[tuple[float, int, float]]:
        """
        Period data for the MF6 TDIS package. Returns a list of
        tuples (perlen, nstp, tsmult), 1 for each stress period.
        """
        return [
            (per, self.nstp[ix], self.tsmult[ix]) for ix, per in enumerate(self.perlen)
        ]

    @property
    def pertim(self) -> list[float]:
        """
        Period time corresponding to the end of each time-step. This is the time
        elapsed since the start of stress period.
        """
        if not self._totim_dict:
            self._set_totim_dict()

        return list(self._pertim_dict.values())

    @property
    def totim(self) -> list[float]:
        """
        Simulation time corresponding to the end of each time-step. This is the
        time elapsed since the start of the simulation.
        """
        if not self._totim_dict:
            self._set_totim_dict()

        return list(self._totim_dict.values())

    @property
    def datetimes(self) -> list[datetime]:
        """
        Timestamp corresponding to the end of each time-step.
        """

        if not self._datetime_dict:
            self._set_datetime_dict()

        return list(self._datetime_dict.values())

    @property
    def kper_kstp(self) -> list[tuple[int, int]]:
        """
        Stress period and time step index corresponding to each time step.
        """

        if not self._totim_dict:
            self._set_totim_dict()
        return list(self._totim_dict.keys())

    @property
    def tslen(self) -> list[float]:
        """Duration of each time step."""

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

        return tslen

    def get_datetime_string(
        self, datetime_obj: str | datetime | np.datetime64 | pd.Timestamp
    ) -> str:
        """
        Get a standardized ISO 8601 compliant datetime string.

        Parameters
        ----------
        datetime_obj : various objects
            user supplied datetime representation. Please see the
            ModelTime.parse_datetime documentation for a list
            of the supported representation types

        Returns
        -------
            str: ISO 8601 compliant datetime string
        """

        dt = ModelTime.parse_datetime(datetime_obj)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _set_totim_dict(self) -> None:
        """
        Set up a dictionary of (kper, kstp): totim. Internal use only.
        """

        delt = []
        per_stp = []
        pertim = []
        perlen_array = self.perlen
        nstp_array = self.nstp
        tsmult_array = self.tsmult
        for per, nstp in enumerate(nstp_array):
            perlen = perlen_array[per]
            tsmult = tsmult_array[per]
            pt = 0
            for stp in range(nstp):
                if stp == 0:
                    if tsmult != 1.0:
                        dt = perlen * (tsmult - 1) / ((tsmult**nstp) - 1)
                    else:
                        dt = perlen / nstp
                else:
                    dt = delt[-1] * tsmult
                pt += dt
                delt.append(dt)
                per_stp.append((per, stp))
                pertim.append(pt)

        totim = np.add.accumulate(delt)
        self._totim_dict = {ps: totim[i] for i, ps in enumerate(per_stp)}
        self._pertim_dict = {ps: pertim[i] for i, ps in enumerate(per_stp)}

    def _set_datetime_dict(self) -> None:
        """
        Set up a dictionary of (kper, kstp): datetime. Internal use only.
        """

        if not self._totim_dict:
            self._set_totim_dict()

        dt_dict = {}
        for ps, totim in self._totim_dict.items():
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
                day_td = timedelta(days=days)

                dt = datetime(
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
                dt = self.start_datetime + timedelta(**kwargs)

            dt_dict[ps] = dt

        self._datetime_dict = dt_dict

    @staticmethod
    def parse_timeunits(units: int | str | None) -> str:
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
    def _get_datetime_string_format(str_datetime: str) -> str:
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
            datetime object
        """
        str_datetime = str_datetime.strip().lower()
        if "/" in str_datetime:
            dsep = "/"
        elif "-" in str_datetime:
            dsep = "-"
        else:
            raise ValueError(
                "Separator type for date part of date time representation "
                "not recognized, supported date separator types include '/' "
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
    def parse_datetime(
        datetime_obj: str | datetime | np.datetime64 | pd.Timestamp | None,
    ) -> datetime:
        """
        Method to create a datetime object from a variety of user
        inputs including the following:

        datetime objects
        numpy.datetime64 objects
        pandas.Timestamp objects
        string objects

        Supported formats for string objects representing November 12th, 2024
        are as follows:

        '11/12/2024'
        '11-12-2024'
        '2024/11/12'
        '2024-11-12'

        ModelTime can also be represented in the string object. Example formats
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
            datetime object
        """
        if datetime_obj is None:
            datetime_obj = datetime(1970, 1, 1)  # unix time zero
        elif isinstance(datetime_obj, np.datetime64):
            unix_time_0 = datetime(1970, 1, 1)
            ts = (datetime_obj - np.datetime64(unix_time_0)) / np.timedelta64(1, "s")
            datetime_obj = datetime.utcfromtimestamp(ts)
        elif isinstance(datetime_obj, pd.Timestamp):
            datetime_obj = datetime_obj.to_pydatetime()
        elif isinstance(datetime_obj, datetime):
            pass
        elif isinstance(datetime_obj, str):
            str_rep = ModelTime._get_datetime_string_format(datetime_obj)
            datetime_obj = datetime.strptime(datetime_obj, str_rep)

        else:
            raise NotImplementedError(
                f"{type(datetime_obj)} date representations are not currently supported"
            )

        return datetime_obj

    def get_elapsed_time(
        self, kper: int, kstp: int | None = None, start: bool = False
    ) -> float:
        """
        Method to get the total simulation time at the end or beginning of a given
        stress-period or stress-period and time-step combination

        Parameters
        ----------
        kper : int
            zero based stress-period number
        kstp : int or None
            optional zero based time-step number
        start : bool
            Boolean flag to get totim at the start of the kper/kstp combination.
            Default is False and returns totim at the end of a kper/kstp combination

        Returns
        -------
            totim : float
        """
        if kstp is None:
            if start:
                kper -= 1
            kstp = self.nstp[kper] - 1
        else:
            if start:
                kstp -= 1

        if not self._totim_dict:
            self._set_totim_dict()

        if (kper, kstp) not in self._totim_dict:
            raise KeyError(
                f"(kper, kstp): ({kper} {kstp}) not a valid combination of "
                f"stress period and time step"
            )

        return self._totim_dict[kper, kstp]

    def get_datetime(
        self, kper: int, kstp: int | None = None, start: bool = False
    ) -> datetime:
        """
        Method to get the datetime at the end or start of a given stress period or
        stress period and time step combination

        Parameters
        ----------
        kper : int
            zero based modflow stress period number
        kstp : int
            zero based timestep number
        start : bool
            boolean flag to get the datetime value at the start of a stress period.
            Default is False and returns the datetime at the end of a stress period

        Returns
        -------
            datetime object
        """
        if self.time_units == "unknown":
            raise AssertionError(
                "time units must be set in order to calculate datetime"
            )

        if not self._datetime_dict:
            self._set_datetime_dict()

        if kstp is None:
            if start:
                kper -= 1
            kstp = self.nstp[kper] - 1
        else:
            if start:
                kstp -= 1

        if (kper, kstp) not in self._datetime_dict:
            raise KeyError(
                f"(kper, kstp): ({kper} {kstp}) not a valid combination of "
                f"stress period and time step"
            )

        return self._datetime_dict[kper, kstp]

    def intersect(
        self,
        datetime_obj: str | datetime | np.datetime64 | pd.Timestamp | None = None,
        totim: float | None = None,
        forgive: bool = False,
    ) -> tuple[int | None, int | None]:
        """
        Method to intersect a datetime or totim value with the model and
        get the model stress-period and time-step associated with that
        time.

        Parameters
        ----------
        datetime_obj : various objects
            user supplied datetime representation. Please see the
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
            datetime_obj = ModelTime.parse_datetime(datetime_obj)
            timedelta = datetime_obj - self.start_datetime

            if self.time_units == "unknown":
                raise AssertionError(
                    "time units must be set in order to intersect datetime "
                    "objects, set time units or use totim for intersection"
                )

            elif self.time_units == "days":
                totim = timedelta.days

            elif self.time_units in {"hours", "minutes", "seconds"}:
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

                dt_iyear = datetime(
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
                return None, None
            if datetime_obj is None:
                msg = f"totim {totim} outside of time domain 0 - {self.totim[-1]}"
            else:
                end_dt = self.get_datetime(self.nper - 1, self.nstp[-1] - 1)
                msg = (
                    f"supplied datetime"
                    f" {datetime_obj.strftime(self._str_format)} is "
                    f"outside of the model's time domain "
                    f"{self.start_datetime.strftime(self._str_format)} - "
                    f"{end_dt}"
                )
            raise ValueError(msg)

        idx = sorted(np.where(np.array(self.totim) >= totim)[0])[0]
        per, stp = self.kper_kstp[idx]

        return per, stp

    @classmethod
    def from_perioddata(
        cls,
        perioddata: np.recarray | dict | pd.DataFrame,
        time_units: int | str | None = None,
        start_datetime: str | datetime | np.datetime64 | pd.Timestamp | None = None,
        steady_state: ArrayLike | None = None,
    ) -> "ModelTime":
        """
        Instantiate a ModelTime class from a TDIS perioddata array.

        Parameters
        ----------
        perioddata : np.recarray, dict, or pandas dataframe
            TDIS perioddata recarray, pandas dataframe or dictionary of lists/arrays.
            recarray or dictionary must have the keys "perlen", "nstp", and "tsmult".
        time_units : int or str
            string or pre-mf6 integer representation (ITMUNI) of time units
        start_datetime : various objects
            user supplied datetime representation. Please see the
            ModelTime.parse_datetime documentation for a list
            of the supported representation types
        steady_state : list, np.ndarray
            optional list or numpy array of boolean flags that identify whether a stress
            period is steady-state or transient

        Returns
        -------
            ModelTime object

        """
        if isinstance(perioddata, pd.DataFrame):
            return cls(
                perlen=perioddata["perlen"].values,
                nstp=perioddata["nstp"].values,
                tsmult=perioddata["tsmult"].values,
                time_units=time_units,
                start_datetime=start_datetime,
                steady_state=steady_state,
            )

        return cls(
            perlen=perioddata["perlen"],
            nstp=perioddata["nstp"],
            tsmult=perioddata["tsmult"],
            time_units=time_units,
            start_datetime=start_datetime,
            steady_state=steady_state,
        )

    @classmethod
    def from_headers(cls, headers: np.recarray) -> "ModelTime":
        """
        Instantiate a ModelTime class from a head or budget file header array.

        Parameters
        ----------
        headers : np.recarray
            head or budget file header array

        Returns
        -------
            ModelTime object
        """

        perlen = {}
        nstp = {}
        tsmult = {}
        tslens = []
        tdiff = 0.0
        totim = 0.0
        kper = 0

        def set_tsmult():
            nonlocal tslens
            nonlocal tsmult
            tslens = [l for l in tslens if l > 0]

            if len(tslens) in {0, 1}:
                tsmult[kper] = 1.0
            else:
                tsmult[kper] = tslens[-1] / tslens[-2]

        for i in range(len(headers)):
            hdr = headers[i]
            if kper != int(hdr["kper"] - 1):
                tslens = []
            kper = int(hdr["kper"] - 1)
            kstp = int(hdr["kstp"] - 1)
            tdiff = float(abs(totim - hdr["totim"]))
            tslens.append(tdiff)
            nstp[kper] = kstp + 1
            if kper in perlen:
                perlen[kper] += tdiff
            else:
                perlen[kper] = tdiff
            set_tsmult()
            totim = hdr["totim"]

        if i == len(headers) - 1:
            if len(perlen) == 1 and perlen[0] == 0.0:
                perlen[0] = totim
            set_tsmult()

        return cls(perlen.values(), nstp.values(), tsmult.values())

    def reverse(self) -> "ModelTime":
        """
        Get a new instance with stress periods and time steps in reverse order.

        Returns
        -------
            ModelTime object with reversed order of stress periods and time steps.
        """
        return ModelTime(
            self.perlen[::-1],
            self.nstp[::-1],
            1 / self.tsmult[::-1] if self.tsmult is not None else None,
            self.time_units,
            self.start_datetime,
            self.steady_state[::-1] if self.steady_state is not None else None,
        )
