class ModelTime:
    """
    Class for MODFLOW simulation time

    Parameters
    ----------
    stress_periods : pandas dataframe
        headings are: perlen, nstp, tsmult
    temporal_reference : TemporalReference
        contains start time and time units information
    """

    def __init__(
        self,
        period_data=None,
        time_units="days",
        start_datetime=None,
        steady_state=None,
    ):
        self._period_data = period_data
        self._time_units = time_units
        self._start_datetime = start_datetime
        self._steady_state = steady_state

    @property
    def time_units(self):
        return self._time_units

    @property
    def start_datetime(self):
        return self._start_datetime

    @property
    def perlen(self):
        return self._period_data["perlen"]

    @property
    def nper(self):
        return len(self._period_data["perlen"])

    @property
    def nstp(self):
        return self._period_data["nstp"]

    @property
    def tsmult(self):
        return self._period_data["tsmult"]

    @property
    def steady_state(self):
        return self._steady_state
