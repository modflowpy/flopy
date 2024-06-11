from ..mfbase import FlopyException

"""
simulationtime module.  Contains the simulationtime and
stress period classes
"""


class StressPeriod:
    """
    Represents a stress period


    Parameters
    ----------


    Attributes
    ----------


    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, perlen, nstp, tsmult, start_time=None):
        self._perlen = perlen
        self._nstp = nstp
        self._tsmult = tsmult
        self._start_time = start_time

    def get_num_stress_periods(self):
        return len(self._perlen)

    def get_period_length(self):
        return self._perlen

    def get_num_steps(self):
        return self._nstp

    def get_mult(self):
        return self._tsmult


class SimulationTime:
    """
    Represents a block in a MF6 input file


    Parameters
    ----------


    Attributes
    ----------


    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, simdata):
        self.simdata = simdata

    def get_time_units(self):
        time_units = self.simdata.mfdata[
            ("tdis", "options", "time_units")
        ].get_data()
        return time_units

    def get_perioddata(self):
        return self.simdata.mfdata[
            ("tdis", "perioddata", "perioddata")
        ].get_data()

    def get_total_time(self):
        period_data = self.simdata.mfdata[
            ("tdis", "perioddata", "perioddata")
        ].get_data()
        total_time = 0.0
        for period in period_data:
            total_time += period[0]
        return total_time

    def get_num_stress_periods(self):
        return self.simdata.mfdata[("tdis", "dimensions", "nper")].get_data()

    def get_sp_time_steps(self, sp_num):
        period_data = self.simdata.mfdata[
            ("tdis", "perioddata", "perioddata")
        ].get_data()
        if len(period_data) <= sp_num:
            raise FlopyException(
                f"Stress period {sp_num} was requested but does not exist."
            )
        return period_data[sp_num][1]
