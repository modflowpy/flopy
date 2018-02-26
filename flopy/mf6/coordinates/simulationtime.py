"""
simulationtime module.  Contains the simulationtime and
stress period classes
"""


class StressPeriod(object):
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

    #def get_ts_start_time(self, timestep):

    #def get_sp_start_time(self, timestep):

    #def get_ts_end_time(self, timestep):

    #def get_sp_end_time(self, timestep):

    #def get_ts_length(self, timestep):


class SimulationTime(object):
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

    #        self.time_units = simdata[('TDIS', 'OPTIONS', 'time_units')]
    #        self.stress_periods = simdata[('TDIS', 'STRESS_PERIODS',
    #        'perlen,nstp,tsmult')
    #        self.calendar_start_time = calendar_start_time

    #    def get_stress_period_array(self):
    #        return np.arange(1, self.get_num_stress_periods(), 1, np.int)

    def get_num_stress_periods(self):
        return self.simdata.mfdata[('tdis', 'dimensions', 'nper')].get_data()

    def get_sp_time_steps(self, sp_num):
        perioddata = self.simdata.mfdata[
            ('tdis', 'perioddata', 'perioddata')].get_data()
        assert (len(perioddata) > sp_num)
        return perioddata[sp_num][1]

    #def get_stress_period(self, sp_num):

    #def remove_stress_period(self, num_stress_period):

    #def copy_append_stress_period(self, sp_num):

    #def split_stress_period(self, sp_num):
