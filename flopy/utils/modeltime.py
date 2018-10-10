class ModelTime():
    """
    Class for MODFLOW simulation time

    Parameters
    ----------
    stress_periods : pandas dataframe
        headings are: perlen, nstp, tsmult
    temporal_reference : TemporalReference
        contains start time and time units information
    """
    def __init__(self, period_data, time_units='days',
                 start_datetime=None):
        self.period_data = period_data
        self.time_units = time_units
        self.start_datetime = start_datetime

    @property
    def perlen(self):
        return self.period_data['perlen'].values

    @property
    def nper(self):
        return len(self.period_data['perlen'].values)

    @property
    def nstp(self):
        return self.period_data['nstp'].values

    @property
    def tsmult(self):
        return self.period_data['tsmult'].values