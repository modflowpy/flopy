"""
Module spatial referencing for flopy model objects

"""
__all__ = ["TemporalReference"]
# all other classes and methods in this module are deprecated


class TemporalReference:
    """
    For now, just a container to hold start time and time units files
    outside of DIS package.
    """

    defaults = {"itmuni": 4, "start_datetime": "01-01-1970"}

    itmuni_values = {
        "undefined": 0,
        "seconds": 1,
        "minutes": 2,
        "hours": 3,
        "days": 4,
        "years": 5,
    }

    itmuni_text = {v: k for k, v in itmuni_values.items()}

    def __init__(self, itmuni=4, start_datetime=None):
        self.itmuni = itmuni
        self.start_datetime = start_datetime

    @property
    def model_time_units(self):
        return self.itmuni_text[self.itmuni]
