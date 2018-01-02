# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowTdis(mfpackage.MFPackage):
    """
    ModflowTdis defines a tdis package.

    Attributes
    ----------
    time_units : (time_units : string)
        time_units : is the time units of the simulation. This is a text string
          that is used as a label within model output files. Values for
          time\_units may be ``unknown'', ``seconds'', ``minutes'', ``hours'',
          ``days'', or ``years''. The default time unit is ``unknown''.
    start_date_time : (start_date_time : string)
        start_date_time : is the starting date and time of the simulation. This
          is a text string that is used as a label within the simulation list
          file. The value has no affect on the simulation. The recommended
          format for the starting date and time is described at
          https://www.w3.org/TR/NOTE-datetime.
    nper : (nper : integer)
        nper : is the number of stress periods for the simulation.
    tdisrecarray : [(perlen : double), (nstp : integer), (tsmult : double)]
        perlen : is the length of a stress period.
        nstp : is the number of time steps in a stress period.
        tsmult : is the multiplier for the length of successive time steps. The
          length of a time step is calculated by multiplying the length of the
          previous time step by TSMULT. The length of the first time step,
          $\Delta t_1$, is related to PERLEN, NSTP, and TSMULT by the relation
          $\Delta t_1= perlen \frac{tsmult - 1{tsmult^{nstp-1$.

    """
    tdisrecarray = ListTemplateGenerator(('tdis', 'perioddata', 
                                          'tdisrecarray'))
    package_abbr = "tdis"
    package_type = "tdis"
    dfn = [["block options", "name time_units", "type string", 
            "reader urword", "optional true"],
           ["block options", "name start_date_time", "type string", 
            "reader urword", "optional true"],
           ["block dimensions", "name nper", "type integer", 
            "reader urword", "optional false"],
           ["block perioddata", "name tdisrecarray", 
            "type recarray perlen nstp tsmult", "reader urword", 
            "optional false"],
           ["block perioddata", "name perlen", "type double precision", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block perioddata", "name nstp", "type integer", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block perioddata", "name tsmult", "type double precision", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"]]

    def __init__(self, simulation, add_to_package_list=True, time_units=None,
                 start_date_time=None, nper=None, tdisrecarray=None,
                 fname=None, pname=None, parent_file=None):
        super(ModflowTdis, self).__init__(simulation, "tdis", fname, pname,
                                          add_to_package_list, parent_file)        

        # set up variables
        self.time_units = self.build_mfdata("time_units",  time_units)
        self.start_date_time = self.build_mfdata("start_date_time", 
                                                 start_date_time)
        self.nper = self.build_mfdata("nper",  nper)
        self.tdisrecarray = self.build_mfdata("tdisrecarray",  tdisrecarray)
