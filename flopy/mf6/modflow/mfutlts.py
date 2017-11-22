from .. import mfpackage
from ..data import mfdatautil


class ModflowUtlts(mfpackage.MFPackage):
    """
    ModflowUtlts defines a ts package within a utl model.

    Attributes
    ----------
    time_series_namerecord : [(time_series_names : string)]
        time_series_names : Name by which a package references a particular time-array series. The name must be unique among all time-array series used in a package.
    interpolation_methodrecord : [(interpolation_method : string)]
        interpolation_method : Interpolation method, which is either STEPWISE or LINEAR.
    interpolation_methodrecord_single : [(interpolation_method_single : string)]
        interpolation_method_single : Interpolation method, which is either STEPWISE or LINEAR.
    sfacrecord : [(sfacval : double)]
        sfacval : Scale factor, which will multiply all array values in time series. SFAC is an optional attribute; if omitted, SFAC = 1.0.
    sfacrecord_single : [(sfacval : double)]
        sfacval : Scale factor, which will multiply all array values in time series. SFAC is an optional attribute; if omitted, SFAC = 1.0.
    time_seriesrecarray : [(tas_time : double), (tas_array : double)]
        tas_time : A numeric time relative to the start of the simulation, in the time unit used in the simulation. Times must be strictly increasing.
        tas_array : A 2-D array of numeric, floating-point values, or a constant value, readable by the U2DREL array-reading utility.

    """
    time_series_namerecord = mfdatautil.ListTemplateGenerator(('ts', 'attributes', 'time_series_namerecord'))
    interpolation_methodrecord = mfdatautil.ListTemplateGenerator(('ts', 'attributes', 'interpolation_methodrecord'))
    interpolation_methodrecord_single = mfdatautil.ListTemplateGenerator(('ts', 'attributes', 'interpolation_methodrecord_single'))
    sfacrecord = mfdatautil.ListTemplateGenerator(('ts', 'attributes', 'sfacrecord'))
    sfacrecord_single = mfdatautil.ListTemplateGenerator(('ts', 'attributes', 'sfacrecord_single'))
    time_seriesrecarray = mfdatautil.ListTemplateGenerator(('ts', 'timeseries', 'time_seriesrecarray'))
    package_abbr = "utlts"

    def __init__(self, model, add_to_package_list=True, time_series_namerecord=None,
                 interpolation_methodrecord=None, interpolation_methodrecord_single=None,
                 sfacrecord=None, sfacrecord_single=None, time_seriesrecarray=None, fname=None,
                 pname=None, parent_file=None):
        super(ModflowUtlts, self).__init__(model, "ts", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.time_series_namerecord = self.build_mfdata("time_series_namerecord", time_series_namerecord)

        self.interpolation_methodrecord = self.build_mfdata("interpolation_methodrecord", interpolation_methodrecord)

        self.interpolation_methodrecord_single = self.build_mfdata("interpolation_methodrecord_single", interpolation_methodrecord_single)

        self.sfacrecord = self.build_mfdata("sfacrecord", sfacrecord)

        self.sfacrecord_single = self.build_mfdata("sfacrecord_single", sfacrecord_single)

        self.time_seriesrecarray = self.build_mfdata("time_seriesrecarray", time_seriesrecarray)


