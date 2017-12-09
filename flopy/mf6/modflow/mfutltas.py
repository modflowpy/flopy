from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtltas(mfpackage.MFPackage):
    """
    ModflowUtltas defines a tas package within a utl model.

    Attributes
    ----------
    time_series_namerecord : [(time_series_name : string)]
        time_series_name : Name by which a package references a particular
          time-array series. The name must be unique among all time-array
          series used in a package.
    interpolation_methodrecord : [(interpolation_method : string)]
        interpolation_method : Interpolation method, which is either STEPWISE
          or LINEAR.
    sfacrecord : [(sfacval : double)]
        sfacval : Scale factor, which will multiply all array values in time
          series. SFAC is an optional attribute; if omitted, SFAC = 1.0.
    tas_array : [(tas_array : double)]
        tas_array : An array of numeric, floating-point values, or a constant
          value, readable by the U2DREL array-reading utility.

    """
    time_series_namerecord = ListTemplateGenerator(('tas', 'attributes', 
                                                    'time_series_namerecord'))
    interpolation_methodrecord = ListTemplateGenerator((
        'tas', 'attributes', 'interpolation_methodrecord'))
    sfacrecord = ListTemplateGenerator(('tas', 'attributes', 
                                        'sfacrecord'))
    tas_array = ArrayTemplateGenerator(('tas', 'time', 'tas_array'))
    package_abbr = "utltas"

    def __init__(self, model, add_to_package_list=True,
                 time_series_namerecord=None, interpolation_methodrecord=None,
                 sfacrecord=None, tas_array=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowUtltas, self).__init__(model, "tas", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.time_series_namerecord = self.build_mfdata(
            "time_series_namerecord",  time_series_namerecord)
        self.interpolation_methodrecord = self.build_mfdata(
            "interpolation_methodrecord",  interpolation_methodrecord)
        self.sfacrecord = self.build_mfdata("sfacrecord",  sfacrecord)
        self.tas_array = self.build_mfdata("tas_array",  tas_array)
