# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlts(mfpackage.MFPackage):
    """
    ModflowUtlts defines a ts package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    time_series_namerecord : [time_series_names]
        * time_series_names (string) Name by which a package references a
          particular time-array series. The name must be unique among all time-
          array series used in a package.
    interpolation_methodrecord : [interpolation_method]
        * interpolation_method (string) Interpolation method, which is either
          STEPWISE or LINEAR.
    interpolation_methodrecord_single : [interpolation_method_single]
        * interpolation_method_single (string) Interpolation method, which is
          either STEPWISE or LINEAR.
    sfacrecord : [sfacval]
        * sfacval (double) Scale factor, which will multiply all array values
          in time series. SFAC is an optional attribute; if omitted, SFAC =
          1.0.
    sfacrecord_single : [sfacval]
        * sfacval (double) Scale factor, which will multiply all array values
          in time series. SFAC is an optional attribute; if omitted, SFAC =
          1.0.
    timeseries : [ts_time, ts_array]
        * ts_time (double) A numeric time relative to the start of the
          simulation, in the time unit used in the simulation. Times must be
          strictly increasing.
        * ts_array (double) A 2-D array of numeric, floating-point values, or a
          constant value, readable by the U2DREL array-reading utility.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    time_series_namerecord = ListTemplateGenerator(
        ("ts", "attributes", "time_series_namerecord")
    )
    interpolation_methodrecord = ListTemplateGenerator(
        ("ts", "attributes", "interpolation_methodrecord")
    )
    interpolation_methodrecord_single = ListTemplateGenerator(
        ("ts", "attributes", "interpolation_methodrecord_single")
    )
    sfacrecord = ListTemplateGenerator(("ts", "attributes", "sfacrecord"))
    sfacrecord_single = ListTemplateGenerator(
        ("ts", "attributes", "sfacrecord_single")
    )
    timeseries = ListTemplateGenerator(("ts", "timeseries", "timeseries"))
    package_abbr = "utlts"
    _package_type = "ts"
    dfn_file_name = "utl-ts.dfn"

    dfn = [
        [
            "block attributes",
            "name time_series_namerecord",
            "type record names time_series_names",
            "shape",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block attributes",
            "name names",
            "other_names name",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name time_series_names",
            "type string",
            "shape any1d",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name interpolation_methodrecord",
            "type record methods interpolation_method",
            "shape",
            "reader urword",
            "tagged false",
            "optional true",
        ],
        [
            "block attributes",
            "name methods",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name interpolation_method",
            "type string",
            "valid stepwise linear linearend",
            "shape time_series_names",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name interpolation_methodrecord_single",
            "type record method interpolation_method_single",
            "shape",
            "reader urword",
            "tagged false",
            "optional true",
        ],
        [
            "block attributes",
            "name method",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name interpolation_method_single",
            "type string",
            "valid stepwise linear linearend",
            "shape",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name sfacrecord",
            "type record sfacs sfacval",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block attributes",
            "name sfacs",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name sfacval",
            "type double precision",
            "shape <time_series_name",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name sfacrecord_single",
            "type record sfac sfacval",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block attributes",
            "name sfac",
            "type keyword",
            "shape",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block timeseries",
            "name timeseries",
            "type recarray ts_time ts_array",
            "shape",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block timeseries",
            "name ts_time",
            "type double precision",
            "shape",
            "tagged false",
            "reader urword",
            "optional false",
            "repeating false",
        ],
        [
            "block timeseries",
            "name ts_array",
            "type double precision",
            "shape time_series_names",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        interpolation_methodrecord_single=None,
        sfacrecord=None,
        sfacrecord_single=None,
        timeseries=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "ts", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.time_series_namerecord = self.build_mfdata(
            "time_series_namerecord", time_series_namerecord
        )
        self.interpolation_methodrecord = self.build_mfdata(
            "interpolation_methodrecord", interpolation_methodrecord
        )
        self.interpolation_methodrecord_single = self.build_mfdata(
            "interpolation_methodrecord_single",
            interpolation_methodrecord_single,
        )
        self.sfacrecord = self.build_mfdata("sfacrecord", sfacrecord)
        self.sfacrecord_single = self.build_mfdata(
            "sfacrecord_single", sfacrecord_single
        )
        self.timeseries = self.build_mfdata("timeseries", timeseries)
        self._init_complete = True


class UtltsPackages(mfpackage.MFChildPackages):
    """
    UtltsPackages is a container class for the ModflowUtlts class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtlts package removing any sibling child
        packages attached to the same parent package. See ModflowUtlts init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtlts package to the container. See ModflowUtlts
        init documentation for definition of parameters.
    """

    package_abbr = "utltspackages"

    def initialize(
        self,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        interpolation_methodrecord_single=None,
        sfacrecord=None,
        sfacrecord_single=None,
        timeseries=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtlts(
            self._model,
            time_series_namerecord=time_series_namerecord,
            interpolation_methodrecord=interpolation_methodrecord,
            interpolation_methodrecord_single=interpolation_methodrecord_single,
            sfacrecord=sfacrecord,
            sfacrecord_single=sfacrecord_single,
            timeseries=timeseries,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._init_package(new_package, filename)

    def append_package(
        self,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        interpolation_methodrecord_single=None,
        sfacrecord=None,
        sfacrecord_single=None,
        timeseries=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtlts(
            self._model,
            time_series_namerecord=time_series_namerecord,
            interpolation_methodrecord=interpolation_methodrecord,
            interpolation_methodrecord_single=interpolation_methodrecord_single,
            sfacrecord=sfacrecord,
            sfacrecord_single=sfacrecord_single,
            timeseries=timeseries,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._append_package(new_package, filename)
