# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtltas(mfpackage.MFPackage):
    """
    ModflowUtltas defines a tas package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    time_series_namerecord : [time_series_name]
        * time_series_name (string) Name by which a package references a
          particular time-array series. The name must be unique among all time-
          array series used in a package.
    interpolation_methodrecord : [interpolation_method]
        * interpolation_method (string) Interpolation method, which is either
          STEPWISE or LINEAR.
    sfacrecord : [sfacval]
        * sfacval (double) Scale factor, which will multiply all array values
          in time series. SFAC is an optional attribute; if omitted, SFAC =
          1.0.
    tas_array : [double]
        * tas_array (double) An array of numeric, floating-point values, or a
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
        ("tas", "attributes", "time_series_namerecord")
    )
    interpolation_methodrecord = ListTemplateGenerator(
        ("tas", "attributes", "interpolation_methodrecord")
    )
    sfacrecord = ListTemplateGenerator(("tas", "attributes", "sfacrecord"))
    tas_array = ArrayTemplateGenerator(("tas", "time", "tas_array"))
    package_abbr = "utltas"
    _package_type = "tas"
    dfn_file_name = "utl-tas.dfn"

    dfn = [
        [
            "block attributes",
            "name time_series_namerecord",
            "type record name time_series_name",
            "shape",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block attributes",
            "name name",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name time_series_name",
            "type string",
            "shape any1d",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name interpolation_methodrecord",
            "type record method interpolation_method",
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
            "name interpolation_method",
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
            "reader urword",
            "optional false",
        ],
        [
            "block attributes",
            "name sfacval",
            "type double precision",
            "shape time_series_name",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block time",
            "name time_from_model_start",
            "type double precision",
            "block_variable True",
            "in_record true",
            "shape",
            "tagged false",
            "valid",
            "reader urword",
            "optional false",
        ],
        [
            "block time",
            "name tas_array",
            "type double precision",
            "tagged false",
            "just_data true",
            "shape (unknown)",
            "reader readarray",
            "optional false",
            "repeating true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        sfacrecord=None,
        tas_array=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "tas", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.time_series_namerecord = self.build_mfdata(
            "time_series_namerecord", time_series_namerecord
        )
        self.interpolation_methodrecord = self.build_mfdata(
            "interpolation_methodrecord", interpolation_methodrecord
        )
        self.sfacrecord = self.build_mfdata("sfacrecord", sfacrecord)
        self.tas_array = self.build_mfdata("tas_array", tas_array)
        self._init_complete = True


class UtltasPackages(mfpackage.MFChildPackages):
    """
    UtltasPackages is a container class for the ModflowUtltas class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtltas package removing any sibling child
        packages attached to the same parent package. See ModflowUtltas init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtltas package to the container. See ModflowUtltas
        init documentation for definition of parameters.
    """

    package_abbr = "utltaspackages"

    def initialize(
        self,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        sfacrecord=None,
        tas_array=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtltas(
            self._model,
            time_series_namerecord=time_series_namerecord,
            interpolation_methodrecord=interpolation_methodrecord,
            sfacrecord=sfacrecord,
            tas_array=tas_array,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._init_package(new_package, filename)

    def append_package(
        self,
        time_series_namerecord=None,
        interpolation_methodrecord=None,
        sfacrecord=None,
        tas_array=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtltas(
            self._model,
            time_series_namerecord=time_series_namerecord,
            interpolation_methodrecord=interpolation_methodrecord,
            sfacrecord=sfacrecord,
            tas_array=tas_array,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._append_package(new_package, filename)
