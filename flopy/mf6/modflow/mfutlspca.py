# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowUtlspca(mfpackage.MFPackage):
    """
    ModflowUtlspca defines a spca package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    readasarrays : boolean
        * readasarrays (boolean) indicates that array-based input will be used
          for the SPC Package. This keyword must be specified to use array-
          based input.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of spc
          information will be written to the listing file immediately after it
          is read.
    timearrayseries : {varname:data} or tas_array data
        * Contains data for the tas package. Data can be stored in a dictionary
          containing data for the tas package with variable names as keys and
          package data as values. Data just for the timearrayseries variable is
          also acceptable. See tas package documentation for more information.
    concentration : [double]
        * concentration (double) is the concentration of the associated
          Recharge or Evapotranspiration stress package. The concentration
          array may be defined by a time-array series (see the "Using Time-
          Array Series in a Package" section).
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    tas_filerecord = ListTemplateGenerator(
        ("spca", "options", "tas_filerecord")
    )
    concentration = ArrayTemplateGenerator(("spca", "period", "concentration"))
    package_abbr = "utlspca"
    _package_type = "spca"
    dfn_file_name = "utl-spca.dfn"

    dfn = [
        [
            "header",
            "multi-package",
        ],
        [
            "block options",
            "name readasarrays",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
            "default_value True",
        ],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name tas_filerecord",
            "type record tas6 filein tas6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package tas",
            "construct_data tas_array",
            "parameter_name timearrayseries",
        ],
        [
            "block options",
            "name tas6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name tas6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block period",
            "name iper",
            "type integer",
            "block_variable True",
            "in_record true",
            "tagged false",
            "shape",
            "valid",
            "reader urword",
            "optional false",
        ],
        [
            "block period",
            "name concentration",
            "type double precision",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
            "default_value 0.",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        readasarrays=True,
        print_input=None,
        timearrayseries=None,
        concentration=0.0,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "spca", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.readasarrays = self.build_mfdata("readasarrays", readasarrays)
        self.print_input = self.build_mfdata("print_input", print_input)
        self._tas_filerecord = self.build_mfdata("tas_filerecord", None)
        self._tas_package = self.build_child_package(
            "tas", timearrayseries, "tas_array", self._tas_filerecord
        )
        self.concentration = self.build_mfdata("concentration", concentration)
        self._init_complete = True
