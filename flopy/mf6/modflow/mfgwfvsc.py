# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfvsc(mfpackage.MFPackage):
    """
    ModflowGwfvsc defines a vsc package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    viscref : double
        * viscref (double) fluid reference viscosity used in the equation of
          state. This value is set to 1.0 if not specified as an option.
    temperature_species_name : string
        * temperature_species_name (string) string used to identify the
          auxspeciesname in PACKAGEDATA that corresponds to the temperature
          species. There can be only one occurrence of this temperature species
          name in the PACKAGEDATA block or the program will terminate with an
          error. This value has no effect if viscosity does not depend on
          temperature.
    thermal_formulation : string
        * thermal_formulation (string) may be used for specifying which
          viscosity formulation to use for the temperature species. Can be
          either LINEAR or NONLINEAR. The LINEAR viscosity formulation is the
          default.
    thermal_a2 : double
        * thermal_a2 (double) is an empirical parameter specified by the user
          for calculating viscosity using a nonlinear formulation. If A2 is not
          specified, a default value of 10.0 is assigned (Voss, 1984).
    thermal_a3 : double
        * thermal_a3 (double) is an empirical parameter specified by the user
          for calculating viscosity using a nonlinear formulation. If A3 is not
          specified, a default value of 248.37 is assigned (Voss, 1984).
    thermal_a4 : double
        * thermal_a4 (double) is an empirical parameter specified by the user
          for calculating viscosity using a nonlinear formulation. If A4 is not
          specified, a default value of 133.15 is assigned (Voss, 1984).
    viscosity_filerecord : [viscosityfile]
        * viscosityfile (string) name of the binary output file to write
          viscosity information. The viscosity file has the same format as the
          head file. Viscosity values will be written to the viscosity file
          whenever heads are written to the binary head file. The settings for
          controlling head output are contained in the Output Control option.
    nviscspecies : integer
        * nviscspecies (integer) number of species used in the viscosity
          equation of state. If either concentrations or temperature (or both)
          are used to update viscosity then then nrhospecies needs to be at
          least one.
    packagedata : [iviscspec, dviscdc, cviscref, modelname, auxspeciesname]
        * iviscspec (integer) integer value that defines the species number
          associated with the specified PACKAGEDATA data entered on each line.
          IVISCSPECIES must be greater than zero and less than or equal to
          NVISCSPECIES. Information must be specified for each of the
          NVISCSPECIES species or the program will terminate with an error. The
          program will also terminate with an error if information for a
          species is specified more than once. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
        * dviscdc (double) real value that defines the slope of the line
          defining the linear relationship between viscosity and temperature or
          between viscosity and concentration, depending on the type of species
          entered on each line. If the value of AUXSPECIESNAME entered on a
          line corresponds to TEMPERATURE_SPECIES_NAME (in the OPTIONS block),
          this value will be used when VISCOSITY_FUNC is equal to LINEAR (the
          default) in the OPTIONS block. When VISCOSITY_FUNC is set to
          NONLINEAR, a value for DVISCDC must be specified though it is not
          used.
        * cviscref (double) real value that defines the reference temperature
          or reference concentration value used for this species in the
          viscosity equation of state. If AUXSPECIESNAME entered on a line
          corresponds to TEMPERATURE_SPECIES_NAME (in the OPTIONS block), then
          CVISCREF refers to a reference temperature, otherwise it refers to a
          reference concentration.
        * modelname (string) name of a GWT model used to simulate a species
          that will be used in the viscosity equation of state. This name will
          have no effect if the simulation does not include a GWT model that
          corresponds to this GWF model.
        * auxspeciesname (string) name of an auxiliary variable in a GWF stress
          package that will be used for this species to calculate the viscosity
          values. If a viscosity value is needed by the Viscosity Package then
          it will use the temperature or concentration values associated with
          this AUXSPECIESNAME in the viscosity equation of state. For advanced
          stress packages (LAK, SFR, MAW, and UZF) that have an associated
          advanced transport package (LKT, SFT, MWT, and UZT), the
          FLOW_PACKAGE_AUXILIARY_NAME option in the advanced transport package
          can be used to transfer simulated temperature or concentration(s)
          into the flow package auxiliary variable. In this manner, the
          Viscosity Package can calculate viscosity values for lakes, streams,
          multi-aquifer wells, and unsaturated zone flow cells using simulated
          concentrations.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    viscosity_filerecord = ListTemplateGenerator(
        ("gwf6", "vsc", "options", "viscosity_filerecord")
    )
    packagedata = ListTemplateGenerator(
        ("gwf6", "vsc", "packagedata", "packagedata")
    )
    package_abbr = "gwfvsc"
    _package_type = "vsc"
    dfn_file_name = "gwf-vsc.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name viscref",
            "type double",
            "reader urword",
            "optional true",
            "default_value 1.0",
        ],
        [
            "block options",
            "name temperature_species_name",
            "type string",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name thermal_formulation",
            "type string",
            "shape",
            "reader urword",
            "optional true",
            "valid linear nonlinear",
        ],
        [
            "block options",
            "name thermal_a2",
            "type double",
            "reader urword",
            "optional true",
            "default_value 10.",
        ],
        [
            "block options",
            "name thermal_a3",
            "type double",
            "reader urword",
            "optional true",
            "default_value 248.37",
        ],
        [
            "block options",
            "name thermal_a4",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 133.15",
        ],
        [
            "block options",
            "name viscosity_filerecord",
            "type record viscosity fileout viscosityfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name viscosity",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name fileout",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name viscosityfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block dimensions",
            "name nviscspecies",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray iviscspec dviscdc cviscref modelname auxspeciesname",
            "shape (nrhospecies)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name iviscspec",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name dviscdc",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name cviscref",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name modelname",
            "type string",
            "in_record true",
            "tagged false",
            "shape",
            "reader urword",
        ],
        [
            "block packagedata",
            "name auxspeciesname",
            "type string",
            "in_record true",
            "tagged false",
            "shape",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        viscref=1.0,
        temperature_species_name=None,
        thermal_formulation=None,
        thermal_a2=10.0,
        thermal_a3=248.37,
        thermal_a4=133.15,
        viscosity_filerecord=None,
        nviscspecies=None,
        packagedata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "vsc", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.viscref = self.build_mfdata("viscref", viscref)
        self.temperature_species_name = self.build_mfdata(
            "temperature_species_name", temperature_species_name
        )
        self.thermal_formulation = self.build_mfdata(
            "thermal_formulation", thermal_formulation
        )
        self.thermal_a2 = self.build_mfdata("thermal_a2", thermal_a2)
        self.thermal_a3 = self.build_mfdata("thermal_a3", thermal_a3)
        self.thermal_a4 = self.build_mfdata("thermal_a4", thermal_a4)
        self.viscosity_filerecord = self.build_mfdata(
            "viscosity_filerecord", viscosity_filerecord
        )
        self.nviscspecies = self.build_mfdata("nviscspecies", nviscspecies)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self._init_complete = True
