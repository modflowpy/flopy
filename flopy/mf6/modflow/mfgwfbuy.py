# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfbuy(mfpackage.MFPackage):
    """
    ModflowGwfbuy defines a buy package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    hhformulation_rhs : boolean
        * hhformulation_rhs (boolean) use the variable-density hydraulic head
          formulation and add off-diagonal terms to the right-hand. This option
          will prevent the BUY Package from adding asymmetric terms to the flow
          matrix.
    denseref : double
        * denseref (double) fluid reference density used in the equation of
          state. This value is set to 1000. if not specified as an option.
    density_filerecord : [densityfile]
        * densityfile (string) name of the binary output file to write density
          information. The density file has the same format as the head file.
          Density values will be written to the density file whenever heads are
          written to the binary head file. The settings for controlling head
          output are contained in the Output Control option.
    dev_efh_formulation : boolean
        * dev_efh_formulation (boolean) use the variable-density equivalent
          freshwater head formulation instead of the hydraulic head head
          formulation. This dev option has only been implemented for confined
          aquifer conditions and should generally not be used.
    nrhospecies : integer
        * nrhospecies (integer) number of species used in density equation of
          state. This value must be one or greater. The value must be one if
          concentrations are specified using the CONCENTRATION keyword in the
          PERIOD block below.
    packagedata : [irhospec, drhodc, crhoref, modelname, auxspeciesname]
        * irhospec (integer) integer value that defines the species number
          associated with the specified PACKAGEDATA data on the line.
          IRHOSPECIES must be greater than zero and less than or equal to
          NRHOSPECIES. Information must be specified for each of the
          NRHOSPECIES species or the program will terminate with an error. The
          program will also terminate with an error if information for a
          species is specified more than once. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
        * drhodc (double) real value that defines the slope of the density-
          concentration line for this species used in the density equation of
          state.
        * crhoref (double) real value that defines the reference concentration
          value used for this species in the density equation of state.
        * modelname (string) name of GWT model used to simulate a species that
          will be used in the density equation of state. This name will have no
          effect if the simulation does not include a GWT model that
          corresponds to this GWF model.
        * auxspeciesname (string) name of an auxiliary variable in a GWF stress
          package that will be used for this species to calculate a density
          value. If a density value is needed by the Buoyancy Package then it
          will use the concentration values in this AUXSPECIESNAME column in
          the density equation of state. For advanced stress packages (LAK,
          SFR, MAW, and UZF) that have an associated advanced transport package
          (LKT, SFT, MWT, and UZT), the FLOW_PACKAGE_AUXILIARY_NAME option in
          the advanced transport package can be used to transfer simulated
          concentrations into the flow package auxiliary variable. In this
          manner, the Buoyancy Package can calculate density values for lakes,
          streams, multi-aquifer wells, and unsaturated zone flow cells using
          simulated concentrations.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    density_filerecord = ListTemplateGenerator(
        ("gwf6", "buy", "options", "density_filerecord")
    )
    packagedata = ListTemplateGenerator(
        ("gwf6", "buy", "packagedata", "packagedata")
    )
    package_abbr = "gwfbuy"
    _package_type = "buy"
    dfn_file_name = "gwf-buy.dfn"

    dfn = [
        [
            "block options",
            "name hhformulation_rhs",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name denseref",
            "type double",
            "reader urword",
            "optional true",
            "default_value 1000.",
        ],
        [
            "block options",
            "name density_filerecord",
            "type record density fileout densityfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name density",
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
            "name densityfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name dev_efh_formulation",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name nrhospecies",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray irhospec drhodc crhoref modelname auxspeciesname",
            "shape (nrhospecies)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name irhospec",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name drhodc",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name crhoref",
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
        hhformulation_rhs=None,
        denseref=1000.0,
        density_filerecord=None,
        dev_efh_formulation=None,
        nrhospecies=None,
        packagedata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "buy", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.hhformulation_rhs = self.build_mfdata(
            "hhformulation_rhs", hhformulation_rhs
        )
        self.denseref = self.build_mfdata("denseref", denseref)
        self.density_filerecord = self.build_mfdata(
            "density_filerecord", density_filerecord
        )
        self.dev_efh_formulation = self.build_mfdata(
            "dev_efh_formulation", dev_efh_formulation
        )
        self.nrhospecies = self.build_mfdata("nrhospecies", nrhospecies)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self._init_complete = True
