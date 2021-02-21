# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowMvr(mfpackage.MFPackage):
    """
    ModflowMvr defines a mvr package. This package can only be used to move
    water between two different models. To move water between two packages
    in the same model use the "model level" mover package (ex. ModflowGwfmvr).

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of MVR
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of MVR flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    modelnames : boolean
        * modelnames (boolean) keyword to indicate that all package names will
          be preceded by the model name for the package. Model names are
          required when the Mover Package is used with a GWF-GWF Exchange. The
          MODELNAME keyword should not be used for a Mover Package that is for
          a single GWF Model.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    maxmvr : integer
        * maxmvr (integer) integer value specifying the maximum number of water
          mover entries that will specified for any stress period.
    maxpackages : integer
        * maxpackages (integer) integer value specifying the number of unique
          packages that are included in this water mover input file.
    packages : [mname, pname]
        * mname (string) name of model containing the package. Model names are
          assigned by the user in the simulation name file.
        * pname (string) is the name of a package that may be included in a
          subsequent stress period block. The package name is assigned in the
          name file for the GWF Model. Package names are optionally provided in
          the name file. If they are not provided by the user, then packages
          are assigned a default value, which is the package acronym followed
          by a hyphen and the package number. For example, the first Drain
          Package is named DRN-1. The second Drain Package is named DRN-2, and
          so forth.
    perioddata : [mname1, pname1, id1, mname2, pname2, id2, mvrtype, value]
        * mname1 (string) name of model containing the package, PNAME1.
        * pname1 (string) is the package name for the provider. The package
          PNAME1 must be designated to provide water through the MVR Package by
          specifying the keyword "MOVER" in its OPTIONS block.
        * id1 (integer) is the identifier for the provider. For the standard
          boundary packages, the provider identifier is the number of the
          boundary as it is listed in the package input file. (Note that the
          order of these boundaries may change by stress period, which must be
          accounted for in the Mover Package.) So the first well has an
          identifier of one. The second is two, and so forth. For the advanced
          packages, the identifier is the reach number (SFR Package), well
          number (MAW Package), or UZF cell number. For the Lake Package, ID1
          is the lake outlet number. Thus, outflows from a single lake can be
          routed to different streams, for example. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
        * mname2 (string) name of model containing the package, PNAME2.
        * pname2 (string) is the package name for the receiver. The package
          PNAME2 must be designated to receive water from the MVR Package by
          specifying the keyword "MOVER" in its OPTIONS block.
        * id2 (integer) is the identifier for the receiver. The receiver
          identifier is the reach number (SFR Package), Lake number (LAK
          Package), well number (MAW Package), or UZF cell number. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * mvrtype (string) is the character string signifying the method for
          determining how much water will be moved. Supported values are
          "FACTOR" "EXCESS" "THRESHOLD" and "UPTO". These four options
          determine how the receiver flow rate, :math:`Q_R`, is calculated.
          These options mirror the options defined for the cprior variable in
          the SFR package, with the term "FACTOR" being functionally equivalent
          to the "FRACTION" option for cprior.
        * value (double) is the value to be used in the equation for
          calculating the amount of water to move. For the "FACTOR" option,
          VALUE is the :math:`\\alpha` factor. For the remaining options, VALUE
          is the specified flow rate, :math:`Q_S`.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    budget_filerecord = ListTemplateGenerator(
        ("mvr", "options", "budget_filerecord")
    )
    packages = ListTemplateGenerator(("mvr", "packages", "packages"))
    perioddata = ListTemplateGenerator(("mvr", "period", "perioddata"))
    package_abbr = "mvr"
    _package_type = "mvr"
    dfn_file_name = "gwf-mvr.dfn"

    dfn = [
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name modelnames",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name budget_filerecord",
            "type record budget fileout budgetfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budget",
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
            "name budgetfile",
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
            "name maxmvr",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name maxpackages",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packages",
            "name packages",
            "type recarray mname pname",
            "reader urword",
            "shape (npackages)",
            "optional false",
        ],
        [
            "block packages",
            "name mname",
            "type string",
            "reader urword",
            "shape",
            "tagged false",
            "in_record true",
            "optional true",
        ],
        [
            "block packages",
            "name pname",
            "type string",
            "reader urword",
            "shape",
            "tagged false",
            "in_record true",
            "optional false",
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
            "name perioddata",
            "type recarray mname1 pname1 id1 mname2 pname2 id2 mvrtype value",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block period",
            "name mname1",
            "type string",
            "reader urword",
            "shape",
            "tagged false",
            "in_record true",
            "optional true",
        ],
        [
            "block period",
            "name pname1",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name id1",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name mname2",
            "type string",
            "reader urword",
            "shape",
            "tagged false",
            "in_record true",
            "optional true",
        ],
        [
            "block period",
            "name pname2",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name id2",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name mvrtype",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name value",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        print_input=None,
        print_flows=None,
        modelnames=None,
        budget_filerecord=None,
        maxmvr=None,
        maxpackages=None,
        packages=None,
        perioddata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowMvr, self).__init__(
            simulation, "mvr", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.modelnames = self.build_mfdata("modelnames", modelnames)
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.maxmvr = self.build_mfdata("maxmvr", maxmvr)
        self.maxpackages = self.build_mfdata("maxpackages", maxpackages)
        self.packages = self.build_mfdata("packages", packages)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
