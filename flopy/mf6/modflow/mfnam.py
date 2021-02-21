# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowNam(mfpackage.MFPackage):
    """
    ModflowNam defines a nam package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    continue_ : boolean
        * continue (boolean) keyword flag to indicate that the simulation
          should continue even if one or more solutions do not converge.
    nocheck : boolean
        * nocheck (boolean) keyword flag to indicate that the model input check
          routines should not be called prior to each time step. Checks are
          performed by default.
    memory_print_option : string
        * memory_print_option (string) is a flag that controls printing of
          detailed memory manager usage to the end of the simulation list file.
          NONE means do not print detailed information. SUMMARY means print
          only the total memory for each simulation component. ALL means print
          information for each variable stored in the memory manager. NONE is
          default if MEMORY_PRINT_OPTION is not specified.
    maxerrors : integer
        * maxerrors (integer) maximum number of errors that will be stored and
          printed.
    tdis6 : string
        * tdis6 (string) is the name of the Temporal Discretization (TDIS)
          Input File.
    models : [mtype, mfname, mname]
        * mtype (string) is the type of model to add to simulation.
        * mfname (string) is the file name of the model name file.
        * mname (string) is the user-assigned name of the model. The model name
          cannot exceed 16 characters and must not have blanks within the name.
          The model name is case insensitive; any lowercase letters are
          converted and stored as upper case letters.
    exchanges : [exgtype, exgfile, exgmnamea, exgmnameb]
        * exgtype (string) is the exchange type.
        * exgfile (string) is the input file for the exchange.
        * exgmnamea (string) is the name of the first model that is part of
          this exchange.
        * exgmnameb (string) is the name of the second model that is part of
          this exchange.
    mxiter : integer
        * mxiter (integer) is the maximum number of outer iterations for this
          solution group. The default value is 1. If there is only one solution
          in the solution group, then MXITER must be 1.
    solutiongroup : [slntype, slnfname, slnmnames]
        * slntype (string) is the type of solution. The Integrated Model
          Solution (IMS6) is the only supported option in this version.
        * slnfname (string) name of file containing solution input.
        * slnmnames (string) is the array of model names to add to this
          solution. The number of model names is determined by the number of
          model names the user provides on this line.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    models = ListTemplateGenerator(("nam", "models", "models"))
    exchanges = ListTemplateGenerator(("nam", "exchanges", "exchanges"))
    solutiongroup = ListTemplateGenerator(
        ("nam", "solutiongroup", "solutiongroup")
    )
    package_abbr = "nam"
    _package_type = "nam"
    dfn_file_name = "sim-nam.dfn"

    dfn = [
        [
            "block options",
            "name continue",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name nocheck",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name memory_print_option",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name maxerrors",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block timing",
            "name tdis6",
            "preserve_case true",
            "type string",
            "reader urword",
            "optional",
        ],
        [
            "block models",
            "name models",
            "type recarray mtype mfname mname",
            "reader urword",
            "optional",
        ],
        [
            "block models",
            "name mtype",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block models",
            "name mfname",
            "in_record true",
            "type string",
            "preserve_case true",
            "tagged false",
            "reader urword",
        ],
        [
            "block models",
            "name mname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block exchanges",
            "name exchanges",
            "type recarray exgtype exgfile exgmnamea exgmnameb",
            "reader urword",
            "optional",
        ],
        [
            "block exchanges",
            "name exgtype",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block exchanges",
            "name exgfile",
            "in_record true",
            "type string",
            "preserve_case true",
            "tagged false",
            "reader urword",
        ],
        [
            "block exchanges",
            "name exgmnamea",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block exchanges",
            "name exgmnameb",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block solutiongroup",
            "name group_num",
            "type integer",
            "block_variable True",
            "in_record true",
            "tagged false",
            "shape",
            "reader urword",
        ],
        [
            "block solutiongroup",
            "name mxiter",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block solutiongroup",
            "name solutiongroup",
            "type recarray slntype slnfname slnmnames",
            "reader urword",
        ],
        [
            "block solutiongroup",
            "name slntype",
            "type string",
            "valid ims6",
            "in_record true",
            "tagged false",
            "reader urword",
        ],
        [
            "block solutiongroup",
            "name slnfname",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
        ],
        [
            "block solutiongroup",
            "name slnmnames",
            "type string",
            "in_record true",
            "shape (:)",
            "tagged false",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        continue_=None,
        nocheck=None,
        memory_print_option=None,
        maxerrors=None,
        tdis6=None,
        models=None,
        exchanges=None,
        mxiter=None,
        solutiongroup=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowNam, self).__init__(
            simulation, "nam", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.continue_ = self.build_mfdata("continue", continue_)
        self.nocheck = self.build_mfdata("nocheck", nocheck)
        self.memory_print_option = self.build_mfdata(
            "memory_print_option", memory_print_option
        )
        self.maxerrors = self.build_mfdata("maxerrors", maxerrors)
        self.tdis6 = self.build_mfdata("tdis6", tdis6)
        self.models = self.build_mfdata("models", models)
        self.exchanges = self.build_mfdata("exchanges", exchanges)
        self.mxiter = self.build_mfdata("mxiter", mxiter)
        self.solutiongroup = self.build_mfdata("solutiongroup", solutiongroup)
        self._init_complete = True
