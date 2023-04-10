# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on April 10, 2023 20:25:00 UTC
from flopy.mf6 import mfpackage
from flopy.mf6.data.mfdatautil import ListTemplateGenerator


class ModflowGwffp_Rvp(mfpackage.MFPackage):
    """
    ModflowGwffp_Rvp defines a fp_rvp package that is a flopy plugin extension
    of a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of river
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of river flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that river flow terms will
          be written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    auxiliary : [string]
        * auxiliary (string) defines an array of one or more auxiliary variable
          names. There is no limit on the number of auxiliary variables that
          can be provided on this line; however, lists of information provided
          in subsequent blocks must have a column of data for each auxiliary
          variable name defined here. The number of auxiliary variables
          detected on this line determines the value for naux. Comments cannot
          be provided anywhere on this line as they will be interpreted as
          auxiliary variable names. Auxiliary variables may not be used by the
          package, but they will be available for use by other parts of the
          program. The program will terminate with an error if auxiliary
          variables are specified on more than one line in the options block.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of river cells.
    maxbound : integer
        * maxbound (integer) integer value specifying the maximum number of
          rivers cells that will be specified for use during any stress period.
    stress_period_data : [cellid, stage, cond_up, cond_down, rbot, aux,
      boundname]
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * stage (double) is the head in the river.
        * cond_up (double) is the riverbed hydraulic conductance for upward
          flow.
        * cond_down (double) is the riverbed hydraulic conductance for downward
          flow.
        * rbot (double) is the elevation of the bottom of the riverbed.
        * aux (double) auxiliary variables
        * boundname (string) name of the river cell. BOUNDNAME is an ASCII
          character variable that can contain as many as 40 characters. If
          BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(
        ("gwf6", "fp_rvp", "options", "auxiliary")
    )
    stress_period_data = ListTemplateGenerator(
        ("gwf6", "fp_rvp", "period", "stress_period_data")
    )
    package_abbr = "gwffp_rvp"
    _package_type = "fp_rvp"
    dfn_file_name = "gwf-fp_rvp.dfn"

    dfn = [
        [
            "header",
            "flopy-plugin rvp",
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
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name auxiliary",
            "type string",
            "shape (naux)",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name boundnames",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name maxbound",
            "type integer",
            "reader urword",
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
            "name stress_period_data",
            "type recarray cellid stage cond_up cond_down rbot aux boundname",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block period",
            "name cellid",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name stage",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name cond_up",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name cond_down",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name rbot",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name aux",
            "type double precision",
            "shape (naux)",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional True",
        ],
        [
            "block period",
            "name boundname",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional True",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_input=None,
        print_flows=None,
        save_flows=None,
        auxiliary=None,
        boundnames=None,
        maxbound=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        **kwargs
    ):
        super().__init__(
            model, "fp_rvp", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
