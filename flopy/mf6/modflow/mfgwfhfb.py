# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfhfb(mfpackage.MFPackage):
    """
    ModflowGwfhfb defines a hfb package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of horizontal
          flow barriers will be written to the listing file immediately after
          it is read.
    maxhfb : integer
        * maxhfb (integer) integer value specifying the maximum number of
          horizontal flow barriers that will be entered in this input file. The
          value of MAXHFB is used to allocate memory for the horizontal flow
          barriers.
    stress_period_data : [cellid1, cellid2, hydchr]
        * cellid1 ((integer, ...)) identifier for the first cell. For a
          structured grid that uses the DIS input file, CELLID1 is the layer,
          row, and column numbers of the cell. For a grid that uses the DISV
          input file, CELLID1 is the layer number and CELL2D number for the two
          cells. If the model uses the unstructured discretization (DISU) input
          file, then CELLID1 is the node numbers for the cell. The barrier is
          located between cells designated as CELLID1 and CELLID2. For models
          that use the DIS and DISV grid types, the layer number for CELLID1
          and CELLID2 must be the same. For all grid types, cells must be
          horizontally adjacent or the program will terminate with an error.
          This argument is an index variable, which means that it should be
          treated as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * cellid2 ((integer, ...)) identifier for the second cell. See CELLID1
          for description of how to specify. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
        * hydchr (double) is the hydraulic characteristic of the horizontal-
          flow barrier. The hydraulic characteristic is the barrier hydraulic
          conductivity divided by the width of the horizontal-flow barrier. If
          the hydraulic characteristic is negative, then the absolute value of
          HYDCHR acts as a multiplier to the conductance between the two model
          cells specified as containing the barrier. For example, if the value
          for HYDCHR was specified as -1.5, the conductance calculated for the
          two cells would be multiplied by 1.5.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    stress_period_data = ListTemplateGenerator(
        ("gwf6", "hfb", "period", "stress_period_data")
    )
    package_abbr = "gwfhfb"
    _package_type = "hfb"
    dfn_file_name = "gwf-hfb.dfn"

    dfn = [
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name maxhfb",
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
            "type recarray cellid1 cellid2 hydchr",
            "shape (maxhfb)",
            "reader urword",
        ],
        [
            "block period",
            "name cellid1",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name cellid2",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name hydchr",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_input=None,
        maxhfb=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "hfb", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.maxhfb = self.build_mfdata("maxhfb", maxhfb)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
