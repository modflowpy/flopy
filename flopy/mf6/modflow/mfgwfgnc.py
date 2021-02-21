# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfgnc(mfpackage.MFPackage):
    """
    ModflowGwfgnc defines a gnc package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of GNC
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of GNC flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    explicit : boolean
        * explicit (boolean) keyword to indicate that the ghost node correction
          is applied in an explicit manner on the right-hand side of the
          matrix. The explicit approach will likely require additional outer
          iterations. If the keyword is not specified, then the correction will
          be applied in an implicit manner on the left-hand side. The implicit
          approach will likely converge better, but may require additional
          memory. If the EXPLICIT keyword is not specified, then the BICGSTAB
          linear acceleration option should be specified within the LINEAR
          block of the Sparse Matrix Solver.
    numgnc : integer
        * numgnc (integer) is the number of GNC entries.
    numalphaj : integer
        * numalphaj (integer) is the number of contributing factors.
    gncdata : [cellidn, cellidm, cellidsj, alphasj]
        * cellidn ((integer, ...)) is the cellid of the cell, :math:`n`, in
          which the ghost node is located. For a structured grid that uses the
          DIS input file, CELLIDN is the layer, row, and column numbers of the
          cell. For a grid that uses the DISV input file, CELLIDN is the layer
          number and CELL2D number for the two cells. If the model uses the
          unstructured discretization (DISU) input file, then CELLIDN is the
          node number for the cell. This argument is an index variable, which
          means that it should be treated as zero-based when working with FloPy
          and Python. Flopy will automatically subtract one when loading index
          variables and add one when writing index variables.
        * cellidm ((integer, ...)) is the cellid of the connecting cell,
          :math:`m`, to which flow occurs from the ghost node. For a structured
          grid that uses the DIS input file, CELLIDM is the layer, row, and
          column numbers of the cell. For a grid that uses the DISV input file,
          CELLIDM is the layer number and CELL2D number for the two cells. If
          the model uses the unstructured discretization (DISU) input file,
          then CELLIDM is the node number for the cell. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * cellidsj ((integer, ...)) is the array of CELLIDS for the
          contributing j cells, which contribute to the interpolated head value
          at the ghost node. This item contains one CELLID for each of the
          contributing cells of the ghost node. Note that if the number of
          actual contributing cells needed by the user is less than NUMALPHAJ
          for any ghost node, then a dummy CELLID of zero(s) should be inserted
          with an associated contributing factor of zero. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column
          numbers of the cell. For a grid that uses the DISV input file, CELLID
          is the layer number and cell2d number for the two cells. If the model
          uses the unstructured discretization (DISU) input file, then CELLID
          is the node number for the cell. This argument is an index variable,
          which means that it should be treated as zero-based when working with
          FloPy and Python. Flopy will automatically subtract one when loading
          index variables and add one when writing index variables.
        * alphasj (double) is the contributing factors for each contributing
          node in CELLIDSJ. Note that if the number of actual contributing
          cells is less than NUMALPHAJ for any ghost node, then dummy CELLIDS
          should be inserted with an associated contributing factor of zero.
          The sum of ALPHASJ should be less than one. This is because one minus
          the sum of ALPHASJ is equal to the alpha term (alpha n in equation
          4-61 of the GWF Model report) that is multiplied by the head in cell
          n.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    gncdata = ListTemplateGenerator(("gwf6", "gnc", "gncdata", "gncdata"))
    package_abbr = "gwfgnc"
    _package_type = "gnc"
    dfn_file_name = "gwf-gnc.dfn"

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
            "name explicit",
            "type keyword",
            "tagged true",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name numgnc",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name numalphaj",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block gncdata",
            "name gncdata",
            "type recarray cellidn cellidm cellidsj alphasj",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block gncdata",
            "name cellidn",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block gncdata",
            "name cellidm",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block gncdata",
            "name cellidsj",
            "type integer",
            "shape (numalphaj)",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block gncdata",
            "name alphasj",
            "type double precision",
            "shape (numalphaj)",
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
        print_flows=None,
        explicit=None,
        numgnc=None,
        numalphaj=None,
        gncdata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwfgnc, self).__init__(
            model, "gnc", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.explicit = self.build_mfdata("explicit", explicit)
        self.numgnc = self.build_mfdata("numgnc", numgnc)
        self.numalphaj = self.build_mfdata("numalphaj", numalphaj)
        self.gncdata = self.build_mfdata("gncdata", gncdata)
        self._init_complete = True
