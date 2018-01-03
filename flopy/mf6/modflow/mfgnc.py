# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGnc(mfpackage.MFPackage):
    """
    ModflowGnc defines a gnc package.

    Parameters
    ----------
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of GNC
        information will be written to the listing file immediately after it is
        read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of GNC flow
        rates will be printed to the listing file for every stress period time
        step in which ``BUDGET PRINT'' is specified in Output Control. If there
        is no Output Control option and PRINT\_FLOWS is specified, then flow
        rates are printed for the last time step of each stress period.
    explicit : boolean
        * explicit (boolean) keyword to indicate that the ghost node correction
        is applied in an explicit manner on the right-hand side of the matrix.
        The explicit approach will likely require additional outer iterations.
        If the keyword is not specified, then the correction will be applied in
        an implicit manner on the left-hand side. The implicit approach will
        likely converge better, but may require additional memory. If the
        EXPLICIT keyword is not specified, then the BICGSTAB linear
        acceleration option should be specified within the LINEAR block of the
        Sparse Matrix Solver.
    numgnc : integer
        * numgnc (integer) is the number of GNC entries.
    numalphaj : integer
        * numalphaj (integer) is the number of contributing factors.
    gncdatarecarray : [cellidn, cellidm, cellidsj, alphasj]
        * cellidn ((integer, ...)) is the cellid of the cell, $n$, in which the
        ghost node is located. For a structured grid that uses the DIS input
        file, cellidn is the layer, row, and column numbers of the
        cell. For a grid that uses the DISV input file, cellidn is the
        layer number and cell2d number for the two cells. If the model uses the
        unstructured discretization (DISU) input file, then cellidn is
        the node number for the cell.
        * cellidm ((integer, ...)) is the cellid of the connecting cell, $m$,
        to which flow occurs from the ghost node. For a structured grid that
        uses the DIS input file, cellidm is the layer, row, and column
        numbers of the cell. For a grid that uses the DISV input file,
        cellidm is the layer number and cell2d number for the two
        cells. If the model uses the unstructured discretization (DISU) input
        file, then cellidm is the node number for the cell.
        * cellidsj ((integer, ...)) is the array of cellids for the
        contributing $j$ cells, which contribute to the interpolated head value
        at the ghost node. This item contains one cellid for each of the
        contributing cells of the ghost node. Note that if the number of actual
        contributing cells needed by the user is less than numalphaj
        for any ghost node, then a dummy cellid of zero(s) should be inserted
        with an associated contributing factor of zero. For a structured grid
        that uses the DIS input file, cellid is the layer, row, and
        column numbers of the cell. For a grid that uses the DISV input file,
        cellid is the layer number and cell2d number for the two
        cells. If the model uses the unstructured discretization (DISU) input
        file, then cellid is the node number for the cell.
        * alphasj (double) is the contributing factors for each contributing
        node in cellidsj. Note that if the number of actual
        contributing cells is less than numalphaj for any ghost node,
        then dummy cellids should be inserted with an associated contributing
        factor of zero.

    """
    gncdatarecarray = ListTemplateGenerator(('gnc', 'gncdata', 
                                             'gncdatarecarray'))
    package_abbr = "gnc"
    package_type = "gnc"
    dfn = [["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name explicit", "type keyword", "tagged true", 
            "reader urword", "optional true"],
           ["block dimensions", "name numgnc", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name numalphaj", "type integer", 
            "reader urword", "optional false"],
           ["block gncdata", "name gncdatarecarray", 
            "type recarray cellidn cellidm cellidsj alphasj", 
            "shape (maxbound)", "reader urword"],
           ["block gncdata", "name cellidn", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block gncdata", "name cellidm", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block gncdata", "name cellidsj", "type integer", 
            "shape (numalphaj)", "tagged false", "in_record true", 
            "reader urword"],
           ["block gncdata", "name alphasj", "type double precision", 
            "shape (numalphaj)", "tagged false", "in_record true", 
            "reader urword"]]

    def __init__(self, simulation, add_to_package_list=True, print_input=None,
                 print_flows=None, explicit=None, numgnc=None, numalphaj=None,
                 gncdatarecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGnc, self).__init__(simulation, "gnc", fname, pname,
                                         add_to_package_list, parent_file)        

        # set up variables
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.explicit = self.build_mfdata("explicit",  explicit)
        self.numgnc = self.build_mfdata("numgnc",  numgnc)
        self.numalphaj = self.build_mfdata("numalphaj",  numalphaj)
        self.gncdatarecarray = self.build_mfdata("gncdatarecarray", 
                                                 gncdatarecarray)
