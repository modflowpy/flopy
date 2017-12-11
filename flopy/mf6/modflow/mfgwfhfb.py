from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfhfb(mfpackage.MFPackage):
    """
    ModflowGwfhfb defines a hfb package within a gwf6 model.

    Attributes
    ----------
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of horizontal flow
          barriers will be written to the listing file immediately after it is
          read.
    maxhfb : (maxhfb : integer)
        maxhfb : integer value specifying the maximum number of horizontal flow
          barriers that will be entered in this input file. The value of
          maxhfb is used to allocate memory for the horizontal flow
          barriers.
    hfbrecarray : [(cellid1 : (integer, ...)), (cellid2 : (integer, ...)), (hydchr
      : double)]
        cellid1 : identifier for the first cell. For a structured grid that
          uses the DIS input file, cellid1 is the layer, row, and
          column numbers of the cell. For a grid that uses the DISV input file,
          cellid1 is the layer number and cell2d number for the two
          cells. If the model uses the unstructured discretization (DISU) input
          file, then cellid1 is the node numbers for the cell. The
          barrier is located between cells designated as cellid1 and
          cellid2. For models that use the DIS and DISV grid types,
          the layer number for cellid1 and cellid2 must be
          the same. For all grid types, cells must be horizontally adjacent or
          the program will terminate with an error.
        cellid2 : identifier for the second cell. See cellid1 for
          description of how to specify.
        hydchr : is the hydraulic characteristic of the horizontal-flow
          barrier. The hydraulic characteristic is the barrier hydraulic
          conductivity divided by the width of the horizontal-flow barrier. If
          hydraulic characteristic is negative, then it acts as a multiplier to
          the conductance between the two model cells specified as containing a
          barrier. For example, if the value for hydchr was specified
          as 1.5, the conductance calculated for the two cells would be
          multiplied by 1.5.

    """
    hfbrecarray = ListTemplateGenerator(('gwf6', 'hfb', 'period', 
                                         'hfbrecarray'))
    package_abbr = "gwfhfb"

    def __init__(self, model, add_to_package_list=True, print_input=None,
                 maxhfb=None, hfbrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfhfb, self).__init__(model, "hfb", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.maxhfb = self.build_mfdata("maxhfb",  maxhfb)
        self.hfbrecarray = self.build_mfdata("hfbrecarray",  hfbrecarray)
