# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowOlfgwf(mfpackage.MFPackage):
    """
    ModflowOlfgwf defines a olfgwf package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    exgtype : <string>
        * is the exchange type (GWF-GWF or GWF-GWT).
    exgmnamea : <string>
        * is the name of the first model that is part of this exchange.
    exgmnameb : <string>
        * is the name of the second model that is part of this exchange.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of exchange
          entries will be echoed to the listing file immediately after it is
          read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of exchange
          flow rates will be printed to the listing file for every stress
          period in which "SAVE BUDGET" is specified in Output Control.
    fixed_conductance : boolean
        * fixed_conductance (boolean) keyword to indicate that the product of
          the bedleak and cfact input variables in the exchangedata block
          represents conductance. This conductance is fixed and does not change
          as a function of head in the surface water and groundwater models.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    nexg : integer
        * nexg (integer) keyword and integer value specifying the number of
          SWF-GWF exchanges.
    exchangedata : [cellidm1, cellidm2, bedleak, cfact]
        * cellidm1 ((integer, ...)) is the cellid of the cell in model 1, which
          must be the surface water model. For a structured grid that uses the
          DIS input file, CELLIDM1 is the layer, row, and column numbers of the
          cell. For a grid that uses the DISV input file, CELLIDM1 is the layer
          number and CELL2D number for the two cells. If the model uses the
          unstructured discretization (DISU) input file, then CELLIDM1 is the
          node number for the cell. This argument is an index variable, which
          means that it should be treated as zero-based when working with FloPy
          and Python. Flopy will automatically subtract one when loading index
          variables and add one when writing index variables.
        * cellidm2 ((integer, ...)) is the cellid of the cell in model 2, which
          must be the groundwater model. For a structured grid that uses the
          DIS input file, CELLIDM2 is the layer, row, and column numbers of the
          cell. For a grid that uses the DISV input file, CELLIDM2 is the layer
          number and CELL2D number for the two cells. If the model uses the
          unstructured discretization (DISU) input file, then CELLIDM2 is the
          node number for the cell. This argument is an index variable, which
          means that it should be treated as zero-based when working with FloPy
          and Python. Flopy will automatically subtract one when loading index
          variables and add one when writing index variables.
        * bedleak (double) is the leakance between the surface water and
          groundwater. bedleak has dimensions of 1/T and is equal to the
          hydraulic conductivity of the bed sediments divided by the thickness
          of the bed sediments.
        * cfact (double) is the factor used for the conductance calculation.
          The definition for this parameter depends the type of surface water
          model and whether or not the fixed_conductance option is specified.
          If the fixed_conductance option is specified, then the hydraulic
          conductance is calculated as the product of bedleak and cfact. In
          this case, the conductance is fixed and does not change as a function
          of the calculated surface water and groundwater head. If the
          fixed_conductance option is not specified, then the definition of
          cfact depends on whether the surface water model represents one-
          dimensional channel flow or two-dimensional overland flow. If the
          surface water model represents one-dimensional channel flow, then
          cfact is the length of the channel cell in the groundwater model
          cell. If the surface water model represents two-dimensional overland
          flow, then cfact is the intersection area of the overland flow cell
          and the underlying groundwater model cell.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    obs_filerecord = ListTemplateGenerator(('olfgwf', 'options',
                                            'obs_filerecord'))
    exchangedata = ListTemplateGenerator(('olfgwf', 'exchangedata',
                                          'exchangedata'))
    package_abbr = "olfgwf"
    _package_type = "olfgwf"
    dfn_file_name = "exg-olfgwf.dfn"

    dfn = [
           ["header", 
            "multi-package", ],
           ["block options", "name print_input", "type keyword",
            "reader urword", "optional true", "mf6internal ipr_input"],
           ["block options", "name print_flows", "type keyword",
            "reader urword", "optional true", "mf6internal ipr_flow"],
           ["block options", "name fixed_conductance", "type keyword",
            "reader urword", "optional true", "mf6internal ifixedcond"],
           ["block options", "name obs_filerecord",
            "type record obs6 filein obs6_filename", "shape", "reader urword",
            "tagged true", "optional true", "construct_package obs",
            "construct_data continuous", "parameter_name observations"],
           ["block options", "name obs6", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name filein", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name obs6_filename", "type string",
            "preserve_case true", "in_record true", "tagged false",
            "reader urword", "optional false"],
           ["block dimensions", "name nexg", "type integer",
            "reader urword", "optional false"],
           ["block exchangedata", "name exchangedata",
            "type recarray cellidm1 cellidm2 bedleak cfact", "shape (nexg)",
            "reader urword", "optional false"],
           ["block exchangedata", "name cellidm1", "type integer",
            "in_record true", "tagged false", "reader urword",
            "optional false", "numeric_index true"],
           ["block exchangedata", "name cellidm2", "type integer",
            "in_record true", "tagged false", "reader urword",
            "optional false", "numeric_index true"],
           ["block exchangedata", "name bedleak", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"],
           ["block exchangedata", "name cfact", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"]]

    def __init__(self, simulation, loading_package=False, exgtype="OLF6-GWF6",
                 exgmnamea=None, exgmnameb=None, print_input=None,
                 print_flows=None, fixed_conductance=None, observations=None,
                 nexg=None, exchangedata=None, filename=None, pname=None,
                 **kwargs):
        super().__init__(simulation, "olfgwf", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.fixed_conductance = self.build_mfdata("fixed_conductance",
                                                   fixed_conductance)
        self._obs_filerecord = self.build_mfdata("obs_filerecord",
                                                 None)
        self._obs_package = self.build_child_package("obs", observations,
                                                     "continuous",
                                                     self._obs_filerecord)
        self.nexg = self.build_mfdata("nexg", nexg)
        self.exchangedata = self.build_mfdata("exchangedata", exchangedata)
        self._init_complete = True
