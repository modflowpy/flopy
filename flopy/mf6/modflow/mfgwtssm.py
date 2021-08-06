# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwtssm(mfpackage.MFPackage):
    """
    ModflowGwtssm defines a ssm package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of SSM flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that SSM flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    sources : [pname, srctype, auxname]
        * pname (string) name of the flow package for which an auxiliary
          variable contains a source concentration. If this flow package is
          represented using an advanced transport package (SFT, LKT, MWT, or
          UZT), then the advanced transport package will override SSM terms
          specified here.
        * srctype (string) keyword indicating how concentration will be
          assigned for sources and sinks. Keyword must be specified as either
          AUX or AUXMIXED. For both options the user must provide an auxiliary
          variable in the corresponding flow package. The auxiliary variable
          must have the same name as the AUXNAME value that follows. If the AUX
          keyword is specified, then the auxiliary variable specified by the
          user will be assigned as the concenration value for groundwater
          sources (flows with a positive sign). For negative flow rates
          (sinks), groundwater will be withdrawn from the cell at the simulated
          concentration of the cell. The AUXMIXED option provides an
          alternative method for how to determine the concentration of sinks.
          If the cell concentration is larger than the user-specified auxiliary
          concentration, then the concentration of groundwater withdrawn from
          the cell will be assigned as the user-specified concentration.
          Alternatively, if the user-specified auxiliary concentration is
          larger than the cell concentration, then groundwater will be
          withdrawn at the cell concentration. Thus, the AUXMIXED option is
          designed to work with the Evapotranspiration (EVT) and Recharge (RCH)
          Packages where water may be withdrawn at a concentration that is less
          than the cell concentration.
        * auxname (string) name of the auxiliary variable in the package PNAME.
          This auxiliary variable must exist and be specified by the user in
          that package. The values in this auxiliary variable will be used to
          set the concentration associated with the flows for that boundary
          package.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    sources = ListTemplateGenerator(("gwt6", "ssm", "sources", "sources"))
    package_abbr = "gwtssm"
    _package_type = "ssm"
    dfn_file_name = "gwt-ssm.dfn"

    dfn = [
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
            "block sources",
            "name sources",
            "type recarray pname srctype auxname",
            "reader urword",
            "optional false",
        ],
        [
            "block sources",
            "name pname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block sources",
            "name srctype",
            "in_record true",
            "type string",
            "tagged false",
            "optional false",
            "reader urword",
        ],
        [
            "block sources",
            "name auxname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_flows=None,
        save_flows=None,
        sources=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "ssm", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.sources = self.build_mfdata("sources", sources)
        self._init_complete = True
