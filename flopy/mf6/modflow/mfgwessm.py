# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwessm(mfpackage.MFPackage):
    """
    ModflowGwessm defines a ssm package within a gwe6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
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
          variable contains a source temperature. If this flow package is
          represented using an advanced transport package (SFE, LKE, MWE, or
          UZE), then the advanced transport package will override SSM terms
          specified here.
        * srctype (string) keyword indicating how temperature will be assigned
          for sources and sinks. Keyword must be specified as either AUX or
          AUXMIXED. For both options the user must provide an auxiliary
          variable in the corresponding flow package. The auxiliary variable
          must have the same name as the AUXNAME value that follows. If the AUX
          keyword is specified, then the auxiliary variable specified by the
          user will be assigned as the concenration value for groundwater
          sources (flows with a positive sign). For negative flow rates
          (sinks), groundwater will be withdrawn from the cell at the simulated
          temperature of the cell. The AUXMIXED option provides an alternative
          method for how to determine the temperature of sinks. If the cell
          temperature is larger than the user-specified auxiliary temperature,
          then the temperature of groundwater withdrawn from the cell will be
          assigned as the user-specified temperature. Alternatively, if the
          user-specified auxiliary temperature is larger than the cell
          temperature, then groundwater will be withdrawn at the cell
          temperature. Thus, the AUXMIXED option is designed to work with the
          Evapotranspiration (EVT) and Recharge (RCH) Packages where water may
          be withdrawn at a temperature that is less than the cell temperature.
        * auxname (string) name of the auxiliary variable in the package PNAME.
          This auxiliary variable must exist and be specified by the user in
          that package. The values in this auxiliary variable will be used to
          set the temperature associated with the flows for that boundary
          package.
    fileinput : [pname, spt6_filename]
        * pname (string) name of the flow package for which an SPT6 input file
          contains a source temperature. If this flow package is represented
          using an advanced transport package (SFE, LKE, MWE, or UZE), then the
          advanced transport package will override SSM terms specified here.
        * spt6_filename (string) character string that defines the path and
          filename for the file containing source and sink input data for the
          flow package. The SPT6_FILENAME file is a flexible input file that
          allows temperatures to be specified by stress period and with time
          series. Instructions for creating the SPT6_FILENAME input file are
          provided in the next section on file input for boundary temperatures.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    sources = ListTemplateGenerator(("gwe6", "ssm", "sources", "sources"))
    fileinput = ListTemplateGenerator(
        ("gwe6", "ssm", "fileinput", "fileinput")
    )
    package_abbr = "gwessm"
    _package_type = "ssm"
    dfn_file_name = "gwe-ssm.dfn"

    dfn = [
        [
            "header",
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
        [
            "block fileinput",
            "name fileinput",
            "type recarray pname spt6 filein spt6_filename mixed",
            "reader urword",
        ],
        [
            "block fileinput",
            "name pname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block fileinput",
            "name spt6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block fileinput",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block fileinput",
            "name spt6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block fileinput",
            "name mixed",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_flows=None,
        save_flows=None,
        sources=None,
        fileinput=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "ssm", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.sources = self.build_mfdata("sources", sources)
        self.fileinput = self.build_mfdata("fileinput", fileinput)
        self._init_complete = True
