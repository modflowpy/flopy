# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwemwe(mfpackage.MFPackage):
    """
    ModflowGwemwe defines a mwe package within a gwe6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    flow_package_name : string
        * flow_package_name (string) keyword to specify the name of the
          corresponding flow package. If not specified, then the corresponding
          flow package must have the same name as this advanced transport
          package (the name associated with this package in the GWE name file).
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
    flow_package_auxiliary_name : string
        * flow_package_auxiliary_name (string) keyword to specify the name of
          an auxiliary variable in the corresponding flow package. If
          specified, then the simulated temperatures from this advanced
          transport package will be copied into the auxiliary variable
          specified with this name. Note that the flow package must have an
          auxiliary variable with this name or the program will terminate with
          an error. If the flows for this advanced transport package are read
          from a file, then this option will have no effect.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of well cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of well
          information will be written to the listing file immediately after it
          is read.
    print_temperature : boolean
        * print_temperature (boolean) keyword to indicate that the list of well
          temperature will be printed to the listing file for every stress
          period in which "TEMPERATURE PRINT" is specified in Output Control.
          If there is no Output Control option and PRINT_TEMPERATURE is
          specified, then temperature are printed for the last time step of
          each stress period.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of well flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that well flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    temperature_filerecord : [tempfile]
        * tempfile (string) name of the binary output file to write temperature
          information.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
    timeseries : {varname:data} or timeseries data
        * Contains data for the ts package. Data can be stored in a dictionary
          containing data for the ts package with variable names as keys and
          package data as values. Data just for the timeseries variable is also
          acceptable. See ts package documentation for more information.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    packagedata : [mawno, strt, ktf, fthk, aux, boundname]
        * mawno (integer) integer value that defines the well number associated
          with the specified PACKAGEDATA data on the line. MAWNO must be
          greater than zero and less than or equal to NMAWWELLS. Well
          information must be specified for every well or the program will
          terminate with an error. The program will also terminate with an
          error if information for a well is specified more than once. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * strt (double) real value that defines the starting temperature for
          the well.
        * ktf (double) is the thermal conductivity of the of the interface
          between the aquifer cell and the feature.
        * fthk (double) real value that defines the thickness of the material
          through which conduction occurs. Must be greater than 0.
        * aux (double) represents the values of the auxiliary variables for
          each well. The values of auxiliary variables must be present for each
          well. The values must be specified in the order of the auxiliary
          variables specified in the OPTIONS block. If the package supports
          time series and the Options block includes a TIMESERIESFILE entry
          (see the "Time-Variable Input" section), values can be obtained from
          a time series by entering the time-series name in place of a numeric
          value.
        * boundname (string) name of the well cell. BOUNDNAME is an ASCII
          character variable that can contain as many as 40 characters. If
          BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    mweperioddata : [mawno, mwesetting]
        * mawno (integer) integer value that defines the well number associated
          with the specified PERIOD data on the line. MAWNO must be greater
          than zero and less than or equal to NMAWWELLS. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * mwesetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          MWESETTING string include: STATUS, TEMPERATURE, RAINFALL,
          EVAPORATION, RUNOFF, and AUXILIARY. These settings are used to assign
          the temperature of associated with the corresponding flow terms.
          Temperatures cannot be specified for all flow terms. For example, the
          Multi-Aquifer Well Package supports a "WITHDRAWAL" flow term. If this
          withdrawal term is active, then water will be withdrawn from the well
          at the calculated temperature of the well.
            status : [string]
                * status (string) keyword option to define well status. STATUS
                  can be ACTIVE, INACTIVE, or CONSTANT. By default, STATUS is
                  ACTIVE, which means that temperature will be calculated for
                  the well. If a well is inactive, then there will be no solute
                  mass fluxes into or out of the well and the inactive value
                  will be written for the well temperature. If a well is
                  constant, then the temperature for the well will be fixed at
                  the user specified value.
            temperature : [string]
                * temperature (string) real or character value that defines the
                  temperature for the well. The specified TEMPERATURE is only
                  applied if the well is a constant temperature well. If the
                  Options block includes a TIMESERIESFILE entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.
            rate : [string]
                * rate (string) real or character value that defines the
                  injection solute temperature :math:`^{\\circ}C` for the well.
                  If the Options block includes a TIMESERIESFILE entry (see the
                  "Time-Variable Input" section), values can be obtained from a
                  time series by entering the time-series name in place of a
                  numeric value.
            auxiliaryrecord : [auxname, auxval]
                * auxname (string) name for the auxiliary variable to be
                  assigned AUXVAL. AUXNAME must match one of the auxiliary
                  variable names defined in the OPTIONS block. If AUXNAME does
                  not match one of the auxiliary variable names defined in the
                  OPTIONS block the data are ignored.
                * auxval (double) value for the auxiliary variable. If the
                  Options block includes a TIMESERIESFILE entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(("gwe6", "mwe", "options", "auxiliary"))
    temperature_filerecord = ListTemplateGenerator(
        ("gwe6", "mwe", "options", "temperature_filerecord")
    )
    budget_filerecord = ListTemplateGenerator(
        ("gwe6", "mwe", "options", "budget_filerecord")
    )
    budgetcsv_filerecord = ListTemplateGenerator(
        ("gwe6", "mwe", "options", "budgetcsv_filerecord")
    )
    ts_filerecord = ListTemplateGenerator(
        ("gwe6", "mwe", "options", "ts_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwe6", "mwe", "options", "obs_filerecord")
    )
    packagedata = ListTemplateGenerator(
        ("gwe6", "mwe", "packagedata", "packagedata")
    )
    mweperioddata = ListTemplateGenerator(
        ("gwe6", "mwe", "period", "mweperioddata")
    )
    package_abbr = "gwemwe"
    _package_type = "mwe"
    dfn_file_name = "gwe-mwe.dfn"

    dfn = [
        [
            "header",
            "multi-package",
        ],
        [
            "block options",
            "name flow_package_name",
            "type string",
            "shape",
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
            "name flow_package_auxiliary_name",
            "type string",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name boundnames",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
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
            "name print_temperature",
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
            "name temperature_filerecord",
            "type record temperature fileout tempfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name temperature",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name tempfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
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
            "block options",
            "name budgetcsv_filerecord",
            "type record budgetcsv fileout budgetcsvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budgetcsv",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name budgetcsvfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name ts_filerecord",
            "type record ts6 filein ts6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package ts",
            "construct_data timeseries",
            "parameter_name timeseries",
        ],
        [
            "block options",
            "name ts6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name ts6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block options",
            "name obs_filerecord",
            "type record obs6 filein obs6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package obs",
            "construct_data continuous",
            "parameter_name observations",
        ],
        [
            "block options",
            "name obs6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name obs6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray mawno strt ktf fthk aux boundname",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name mawno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name strt",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name ktf",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name fthk",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name aux",
            "type double precision",
            "in_record true",
            "tagged false",
            "shape (naux)",
            "reader urword",
            "time_series true",
            "optional true",
        ],
        [
            "block packagedata",
            "name boundname",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
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
            "name mweperioddata",
            "type recarray mawno mwesetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name mawno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name mwesetting",
            "type keystring status temperature rate auxiliaryrecord",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name status",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name temperature",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "time_series true",
            "reader urword",
        ],
        [
            "block period",
            "name rate",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name auxiliaryrecord",
            "type record auxiliary auxname auxval",
            "shape",
            "tagged",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name auxiliary",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name auxname",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name auxval",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        flow_package_name=None,
        auxiliary=None,
        flow_package_auxiliary_name=None,
        boundnames=None,
        print_input=None,
        print_temperature=None,
        print_flows=None,
        save_flows=None,
        temperature_filerecord=None,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        timeseries=None,
        observations=None,
        packagedata=None,
        mweperioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "mwe", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.flow_package_name = self.build_mfdata(
            "flow_package_name", flow_package_name
        )
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.flow_package_auxiliary_name = self.build_mfdata(
            "flow_package_auxiliary_name", flow_package_auxiliary_name
        )
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_temperature = self.build_mfdata(
            "print_temperature", print_temperature
        )
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.temperature_filerecord = self.build_mfdata(
            "temperature_filerecord", temperature_filerecord
        )
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.budgetcsv_filerecord = self.build_mfdata(
            "budgetcsv_filerecord", budgetcsv_filerecord
        )
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.mweperioddata = self.build_mfdata("mweperioddata", mweperioddata)
        self._init_complete = True
