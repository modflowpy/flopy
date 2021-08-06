# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfmaw(mfpackage.MFPackage):
    """
    ModflowGwfmaw defines a maw package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
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
          provided with the list of multi-aquifer well cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of multi-
          aquifer well information will be written to the listing file
          immediately after it is read.
    print_head : boolean
        * print_head (boolean) keyword to indicate that the list of multi-
          aquifer well heads will be printed to the listing file for every
          stress period in which "HEAD PRINT" is specified in Output Control.
          If there is no Output Control option and PRINT_HEAD is specified,
          then heads are printed for the last time step of each stress period.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of multi-
          aquifer well flow rates will be printed to the listing file for every
          stress period time step in which "BUDGET PRINT" is specified in
          Output Control. If there is no Output Control option and
          "PRINT_FLOWS" is specified, then flow rates are printed for the last
          time step of each stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that multi-aquifer well flow
          terms will be written to the file specified with "BUDGET FILEOUT" in
          Output Control.
    head_filerecord : [headfile]
        * headfile (string) name of the binary output file to write head
          information.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    no_well_storage : boolean
        * no_well_storage (boolean) keyword that deactivates inclusion of well
          storage contributions to the multi-aquifer well package continuity
          equation.
    flow_correction : boolean
        * flow_correction (boolean) keyword that activates flow corrections in
          cases where the head in a multi-aquifer well is below the bottom of
          the screen for a connection or the head in a convertible cell
          connected to a multi-aquifer well is below the cell bottom. When flow
          corrections are activated, unit head gradients are used to calculate
          the flow between a multi-aquifer well and a connected GWF cell. By
          default, flow corrections are not made.
    flowing_wells : boolean
        * flowing_wells (boolean) keyword that activates the flowing wells
          option for the multi-aquifer well package.
    shutdown_theta : double
        * shutdown_theta (double) value that defines the weight applied to
          discharge rate for wells that limit the water level in a discharging
          well (defined using the HEAD_LIMIT keyword in the stress period
          data). SHUTDOWN_THETA is used to control discharge rate oscillations
          when the flow rate from the aquifer is less than the specified flow
          rate from the aquifer to the well. Values range between 0.0 and 1.0,
          and larger values increase the weight (decrease under-relaxation)
          applied to the well discharge rate. The HEAD_LIMIT option has been
          included to facilitate backward compatibility with previous versions
          of MODFLOW but use of the RATE_SCALING option instead of the
          HEAD_LIMIT option is recommended. By default, SHUTDOWN_THETA is 0.7.
    shutdown_kappa : double
        * shutdown_kappa (double) value that defines the weight applied to
          discharge rate for wells that limit the water level in a discharging
          well (defined using the HEAD_LIMIT keyword in the stress period
          data). SHUTDOWN_KAPPA is used to control discharge rate oscillations
          when the flow rate from the aquifer is less than the specified flow
          rate from the aquifer to the well. Values range between 0.0 and 1.0,
          and larger values increase the weight applied to the well discharge
          rate. The HEAD_LIMIT option has been included to facilitate backward
          compatibility with previous versions of MODFLOW but use of the
          RATE_SCALING option instead of the HEAD_LIMIT option is recommended.
          By default, SHUTDOWN_KAPPA is 0.0001.
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
    mover : boolean
        * mover (boolean) keyword to indicate that this instance of the MAW
          Package can be used with the Water Mover (MVR) Package. When the
          MOVER option is specified, additional memory is allocated within the
          package to store the available, provided, and received water.
    nmawwells : integer
        * nmawwells (integer) integer value specifying the number of multi-
          aquifer wells that will be simulated for all stress periods.
    packagedata : [wellno, radius, bottom, strt, condeqn, ngwfnodes, aux,
      boundname]
        * wellno (integer) integer value that defines the well number
          associated with the specified PACKAGEDATA data on the line. WELLNO
          must be greater than zero and less than or equal to NMAWWELLS. Multi-
          aquifer well information must be specified for every multi-aquifer
          well or the program will terminate with an error. The program will
          also terminate with an error if information for a multi-aquifer well
          is specified more than once. This argument is an index variable,
          which means that it should be treated as zero-based when working with
          FloPy and Python. Flopy will automatically subtract one when loading
          index variables and add one when writing index variables.
        * radius (double) radius for the multi-aquifer well. The program will
          terminate with an error if the radius is less than or equal to zero.
        * bottom (double) bottom elevation of the multi-aquifer well. If
          CONDEQN is SPECIFIED, THIEM, SKIN, or COMPOSITE, BOTTOM is set to the
          cell bottom in the lowermost GWF cell connection in cases where the
          specified well bottom is above the bottom of this GWF cell. If
          CONDEQN is MEAN, BOTTOM is set to the lowermost GWF cell connection
          screen bottom in cases where the specified well bottom is above this
          value. The bottom elevation defines the lowest well head that will be
          simulated when the NEWTON UNDER_RELAXATION option is specified in the
          GWF model name file. The bottom elevation is also used to calculate
          volumetric storage in the well.
        * strt (double) starting head for the multi-aquifer well. The program
          will terminate with an error if the starting head is less than the
          specified well bottom.
        * condeqn (string) character string that defines the conductance
          equation that is used to calculate the saturated conductance for the
          multi-aquifer well. Possible multi-aquifer well CONDEQN strings
          include: SPECIFIED--character keyword to indicate the multi-aquifer
          well saturated conductance will be specified. THIEM--character
          keyword to indicate the multi-aquifer well saturated conductance will
          be calculated using the Thiem equation, which considers the cell top
          and bottom, aquifer hydraulic conductivity, and effective cell and
          well radius. SKIN--character keyword to indicate that the multi-
          aquifer well saturated conductance will be calculated using the cell
          top and bottom, aquifer and screen hydraulic conductivity, and well
          and skin radius. CUMULATIVE--character keyword to indicate that the
          multi-aquifer well saturated conductance will be calculated using a
          combination of the Thiem and SKIN equations. MEAN--character keyword
          to indicate the multi-aquifer well saturated conductance will be
          calculated using the aquifer and screen top and bottom, aquifer and
          screen hydraulic conductivity, and well and skin radius. The
          CUMULATIVE conductance equation is identical to the SKIN LOSSTYPE in
          the Multi-Node Well (MNW2) package for MODFLOW-2005. The program will
          terminate with an error condition if CONDEQN is SKIN or CUMULATIVE
          and the calculated saturated conductance is less than zero; if an
          error condition occurs, it is suggested that the THEIM or MEAN
          conductance equations be used for these multi-aquifer wells.
        * ngwfnodes (integer) integer value that defines the number of GWF
          nodes connected to this (WELLNO) multi-aquifer well. NGWFNODES must
          be greater than zero.
        * aux (double) represents the values of the auxiliary variables for
          each multi-aquifer well. The values of auxiliary variables must be
          present for each multi-aquifer well. The values must be specified in
          the order of the auxiliary variables specified in the OPTIONS block.
          If the package supports time series and the Options block includes a
          TIMESERIESFILE entry (see the "Time-Variable Input" section), values
          can be obtained from a time series by entering the time-series name
          in place of a numeric value.
        * boundname (string) name of the multi-aquifer well cell. BOUNDNAME is
          an ASCII character variable that can contain as many as 40
          characters. If BOUNDNAME contains spaces in it, then the entire name
          must be enclosed within single quotes.
    connectiondata : [wellno, icon, cellid, scrn_top, scrn_bot, hk_skin,
      radius_skin]
        * wellno (integer) integer value that defines the well number
          associated with the specified CONNECTIONDATA data on the line. WELLNO
          must be greater than zero and less than or equal to NMAWWELLS. Multi-
          aquifer well connection information must be specified for every
          multi-aquifer well connection to the GWF model (NGWFNODES) or the
          program will terminate with an error. The program will also terminate
          with an error if connection information for a multi-aquifer well
          connection to the GWF model is specified more than once. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * icon (integer) integer value that defines the GWF connection number
          for this multi-aquifer well connection entry. ICONN must be greater
          than zero and less than or equal to NGWFNODES for multi-aquifer well
          WELLNO. This argument is an index variable, which means that it
          should be treated as zero-based when working with FloPy and Python.
          Flopy will automatically subtract one when loading index variables
          and add one when writing index variables.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. One or
          more screened intervals can be connected to the same CELLID if
          CONDEQN for a well is MEAN. The program will terminate with an error
          if MAW wells using SPECIFIED, THIEM, SKIN, or CUMULATIVE conductance
          equations have more than one connection to the same CELLID. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * scrn_top (double) value that defines the top elevation of the screen
          for the multi-aquifer well connection. If CONDEQN is SPECIFIED,
          THIEM, SKIN, or COMPOSITE, SCRN_TOP can be any value and is set to
          the top of the cell. If CONDEQN is MEAN, SCRN_TOP is set to the
          multi-aquifer well connection cell top if the specified value is
          greater than the cell top. The program will terminate with an error
          if the screen top is less than the screen bottom.
        * scrn_bot (double) value that defines the bottom elevation of the
          screen for the multi-aquifer well connection. If CONDEQN is
          SPECIFIED, THIEM, SKIN, or COMPOSITE, SCRN_BOT can be any value is
          set to the bottom of the cell. If CONDEQN is MEAN, SCRN_BOT is set to
          the multi-aquifer well connection cell bottom if the specified value
          is less than the cell bottom. The program will terminate with an
          error if the screen bottom is greater than the screen top.
        * hk_skin (double) value that defines the skin (filter pack) hydraulic
          conductivity (if CONDEQN for the multi-aquifer well is SKIN,
          CUMULATIVE, or MEAN) or conductance (if CONDEQN for the multi-aquifer
          well is SPECIFIED) for each GWF node connected to the multi-aquifer
          well (NGWFNODES). If CONDEQN is SPECIFIED, HK_SKIN must be greater
          than or equal to zero. HK_SKIN can be any value if CONDEQN is THIEM.
          Otherwise, HK_SKIN must be greater than zero. If CONDEQN is SKIN, the
          contrast between the cell transmissivity (the product of geometric
          mean horizontal hydraulic conductivity and the cell thickness) and
          the well transmissivity (the product of HK_SKIN and the screen
          thicknesses) must be greater than one in node CELLID or the program
          will terminate with an error condition; if an error condition occurs,
          it is suggested that the HK_SKIN be reduced to a value less than K11
          and K22 in node CELLID or the THEIM or MEAN conductance equations be
          used for these multi-aquifer wells.
        * radius_skin (double) real value that defines the skin radius (filter
          pack radius) for the multi-aquifer well. RADIUS_SKIN can be any value
          if CONDEQN is SPECIFIED or THIEM. If CONDEQN is SKIN, CUMULATIVE, or
          MEAN, the program will terminate with an error if RADIUS_SKIN is less
          than or equal to the RADIUS for the multi-aquifer well.
    perioddata : [wellno, mawsetting]
        * wellno (integer) integer value that defines the well number
          associated with the specified PERIOD data on the line. WELLNO must be
          greater than zero and less than or equal to NMAWWELLS. This argument
          is an index variable, which means that it should be treated as zero-
          based when working with FloPy and Python. Flopy will automatically
          subtract one when loading index variables and add one when writing
          index variables.
        * mawsetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          MAWSETTING string include: STATUS, FLOWING_WELL, RATE, WELL_HEAD,
          HEAD_LIMIT, SHUT_OFF, RATE_SCALING, and AUXILIARY.
            status : [string]
                * status (string) keyword option to define well status. STATUS
                  can be ACTIVE, INACTIVE, or CONSTANT. By default, STATUS is
                  ACTIVE.
            flowing_wellrecord : [fwelev, fwcond, fwrlen]
                * fwelev (double) elevation used to determine whether or not
                  the well is flowing.
                * fwcond (double) conductance used to calculate the discharge
                  of a free flowing well. Flow occurs when the head in the well
                  is above the well top elevation (FWELEV).
                * fwrlen (double) length used to reduce the conductance of the
                  flowing well. When the head in the well drops below the well
                  top plus the reduction length, then the conductance is
                  reduced. This reduction length can be used to improve the
                  stability of simulations with flowing wells so that there is
                  not an abrupt change in flowing well rates.
            rate : [double]
                * rate (double) is the volumetric pumping rate for the multi-
                  aquifer well. A positive value indicates recharge and a
                  negative value indicates discharge (pumping). RATE only
                  applies to active (IBOUND > 0) multi-aquifer wells. If the
                  Options block includes a TIMESERIESFILE entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value. By default, the RATE for each multi-aquifer well is
                  zero.
            well_head : [double]
                * well_head (double) is the head in the multi-aquifer well.
                  WELL_HEAD is only applied to constant head (STATUS is
                  CONSTANT) and inactive (STATUS is INACTIVE) multi-aquifer
                  wells. If the Options block includes a TIMESERIESFILE entry
                  (see the "Time-Variable Input" section), values can be
                  obtained from a time series by entering the time-series name
                  in place of a numeric value. The program will terminate with
                  an error if WELL_HEAD is less than the bottom of the well.
            head_limit : [string]
                * head_limit (string) is the limiting water level (head) in the
                  well, which is the minimum of the well RATE or the well
                  inflow rate from the aquifer. HEAD_LIMIT can be applied to
                  extraction wells (RATE < 0) or injection wells (RATE > 0).
                  HEAD_LIMIT can be deactivated by specifying the text string
                  'OFF'. The HEAD_LIMIT option is based on the HEAD_LIMIT
                  functionality available in the MNW2 (Konikow et al., 2009)
                  package for MODFLOW-2005. The HEAD_LIMIT option has been
                  included to facilitate backward compatibility with previous
                  versions of MODFLOW but use of the RATE_SCALING option
                  instead of the HEAD_LIMIT option is recommended. By default,
                  HEAD_LIMIT is 'OFF'.
            shutoffrecord : [minrate, maxrate]
                * minrate (double) is the minimum rate that a well must exceed
                  to shutoff a well during a stress period. The well will shut
                  down during a time step if the flow rate to the well from the
                  aquifer is less than MINRATE. If a well is shut down during a
                  time step, reactivation of the well cannot occur until the
                  next time step to reduce oscillations. MINRATE must be less
                  than maxrate.
                * maxrate (double) is the maximum rate that a well must exceed
                  to reactivate a well during a stress period. The well will
                  reactivate during a timestep if the well was shutdown during
                  the previous time step and the flow rate to the well from the
                  aquifer exceeds maxrate. Reactivation of the well cannot
                  occur until the next time step if a well is shutdown to
                  reduce oscillations. maxrate must be greater than MINRATE.
            rate_scalingrecord : [pump_elevation, scaling_length]
                * pump_elevation (double) is the elevation of the multi-aquifer
                  well pump (PUMP_ELEVATION). PUMP_ELEVATION should not be less
                  than the bottom elevation (BOTTOM) of the multi-aquifer well.
                * scaling_length (double) height above the pump elevation
                  (SCALING_LENGTH). If the simulated well head is below this
                  elevation (pump elevation plus the scaling length), then the
                  pumping rate is reduced.
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

    auxiliary = ListTemplateGenerator(("gwf6", "maw", "options", "auxiliary"))
    head_filerecord = ListTemplateGenerator(
        ("gwf6", "maw", "options", "head_filerecord")
    )
    budget_filerecord = ListTemplateGenerator(
        ("gwf6", "maw", "options", "budget_filerecord")
    )
    ts_filerecord = ListTemplateGenerator(
        ("gwf6", "maw", "options", "ts_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwf6", "maw", "options", "obs_filerecord")
    )
    packagedata = ListTemplateGenerator(
        ("gwf6", "maw", "packagedata", "packagedata")
    )
    connectiondata = ListTemplateGenerator(
        ("gwf6", "maw", "connectiondata", "connectiondata")
    )
    perioddata = ListTemplateGenerator(("gwf6", "maw", "period", "perioddata"))
    package_abbr = "gwfmaw"
    _package_type = "maw"
    dfn_file_name = "gwf-maw.dfn"

    dfn = [
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
            "name print_head",
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
            "name head_filerecord",
            "type record head fileout headfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name head",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name headfile",
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
            "name no_well_storage",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name flow_correction",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name flowing_wells",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name shutdown_theta",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name shutdown_kappa",
            "type double precision",
            "reader urword",
            "optional true",
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
            "block options",
            "name mover",
            "type keyword",
            "tagged true",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name nmawwells",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray wellno radius bottom strt condeqn ngwfnodes aux "
            "boundname",
            "shape (nmawwells)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name wellno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name radius",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name bottom",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
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
            "name condeqn",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name ngwfnodes",
            "type integer",
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
            "block connectiondata",
            "name connectiondata",
            "type recarray wellno icon cellid scrn_top scrn_bot hk_skin "
            "radius_skin",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name wellno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block connectiondata",
            "name icon",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block connectiondata",
            "name cellid",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name scrn_top",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name scrn_bot",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name hk_skin",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name radius_skin",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
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
            "name perioddata",
            "type recarray wellno mawsetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name wellno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name mawsetting",
            "type keystring status flowing_wellrecord rate well_head "
            "head_limit shutoffrecord rate_scalingrecord auxiliaryrecord",
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
            "name flowing_wellrecord",
            "type record flowing_well fwelev fwcond fwrlen",
            "shape",
            "tagged",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name flowing_well",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name fwelev",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name fwcond",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name fwrlen",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name rate",
            "type double precision",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name well_head",
            "type double precision",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name head_limit",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name shutoffrecord",
            "type record shut_off minrate maxrate",
            "shape",
            "tagged",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name shut_off",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name minrate",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name maxrate",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name rate_scalingrecord",
            "type record rate_scaling pump_elevation scaling_length",
            "shape",
            "tagged",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name rate_scaling",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name pump_elevation",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name scaling_length",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
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
        auxiliary=None,
        boundnames=None,
        print_input=None,
        print_head=None,
        print_flows=None,
        save_flows=None,
        head_filerecord=None,
        budget_filerecord=None,
        no_well_storage=None,
        flow_correction=None,
        flowing_wells=None,
        shutdown_theta=None,
        shutdown_kappa=None,
        timeseries=None,
        observations=None,
        mover=None,
        nmawwells=None,
        packagedata=None,
        connectiondata=None,
        perioddata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "maw", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_head = self.build_mfdata("print_head", print_head)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.head_filerecord = self.build_mfdata(
            "head_filerecord", head_filerecord
        )
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.no_well_storage = self.build_mfdata(
            "no_well_storage", no_well_storage
        )
        self.flow_correction = self.build_mfdata(
            "flow_correction", flow_correction
        )
        self.flowing_wells = self.build_mfdata("flowing_wells", flowing_wells)
        self.shutdown_theta = self.build_mfdata(
            "shutdown_theta", shutdown_theta
        )
        self.shutdown_kappa = self.build_mfdata(
            "shutdown_kappa", shutdown_kappa
        )
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.mover = self.build_mfdata("mover", mover)
        self.nmawwells = self.build_mfdata("nmawwells", nmawwells)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.connectiondata = self.build_mfdata(
            "connectiondata", connectiondata
        )
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
