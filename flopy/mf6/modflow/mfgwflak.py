# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwflak(mfpackage.MFPackage):
    """
    ModflowGwflak defines a lak package within a gwf6 model.

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
          provided with the list of lake cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of lake
          information will be written to the listing file immediately after it
          is read.
    print_stage : boolean
        * print_stage (boolean) keyword to indicate that the list of lake
          stages will be printed to the listing file for every stress period in
          which "HEAD PRINT" is specified in Output Control. If there is no
          Output Control option and PRINT_STAGE is specified, then stages are
          printed for the last time step of each stress period.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of lake flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that lake flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    stage_filerecord : [stagefile]
        * stagefile (string) name of the binary output file to write stage
          information.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    ts_filerecord : [ts6_filename]
        * ts6_filename (string) defines a time-series file defining time series
          that can be used to assign time-varying values. See the "Time-
          Variable Input" section for instructions on using the time-series
          capability.
    obs_filerecord : [obs6_filename]
        * obs6_filename (string) name of input file to define observations for
          the LAK package. See the "Observation utility" section for
          instructions for preparing observation input files. Table
          reftable:obstype lists observation type(s) supported by the LAK
          package.
    mover : boolean
        * mover (boolean) keyword to indicate that this instance of the LAK
          Package can be used with the Water Mover (MVR) Package. When the
          MOVER option is specified, additional memory is allocated within the
          package to store the available, provided, and received water.
    surfdep : double
        * surfdep (double) real value that defines the surface depression depth
          for VERTICAL lake-GWF connections. If specified, SURFDEP must be
          greater than or equal to zero. If SURFDEP is not specified, a default
          value of zero is used for all vertical lake-GWF connections.
    time_conversion : double
        * time_conversion (double) value that is used in converting outlet flow
          terms that use Manning's equation or gravitational acceleration to
          consistent time units. TIME_CONVERSION should be set to 1.0, 60.0,
          3,600.0, 86,400.0, and 31,557,600.0 when using time units
          (TIME_UNITS) of seconds, minutes, hours, days, or years in the
          simulation, respectively. CONVTIME does not need to be specified if
          no lake outlets are specified or TIME_UNITS are seconds.
    length_conversion : double
        * length_conversion (double) real value that is used in converting
          outlet flow terms that use Manning's equation or gravitational
          acceleration to consistent length units. LENGTH_CONVERSION should be
          set to 3.28081, 1.0, and 100.0 when using length units (LENGTH_UNITS)
          of feet, meters, or centimeters in the simulation, respectively.
          LENGTH_CONVERSION does not need to be specified if no lake outlets
          are specified or LENGTH_UNITS are meters.
    nlakes : integer
        * nlakes (integer) value specifying the number of lakes that will be
          simulated for all stress periods.
    noutlets : integer
        * noutlets (integer) value specifying the number of outlets that will
          be simulated for all stress periods. If NOUTLETS is not specified, a
          default value of zero is used.
    ntables : integer
        * ntables (integer) value specifying the number of lakes tables that
          will be used to define the lake stage, volume relation, and surface
          area. If NTABLES is not specified, a default value of zero is used.
    packagedata : [lakeno, strt, nlakeconn, aux, boundname]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified PACKAGEDATA data on the line. LAKENO
          must be greater than zero and less than or equal to NLAKES. Lake
          information must be specified for every lake or the program will
          terminate with an error. The program will also terminate with an
          error if information for a lake is specified more than once.
        * strt (double) real value that defines the starting stage for the
          lake.
        * nlakeconn (integer) integer value that defines the number of GWF
          cells connected to this (LAKENO) lake. There can only be one vertical
          lake connection to each GWF cell. NLAKECONN must be greater than
          zero.
        * aux (double) represents the values of the auxiliary variables for
          each lake. The values of auxiliary variables must be present for each
          lake. The values must be specified in the order of the auxiliary
          variables specified in the OPTIONS block. If the package supports
          time series and the Options block includes a TIMESERIESFILE entry
          (see the "Time-Variable Input" section), values can be obtained from
          a time series by entering the time-series name in place of a numeric
          value.
        * boundname (string) name of the lake cell. BOUNDNAME is an ASCII
          character variable that can contain as many as 40 characters. If
          BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    connectiondata : [lakeno, iconn, cellid, claktype, bedleak, belev, telev,
      connlen, connwidth]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified CONNECTIONDATA data on the line. LAKENO
          must be greater than zero and less than or equal to NLAKES. Lake
          connection information must be specified for every lake connection to
          the GWF model (NLAKECONN) or the program will terminate with an
          error. The program will also terminate with an error if connection
          information for a lake connection to the GWF model is specified more
          than once.
        * iconn (integer) integer value that defines the GWF connection number
          for this lake connection entry. ICONN must be greater than zero and
          less than or equal to NLAKECONN for lake LAKENO.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell.
        * claktype (string) character string that defines the lake-GWF
          connection type for the lake connection. Possible lake-GWF connection
          type strings include: VERTICAL--character keyword to indicate the
          lake-GWF connection is vertical and connection conductance
          calculations use the hydraulic conductivity corresponding to the
          :math:`K_{33}` tensor component defined for CELLID in the NPF
          package. HORIZONTAL--character keyword to indicate the lake-GWF
          connection is horizontal and connection conductance calculations use
          the hydraulic conductivity corresponding to the :math:`K_{11}` tensor
          component defined for CELLID in the NPF package. EMBEDDEDH--character
          keyword to indicate the lake-GWF connection is embedded in a single
          cell and connection conductance calculations use the hydraulic
          conductivity corresponding to the :math:`K_{11}` tensor component
          defined for CELLID in the NPF package. EMBEDDEDV--character keyword
          to indicate the lake-GWF connection is embedded in a single cell and
          connection conductance calculations use the hydraulic conductivity
          corresponding to the :math:`K_{33}` tensor component defined for
          CELLID in the NPF package. Embedded lakes can only be connected to a
          single cell (NLAKCONN = 1) and there must be a lake table associated
          with each embedded lake.
        * bedleak (double) character string or real value that defines the bed
          leakance for the lake-GWF connection. BEDLEAK must be greater than or
          equal to zero or specified to be NONE. If BEDLEAK is specified to be
          NONE, the lake-GWF connection conductance is solely a function of
          aquifer properties in the connected GWF cell and lakebed sediments
          are assumed to be absent.
        * belev (double) real value that defines the bottom elevation for a
          HORIZONTAL lake-GWF connection. Any value can be specified if
          CLAKTYPE is VERTICAL, EMBEDDEDH, or EMBEDDEDV. If CLAKTYPE is
          HORIZONTAL and BELEV is not equal to TELEV, BELEV must be greater
          than or equal to the bottom of the GWF cell CELLID. If BELEV is equal
          to TELEV, BELEV is reset to the bottom of the GWF cell CELLID.
        * telev (double) real value that defines the top elevation for a
          HORIZONTAL lake-GWF connection. Any value can be specified if
          CLAKTYPE is VERTICAL, EMBEDDEDH, or EMBEDDEDV. If CLAKTYPE is
          HORIZONTAL and TELEV is not equal to BELEV, TELEV must be less than
          or equal to the top of the GWF cell CELLID. If TELEV is equal to
          BELEV, TELEV is reset to the top of the GWF cell CELLID.
        * connlen (double) real value that defines the distance between the
          connected GWF CELLID node and the lake for a HORIZONTAL, EMBEDDEDH,
          or EMBEDDEDV lake-GWF connection. CONLENN must be greater than zero
          for a HORIZONTAL, EMBEDDEDH, or EMBEDDEDV lake-GWF connection. Any
          value can be specified if CLAKTYPE is VERTICAL.
        * connwidth (double) real value that defines the connection face width
          for a HORIZONTAL lake-GWF connection. CONNWIDTH must be greater than
          zero for a HORIZONTAL lake-GWF connection. Any value can be specified
          if CLAKTYPE is VERTICAL, EMBEDDEDH, or EMBEDDEDV.
    tables : [lakeno, tab6_filename]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified TABLES data on the line. LAKENO must be
          greater than zero and less than or equal to NLAKES. The program will
          terminate with an error if table information for a lake is specified
          more than once or the number of specified tables is less than
          NTABLES.
        * tab6_filename (string) character string that defines the path and
          filename for the file containing lake table data for the lake
          connection. The CTABNAME file includes the number of entries in the
          file and the relation between stage, surface area, and volume for
          each entry in the file. Lake table files for EMBEDDEDH and EMBEDDEDV
          lake-GWF connections also include lake-GWF exchange area data for
          each entry in the file. Input instructions for the CTABNAME file is
          included at the LAK package lake table file input instructions
          section.
    outlets : [outletno, lakein, lakeout, couttype, invert, width, rough,
      slope]
        * outletno (integer) integer value that defines the outlet number
          associated with the specified OUTLETS data on the line. OUTLETNO must
          be greater than zero and less than or equal to NOUTLETS. Outlet
          information must be specified for every outlet or the program will
          terminate with an error. The program will also terminate with an
          error if information for a outlet is specified more than once.
        * lakein (integer) integer value that defines the lake number that
          outlet is connected to. LAKEIN must be greater than zero and less
          than or equal to NLAKES.
        * lakeout (integer) integer value that defines the lake number that
          outlet discharge from lake outlet OUTLETNO is routed to. LAKEOUT must
          be greater than or equal to zero and less than or equal to NLAKES. If
          LAKEOUT is zero, outlet discharge from lake outlet OUTLETNO is
          discharged to an external boundary.
        * couttype (string) character string that defines the outlet type for
          the outlet OUTLETNO. Possible COUTTYPE strings include: SPECIFIED--
          character keyword to indicate the outlet is defined as a specified
          flow. MANNING--character keyword to indicate the outlet is defined
          using Manning's equation. WEIR--character keyword to indicate the
          outlet is defined using a sharp weir equation.
        * invert (double) real value that defines the invert elevation for the
          lake outlet. Any value can be specified if COUTTYPE is SPECIFIED. If
          the Options block includes a TIMESERIESFILE entry (see the "Time-
          Variable Input" section), values can be obtained from a time series
          by entering the time-series name in place of a numeric value.
        * width (double) real value that defines the width of the lake outlet.
          Any value can be specified if COUTTYPE is SPECIFIED. If the Options
          block includes a TIMESERIESFILE entry (see the "Time-Variable Input"
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        * rough (double) real value that defines the roughness coefficient for
          the lake outlet. Any value can be specified if COUTTYPE is not
          MANNING. If the Options block includes a TIMESERIESFILE entry (see
          the "Time-Variable Input" section), values can be obtained from a
          time series by entering the time-series name in place of a numeric
          value.
        * slope (double) real value that defines the bed slope for the lake
          outlet. Any value can be specified if COUTTYPE is not MANNING. If the
          Options block includes a TIMESERIESFILE entry (see the "Time-Variable
          Input" section), values can be obtained from a time series by
          entering the time-series name in place of a numeric value.
    lakeperioddata : [lakeno, laksetting]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified PERIOD data on the line. LAKENO must be
          greater than zero and less than or equal to NLAKES.
        * laksetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          LAKSETTING string include: STATUS, STAGE, STAGE, EVAPORATION,
          RUNOFFON, WITHDRAWAL, and AUXILIARY.
            status : [string]
                * status (string) keyword option to define lake status. STATUS
                  can be ACTIVE, INACTIVE, or CONSTANT. By default, STATUS is
                  ACTIVE.
            stage : [string]
                * stage (string) real or character value that defines the stage
                  for the lake. The specified STAGE is only applied if the lake
                  is a constant stage lake. If the Options block includes a
                  TIMESERIESFILE entry (see the "Time-Variable Input" section),
                  values can be obtained from a time series by entering the
                  time-series name in place of a numeric value.
            rainfall : [string]
                * rainfall (string) real or character value that defines the
                  rainfall rate :math:`(LT^{-1})` for the lake. Value must be
                  greater than or equal to zero. If the Options block includes
                  a TIMESERIESFILE entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            evaporation : [string]
                * evaporation (string) real or character value that defines the
                  maximum evaporation rate :math:`(LT^{-1})` for the lake.
                  Value must be greater than or equal to zero. If the Options
                  block includes a TIMESERIESFILE entry (see the "Time-Variable
                  Input" section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            runoff : [string]
                * runoff (string) real or character value that defines the
                  runoff rate :math:`(L^3 T^{-1})` for the lake. Value must be
                  greater than or equal to zero. If the Options block includes
                  a TIMESERIESFILE entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            withdrawal : [string]
                * withdrawal (string) real or character value that defines the
                  maximum withdrawal rate :math:`(L^3 T^{-1})` for the lake.
                  Value must be greater than or equal to zero. If the Options
                  block includes a TIMESERIESFILE entry (see the "Time-Variable
                  Input" section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
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
    outletperioddata : [outletno, outletsetting]
        * outletno (integer) integer value that defines the outlet number
          associated with the specified PERIOD data on the line. OUTLETNO must
          be greater than zero and less than or equal to NOUTLETS.
        * outletsetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          OUTLETSETTING string include: RATE, INVERT, WIDTH, SLOPE, and ROUGH.
            rate : [string]
                * rate (string) real or character value that defines the
                  extraction rate for the lake outflow. A positive value
                  indicates inflow and a negative value indicates outflow from
                  the lake. RATE only applies to active (IBOUND :math:`>` 0)
                  lakes. A specified RATE is only applied if COUTTYPE for the
                  OUTLETNO is SPECIFIED. If the Options block includes a
                  TIMESERIESFILE entry (see the "Time-Variable Input" section),
                  values can be obtained from a time series by entering the
                  time-series name in place of a numeric value. By default, the
                  RATE for each SPECIFIED lake outlet is zero.
            invert : [string]
                * invert (string) real or character value that defines the
                  invert elevation for the lake outlet. A specified INVERT
                  value is only used for active lakes if COUTTYPE for lake
                  outlet OUTLETNO is not SPECIFIED. If the Options block
                  includes a TIMESERIESFILE entry (see the "Time-Variable
                  Input" section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            width : [string]
                * width (string) real or character value that defines the width
                  of the lake outlet. A specified WIDTH value is only used for
                  active lakes if COUTTYPE for lake outlet OUTLETNO is not
                  SPECIFIED. If the Options block includes a TIMESERIESFILE
                  entry (see the "Time-Variable Input" section), values can be
                  obtained from a time series by entering the time-series name
                  in place of a numeric value.
            slope : [string]
                * slope (string) real or character value that defines the bed
                  slope for the lake outlet. A specified SLOPE value is only
                  used for active lakes if COUTTYPE for lake outlet OUTLETNO is
                  MANNING. If the Options block includes a TIMESERIESFILE entry
                  (see the "Time-Variable Input" section), values can be
                  obtained from a time series by entering the time-series name
                  in place of a numeric value.
            rough : [string]
                * rough (string) real or character value that defines the width
                  of the lake outlet. A specified WIDTH value is only used for
                  active lakes if COUTTYPE for lake outlet OUTLETNO is not
                  SPECIFIED. If the Options block includes a TIMESERIESFILE
                  entry (see the "Time-Variable Input" section), values can be
                  obtained from a time series by entering the time-series name
                  in place of a numeric value.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    auxiliary = ListTemplateGenerator(('gwf6', 'lak', 'options', 
                                       'auxiliary'))
    stage_filerecord = ListTemplateGenerator(('gwf6', 'lak', 'options', 
                                              'stage_filerecord'))
    budget_filerecord = ListTemplateGenerator(('gwf6', 'lak', 'options', 
                                               'budget_filerecord'))
    ts_filerecord = ListTemplateGenerator(('gwf6', 'lak', 'options', 
                                           'ts_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwf6', 'lak', 'options', 
                                            'obs_filerecord'))
    packagedata = ListTemplateGenerator(('gwf6', 'lak', 'packagedata', 
                                         'packagedata'))
    connectiondata = ListTemplateGenerator(('gwf6', 'lak', 
                                            'connectiondata', 
                                            'connectiondata'))
    tables = ListTemplateGenerator(('gwf6', 'lak', 'tables', 'tables'))
    outlets = ListTemplateGenerator(('gwf6', 'lak', 'outlets', 
                                     'outlets'))
    lakeperioddata = ListTemplateGenerator(('gwf6', 'lak', 'period', 
                                            'lakeperioddata'))
    outletperioddata = ListTemplateGenerator(('gwf6', 'lak', 'period', 
                                              'outletperioddata'))
    package_abbr = "gwflak"
    package_type = "lak"
    dfn_file_name = "gwf-lak.dfn"

    dfn = [["block options", "name auxiliary", "type string", 
            "shape (naux)", "reader urword", "optional true"],
           ["block options", "name boundnames", "type keyword", "shape", 
            "reader urword", "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_stage", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name stage_filerecord", 
            "type record stage fileout stagefile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name stage", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name stagefile", "type string", 
            "preserve_case true", "shape", "in_record true", "reader urword", 
            "tagged false", "optional false"],
           ["block options", "name budget_filerecord", 
            "type record budget fileout budgetfile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name budget", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name fileout", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name budgetfile", "type string", 
            "preserve_case true", "shape", "in_record true", "reader urword", 
            "tagged false", "optional false"],
           ["block options", "name ts_filerecord", 
            "type record ts6 filein ts6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name ts6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name ts6_filename", "type string", 
            "preserve_case true", "in_record true", "reader urword", 
            "optional false", "tagged false"],
           ["block options", "name obs_filerecord", 
            "type record obs6 filein obs6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name obs6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name obs6_filename", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword", "optional false"],
           ["block options", "name mover", "type keyword", "tagged true", 
            "reader urword", "optional true"],
           ["block options", "name surfdep", "type double precision", 
            "reader urword", "optional true"],
           ["block options", "name time_conversion", 
            "type double precision", "reader urword", "optional true"],
           ["block options", "name length_conversion", 
            "type double precision", "reader urword", "optional true"],
           ["block dimensions", "name nlakes", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name noutlets", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name ntables", "type integer", 
            "reader urword", "optional false"],
           ["block packagedata", "name packagedata", 
            "type recarray lakeno strt nlakeconn aux boundname", 
            "shape (maxbound)", "reader urword"],
           ["block packagedata", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block packagedata", "name strt", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name nlakeconn", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name aux", "type double precision", 
            "in_record true", "tagged false", "shape (naux)", "reader urword", 
            "time_series true", "optional true"],
           ["block packagedata", "name boundname", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "optional true"],
           ["block connectiondata", "name connectiondata", 
            "type recarray lakeno iconn cellid claktype bedleak belev telev " 
            "connlen connwidth", 
            "shape (sum(nlakecon))", "reader urword"],
           ["block connectiondata", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block connectiondata", "name iconn", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block connectiondata", "name cellid", "type integer", 
            "shape (ncelldim)", "tagged false", "in_record true", 
            "reader urword"],
           ["block connectiondata", "name claktype", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name bedleak", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name belev", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name telev", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name connlen", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name connwidth", 
            "type double precision", "shape", "tagged false", 
            "in_record true", "reader urword"],
           ["block tables", "name tables", 
            "type recarray lakeno tab6 filein tab6_filename", 
            "shape (ntables)", "reader urword"],
           ["block tables", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block tables", "name tab6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block tables", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block tables", "name tab6_filename", "type string", 
            "preserve_case true", "in_record true", "reader urword", 
            "optional false", "tagged false"],
           ["block outlets", "name outlets", 
            "type recarray outletno lakein lakeout couttype invert width " 
            "rough slope", 
            "shape (noutlets)", "reader urword"],
           ["block outlets", "name outletno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block outlets", "name lakein", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block outlets", "name lakeout", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block outlets", "name couttype", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block outlets", "name invert", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block outlets", "name width", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block outlets", "name rough", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block outlets", "name slope", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name lakeperioddata", 
            "type recarray lakeno laksetting", "shape", "reader urword"],
           ["block period", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block period", "name laksetting", 
            "type keystring status stage rainfall evaporation runoff " 
            "withdrawal auxiliaryrecord", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name status", "type string", "shape", 
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name stage", "type string", "shape", 
            "tagged true", "in_record true", "time_series true", 
            "reader urword"],
           ["block period", "name rainfall", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name evaporation", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name runoff", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name withdrawal", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name auxiliaryrecord", 
            "type record auxiliary auxname auxval", "shape", "tagged", 
            "in_record true", "reader urword"],
           ["block period", "name auxiliary", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name auxname", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name auxval", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name outletperioddata", 
            "type recarray outletno outletsetting", "shape", "reader urword"],
           ["block period", "name outletno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block period", "name outletsetting", 
            "type keystring rate invert width slope rough", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name rate", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name invert", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name rough", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name width", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name slope", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"]]

    def __init__(self, model, loading_package=False, auxiliary=None,
                 boundnames=None, print_input=None, print_stage=None,
                 print_flows=None, save_flows=None, stage_filerecord=None,
                 budget_filerecord=None, ts_filerecord=None,
                 obs_filerecord=None, mover=None, surfdep=None,
                 time_conversion=None, length_conversion=None, nlakes=None,
                 noutlets=None, ntables=None, packagedata=None,
                 connectiondata=None, tables=None, outlets=None,
                 lakeperioddata=None, outletperioddata=None, fname=None,
                 pname=None, parent_file=None):
        super(ModflowGwflak, self).__init__(model, "lak", fname, pname,
                                            loading_package, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary",  auxiliary)
        self.boundnames = self.build_mfdata("boundnames",  boundnames)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_stage = self.build_mfdata("print_stage",  print_stage)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.stage_filerecord = self.build_mfdata("stage_filerecord", 
                                                  stage_filerecord)
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.ts_filerecord = self.build_mfdata("ts_filerecord",  ts_filerecord)
        self.obs_filerecord = self.build_mfdata("obs_filerecord", 
                                                obs_filerecord)
        self.mover = self.build_mfdata("mover",  mover)
        self.surfdep = self.build_mfdata("surfdep",  surfdep)
        self.time_conversion = self.build_mfdata("time_conversion", 
                                                 time_conversion)
        self.length_conversion = self.build_mfdata("length_conversion", 
                                                   length_conversion)
        self.nlakes = self.build_mfdata("nlakes",  nlakes)
        self.noutlets = self.build_mfdata("noutlets",  noutlets)
        self.ntables = self.build_mfdata("ntables",  ntables)
        self.packagedata = self.build_mfdata("packagedata",  packagedata)
        self.connectiondata = self.build_mfdata("connectiondata", 
                                                connectiondata)
        self.tables = self.build_mfdata("tables",  tables)
        self.outlets = self.build_mfdata("outlets",  outlets)
        self.lakeperioddata = self.build_mfdata("lakeperioddata", 
                                                lakeperioddata)
        self.outletperioddata = self.build_mfdata("outletperioddata", 
                                                  outletperioddata)
