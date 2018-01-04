# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwflak(mfpackage.MFPackage):
    """
    ModflowGwflak defines a lak package within a gwf6 model.

    Parameters
    ----------
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
          there is no Output Control option and PRINT_FLOWS is specified, then
          flow rates are printed for the last time step of each stress period.
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
          for texttt{VERTICAL} lake-texttt{GWF} connections. If specified,
          texttt{surfdep} must be greater than or equal to zero. If
          texttt{SURFDEP} is not specified, a default value of zero is used for
          all vertical lake-texttt{GWF} connections.
    time_conversion : double
        * time_conversion (double) value that is used in converting outlet flow
          terms that use Manning's equation or gravitational acceleration to
          consistent time units. texttt{time_conversion} should be set to 1.0,
          60.0, 3,600.0, 86,400.0, and 31,557,600.0 when using time units
          (texttt{time_units}) of seconds, minutes, hours, days, or years in
          the simulation, respectively. texttt{convtime} does not need to be
          specified if no lake outlets are specified or texttt{time_units} are
          seconds.
    length_conversion : double
        * length_conversion (double) real value that is used in converting
          outlet flow terms that use Manning's equation or gravitational
          acceleration to consistent length units. texttt{length_conversion}
          should be set to 3.28081, 1.0, and 100.0 when using length units
          (texttt{length_units}) of feet, meters, or centimeters in the
          simulation, respectively. texttt{length_conversion} does not need to
          be specified if no lake outlets are specified or texttt{length_units}
          are meters.
    nlakes : integer
        * nlakes (integer) value specifying the number of lakes that will be
          simulated for all stress periods.
    noutlets : integer
        * noutlets (integer) value specifying the number of outlets that will
          be simulated for all stress periods. If texttt{NOUTLETS} is not
          specified, a default value of zero is used.
    ntables : integer
        * ntables (integer) value specifying the number of lakes tables that
          will be used to define the lake stage, volume relation, and surface
          area. If texttt{NTABLES} is not specified, a default value of zero is
          used.
    lakrecarray_package : [lakeno, strt, nlakeconn, aux, boundname]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified PACKAGEDATA data on the line.
          texttt{lakeno} must be greater than zero and less than or equal to
          texttt{nlakes}. Lake information must be specified for every lake or
          the program will terminate with an error. The program will also
          terminate with an error if information for a lake is specified more
          than once.
        * strt (double) real value that defines the starting stage for the
          lake.
        * nlakeconn (integer) integer value that defines the number of
          texttt{GWF} nodes connected to this (texttt{lakeno}) lake. There can
          only be one vertical lake connection to each texttt{GWF} node.
          texttt{nlakeconn} must be greater than zero.
        * aux (double) represents the values of the auxiliary variables for
          each lake. The values of auxiliary variables must be present for each
          lake. The values must be specified in the order of the auxiliary
          variables specified in the OPTIONS block. If the package supports
          time series and the Options block includes a TIMESERIESFILE entry
          (see the "Time-Variable Input" section), values can be obtained from
          a time series by entering the time-series name in place of a numeric
          value.
        * boundname (string) name of the lake cell. boundname is an ASCII
          character variable that can contain as many as 40 characters. If
          boundname contains spaces in it, then the entire name must be
          enclosed within single quotes.
    lakrecarray : [lakeno, iconn, cellid, claktype, bedleak, belev, telev,
      connlen, connwidth]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified CONNECTIONDATA data on the line.
          texttt{lakeno} must be greater than zero and less than or equal to
          texttt{nlakes}. Lake connection information must be specified for
          every lake connection to the GWF model (texttt{nlakeconn}) or the
          program will terminate with an error. The program will also terminate
          with an error if connection information for a lake connection to the
          GWF model is specified more than once.
        * iconn (integer) integer value that defines the texttt{GWF} connection
          number for this lake connection entry. texttt{iconn} must be greater
          than zero and less than or equal to texttt{nlakeconn} for lake
          texttt{lakeno}.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, cellid is the layer, row, and column.
          For a grid that uses the DISV input file, cellid is the layer and
          cell2d number. If the model uses the unstructured discretization
          (DISU) input file, then cellid is the node number for the cell.
        * claktype (string) character string that defines the lake-texttt{GWF}
          connection type for the lake connection. Possible lake-texttt{GWF}
          connection type strings include: texttt{VERTICAL}--character keyword
          to indicate the lake-texttt{GWF} connection is vertical and
          connection conductance calculations use the hydraulic conductivity
          corresponding to the :math:`K_{33}` tensor component defined for
          texttt{cellid} in the NPF package. texttt{HORIZONTAL}--character
          keyword to indicate the lake-texttt{GWF} connection is horizontal and
          connection conductance calculations use the hydraulic conductivity
          corresponding to the :math:`K_{11}` tensor component defined for
          texttt{cellid} in the NPF package. texttt{EMBEDDEDH}--character
          keyword to indicate the lake-texttt{GWF} connection is embedded in a
          single cell and connection conductance calculations use the hydraulic
          conductivity corresponding to the :math:`K_{11}` tensor component
          defined for texttt{cellid} in the NPF package.
          texttt{EMBEDDEDV}--character keyword to indicate the lake-texttt{GWF}
          connection is embedded in a single cell and connection conductance
          calculations use the hydraulic conductivity corresponding to the
          :math:`K_{33}` tensor component defined for \texttt{cellid} in the
          NPF package. Embedded lakes can only be connected to a single cell
          (\texttt{nlakconn = 1}) and there must be a lake table associated
          with each embedded lake.
        * bedleak (double) character string or real value that defines the bed
          leakance for the lake-texttt{GWF} connection. texttt{bedleak} must be
          greater than or equal to zero or specified to be texttt{none}. If
          texttt{bedleak} is specified to be texttt{none}, the lake-texttt{GWF}
          connection conductance is solely a function of aquifer properties in
          the connected texttt{GWF} cell and lakebed sediments are assumed to
          be absent.
        * belev (double) real value that defines the bottom elevation for a
          texttt{HORIZONTAL} lake-texttt{GWF} connection. Any value can be
          specified if texttt{claktype} is texttt{VERTICAL}, texttt{EMBEDDEDH},
          or texttt{EMBEDDEDV}. If texttt{claktype} is texttt{HORIZONTAL} and
          texttt{belev} is not equal to texttt{telev}, texttt{belev} must be
          greater than or equal to the bottom of the texttt{GWF} cell
          texttt{cellid}. If texttt{belev} is equal to texttt{telev},
          texttt{belev} is reset to the bottom of the texttt{GWF} cell
          texttt{cellid}.
        * telev (double) real value that defines the top elevation for a
          texttt{HORIZONTAL} lake-texttt{GWF} connection. Any value can be
          specified if texttt{claktype} is texttt{VERTICAL}, texttt{EMBEDDEDH},
          or texttt{EMBEDDEDV}. If texttt{claktype} is texttt{HORIZONTAL} and
          texttt{telev} is not equal to texttt{belev}, texttt{telev} must be
          less than or equal to the top of the texttt{GWF} cell texttt{cellid}.
          If texttt{telev} is equal to texttt{belev}, texttt{telev} is reset to
          the top of the texttt{GWF} cell texttt{cellid}.
        * connlen (double) real value that defines the distance between the
          connected texttt{GWF} texttt{cellid} node and the lake for a
          texttt{HORIZONTAL}, texttt{EMBEDDEDH}, or texttt{EMBEDDEDV} lake-
          texttt{GWF} connection. texttt{connlen} must be greater than zero for
          a texttt{HORIZONTAL}, texttt{EMBEDDEDH}, or texttt{EMBEDDEDV} lake-
          texttt{GWF} connection. Any value can be specified if
          texttt{claktype} is texttt{VERTICAL}.
        * connwidth (double) real value that defines the connection face width
          for a texttt{HORIZONTAL} lake-texttt{GWF} connection.
          texttt{connwidth} must be greater than zero for a texttt{HORIZONTAL}
          lake-texttt{GWF} connection. Any value can be specified if
          texttt{claktype} is texttt{VERTICAL}, texttt{EMBEDDEDH}, or
          texttt{EMBEDDEDV}.
    lake_tablesrecarray : [lakeno, tab6_filename]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified TABLES data on the line. texttt{lakeno}
          must be greater than zero and less than or equal to texttt{nlakes}.
          The program will terminate with an error if table information for a
          lake is specified more than once or the number of specified tables is
          less than texttt{ntables}.
        * tab6_filename (string) character string that defines the path and
          filename for the file containing lake table data for the lake
          connection. The texttt{ctabname} file includes the number of entries
          in the file and the relation between stage, surface area, and volume
          for each entry in the file. Lake table files for texttt{EMBEDDEDH}
          and texttt{EMBEDDEDV} lake-texttt{GWF} connections also include lake-
          texttt{GWF} exchange area data for each entry in the file. Input
          instructions for the texttt{ctabname} file is included at the LAK
          package lake table file input instructions section.
    outletsrecarray : [outletno, lakein, lakeout, couttype, invert, width,
      rough, slope]
        * outletno (integer) integer value that defines the outlet number
          associated with the specified OUTLETS data on the line.
          texttt{outletno} must be greater than zero and less than or equal to
          texttt{noutlets}. Outlet information must be specified for every
          outlet or the program will terminate with an error. The program will
          also terminate with an error if information for a outlet is specified
          more than once.
        * lakein (integer) integer value that defines the lake number that
          outlet is connected to. texttt{lakein} must be greater than zero and
          less than or equal to texttt{nlakes}.
        * lakeout (integer) integer value that defines the lake number that
          outlet discharge from lake outlet texttt{outletno} is routed to.
          texttt{lakeout} must be greater than or equal to zero and less than
          or equal to texttt{nlakes}. If texttt{lakeout} is zero, outlet
          discharge from lake outlet texttt{outletno} is discharged to an
          external boundary.
        * couttype (string) character string that defines the outlet type for
          the outlet texttt{outletno}. Possible texttt{couttype} strings
          include: texttt{SPECIFIED}--character keyword to indicate the outlet
          is defined as a specified flow. texttt{MANNING}--character keyword to
          indicate the outlet is defined using Manning's equation.
          texttt{WEIR}--character keyword to indicate the outlet is defined
          using a sharp weir equation.
        * invert (double) real value that defines the invert elevation for the
          lake outlet. Any value can be specified if texttt{couttype} is
          texttt{SPECIFIED}. If the Options block includes a
          texttt{TIMESERIESFILE} entry (see the "Time-Variable Input" section),
          values can be obtained from a time series by entering the time-series
          name in place of a numeric value.
        * width (double) real value that defines the width of the lake outlet.
          Any value can be specified if texttt{couttype} is texttt{SPECIFIED}.
          If the Options block includes a texttt{TIMESERIESFILE} entry (see the
          "Time-Variable Input" section), values can be obtained from a time
          series by entering the time-series name in place of a numeric value.
        * rough (double) real value that defines the roughness coefficient for
          the lake outlet. Any value can be specified if texttt{couttype} is
          not texttt{MANNING}. If the Options block includes a
          texttt{TIMESERIESFILE} entry (see the "Time-Variable Input" section),
          values can be obtained from a time series by entering the time-series
          name in place of a numeric value.
        * slope (double) real value that defines the bed slope for the lake
          outlet. Any value can be specified if texttt{couttype} is not
          texttt{MANNING}. If the Options block includes a
          texttt{TIMESERIESFILE} entry (see the "Time-Variable Input" section),
          values can be obtained from a time series by entering the time-series
          name in place of a numeric value.
    lakeperiodrecarray : [lakeno, laksetting]
        * lakeno (integer) integer value that defines the lake number
          associated with the specified PERIOD data on the line. texttt{lakeno}
          must be greater than zero and less than or equal to texttt{nlakes}.
        * laksetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          texttt{laksetting} string include: texttt{STATUS}, texttt{STAGE},
          texttt{RAINFALL}, texttt{EVAPORATION}, texttt{RUNOFF},
          texttt{WITHDRAWAL}, and texttt{AUXILIARY}.
            runoff : [string]
                * runoff (string) real or character value that defines the
                  runoff rate for the lake. texttt{value} must be greater than
                  or equal to zero. If the Options block includes a
                  texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            evaporation : [string]
                * evaporation (string) real or character value that defines the
                  maximum evaporation rate for the lake. texttt{value} must be
                  greater than or equal to zero. If the Options block includes
                  a texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            rainfall : [string]
                * rainfall (string) real or character value that defines the
                  rainfall rate for the lake. texttt{value} must be greater
                  than or equal to zero. If the Options block includes a
                  texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            status : [string]
                * status (string) keyword option to define lake status.
                  texttt{status} can be texttt{ACTIVE}, texttt{INACTIVE}, or
                  texttt{CONSTANT}. By default, texttt{status} is
                  texttt{ACTIVE}.
            auxiliaryrecord : [auxname, auxval]
                * auxname (string) name for the auxiliary variable to be
                  assigned texttt{auxval}. texttt{auxname} must match one of
                  the auxiliary variable names defined in the texttt{OPTIONS}
                  block. If texttt{auxname} does not match one of the auxiliary
                  variable names defined in the texttt{OPTIONS} block the data
                  are ignored.
                * auxval (double) value for the auxiliary variable. If the
                  Options block includes a texttt{TIMESERIESFILE} entry (see
                  the "Time-Variable Input" section), values can be obtained
                  from a time series by entering the time-series name in place
                  of a numeric value.
            stage : [string]
                * stage (string) real or character value that defines the stage
                  for the lake. The specified texttt{stage} is only applied if
                  the lake is a constant stage lake. If the Options block
                  includes a texttt{TIMESERIESFILE} entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.
            withdrawal : [string]
                * withdrawal (string) real or character value that defines the
                  maximum withdrawal rate for the lake. texttt{value} must be
                  greater than or equal to zero. If the Options block includes
                  a texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
    rate : string
        * rate (string) real or character value that defines the extraction
          rate for the lake outflow. A positive value indicates inflow and a
          negative value indicates outflow from the lake. texttt{rate} only
          applies to active (texttt{IBOUND}:math:`>0`) lakes. A specified
          \texttt{rate} is only applied if \texttt{couttype} for the
          \texttt{outletno} is \texttt{SPECIFIED}. If the Options block
          includes a \texttt{TIMESERIESFILE} entry (see the "Time-Variable
          Input" section), values can be obtained from a time series by
          entering the time-series name in place of a numeric value. By
          default, the \texttt{rate} for each \texttt{SPECIFIED} lake outlet is
          zero.
    outletperiodrecarray : [outletno, outletsetting]
        * outletno (integer) integer value that defines the outlet number
          associated with the specified PERIOD data on the line.
          texttt{outletno} must be greater than zero and less than or equal to
          texttt{noutlets}.
        * outletsetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          texttt{outletsetting} string include: texttt{RATE}, texttt{INVERT},
          texttt{WIDTH}, texttt{SLOPE}, and texttt{ROUGH}.
            rate : [string]
                * rate (string) real or character value that defines the
                  extraction rate for the lake outflow. A positive value
                  indicates inflow and a negative value indicates outflow from
                  the lake. texttt{rate} only applies to active
                  (texttt{IBOUND}:math:`>0`) lakes. A specified \texttt{rate}
                  is only applied if \texttt{couttype} for the
                  \texttt{outletno} is \texttt{SPECIFIED}. If the Options block
                  includes a \texttt{TIMESERIESFILE} entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value. By default, the \texttt{rate} for each
                  \texttt{SPECIFIED} lake outlet is zero.
            invert : [string]
                * invert (string) real or character value that defines the
                  invert elevation for the lake outlet. A specified
                  texttt{invert} value is only used for active lakes if
                  texttt{couttype} for lake outlet texttt{outletno} is not
                  texttt{SPECIFIED}. If the Options block includes a
                  texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            width : [string]
                * width (string) real or character value that defines the width
                  of the lake outlet. A specified texttt{width} value is only
                  used for active lakes if texttt{couttype} for lake outlet
                  texttt{outletno} is not texttt{SPECIFIED}. If the Options
                  block includes a texttt{TIMESERIESFILE} entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.
            slope : [string]
                * slope (string) real or character value that defines the bed
                  slope for the lake outlet. A specified texttt{slope} value is
                  only used for active lakes if texttt{couttype} for lake
                  outlet texttt{outletno} is texttt{MANNING}. If the Options
                  block includes a texttt{TIMESERIESFILE} entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.
            rough : [string]
                * rough (string) real or character value that defines the width
                  of the lake outlet. A specified texttt{width} value is only
                  used for active lakes if texttt{couttype} for lake outlet
                  texttt{outletno} is not texttt{SPECIFIED}. If the Options
                  block includes a texttt{TIMESERIESFILE} entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value.

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
    lakrecarray_package = ListTemplateGenerator(('gwf6', 'lak', 
                                                 'packagedata', 
                                                 'lakrecarray_package'))
    lakrecarray = ListTemplateGenerator(('gwf6', 'lak', 'connectiondata', 
                                         'lakrecarray'))
    lake_tablesrecarray = ListTemplateGenerator(('gwf6', 'lak', 'tables', 
                                                 'lake_tablesrecarray'))
    outletsrecarray = ListTemplateGenerator(('gwf6', 'lak', 'outlets', 
                                             'outletsrecarray'))
    lakeperiodrecarray = ListTemplateGenerator(('gwf6', 'lak', 'period', 
                                                'lakeperiodrecarray'))
    outletperiodrecarray = ListTemplateGenerator(('gwf6', 'lak', 
                                                  'period', 
                                                  'outletperiodrecarray'))
    package_abbr = "gwflak"
    package_type = "lak"
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
           ["block packagedata", "name lakrecarray_package", 
            "type recarray lakeno strt nlakeconn aux boundname", 
            "shape (maxbound)", "reader urword"],
           ["block packagedata", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
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
           ["block connectiondata", "name lakrecarray", 
            "type recarray lakeno iconn cellid claktype bedleak belev telev " 
            "connlen connwidth", 
            "shape (sum(nlakecon))", "reader urword"],
           ["block connectiondata", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name iconn", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
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
           ["block tables", "name lake_tablesrecarray", 
            "type recarray lakeno tab6 filein tab6_filename", 
            "shape (ntables)", "reader urword"],
           ["block tables", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block tables", "name tab6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block tables", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block tables", "name tab6_filename", "type string", 
            "preserve_case true", "in_record true", "reader urword", 
            "optional false", "tagged false"],
           ["block outlets", "name outletsrecarray", 
            "type recarray outletno lakein lakeout couttype invert width " 
            "rough slope", 
            "shape (noutlets)", "reader urword"],
           ["block outlets", "name outletno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block outlets", "name lakein", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block outlets", "name lakeout", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
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
           ["block period", "name lakeperiodrecarray", 
            "type recarray lakeno laksetting", "shape", "reader urword"],
           ["block period", "name lakeno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name laksetting", 
            "type keystring status stage rainfall evaporation runoff " 
            "withdrawal auxiliaryrecord", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name status", "type string", "shape", 
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name rate", "type string", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
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
           ["block period", "name outletperiodrecarray", 
            "type recarray outletno outletsetting", "shape", "reader urword"],
           ["block period", "name outletno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
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

    def __init__(self, model, add_to_package_list=True, auxiliary=None,
                 boundnames=None, print_input=None, print_stage=None,
                 print_flows=None, save_flows=None, stage_filerecord=None,
                 budget_filerecord=None, ts_filerecord=None,
                 obs_filerecord=None, mover=None, surfdep=None,
                 time_conversion=None, length_conversion=None, nlakes=None,
                 noutlets=None, ntables=None, lakrecarray_package=None,
                 lakrecarray=None, lake_tablesrecarray=None,
                 outletsrecarray=None, lakeperiodrecarray=None, rate=None,
                 outletperiodrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwflak, self).__init__(model, "lak", fname, pname,
                                            add_to_package_list, parent_file)        

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
        self.lakrecarray_package = self.build_mfdata("lakrecarray_package", 
                                                     lakrecarray_package)
        self.lakrecarray = self.build_mfdata("lakrecarray",  lakrecarray)
        self.lake_tablesrecarray = self.build_mfdata("lake_tablesrecarray", 
                                                     lake_tablesrecarray)
        self.outletsrecarray = self.build_mfdata("outletsrecarray", 
                                                 outletsrecarray)
        self.lakeperiodrecarray = self.build_mfdata("lakeperiodrecarray", 
                                                    lakeperiodrecarray)
        self.rate = self.build_mfdata("rate",  rate)
        self.outletperiodrecarray = self.build_mfdata("outletperiodrecarray", 
                                                      outletperiodrecarray)
