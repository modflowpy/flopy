from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwflak(mfpackage.MFPackage):
    """
    ModflowGwflak defines a lak package within a gwf6 model.

    Attributes
    ----------
    auxiliary : [(auxiliary : string)]
        auxiliary : defines an array of one or more auxiliary variable names.
          There is no limit on the number of auxiliary variables that can be
          provided on this line; however, lists of information provided in
          subsequent blocks must have a column of data for each auxiliary
          variable name defined here. The number of auxiliary variables
          detected on this line determines the value for naux. Comments cannot
          be provided anywhere on this line as they will be interpreted as
          auxiliary variable names. Auxiliary variables may not be used by the
          package, but they will be available for use by other parts of the
          program. The program will terminate with an error if auxiliary
          variables are specified on more than one line in the options block.
    boundnames : (boundnames : boolean)
        boundnames : keyword to indicate that boundary names may be provided
          with the list of lake cells.
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of lake information
          will be written to the listing file immediately after it is read.
    print_stage : (print_stage : boolean)
        print_stage : keyword to indicate that the list of lake stages will be
          printed to the listing file for every stress period in which ``HEAD
          PRINT'' is specified in Output Control. If there is no Output Control
          option and PRINT\_STAGE is specified, then stages are printed for the
          last time step of each stress period.
    print_flows : (print_flows : boolean)
        print_flows : keyword to indicate that the list of lake flow rates will
          be printed to the listing file for every stress period time step in
          which ``BUDGET PRINT'' is specified in Output Control. If there is no
          Output Control option and PRINT\_FLOWS is specified, then flow rates
          are printed for the last time step of each stress period.
    save_flows : (save_flows : boolean)
        save_flows : keyword to indicate that lake flow terms will be written
          to the file specified with ``BUDGET FILEOUT'' in Output Control.
    stage_filerecord : [(stagefile : string)]
        stagefile : name of the binary output file to write stage information.
    budget_filerecord : [(budgetfile : string)]
        budgetfile : name of the binary output file to write budget
          information.
    ts_filerecord : [(ts6_filename : string)]
        ts6_filename : defines a time-series file defining time series that can
          be used to assign time-varying values. See the ``Time-Variable
          Input'' section for instructions on using the time-series capability.
    obs_filerecord : [(obs6_filename : string)]
        obs6_filename : name of input file to define observations for the LAK
          package. See the ``Observation utility'' section for instructions for
          preparing observation input files. Table obstype lists
          observation type(s) supported by the LAK package.
    mover : (mover : boolean)
        mover : keyword to indicate that this instance of the LAK Package can
          be used with the Water Mover (MVR) Package. When the MOVER option is
          specified, additional memory is allocated within the package to store
          the available, provided, and received water.
    surfdep : (surfdep : double)
        surfdep : real value that defines the surface depression depth for
          VERTICAL lake-GWF connections. If specified,
          surfdep must be greater than or equal to zero. If
          SURFDEP is not specified, a default value of zero is used
          for all vertical lake-GWF connections.
    time_conversion : (time_conversion : double)
        time_conversion : value that is used in converting outlet flow terms
          that use Manning's equation or gravitational acceleration to
          consistent time units. time\_conversion should be set to
          1.0, 60.0, 3,600.0, 86,400.0, and 31,557,600.0 when using time units
          (time\_units) of seconds, minutes, hours, days, or years in
          the simulation, respectively. convtime does not need to be
          specified if no lake outlets are specified or time\_units
          are seconds.
    length_conversion : (length_conversion : double)
        length_conversion : real value that is used in converting outlet flow
          terms that use Manning's equation or gravitational acceleration to
          consistent length units. length\_conversion should be set to
          3.28081, 1.0, and 100.0 when using length units
          (length\_units) of feet, meters, or centimeters in the
          simulation, respectively. length\_conversion does not need
          to be specified if no lake outlets are specified or
          length\_units are meters.
    nlakes : (nlakes : integer)
        nlakes : value specifying the number of lakes that will be simulated
          for all stress periods.
    noutlets : (noutlets : integer)
        noutlets : value specifying the number of outlets that will be
          simulated for all stress periods. If NOUTLETS is not
          specified, a default value of zero is used.
    ntables : (ntables : integer)
        ntables : value specifying the number of lakes tables that will be used
          to define the lake stage, volume relation, and surface area. If
          NTABLES is not specified, a default value of zero is used.
    lakrecarray_package : [(lakeno : integer), (strt : double), (nlakeconn :
      integer), (aux : double), (boundname : string)]
        lakeno : integer value that defines the lake number associated with the
          specified PACKAGEDATA data on the line. lakeno must be
          greater than zero and less than or equal to nlakes. Lake
          information must be specified for every lake or the program will
          terminate with an error. The program will also terminate with an
          error if information for a lake is specified more than once.
        strt : real value that defines the starting stage for the lake.
        nlakeconn : integer value that defines the number of GWF nodes
          connected to this (lakeno) lake. There can only be one
          vertical lake connection to each GWF node.
          nlakeconn must be greater than zero.
        aux : represents the values of the auxiliary variables for each lake.
          The values of auxiliary variables must be present for each lake. The
          values must be specified in the order of the auxiliary variables
          specified in the OPTIONS block. If the package supports time series
          and the Options block includes a TIMESERIESFILE entry (see the
          ``Time-Variable Input'' section), values can be obtained from a time
          series by entering the time-series name in place of a numeric value.
        boundname : name of the lake cell. boundname is an ASCII character
          variable that can contain as many as 40 characters. If boundname
          contains spaces in it, then the entire name must be enclosed within
          single quotes.
    lakrecarray : [(lakeno : integer), (iconn : integer), (cellid : (integer,
      ...)), (claktype : string), (bedleak : double), (belev : double), (telev
      : double), (connlen : double), (connwidth : double)]
        lakeno : integer value that defines the lake number associated with the
          specified CONNECTIONDATA data on the line. lakeno must be
          greater than zero and less than or equal to nlakes. Lake
          connection information must be specified for every lake connection to
          the GWF model (nlakeconn) or the program will terminate with
          an error. The program will also terminate with an error if connection
          information for a lake connection to the GWF model is specified more
          than once.
        iconn : integer value that defines the GWF connection number
          for this lake connection entry. iconn must be greater than
          zero and less than or equal to nlakeconn for lake
          lakeno.
        cellid : is the cell identifier, and depends on the type of grid that
          is used for the simulation. For a structured grid that uses the DIS
          input file, cellid is the layer, row, and column. For a grid that
          uses the DISV input file, cellid is the layer and cell2d number. If
          the model uses the unstructured discretization (DISU) input file,
          then cellid is the node number for the cell.
        claktype : character string that defines the lake-GWF
          connection type for the lake connection. Possible lake-GWF
          connection type strings include: VERTICAL--character keyword
          to indicate the lake-GWF connection is vertical and
          connection conductance calculations use the hydraulic conductivity
          corresponding to the $K_{33$ tensor component defined for
          cellid in the NPF package. HORIZONTAL--character
          keyword to indicate the lake-GWF connection is horizontal
          and connection conductance calculations use the hydraulic
          conductivity corresponding to the $K_{11$ tensor component defined
          for cellid in the NPF package. EMBEDDEDH--character
          keyword to indicate the lake-GWF connection is embedded in a
          single cell and connection conductance calculations use the hydraulic
          conductivity corresponding to the $K_{11$ tensor component defined
          for cellid in the NPF package. EMBEDDEDV--character
          keyword to indicate the lake-GWF connection is embedded in a
          single cell and connection conductance calculations use the hydraulic
          conductivity corresponding to the $K_{33$ tensor component defined
          for cellid in the NPF package. Embedded lakes can only be
          connected to a single cell (nlakconn = 1) and there must be
          a lake table associated with each embedded lake.
        bedleak : character string or real value that defines the bed leakance
          for the lake-GWF connection. bedleak must be
          greater than or equal to zero or specified to be none. If
          bedleak is specified to be none, the
          lake-GWF connection conductance is solely a function of
          aquifer properties in the connected GWF cell and lakebed
          sediments are assumed to be absent.
        belev : real value that defines the bottom elevation for a
          HORIZONTAL lake-GWF connection. Any value can be
          specified if claktype is VERTICAL,
          EMBEDDEDH, or EMBEDDEDV. If claktype is
          HORIZONTAL and belev is not equal to
          telev, belev must be greater than or equal to the
          bottom of the GWF cell cellid. If belev is
          equal to telev, belev is reset to the bottom of the
          GWF cell cellid.
        telev : real value that defines the top elevation for a
          HORIZONTAL lake-GWF connection. Any value can be
          specified if claktype is VERTICAL,
          EMBEDDEDH, or EMBEDDEDV. If claktype is
          HORIZONTAL and telev is not equal to
          belev, telev must be less than or equal to the top
          of the GWF cell cellid. If telev is equal
          to belev, telev is reset to the top of the
          GWF cell cellid.
        connlen : real value that defines the distance between the connected
          GWF cellid node and the lake for a
          HORIZONTAL, EMBEDDEDH, or EMBEDDEDV
          lake-GWF connection. connlen must be greater than
          zero for a HORIZONTAL, EMBEDDEDH, or
          EMBEDDEDV lake-GWF connection. Any value can be
          specified if claktype is VERTICAL.
        connwidth : real value that defines the connection face width for a
          HORIZONTAL lake-GWF connection. connwidth
          must be greater than zero for a HORIZONTAL lake-GWF
          connection. Any value can be specified if claktype is
          VERTICAL, EMBEDDEDH, or EMBEDDEDV.
    lake_tablesrecarray : [(lakeno : integer), (tab6_filename : string)]
        lakeno : integer value that defines the lake number associated with the
          specified TABLES data on the line. lakeno must be greater
          than zero and less than or equal to nlakes. The program will
          terminate with an error if table information for a lake is specified
          more than once or the number of specified tables is less than
          ntables.
        tab6_filename : character string that defines the path and filename for
          the file containing lake table data for the lake connection. The
          ctabname file includes the number of entries in the file and
          the relation between stage, surface area, and volume for each entry
          in the file. Lake table files for EMBEDDEDH and
          EMBEDDEDV lake-GWF connections also include
          lake-GWF exchange area data for each entry in the file.
          Input instructions for the ctabname file is included at the
          LAK package lake table file input instructions section.
    outletsrecarray : [(outletno : integer), (lakein : integer), (lakeout :
      integer), (couttype : string), (invert : double), (width : double),
      (rough : double), (slope : double)]
        outletno : integer value that defines the outlet number associated with
          the specified OUTLETS data on the line. outletno must be
          greater than zero and less than or equal to noutlets. Outlet
          information must be specified for every outlet or the program will
          terminate with an error. The program will also terminate with an
          error if information for a outlet is specified more than once.
        lakein : integer value that defines the lake number that outlet is
          connected to. lakein must be greater than zero and less than
          or equal to nlakes.
        lakeout : integer value that defines the lake number that outlet
          discharge from lake outlet outletno is routed to.
          lakeout must be greater than or equal to zero and less than
          or equal to nlakes. If lakeout is zero, outlet
          discharge from lake outlet outletno is discharged to an
          external boundary.
        couttype : character string that defines the outlet type for the outlet
          outletno. Possible couttype strings include:
          SPECIFIED--character keyword to indicate the outlet is
          defined as a specified flow. MANNING--character keyword to
          indicate the outlet is defined using Manning's equation.
          WEIR--character keyword to indicate the outlet is defined
          using a sharp weir equation.
        invert : real value that defines the invert elevation for the lake
          outlet. Any value can be specified if couttype is
          SPECIFIED. If the Options block includes a
          TIMESERIESFILE entry (see the ``Time-Variable Input''
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        width : real value that defines the width of the lake outlet. Any value
          can be specified if couttype is SPECIFIED. If the
          Options block includes a TIMESERIESFILE entry (see the
          ``Time-Variable Input'' section), values can be obtained from a time
          series by entering the time-series name in place of a numeric value.
        rough : real value that defines the roughness coefficient for the lake
          outlet. Any value can be specified if couttype is not
          MANNING. If the Options block includes a
          TIMESERIESFILE entry (see the ``Time-Variable Input''
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        slope : real value that defines the bed slope for the lake outlet. Any
          value can be specified if couttype is not MANNING.
          If the Options block includes a TIMESERIESFILE entry (see
          the ``Time-Variable Input'' section), values can be obtained from a
          time series by entering the time-series name in place of a numeric
          value.
    lakeperiodrecarray : [(lakeno : integer), (laksetting : keystring)]
        lakeno : integer value that defines the lake number associated with the
          specified PERIOD data on the line. lakeno must be greater
          than zero and less than or equal to nlakes.
        laksetting : line of information that is parsed into a keyword and
          values. Keyword values that can be used to start the
          laksetting string include: STATUS, STAGE,
          RAINFALL, EVAPORATION, RUNOFF,
          WITHDRAWAL, and AUXILIARY.
    rate : (rate : string)
        rate : real or character value that defines the extraction rate for the
          lake outflow. A positive value indicates inflow and a negative value
          indicates outflow from the lake. rate only applies to active
          (IBOUND$>0$) lakes. A specified rate is only
          applied if couttype for the outletno is
          SPECIFIED. If the Options block includes a
          TIMESERIESFILE entry (see the ``Time-Variable Input''
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value. By default, the
          rate for each SPECIFIED lake outlet is zero.
    auxname : (auxname : string)
        auxname : name for the auxiliary variable to be assigned
          auxval. auxname must match one of the auxiliary
          variable names defined in the OPTIONS block. If
          auxname does not match one of the auxiliary variable names
          defined in the OPTIONS block the data are ignored.
    auxval : (auxval : double)
        auxval : value for the auxiliary variable. If the Options block
          includes a TIMESERIESFILE entry (see the ``Time-Variable
          Input'' section), values can be obtained from a time series by
          entering the time-series name in place of a numeric value.
    outletperiodrecarray : [(outletno : integer), (outletsetting : keystring)]
        outletno : integer value that defines the outlet number associated with
          the specified PERIOD data on the line. outletno must be
          greater than zero and less than or equal to noutlets.
        outletsetting : line of information that is parsed into a keyword and
          values. Keyword values that can be used to start the
          outletsetting string include: RATE,
          INVERT, WIDTH, SLOPE, and ROUGH.

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
                 auxname=None, auxval=None, outletperiodrecarray=None,
                 fname=None, pname=None, parent_file=None):
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
        self.auxname = self.build_mfdata("auxname",  auxname)
        self.auxval = self.build_mfdata("auxval",  auxval)
        self.outletperiodrecarray = self.build_mfdata("outletperiodrecarray", 
                                                      outletperiodrecarray)
