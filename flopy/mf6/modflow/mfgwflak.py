# autogenerated file, do not modify

from os import PathLike, curdir
from typing import Union

from flopy.mf6.data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator
from flopy.mf6.mfpackage import MFChildPackages, MFPackage


class ModflowGwflak(MFPackage):
    """
    ModflowGwflak defines a LAK package.

    Parameters
    ----------
    auxiliary : [string]
        defines an array of one or more auxiliary variable names.  there is no limit on
        the number of auxiliary variables that can be provided on this line; however,
        lists of information provided in subsequent blocks must have a column of data
        for each auxiliary variable name defined here.   the number of auxiliary
        variables detected on this line determines the value for naux.  comments cannot
        be provided anywhere on this line as they will be interpreted as auxiliary
        variable names.  auxiliary variables may not be used by the package, but they
        will be available for use by other parts of the program.  the program will
        terminate with an error if auxiliary variables are specified on more than one
        line in the options block.
    boundnames : keyword
        keyword to indicate that boundary names may be provided with the list of lake
        cells.
    print_input : keyword
        keyword to indicate that the list of lake information will be written to the
        listing file immediately after it is read.
    print_stage : keyword
        keyword to indicate that the list of lake {#2} will be printed to the listing
        file for every stress period in which 'head print' is specified in output
        control.  if there is no output control option and print_{#3} is specified,
        then {#2} are printed for the last time step of each stress period.
    print_flows : keyword
        keyword to indicate that the list of lake flow rates will be printed to the
        listing file for every stress period time step in which 'budget print' is
        specified in output control.  if there is no output control option and
        'print_flows' is specified, then flow rates are printed for the last time step
        of each stress period.
    save_flows : keyword
        keyword to indicate that lake flow terms will be written to the file specified
        with 'budget fileout' in output control.
    stage_filerecord : (stagefile)
        * stagefile : string
                name of the binary output file to write stage information.

    budget_filerecord : (budgetfile)
        * budgetfile : string
                name of the binary output file to write budget information.

    budgetcsv_filerecord : (budgetcsvfile)
        * budgetcsvfile : string
                name of the comma-separated value (CSV) output file to write budget summary
                information.  A budget summary record will be written to this file for each
                time step of the simulation.

    package_convergence_filerecord : (package_convergence_filename)
        * package_convergence_filename : string
                name of the comma spaced values output file to write package convergence
                information.

    timeseries : record ts6 filein ts6_filename
        Contains data for the ts package. Data can be passed as a dictionary to the ts
        package with variable names as keys and package data as values. Data for the
        timeseries variable is also acceptable. See ts package documentation for more
        information.
    observations : record obs6 filein obs6_filename
        Contains data for the obs package. Data can be passed as a dictionary to the
        obs package with variable names as keys and package data as values. Data for
        the observations variable is also acceptable. See obs package documentation for
        more information.
    mover : keyword
        keyword to indicate that this instance of the lak package can be used with the
        water mover (mvr) package.  when the mover option is specified, additional
        memory is allocated within the package to store the available, provided, and
        received water.
    surfdep : double precision
        real value that defines the surface depression depth for vertical lake-gwf
        connections. if specified, surfdep must be greater than or equal to zero. if
        surfdep is not specified, a default value of zero is used for all vertical
        lake-gwf connections.
    maximum_iterations : integer
        integer value that defines the maximum number of newton-raphson iterations
        allowed for a lake. by default, maximum_iterations is equal to 100.
        maximum_iterations would only need to be increased from the default value if
        one or more lakes in a simulation has a large water budget error.
    maximum_stage_change : double precision
        real value that defines the lake stage closure tolerance. by default,
        maximum_stage_change is equal to :math:`1 times 10^{-5}`. the
        maximum_stage_change would only need to be increased or decreased from the
        default value if the water budget error for one or more lakes is too small or
        too large, respectively.
    time_conversion : double precision
        real value that is used to convert user-specified manning's roughness
        coefficients or gravitational acceleration used to calculate outlet flows from
        seconds to model time units. time_conversion should be set to 1.0, 60.0,
        3,600.0, 86,400.0, and 31,557,600.0 when using time units (time_units) of
        seconds, minutes, hours, days, or years in the simulation, respectively.
        convtime does not need to be specified if no lake outlets are specified or
        time_units are seconds.
    length_conversion : double precision
        real value that is used to convert outlet user-specified manning's roughness
        coefficients or gravitational acceleration used to calculate outlet flows from
        meters to model length units. length_conversion should be set to 3.28081, 1.0,
        and 100.0 when using length units (length_units) of feet, meters, or
        centimeters in the simulation, respectively. length_conversion does not need to
        be specified if no lake outlets are specified or length_units are meters.
    nlakes : integer
        value specifying the number of lakes that will be simulated for all stress
        periods.
    noutlets : integer
        value specifying the number of outlets that will be simulated for all stress
        periods. if noutlets is not specified, a default value of zero is used.
    ntables : integer
        value specifying the number of lakes tables that will be used to define the
        lake stage, volume relation, and surface area. if ntables is not specified, a
        default value of zero is used.
    packagedata : [list]
    connectiondata : [list]
    tables : [list]
    outlets : [list]
    perioddata : list

    """

    auxiliary = ArrayTemplateGenerator(("gwf6", "lak", "options", "auxiliary"))
    stage_filerecord = ListTemplateGenerator(
        ("gwf6", "lak", "options", "stage_filerecord")
    )
    budget_filerecord = ListTemplateGenerator(
        ("gwf6", "lak", "options", "budget_filerecord")
    )
    budgetcsv_filerecord = ListTemplateGenerator(
        ("gwf6", "lak", "options", "budgetcsv_filerecord")
    )
    package_convergence_filerecord = ListTemplateGenerator(
        ("gwf6", "lak", "options", "package_convergence_filerecord")
    )
    ts_filerecord = ListTemplateGenerator(("gwf6", "lak", "options", "ts_filerecord"))
    obs_filerecord = ListTemplateGenerator(("gwf6", "lak", "options", "obs_filerecord"))
    packagedata = ListTemplateGenerator(("gwf6", "lak", "packagedata", "packagedata"))
    connectiondata = ListTemplateGenerator(
        ("gwf6", "lak", "connectiondata", "connectiondata")
    )
    tables = ListTemplateGenerator(("gwf6", "lak", "tables", "tables"))
    outlets = ListTemplateGenerator(("gwf6", "lak", "outlets", "outlets"))
    perioddata = ListTemplateGenerator(("gwf6", "lak", "period", "perioddata"))
    package_abbr = "gwflak"
    _package_type = "lak"
    dfn_file_name = "gwf-lak.dfn"
    dfn = [
        ["header", "multi-package", "package-type advanced-stress-package"],
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
            "name print_stage",
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
            "name stage_filerecord",
            "type record stage fileout stagefile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name stage",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name stagefile",
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
            "name package_convergence_filerecord",
            "type record package_convergence fileout package_convergence_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name package_convergence",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name package_convergence_filename",
            "type string",
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
            "construct_data observations",
            "parameter_name continuous",
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
            "block options",
            "name surfdep",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name maximum_iterations",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name maximum_stage_change",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name time_conversion",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name length_conversion",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name nlakes",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name noutlets",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name ntables",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray ifno strt nlakeconn aux boundname",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name ifno",
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
            "name nlakeconn",
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
            "type recarray ifno iconn cellid claktype bedleak belev telev connlen connwidth",
            "shape (sum(nlakeconn))",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name ifno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block connectiondata",
            "name iconn",
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
            "name claktype",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name bedleak",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name belev",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name telev",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name connlen",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block connectiondata",
            "name connwidth",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block tables",
            "name tables",
            "type recarray ifno tab6 filein tab6_filename",
            "shape (ntables)",
            "reader urword",
        ],
        [
            "block tables",
            "name ifno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block tables",
            "name tab6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block tables",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block tables",
            "name tab6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block outlets",
            "name outlets",
            "type recarray outletno lakein lakeout couttype invert width rough slope",
            "shape (noutlets)",
            "reader urword",
        ],
        [
            "block outlets",
            "name outletno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block outlets",
            "name lakein",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block outlets",
            "name lakeout",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block outlets",
            "name couttype",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block outlets",
            "name invert",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block outlets",
            "name width",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block outlets",
            "name rough",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block outlets",
            "name slope",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
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
            "type recarray number laksetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name number",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name laksetting",
            "type keystring status stage rainfall evaporation runoff inflow withdrawal rate invert width slope rough auxiliaryrecord",
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
            "name stage",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "time_series true",
            "reader urword",
        ],
        [
            "block period",
            "name rainfall",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name evaporation",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name runoff",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name inflow",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name withdrawal",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
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
            "name invert",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name rough",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name width",
            "type string",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name slope",
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
        auxiliary=None,
        boundnames=None,
        print_input=None,
        print_stage=None,
        print_flows=None,
        save_flows=None,
        stage_filerecord=None,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        package_convergence_filerecord=None,
        timeseries=None,
        observations=None,
        mover=None,
        surfdep=None,
        maximum_iterations=None,
        maximum_stage_change=None,
        time_conversion=None,
        length_conversion=None,
        nlakes=None,
        noutlets=None,
        ntables=None,
        packagedata=None,
        connectiondata=None,
        tables=None,
        outlets=None,
        perioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        """
        ModflowGwflak defines a LAK package.

        Parameters
        ----------
        model
            Model that this package is a part of. Package is automatically
            added to model when it is initialized.
        loading_package : bool
            Do not set this parameter. It is intended for debugging and internal
            processing purposes only.
        auxiliary : [string]
            defines an array of one or more auxiliary variable names.  there is no limit on
            the number of auxiliary variables that can be provided on this line; however,
            lists of information provided in subsequent blocks must have a column of data
            for each auxiliary variable name defined here.   the number of auxiliary
            variables detected on this line determines the value for naux.  comments cannot
            be provided anywhere on this line as they will be interpreted as auxiliary
            variable names.  auxiliary variables may not be used by the package, but they
            will be available for use by other parts of the program.  the program will
            terminate with an error if auxiliary variables are specified on more than one
            line in the options block.
        boundnames : keyword
            keyword to indicate that boundary names may be provided with the list of lake
            cells.
        print_input : keyword
            keyword to indicate that the list of lake information will be written to the
            listing file immediately after it is read.
        print_stage : keyword
            keyword to indicate that the list of lake {#2} will be printed to the listing
            file for every stress period in which 'head print' is specified in output
            control.  if there is no output control option and print_{#3} is specified,
            then {#2} are printed for the last time step of each stress period.
        print_flows : keyword
            keyword to indicate that the list of lake flow rates will be printed to the
            listing file for every stress period time step in which 'budget print' is
            specified in output control.  if there is no output control option and
            'print_flows' is specified, then flow rates are printed for the last time step
            of each stress period.
        save_flows : keyword
            keyword to indicate that lake flow terms will be written to the file specified
            with 'budget fileout' in output control.
        stage_filerecord : record
        budget_filerecord : record
        budgetcsv_filerecord : record
        package_convergence_filerecord : record
        timeseries : record ts6 filein ts6_filename
            Contains data for the ts package. Data can be passed as a dictionary to the ts
            package with variable names as keys and package data as values. Data for the
            timeseries variable is also acceptable. See ts package documentation for more
            information.
        observations : record obs6 filein obs6_filename
            Contains data for the obs package. Data can be passed as a dictionary to the
            obs package with variable names as keys and package data as values. Data for
            the observations variable is also acceptable. See obs package documentation for
            more information.
        mover : keyword
            keyword to indicate that this instance of the lak package can be used with the
            water mover (mvr) package.  when the mover option is specified, additional
            memory is allocated within the package to store the available, provided, and
            received water.
        surfdep : double precision
            real value that defines the surface depression depth for vertical lake-gwf
            connections. if specified, surfdep must be greater than or equal to zero. if
            surfdep is not specified, a default value of zero is used for all vertical
            lake-gwf connections.
        maximum_iterations : integer
            integer value that defines the maximum number of newton-raphson iterations
            allowed for a lake. by default, maximum_iterations is equal to 100.
            maximum_iterations would only need to be increased from the default value if
            one or more lakes in a simulation has a large water budget error.
        maximum_stage_change : double precision
            real value that defines the lake stage closure tolerance. by default,
            maximum_stage_change is equal to :math:`1 times 10^{-5}`. the
            maximum_stage_change would only need to be increased or decreased from the
            default value if the water budget error for one or more lakes is too small or
            too large, respectively.
        time_conversion : double precision
            real value that is used to convert user-specified manning's roughness
            coefficients or gravitational acceleration used to calculate outlet flows from
            seconds to model time units. time_conversion should be set to 1.0, 60.0,
            3,600.0, 86,400.0, and 31,557,600.0 when using time units (time_units) of
            seconds, minutes, hours, days, or years in the simulation, respectively.
            convtime does not need to be specified if no lake outlets are specified or
            time_units are seconds.
        length_conversion : double precision
            real value that is used to convert outlet user-specified manning's roughness
            coefficients or gravitational acceleration used to calculate outlet flows from
            meters to model length units. length_conversion should be set to 3.28081, 1.0,
            and 100.0 when using length units (length_units) of feet, meters, or
            centimeters in the simulation, respectively. length_conversion does not need to
            be specified if no lake outlets are specified or length_units are meters.
        nlakes : integer
            value specifying the number of lakes that will be simulated for all stress
            periods.
        noutlets : integer
            value specifying the number of outlets that will be simulated for all stress
            periods. if noutlets is not specified, a default value of zero is used.
        ntables : integer
            value specifying the number of lakes tables that will be used to define the
            lake stage, volume relation, and surface area. if ntables is not specified, a
            default value of zero is used.
        packagedata : [list]
        connectiondata : [list]
        tables : [list]
        outlets : [list]
        perioddata : list

        filename : str
            File name for this package.
        pname : str
            Package name for this package.
        parent_file : MFPackage
            Parent package file that references this package. Only needed for
            utility packages (mfutl*). For example, mfutllaktab package must have
            a mfgwflak package parent_file.
        """

        super().__init__(model, "lak", filename, pname, loading_package, **kwargs)

        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_stage = self.build_mfdata("print_stage", print_stage)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.stage_filerecord = self.build_mfdata("stage_filerecord", stage_filerecord)
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.budgetcsv_filerecord = self.build_mfdata(
            "budgetcsv_filerecord", budgetcsv_filerecord
        )
        self.package_convergence_filerecord = self.build_mfdata(
            "package_convergence_filerecord", package_convergence_filerecord
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
        self.surfdep = self.build_mfdata("surfdep", surfdep)
        self.maximum_iterations = self.build_mfdata(
            "maximum_iterations", maximum_iterations
        )
        self.maximum_stage_change = self.build_mfdata(
            "maximum_stage_change", maximum_stage_change
        )
        self.time_conversion = self.build_mfdata("time_conversion", time_conversion)
        self.length_conversion = self.build_mfdata(
            "length_conversion", length_conversion
        )
        self.nlakes = self.build_mfdata("nlakes", nlakes)
        self.noutlets = self.build_mfdata("noutlets", noutlets)
        self.ntables = self.build_mfdata("ntables", ntables)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.connectiondata = self.build_mfdata("connectiondata", connectiondata)
        self.tables = self.build_mfdata("tables", tables)
        self.outlets = self.build_mfdata("outlets", outlets)
        self.perioddata = self.build_mfdata("perioddata", perioddata)

        self._init_complete = True
