# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowPts(mfpackage.MFPackage):
    """
    ModflowPts defines a pts package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_option : string
        * print_option (string) is a flag that controls printing of convergence
          information from the solver. NONE means print nothing. SUMMARY means
          print only the total number of iterations and nonlinear residual
          reduction summaries. ALL means print linear matrix solver convergence
          information to the solution listing file and model specific linear
          matrix solver convergence information to each model listing file in
          addition to SUMMARY information. NONE is default if PRINT_OPTION is
          not specified.
    complexity : string
        * complexity (string) is an optional keyword that defines default non-
          linear and linear solver parameters. SIMPLE - indicates that default
          solver input values will be defined that work well for nearly linear
          models. This would be used for models that do not include nonlinear
          stress packages and models that are either confined or consist of a
          single unconfined layer that is thick enough to contain the water
          table within a single layer. MODERATE - indicates that default solver
          input values will be defined that work well for moderately nonlinear
          models. This would be used for models that include nonlinear stress
          packages and models that consist of one or more unconfined layers.
          The MODERATE option should be used when the SIMPLE option does not
          result in successful convergence. COMPLEX - indicates that default
          solver input values will be defined that work well for highly
          nonlinear models. This would be used for models that include
          nonlinear stress packages and models that consist of one or more
          unconfined layers representing complex geology and surface-
          water/groundwater interaction. The COMPLEX option should be used when
          the MODERATE option does not result in successful convergence. Non-
          linear and linear solver parameters assigned using a specified
          complexity can be modified in the NONLINEAR and LINEAR blocks. If the
          COMPLEXITY option is not specified, NONLINEAR and LINEAR variables
          will be assigned the simple complexity values.
    csv_output_filerecord : [csvfile]
        * csvfile (string) name of the ascii comma separated values output file
          to write solver convergence information. If PRINT_OPTION is NONE or
          SUMMARY, comma separated values output includes maximum head change
          convergence information at the end of each outer iteration for each
          time step. If PRINT_OPTION is ALL, comma separated values output
          includes maximum head change and maximum residual convergence
          information for the solution and each model (if the solution includes
          more than one model) and linear acceleration information for each
          inner iteration.
    csv_outer_output_filerecord : [outer_csvfile]
        * outer_csvfile (string) name of the ascii comma separated values
          output file to write maximum dependent-variable (for example, head)
          change convergence information at the end of each outer iteration for
          each time step.
    csv_inner_output_filerecord : [inner_csvfile]
        * inner_csvfile (string) name of the ascii comma separated values
          output file to write solver convergence information. Comma separated
          values output includes maximum dependent-variable (for example, head)
          change and maximum residual convergence information for the solution
          and each model (if the solution includes more than one model) and
          linear acceleration information for each inner iteration.
    no_ptcrecord : [no_ptc_option]
        * no_ptc_option (string) is an optional keyword that is used to define
          options for disabling pseudo-transient continuation (PTC). FIRST is
          an optional keyword to disable PTC for the first stress period, if
          steady-state and one or more model is using the Newton-Raphson
          formulation. ALL is an optional keyword to disable PTC for all
          steady-state stress periods for models using the Newton-Raphson
          formulation. If NO_PTC_OPTION is not specified, the NO_PTC ALL option
          is used.
    ats_outer_maximum_fraction : double
        * ats_outer_maximum_fraction (double) real value defining the fraction
          of the maximum allowable outer iterations used with the Adaptive Time
          Step (ATS) capability if it is active. If this value is set to zero
          by the user, then this solution will have no effect on ATS behavior.
          This value must be greater than or equal to zero and less than or
          equal to 0.5 or the program will terminate with an error. If it is
          not specified by the user, then it is assigned a default value of one
          third. When the number of outer iterations for this solution is less
          than the product of this value and the maximum allowable outer
          iterations, then ATS will increase the time step length by a factor
          of DTADJ in the ATS input file. When the number of outer iterations
          for this solution is greater than the maximum allowable outer
          iterations minus the product of this value and the maximum allowable
          outer iterations, then the ATS (if active) will decrease the time
          step length by a factor of 1 / DTADJ.
    outer_maximum : integer
        * outer_maximum (integer) integer value defining the maximum number of
          outer (nonlinear) iterations -- that is, calls to the solution
          routine. For a linear problem OUTER_MAXIMUM should be 1.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    csv_output_filerecord = ListTemplateGenerator(
        ("pts", "options", "csv_output_filerecord")
    )
    csv_outer_output_filerecord = ListTemplateGenerator(
        ("pts", "options", "csv_outer_output_filerecord")
    )
    csv_inner_output_filerecord = ListTemplateGenerator(
        ("pts", "options", "csv_inner_output_filerecord")
    )
    no_ptcrecord = ListTemplateGenerator(("pts", "options", "no_ptcrecord"))
    package_abbr = "pts"
    _package_type = "pts"
    dfn_file_name = "sln-pts.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name print_option",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name complexity",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name csv_output_filerecord",
            "type record csv_output fileout csvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "deprecated 6.1.1",
        ],
        [
            "block options",
            "name csv_output",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
            "deprecated 6.1.1",
        ],
        [
            "block options",
            "name csvfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
            "deprecated 6.1.1",
        ],
        [
            "block options",
            "name csv_outer_output_filerecord",
            "type record csv_outer_output fileout outer_csvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name csv_outer_output",
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
            "name outer_csvfile",
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
            "name csv_inner_output_filerecord",
            "type record csv_inner_output fileout inner_csvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name csv_inner_output",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name inner_csvfile",
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
            "name no_ptcrecord",
            "type record no_ptc no_ptc_option",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name no_ptc",
            "type keyword",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged true",
        ],
        [
            "block options",
            "name no_ptc_option",
            "type string",
            "in_record true",
            "reader urword",
            "optional true",
            "tagged false",
        ],
        [
            "block options",
            "name ats_outer_maximum_fraction",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name outer_maximum",
            "type integer",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        print_option=None,
        complexity=None,
        csv_output_filerecord=None,
        csv_outer_output_filerecord=None,
        csv_inner_output_filerecord=None,
        no_ptcrecord=None,
        ats_outer_maximum_fraction=None,
        outer_maximum=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            simulation, "pts", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_option = self.build_mfdata("print_option", print_option)
        self.complexity = self.build_mfdata("complexity", complexity)
        self.csv_output_filerecord = self.build_mfdata(
            "csv_output_filerecord", csv_output_filerecord
        )
        self.csv_outer_output_filerecord = self.build_mfdata(
            "csv_outer_output_filerecord", csv_outer_output_filerecord
        )
        self.csv_inner_output_filerecord = self.build_mfdata(
            "csv_inner_output_filerecord", csv_inner_output_filerecord
        )
        self.no_ptcrecord = self.build_mfdata("no_ptcrecord", no_ptcrecord)
        self.ats_outer_maximum_fraction = self.build_mfdata(
            "ats_outer_maximum_fraction", ats_outer_maximum_fraction
        )
        self.outer_maximum = self.build_mfdata("outer_maximum", outer_maximum)
        self._init_complete = True
