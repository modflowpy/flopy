# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowIms(mfpackage.MFPackage):
    """
    ModflowIms defines a ims package.

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
          equal to 0.5 or the program will terminate with an error. If it not
          specified by the user, then it is assigned a default value of one
          third. When the number of outer iterations for this solution is less
          than the product of this value and the maximum allowable outer
          iterations, then ATS will increase the time step length by a factor
          of DTADJ in the ATS input file. When the number of outer iterations
          for this solution is greater than the maximum allowable outer
          iterations minus the product of this value and the maximum allowable
          outer iterations, then the ATS (if active) will decrease the time
          step length by a factor of 1 / DTADJ.
    outer_hclose : double
        * outer_hclose (double) real value defining the head change criterion
          for convergence of the outer (nonlinear) iterations, in units of
          length. When the maximum absolute value of the head change at all
          nodes during an iteration is less than or equal to OUTER_HCLOSE,
          iteration stops. Commonly, OUTER_HCLOSE equals 0.01. The OUTER_HCLOSE
          option has been deprecated in favor of the more general OUTER_DVCLOSE
          (for dependent variable), however either one can be specified in
          order to maintain backward compatibility.
    outer_dvclose : double
        * outer_dvclose (double) real value defining the dependent-variable
          (for example, head) change criterion for convergence of the outer
          (nonlinear) iterations, in units of the dependent-variable (for
          example, length for head). When the maximum absolute value of the
          dependent-variable change at all nodes during an iteration is less
          than or equal to OUTER_DVCLOSE, iteration stops. Commonly,
          OUTER_DVCLOSE equals 0.01. The keyword, OUTER_HCLOSE can be still be
          specified instead of OUTER_DVCLOSE for backward compatibility with
          previous versions of MODFLOW 6 but eventually OUTER_HCLOSE will be
          deprecated and specification of OUTER_HCLOSE will cause MODFLOW 6 to
          terminate with an error.
    outer_rclosebnd : double
        * outer_rclosebnd (double) real value defining the residual tolerance
          for convergence of model packages that solve a separate equation not
          solved by the IMS linear solver. This value represents the maximum
          allowable residual between successive outer iterations at any single
          model package element. An example of a model package that would use
          OUTER_RCLOSEBND to evaluate convergence is the SFR package which
          solves a continuity equation for each reach. The OUTER_RCLOSEBND
          option is deprecated and has no effect on simulation results as of
          version 6.1.1. The keyword, OUTER_RCLOSEBND can be still be specified
          for backward compatibility with previous versions of MODFLOW 6 but
          eventually specificiation of OUTER_RCLOSEBND will cause MODFLOW 6 to
          terminate with an error.
    outer_maximum : integer
        * outer_maximum (integer) integer value defining the maximum number of
          outer (nonlinear) iterations -- that is, calls to the solution
          routine. For a linear problem OUTER_MAXIMUM should be 1.
    under_relaxation : string
        * under_relaxation (string) is an optional keyword that defines the
          nonlinear under-relaxation schemes used. Under-relaxation is also
          known as dampening, and is used to reduce the size of the calculated
          dependent variable before proceeding to the next outer iteration.
          Under-relaxation can be an effective tool for highly nonlinear models
          when there are large and often counteracting changes in the
          calculated dependent variable between successive outer iterations. By
          default under-relaxation is not used. NONE - under-relaxation is not
          used (default). SIMPLE - Simple under-relaxation scheme with a fixed
          relaxation factor (UNDER_RELAXATION_GAMMA) is used. COOLEY - Cooley
          under-relaxation scheme is used. DBD - delta-bar-delta under-
          relaxation is used. Note that the under-relaxation schemes are often
          used in conjunction with problems that use the Newton-Raphson
          formulation, however, experience has indicated that they also work
          well for non-Newton problems, such as those with the wet/dry options
          of MODFLOW 6.
    under_relaxation_gamma : double
        * under_relaxation_gamma (double) real value defining either the
          relaxation factor for the SIMPLE scheme or the history or memory term
          factor of the Cooley and delta-bar-delta algorithms. For the SIMPLE
          scheme, a value of one indicates that there is no under-relaxation
          and the full head change is applied. This value can be gradually
          reduced from one as a way to improve convergence; for well behaved
          problems, using a value less than one can increase the number of
          outer iterations required for convergence and needlessly increase run
          times. UNDER_RELAXATION_GAMMA must be greater than zero for the
          SIMPLE scheme or the program will terminate with an error. For the
          Cooley and delta-bar-delta schemes, UNDER_RELAXATION_GAMMA is a
          memory term that can range between zero and one. When
          UNDER_RELAXATION_GAMMA is zero, only the most recent history
          (previous iteration value) is maintained. As UNDER_RELAXATION_GAMMA
          is increased, past history of iteration changes has greater influence
          on the memory term. The memory term is maintained as an exponential
          average of past changes. Retaining some past history can overcome
          granular behavior in the calculated function surface and therefore
          helps to overcome cyclic patterns of non-convergence. The value
          usually ranges from 0.1 to 0.3; a value of 0.2 works well for most
          problems. UNDER_RELAXATION_GAMMA only needs to be specified if
          UNDER_RELAXATION is not NONE.
    under_relaxation_theta : double
        * under_relaxation_theta (double) real value defining the reduction
          factor for the learning rate (under-relaxation term) of the delta-
          bar-delta algorithm. The value of UNDER_RELAXATION_THETA is between
          zero and one. If the change in the dependent-variable (for example,
          head) is of opposite sign to that of the previous iteration, the
          under-relaxation term is reduced by a factor of
          UNDER_RELAXATION_THETA. The value usually ranges from 0.3 to 0.9; a
          value of 0.7 works well for most problems. UNDER_RELAXATION_THETA
          only needs to be specified if UNDER_RELAXATION is DBD.
    under_relaxation_kappa : double
        * under_relaxation_kappa (double) real value defining the increment for
          the learning rate (under-relaxation term) of the delta-bar-delta
          algorithm. The value of UNDER_RELAXATION_kappa is between zero and
          one. If the change in the dependent-variable (for example, head) is
          of the same sign to that of the previous iteration, the under-
          relaxation term is increased by an increment of
          UNDER_RELAXATION_KAPPA. The value usually ranges from 0.03 to 0.3; a
          value of 0.1 works well for most problems. UNDER_RELAXATION_KAPPA
          only needs to be specified if UNDER_RELAXATION is DBD.
    under_relaxation_momentum : double
        * under_relaxation_momentum (double) real value defining the fraction
          of past history changes that is added as a momentum term to the step
          change for a nonlinear iteration. The value of
          UNDER_RELAXATION_MOMENTUM is between zero and one. A large momentum
          term should only be used when small learning rates are expected.
          Small amounts of the momentum term help convergence. The value
          usually ranges from 0.0001 to 0.1; a value of 0.001 works well for
          most problems. UNDER_RELAXATION_MOMENTUM only needs to be specified
          if UNDER_RELAXATION is DBD.
    backtracking_number : integer
        * backtracking_number (integer) integer value defining the maximum
          number of backtracking iterations allowed for residual reduction
          computations. If BACKTRACKING_NUMBER = 0 then the backtracking
          iterations are omitted. The value usually ranges from 2 to 20; a
          value of 10 works well for most problems.
    backtracking_tolerance : double
        * backtracking_tolerance (double) real value defining the tolerance for
          residual change that is allowed for residual reduction computations.
          BACKTRACKING_TOLERANCE should not be less than one to avoid getting
          stuck in local minima. A large value serves to check for extreme
          residual increases, while a low value serves to control step size
          more severely. The value usually ranges from 1.0 to 10:math:`^6`; a
          value of 10:math:`^4` works well for most problems but lower values
          like 1.1 may be required for harder problems. BACKTRACKING_TOLERANCE
          only needs to be specified if BACKTRACKING_NUMBER is greater than
          zero.
    backtracking_reduction_factor : double
        * backtracking_reduction_factor (double) real value defining the
          reduction in step size used for residual reduction computations. The
          value of BACKTRACKING_REDUCTION_FACTOR is between zero and one. The
          value usually ranges from 0.1 to 0.3; a value of 0.2 works well for
          most problems. BACKTRACKING_REDUCTION_FACTOR only needs to be
          specified if BACKTRACKING_NUMBER is greater than zero.
    backtracking_residual_limit : double
        * backtracking_residual_limit (double) real value defining the limit to
          which the residual is reduced with backtracking. If the residual is
          smaller than BACKTRACKING_RESIDUAL_LIMIT, then further backtracking
          is not performed. A value of 100 is suitable for large problems and
          residual reduction to smaller values may only slow down computations.
          BACKTRACKING_RESIDUAL_LIMIT only needs to be specified if
          BACKTRACKING_NUMBER is greater than zero.
    inner_maximum : integer
        * inner_maximum (integer) integer value defining the maximum number of
          inner (linear) iterations. The number typically depends on the
          characteristics of the matrix solution scheme being used. For
          nonlinear problems, INNER_MAXIMUM usually ranges from 60 to 600; a
          value of 100 will be sufficient for most linear problems.
    inner_hclose : double
        * inner_hclose (double) real value defining the head change criterion
          for convergence of the inner (linear) iterations, in units of length.
          When the maximum absolute value of the head change at all nodes
          during an iteration is less than or equal to INNER_HCLOSE, the matrix
          solver assumes convergence. Commonly, INNER_HCLOSE is set equal to or
          an order of magnitude less than the OUTER_HCLOSE value specified for
          the NONLINEAR block. The INNER_HCLOSE keyword has been deprecated in
          favor of the more general INNER_DVCLOSE (for dependent variable),
          however either one can be specified in order to maintain backward
          compatibility.
    inner_dvclose : double
        * inner_dvclose (double) real value defining the dependent-variable
          (for example, head) change criterion for convergence of the inner
          (linear) iterations, in units of the dependent-variable (for example,
          length for head). When the maximum absolute value of the dependent-
          variable change at all nodes during an iteration is less than or
          equal to INNER_DVCLOSE, the matrix solver assumes convergence.
          Commonly, INNER_DVCLOSE is set equal to or an order of magnitude less
          than the OUTER_DVCLOSE value specified for the NONLINEAR block. The
          keyword, INNER_HCLOSE can be still be specified instead of
          INNER_DVCLOSE for backward compatibility with previous versions of
          MODFLOW 6 but eventually INNER_HCLOSE will be deprecated and
          specification of INNER_HCLOSE will cause MODFLOW 6 to terminate with
          an error.
    rcloserecord : [inner_rclose, rclose_option]
        * inner_rclose (double) real value that defines the flow residual
          tolerance for convergence of the IMS linear solver and specific flow
          residual criteria used. This value represents the maximum allowable
          residual at any single node. Value is in units of length cubed per
          time, and must be consistent with MODFLOW 6 length and time units.
          Usually a value of :math:`1.0 \\times 10^{-1}` is sufficient for the
          flow-residual criteria when meters and seconds are the defined
          MODFLOW 6 length and time.
        * rclose_option (string) an optional keyword that defines the specific
          flow residual criterion used. STRICT--an optional keyword that is
          used to specify that INNER_RCLOSE represents a infinity-Norm
          (absolute convergence criteria) and that the dependent-variable (for
          example, head) and flow convergence criteria must be met on the first
          inner iteration (this criteria is equivalent to the criteria used by
          the MODFLOW-2005 PCG package (Hill, 1990)). L2NORM_RCLOSE--an
          optional keyword that is used to specify that INNER_RCLOSE represents
          a L-2 Norm closure criteria instead of a infinity-Norm (absolute
          convergence criteria). When L2NORM_RCLOSE is specified, a reasonable
          initial INNER_RCLOSE value is 0.1 times the number of active cells
          when meters and seconds are the defined MODFLOW 6 length and time.
          RELATIVE_RCLOSE--an optional keyword that is used to specify that
          INNER_RCLOSE represents a relative L-2 Norm reduction closure
          criteria instead of a infinity-Norm (absolute convergence criteria).
          When RELATIVE_RCLOSE is specified, a reasonable initial INNER_RCLOSE
          value is :math:`1.0 \\times 10^{-4}` and convergence is achieved for
          a given inner (linear) iteration when :math:`\\Delta h \\le`
          INNER_DVCLOSE and the current L-2 Norm is :math:`\\le` the product of
          the RELATIVE_RCLOSE and the initial L-2 Norm for the current inner
          (linear) iteration. If RCLOSE_OPTION is not specified, an absolute
          residual (infinity-norm) criterion is used.
    linear_acceleration : string
        * linear_acceleration (string) a keyword that defines the linear
          acceleration method used by the default IMS linear solvers. CG -
          preconditioned conjugate gradient method. BICGSTAB - preconditioned
          bi-conjugate gradient stabilized method.
    relaxation_factor : double
        * relaxation_factor (double) optional real value that defines the
          relaxation factor used by the incomplete LU factorization
          preconditioners (MILU(0) and MILUT). RELAXATION_FACTOR is unitless
          and should be greater than or equal to 0.0 and less than or equal to
          1.0. RELAXATION_FACTOR values of about 1.0 are commonly used, and
          experience suggests that convergence can be optimized in some cases
          with relax values of 0.97. A RELAXATION_FACTOR value of 0.0 will
          result in either ILU(0) or ILUT preconditioning (depending on the
          value specified for PRECONDITIONER_LEVELS and/or
          PRECONDITIONER_DROP_TOLERANCE). By default, RELAXATION_FACTOR is
          zero.
    preconditioner_levels : integer
        * preconditioner_levels (integer) optional integer value defining the
          level of fill for ILU decomposition used in the ILUT and MILUT
          preconditioners. Higher levels of fill provide more robustness but
          also require more memory. For optimal performance, it is suggested
          that a large level of fill be applied (7 or 8) with use of a drop
          tolerance. Specification of a PRECONDITIONER_LEVELS value greater
          than zero results in use of the ILUT preconditioner. By default,
          PRECONDITIONER_LEVELS is zero and the zero-fill incomplete LU
          factorization preconditioners (ILU(0) and MILU(0)) are used.
    preconditioner_drop_tolerance : double
        * preconditioner_drop_tolerance (double) optional real value that
          defines the drop tolerance used to drop preconditioner terms based on
          the magnitude of matrix entries in the ILUT and MILUT
          preconditioners. A value of :math:`10^{-4}` works well for most
          problems. By default, PRECONDITIONER_DROP_TOLERANCE is zero and the
          zero-fill incomplete LU factorization preconditioners (ILU(0) and
          MILU(0)) are used.
    number_orthogonalizations : integer
        * number_orthogonalizations (integer) optional integer value defining
          the interval used to explicitly recalculate the residual of the flow
          equation using the solver coefficient matrix, the latest dependent-
          variable (for example, head) estimates, and the right hand side. For
          problems that benefit from explicit recalculation of the residual, a
          number between 4 and 10 is appropriate. By default,
          NUMBER_ORTHOGONALIZATIONS is zero.
    scaling_method : string
        * scaling_method (string) an optional keyword that defines the matrix
          scaling approach used. By default, matrix scaling is not applied.
          NONE - no matrix scaling applied. DIAGONAL - symmetric matrix scaling
          using the POLCG preconditioner scaling method in Hill (1992). L2NORM
          - symmetric matrix scaling using the L2 norm.
    reordering_method : string
        * reordering_method (string) an optional keyword that defines the
          matrix reordering approach used. By default, matrix reordering is not
          applied. NONE - original ordering. RCM - reverse Cuthill McKee
          ordering. MD - minimum degree ordering.
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
        ("ims", "options", "csv_output_filerecord")
    )
    csv_outer_output_filerecord = ListTemplateGenerator(
        ("ims", "options", "csv_outer_output_filerecord")
    )
    csv_inner_output_filerecord = ListTemplateGenerator(
        ("ims", "options", "csv_inner_output_filerecord")
    )
    no_ptcrecord = ListTemplateGenerator(("ims", "options", "no_ptcrecord"))
    rcloserecord = ListTemplateGenerator(("ims", "linear", "rcloserecord"))
    package_abbr = "ims"
    _package_type = "ims"
    dfn_file_name = "sln-ims.dfn"

    dfn = [
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
            "name outer_hclose",
            "type double precision",
            "reader urword",
            "optional true",
            "deprecated 6.1.1",
        ],
        [
            "block nonlinear",
            "name outer_dvclose",
            "type double precision",
            "reader urword",
            "optional false",
        ],
        [
            "block nonlinear",
            "name outer_rclosebnd",
            "type double precision",
            "reader urword",
            "optional true",
            "deprecated 6.1.1",
        ],
        [
            "block nonlinear",
            "name outer_maximum",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block nonlinear",
            "name under_relaxation",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name under_relaxation_gamma",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name under_relaxation_theta",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name under_relaxation_kappa",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name under_relaxation_momentum",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name backtracking_number",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name backtracking_tolerance",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name backtracking_reduction_factor",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block nonlinear",
            "name backtracking_residual_limit",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name inner_maximum",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block linear",
            "name inner_hclose",
            "type double precision",
            "reader urword",
            "optional true",
            "deprecated 6.1.1",
        ],
        [
            "block linear",
            "name inner_dvclose",
            "type double precision",
            "reader urword",
            "optional false",
        ],
        [
            "block linear",
            "name rcloserecord",
            "type record inner_rclose rclose_option",
            "reader urword",
            "optional false",
        ],
        [
            "block linear",
            "name inner_rclose",
            "type double precision",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block linear",
            "name rclose_option",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name linear_acceleration",
            "type string",
            "reader urword",
            "optional false",
        ],
        [
            "block linear",
            "name relaxation_factor",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name preconditioner_levels",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name preconditioner_drop_tolerance",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name number_orthogonalizations",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name scaling_method",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block linear",
            "name reordering_method",
            "type string",
            "reader urword",
            "optional true",
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
        outer_hclose=None,
        outer_dvclose=None,
        outer_rclosebnd=None,
        outer_maximum=None,
        under_relaxation=None,
        under_relaxation_gamma=None,
        under_relaxation_theta=None,
        under_relaxation_kappa=None,
        under_relaxation_momentum=None,
        backtracking_number=None,
        backtracking_tolerance=None,
        backtracking_reduction_factor=None,
        backtracking_residual_limit=None,
        inner_maximum=None,
        inner_hclose=None,
        inner_dvclose=None,
        rcloserecord=None,
        linear_acceleration=None,
        relaxation_factor=None,
        preconditioner_levels=None,
        preconditioner_drop_tolerance=None,
        number_orthogonalizations=None,
        scaling_method=None,
        reordering_method=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            simulation, "ims", filename, pname, loading_package, parent_file
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
        self.outer_hclose = self.build_mfdata("outer_hclose", outer_hclose)
        self.outer_dvclose = self.build_mfdata("outer_dvclose", outer_dvclose)
        self.outer_rclosebnd = self.build_mfdata(
            "outer_rclosebnd", outer_rclosebnd
        )
        self.outer_maximum = self.build_mfdata("outer_maximum", outer_maximum)
        self.under_relaxation = self.build_mfdata(
            "under_relaxation", under_relaxation
        )
        self.under_relaxation_gamma = self.build_mfdata(
            "under_relaxation_gamma", under_relaxation_gamma
        )
        self.under_relaxation_theta = self.build_mfdata(
            "under_relaxation_theta", under_relaxation_theta
        )
        self.under_relaxation_kappa = self.build_mfdata(
            "under_relaxation_kappa", under_relaxation_kappa
        )
        self.under_relaxation_momentum = self.build_mfdata(
            "under_relaxation_momentum", under_relaxation_momentum
        )
        self.backtracking_number = self.build_mfdata(
            "backtracking_number", backtracking_number
        )
        self.backtracking_tolerance = self.build_mfdata(
            "backtracking_tolerance", backtracking_tolerance
        )
        self.backtracking_reduction_factor = self.build_mfdata(
            "backtracking_reduction_factor", backtracking_reduction_factor
        )
        self.backtracking_residual_limit = self.build_mfdata(
            "backtracking_residual_limit", backtracking_residual_limit
        )
        self.inner_maximum = self.build_mfdata("inner_maximum", inner_maximum)
        self.inner_hclose = self.build_mfdata("inner_hclose", inner_hclose)
        self.inner_dvclose = self.build_mfdata("inner_dvclose", inner_dvclose)
        self.rcloserecord = self.build_mfdata("rcloserecord", rcloserecord)
        self.linear_acceleration = self.build_mfdata(
            "linear_acceleration", linear_acceleration
        )
        self.relaxation_factor = self.build_mfdata(
            "relaxation_factor", relaxation_factor
        )
        self.preconditioner_levels = self.build_mfdata(
            "preconditioner_levels", preconditioner_levels
        )
        self.preconditioner_drop_tolerance = self.build_mfdata(
            "preconditioner_drop_tolerance", preconditioner_drop_tolerance
        )
        self.number_orthogonalizations = self.build_mfdata(
            "number_orthogonalizations", number_orthogonalizations
        )
        self.scaling_method = self.build_mfdata(
            "scaling_method", scaling_method
        )
        self.reordering_method = self.build_mfdata(
            "reordering_method", reordering_method
        )
        self._init_complete = True
