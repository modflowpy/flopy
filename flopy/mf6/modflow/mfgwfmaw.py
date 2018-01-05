# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfmaw(mfpackage.MFPackage):
    """
    ModflowGwfmaw defines a maw package within a gwf6 model.

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
          Output Control. If there is no Output Control option and PRINT_FLOWS
          is specified, then flow rates are printed for the last time step of
          each stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that multi-aquifer well flow
          terms will be written to the file specified with "BUDGET FILEOUT" in
          Output Control.
    stage_filerecord : [headfile]
        * headfile (string) name of the binary output file to write stage
          information.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    no_well_storage : boolean
        * no_well_storage (boolean) keyword that deactivates inclusion of well
          storage contributions to the multi-aquifer well package continuity
          equation.
    flowing_wells : boolean
        * flowing_wells (boolean) keyword that activates the flowing wells
          option for the multi-aquifer well package.
    shutdown_theta : double
        * shutdown_theta (double) value that defines the weight applied to
          discharge rate for wells that limit the water level in a discharging
          well (defined using the texttt{HEAD_LIMIT} keyword in the stress
          period data). texttt{SHUTDOWN_THETA} is used to control discharge
          rate oscillations when the flow rate from the aquifer is less than
          the specified flow rate from the aquifer to the well. Values range
          between 0.0 and 1.0, and larger values increase the weight (decrease
          under-relaxation) applied to the well discharge rate. The
          texttt{head_limit} option has been included to facilitate backward
          compatibility with previous versions of MODFLOW but use of the
          texttt{rate_scaling} option instead of the texttt{head_limit} option
          is recommended. By default, texttt{SHUTDOWN_THETA} is 0.7.
    shutdown_kappa : double
        * shutdown_kappa (double) value that defines the weight applied to
          discharge rate for wells that limit the water level in a discharging
          well (defined using the texttt{HEAD_LIMIT} keyword in the stress
          period data). texttt{SHUTDOWN_KAPPA} is used to control discharge
          rate oscillations when the flow rate from the aquifer is less than
          the specified flow rate from the aquifer to the well. Values range
          between 0.0 and 1.0, and larger values increase the weight applied to
          the well discharge rate. The texttt{head_limit} option has been
          included to facilitate backward compatibility with previous versions
          of MODFLOW but use of the texttt{rate_scaling} option instead of the
          texttt{head_limit} option is recommended. By default,
          texttt{SHUTDOWN_KAPPA} is 0.0001.
    ts_filerecord : [ts6_filename]
        * ts6_filename (string) defines a time-series file defining time series
          that can be used to assign time-varying values. See the "Time-
          Variable Input" section for instructions on using the time-series
          capability.
    obs_filerecord : [obs6_filename]
        * obs6_filename (string) name of input file to define observations for
          the MAW package. See the "Observation utility" section for
          instructions for preparing observation input files. Table
          reftable:obstype lists observation type(s) supported by the MAW
          package.
    mover : boolean
        * mover (boolean) keyword to indicate that this instance of the MAW
          Package can be used with the Water Mover (MVR) Package. When the
          MOVER option is specified, additional memory is allocated within the
          package to store the available, provided, and received water.
    nmawwells : integer
        * nmawwells (integer) integer value specifying the number of multi-
          aquifer wells that will be simulated for all stress periods.
    wellrecarray : [wellno, radius, bottom, strt, condeqn, ngwfnodes, aux,
      boundname]
        * wellno (integer) integer value that defines the well number
          associated with the specified PACKAGEDATA data on the line.
          texttt{wellno} must be greater than zero and less than or equal to
          texttt{nmawwells}. Multi-aquifer well information must be specified
          for every multi-aquifer well or the program will terminate with an
          error. The program will also terminate with an error if information
          for a multi-aquifer well is specified more than once.
        * radius (double) radius for the multi-aquifer well.
        * bottom (double) bottom elevation of the multi-aquifer well.
        * strt (double) starting head for the multi-aquifer well.
        * condeqn (string) character string that defines the conductance
          equation that is used to calculate the saturated conductance for the
          multi-aquifer well. Possible multi-aquifer well texttt{condeqn}
          strings include: texttt{SPECIFIED}--character keyword to indicate the
          multi-aquifer well saturated conductance will be specified.
          texttt{THEIM}--character keyword to indicate the multi-aquifer well
          saturated conductance will be calculated using the Theim equation.
          texttt{SKIN}--character keyword to indicate that the multi-aquifer
          well saturated conductance will be calculated using the screen top
          and bottom, screen hydraulic conductivity, and skin radius.
          texttt{CUMULATIVE}--character keyword to indicate that the multi-
          aquifer well saturated conductance will be calculated using a
          combination of the Theim equation and the screen top and bottom,
          screen hydraulic conductivity, and skin radius.
          texttt{MEAN}--character keyword to indicate the multi-aquifer well
          saturated conductance will be calculated using using the aquifer and
          screen top and bottom, aquifer and screen hydraulic conductivity, and
          well and skin radius.
        * ngwfnodes (integer) integer value that defines the number of
          texttt{GWF} nodes connected to this (texttt{wellno}) multi-aquifer
          well. One or more screened intervals can be connected to the same
          texttt{GWF} node. texttt{ngwfnodes} must be greater than zero.
        * aux (double) represents the values of the auxiliary variables for
          each multi-aquifer well. The values of auxiliary variables must be
          present for each multi-aquifer well. The values must be specified in
          the order of the auxiliary variables specified in the OPTIONS block.
          If the package supports time series and the Options block includes a
          TIMESERIESFILE entry (see the "Time-Variable Input" section), values
          can be obtained from a time series by entering the time-series name
          in place of a numeric value.
        * boundname (string) name of the multi-aquifer well cell. boundname is
          an ASCII character variable that can contain as many as 40
          characters. If boundname contains spaces in it, then the entire name
          must be enclosed within single quotes.
    wellconnectionsrecarray : [wellno, icon, cellid, scrn_top, scrn_bot,
      hk_skin, radius_skin]
        * wellno (integer) integer value that defines the well number
          associated with the specified CONNECTIONDATA data on the line.
          texttt{wellno} must be greater than zero and less than or equal to
          texttt{nmawwells}. Multi-aquifer well connection information must be
          specified for every multi-aquifer well connection to the GWF model
          (texttt{ngwfnodes}) or the program will terminate with an error. The
          program will also terminate with an error if connection information
          for a multi-aquifer well connection to the GWF model is specified
          more than once.
        * icon (integer) integer value that defines the texttt{GWF} connection
          number for this multi-aquifer well connection entry. texttt{iconn}
          must be greater than zero and less than or equal to texttt{ngwfnodes}
          for multi-aquifer well texttt{wellno}.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, cellid is the layer, row, and column.
          For a grid that uses the DISV input file, cellid is the layer and
          cell2d number. If the model uses the unstructured discretization
          (DISU) input file, then cellid is the node number for the cell.
        * scrn_top (double) value that defines the top elevation of the screen
          for the multi-aquifer well connection. texttt{scrn_top} can be any
          value if texttt{condeqn} is texttt{SPECIFIED} or texttt{THEIM}. If
          the specified texttt{scrn_top} is greater than the top of the
          texttt{GWF} cell it is set equal to the top of the cell.
        * scrn_bot (double) value that defines the bottom elevation of the
          screen for the multi-aquifer well connection. texttt{scrn_bot} can be
          any value if texttt{condeqn} is texttt{SPECIFIED} or texttt{THEIM}.
          If the specified texttt{scrn_bot} is less than the bottom of the
          texttt{GWF} cell it is set equal to the bottom of the cell.
        * hk_skin (double) value that defines the skin (filter pack) hydraulic
          conductivity (if texttt{condeqn} for the multi-aquifer well is
          texttt{SKIN}, texttt{CUMULATIVE}, or texttt{MEAN}) or conductance (if
          texttt{condeqn} for the multi-aquifer well is texttt{SPECIFIED}) for
          each texttt{GWF} node connected to the multi-aquifer well
          (texttt{ngwfnodes}). texttt{hk_skin} can be any value if
          texttt{condeqn} is texttt{THEIM}.
        * radius_skin (double) real value that defines the skin radius (filter
          pack radius) for the multi-aquifer well. texttt{radius_skin} can be
          any value if texttt{condeqn} is texttt{SPECIFIED} or texttt{THEIM}.
          Otherwise, texttt{radius_skin} must be greater than texttt{radius}
          for the multi-aquifer well.
    wellperiodrecarray : [wellno, mawsetting]
        * wellno (integer) integer value that defines the well number
          associated with the specified PERIOD data on the line. texttt{wellno}
          must be greater than zero and less than or equal to
          texttt{nmawwells}.
        * mawsetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          texttt{mawsetting} string include: texttt{STATUS},
          texttt{FLOWING_WELL}, texttt{RATE}, texttt{WELL_HEAD},
          texttt{HEAD_LIMIT}, texttt{SHUT_OFF}, texttt{RATE_SCALING}, and
          texttt{AUXILIARY}.
            well_head : [double]
                * well_head (double) is the head in the multi-aquifer well.
                  texttt{well_head} is only applied to constant head
                  (texttt{STATUS} is texttt{CONSTANT}) and inactive
                  (texttt{STATUS} is texttt{INACTIVE}) multi-aquifer wells. If
                  the Options block includes a texttt{TIMESERIESFILE} entry
                  (see the "Time-Variable Input" section), values can be
                  obtained from a time series by entering the time-series name
                  in place of a numeric value.
            rate : [double]
                * rate (double) is the volumetric pumping rate for the multi-
                  aquifer well. A positive value indicates recharge and a
                  negative value indicates discharge (pumping). texttt{rate}
                  only applies to active (texttt{IBOUND}:math:`>0`) multi-
                  aquifer wells. If the Options block includes a
                  \texttt{TIMESERIESFILE} entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value. By
                  default, the \texttt{rate} for each multi-aquifer well is
                  zero.
            shutoffrecord : [minrate, maxrate]
                * minrate (double) is the minimum rate that a well must exceed
                  to shutoff a well during a stress period. The well will shut
                  down during a time step if the flow rate to the well from the
                  aquifer is less than texttt{minrate}. If a well is shut down
                  during a time step, reactivation of the well cannot occur
                  until the next time step to reduce oscillations.
                  texttt{minrate} must be less than texttt{maxrate}.
                * maxrate (double) is the maximum rate that a well must exceed
                  to reactivate a well during a stress period. The well will
                  reactivate during a timestep if the well was shutdown during
                  the previous time step and the flow rate to the well from the
                  aquifer exceeds texttt{maxrate}. Reactivation of the well
                  cannot occur until the next time step if a well is shutdown
                  to reduce oscillations. texttt{maxrate} must be greater than
                  texttt{minrate}.
            head_limit : [string]
                * head_limit (string) is the limiting water level (head) in the
                  well, which is the minimum of the well texttt{rate} or the
                  well inflow rate from the aquifer. texttt{head_limit} is only
                  applied to discharging wells (texttt{rate}:math:`<0`).
                  \texttt{head\_limit} can be deactivated by specifying the
                  text string `\texttt{off}'. The \texttt{head\_limit} option
                  is based on the \texttt{head\_limit} functionality available
                  in the MNW2~\citep{konikow2009} package for MODFLOW-2005. The
                  \texttt{head\_limit} option has been included to facilitate
                  backward compatibility with previous versions of MODFLOW but
                  use of the \texttt{rate\_scaling} option instead of the
                  \texttt{head\_limit} option is recommended. By default,
                  \texttt{head\_limit} is `\texttt{off}'.
            rate_scalingrecord : [pump_elevation, scaling_length]
                * pump_elevation (double) is the elevation of the multi-aquifer
                  well pump (texttt{pump_elevation}). texttt{pump_elevation}
                  cannot be less than the bottom elevation (texttt{bottom}) of
                  the multi-aquifer well. By default, texttt{pump_elevation} is
                  set equal to the bottom of the largest texttt{GWF} node
                  number connected to a MAW well.
                * scaling_length (double) height above the pump elevation
                  (texttt{scaling_length}) below which the pumping rate is
                  reduced. The default value for texttt{scaling_length} is the
                  well radius.
            status : [string]
                * status (string) keyword option to define well status.
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
            flowing_wellrecord : [fwelev, fwcond, fwrlen]
                * fwelev (double) elevation used to determine whether or not
                  the well is flowing.
                * fwcond (double) conductance used to calculate the discharge
                  of a free flowing well. Flow occurs when the head in the well
                  is above the well top elevation (texttt{fwelev}).
                * fwrlen (double) length used to reduce the conductance of the
                  flowing well. When the head in the well drops below the well
                  top plus the reduction length, then the conductance is
                  reduced. This reduction length can be used to improve the
                  stability of simulations with flowing wells so that there is
                  not an abrupt change in flowing well rates.

    """
    auxiliary = ListTemplateGenerator(('gwf6', 'maw', 'options', 
                                       'auxiliary'))
    stage_filerecord = ListTemplateGenerator(('gwf6', 'maw', 'options', 
                                              'stage_filerecord'))
    budget_filerecord = ListTemplateGenerator(('gwf6', 'maw', 'options', 
                                               'budget_filerecord'))
    ts_filerecord = ListTemplateGenerator(('gwf6', 'maw', 'options', 
                                           'ts_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwf6', 'maw', 'options', 
                                            'obs_filerecord'))
    wellrecarray = ListTemplateGenerator(('gwf6', 'maw', 'packagedata', 
                                          'wellrecarray'))
    wellconnectionsrecarray = ListTemplateGenerator(('gwf6', 'maw', 
                                                     'connectiondata', 
                                                     'wellconnectionsrecarray'))
    wellperiodrecarray = ListTemplateGenerator(('gwf6', 'maw', 'period', 
                                                'wellperiodrecarray'))
    package_abbr = "gwfmaw"
    package_type = "maw"
    dfn = [["block options", "name auxiliary", "type string", 
            "shape (naux)", "reader urword", "optional true"],
           ["block options", "name boundnames", "type keyword", "shape", 
            "reader urword", "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_head", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name stage_filerecord", 
            "type record head fileout headfile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name head", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name headfile", "type string", 
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
           ["block options", "name no_well_storage", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name flowing_wells", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name shutdown_theta", "type double precision", 
            "reader urword", "optional true"],
           ["block options", "name shutdown_kappa", "type double precision", 
            "reader urword", "optional true"],
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
           ["block dimensions", "name nmawwells", "type integer", 
            "reader urword", "optional false"],
           ["block packagedata", "name wellrecarray", 
            "type recarray wellno radius bottom strt condeqn ngwfnodes aux " 
            "boundname", 
            "shape (nmawwells)", "reader urword"],
           ["block packagedata", "name wellno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name radius", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name bottom", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name strt", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name condeqn", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name ngwfnodes", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name aux", "type double precision", 
            "in_record true", "tagged false", "shape (naux)", "reader urword", 
            "time_series true", "optional true"],
           ["block packagedata", "name boundname", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "optional true"],
           ["block connectiondata", "name wellconnectionsrecarray", 
            "type recarray wellno icon cellid scrn_top scrn_bot hk_skin " 
            "radius_skin", 
            "shape (nmawwells)", "reader urword"],
           ["block connectiondata", "name wellno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name icon", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name cellid", "type integer", 
            "shape (ncelldim)", "tagged false", "in_record true", 
            "reader urword"],
           ["block connectiondata", "name scrn_top", 
            "type double precision", "shape", "tagged false", 
            "in_record true", "reader urword"],
           ["block connectiondata", "name scrn_bot", 
            "type double precision", "shape", "tagged false", 
            "in_record true", "reader urword"],
           ["block connectiondata", "name hk_skin", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block connectiondata", "name radius_skin", 
            "type double precision", "shape", "tagged false", 
            "in_record true", "reader urword"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name wellperiodrecarray", 
            "type recarray wellno mawsetting", "shape", "reader urword"],
           ["block period", "name wellno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name mawsetting", 
            "type keystring status flowing_wellrecord rate well_head " 
            "head_limit shutoffrecord rate_scalingrecord auxiliaryrecord", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name status", "type string", "shape", 
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name flowing_wellrecord", 
            "type record flowing_well fwelev fwcond fwrlen", "shape", 
            "tagged", "in_record true", "reader urword"],
           ["block period", "name flowing_well", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name fwelev", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name fwcond", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name fwrlen", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name rate", "type double precision", "shape", 
            "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name well_head", "type double precision", 
            "shape", "tagged true", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name head_limit", "type string", "shape", 
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name shutoffrecord", 
            "type record shut_off minrate maxrate", "shape", "tagged", 
            "in_record true", "reader urword"],
           ["block period", "name shut_off", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name minrate", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name maxrate", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name rate_scalingrecord", 
            "type record rate_scaling pump_elevation scaling_length", "shape", 
            "tagged", "in_record true", "reader urword"],
           ["block period", "name rate_scaling", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name pump_elevation", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name scaling_length", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block period", "name auxiliaryrecord", 
            "type record auxiliary auxname auxval", "shape", "tagged", 
            "in_record true", "reader urword"],
           ["block period", "name auxiliary", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name auxname", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name auxval", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"]]

    def __init__(self, model, add_to_package_list=True, auxiliary=None,
                 boundnames=None, print_input=None, print_head=None,
                 print_flows=None, save_flows=None, stage_filerecord=None,
                 budget_filerecord=None, no_well_storage=None,
                 flowing_wells=None, shutdown_theta=None, shutdown_kappa=None,
                 ts_filerecord=None, obs_filerecord=None, mover=None,
                 nmawwells=None, wellrecarray=None,
                 wellconnectionsrecarray=None, wellperiodrecarray=None,
                 fname=None, pname=None, parent_file=None):
        super(ModflowGwfmaw, self).__init__(model, "maw", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary",  auxiliary)
        self.boundnames = self.build_mfdata("boundnames",  boundnames)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_head = self.build_mfdata("print_head",  print_head)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.stage_filerecord = self.build_mfdata("stage_filerecord", 
                                                  stage_filerecord)
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.no_well_storage = self.build_mfdata("no_well_storage", 
                                                 no_well_storage)
        self.flowing_wells = self.build_mfdata("flowing_wells",  flowing_wells)
        self.shutdown_theta = self.build_mfdata("shutdown_theta", 
                                                shutdown_theta)
        self.shutdown_kappa = self.build_mfdata("shutdown_kappa", 
                                                shutdown_kappa)
        self.ts_filerecord = self.build_mfdata("ts_filerecord",  ts_filerecord)
        self.obs_filerecord = self.build_mfdata("obs_filerecord", 
                                                obs_filerecord)
        self.mover = self.build_mfdata("mover",  mover)
        self.nmawwells = self.build_mfdata("nmawwells",  nmawwells)
        self.wellrecarray = self.build_mfdata("wellrecarray",  wellrecarray)
        self.wellconnectionsrecarray = self.build_mfdata(
            "wellconnectionsrecarray",  wellconnectionsrecarray)
        self.wellperiodrecarray = self.build_mfdata("wellperiodrecarray", 
                                                    wellperiodrecarray)
