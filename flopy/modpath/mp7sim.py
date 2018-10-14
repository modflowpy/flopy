"""
mpsim module.  Contains the ModpathSim class. Note that the user can access
the ModpathSim class as `flopy.modpath.ModpathSim`.

Additional information for this MODFLOW/MODPATH package can be found at the
`Online MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?dis.htm>`_.

"""
from enum import Enum
import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Util3d, check
from .mp7particlegroup import ParticleGroup, ParticleGroupLRCTemplate,\
    ParticleGroupNodeTemplate


class simType(Enum):
    """
    Enumeration of different simulation types
    """
    endpoint = 1
    pathline = 2
    timeseries = 3
    combined = 4


class trackDir(Enum):
    """
    Enumeration of different tracking directions
    """
    forward = 1
    backward = 2


class weakOpt(Enum):
    """
    Enumeration of different weak sink and source options
    """
    pass_through = 1
    stop_at = 2


class budgetOpt(Enum):
    """
    Enumeration of different budget output options
    """
    no = 0
    summary = 1
    record_summary = 2


class stopOpt(Enum):
    """
    Enumeration of different stop time options
    """
    total = 1
    extend = 2
    specified = 3


class onoffOpt(Enum):
    """
    Enumeration of on-off options
    """
    off = 1
    on = 2


class Modpath7Sim(Package):
    """
    MODPATH Simulation File Package Class.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modpath.Modpath7`) to
            which this package will be added.
        mpnamefilename : str
            Filename of the MODPATH 7 name file. If mpnamefilename is not
            defined it will be generated from the model name
            (default is None).
        listingfilename : str
            Filename of the MODPATH 7 listing file. If listingfilename is not
            defined it will be generated from the model name
            (default is None).
        endpointfilename : str
            Filename of the MODPATH 7 endpoint file. If endpointfilename is
            not defined it will be generated from the model name
            (default is None).
        pathlinefilename : str
            Filename of the MODPATH 7 pathline file. If pathlinefilename is
            not defined it will be generated from the model name
            (default is None).
        timeseriesfilename : str
            Filename of the MODPATH 7 timeseries file. If timeseriesfilename
            is not defined it will be generated from the model name
            (default is None).
        tracefilename : str
            Filename of the MODPATH 7 tracefile file. If tracefilename is not
            defined it will be generated from the model name
            (default is None).
        simulationtype : str
            MODPATH 7 simulation type. Valid simulation types are 'endpoint',
            'pathline', 'timeseries', or 'combined' (default is 'pathline').
        trackingdirection : str
            MODPATH 7 tracking direction. Valid tracking directions are
            'forward' or 'backward' (default os 'forward').
        weaksinkoption : str
            MODPATH 7 weak sink option. Valid weak sink options are
            'pass_through' or 'stop_at' (default value is 'stop_at').
        weaksourceoption : str
            MODPATH 7 weak source option. Valid weak source options are
            'pass_through' or 'stop_at' (default value is 'stop_at').
        budgetoutputoption : str
            MODPATH 7 budget output option. Valid budget output options are
            'no' - individual cell water balance errors are not computed
            and budget record headers are not printed, 'summary' - a summary
            of individual cell water balance errors for each time step is
            printed in the listing file without record headers, or
            'record_summary' -  a summary of individual cell water balance
            errors for each time step is printed in the listing file with
            record headers (default is 'summary').
        traceparticledata : list or tuple
            List or tuple with two ints that define the particle group and
            particle id (zero-based) of the specified particle that is
            followed in detail. If traceparticledata is None, trace mode is
            off (default is None).
        budgetcellnumbers : int, list of ints, tuple of ints, or np.ndarray
            Cell numbers (zero-based) for which detailed water budgets are
            computed. If budgetcellnumbers is None, detailed water budgets are
            not calculated (default is None).
        referencetime : float, list, or tuple
            Specified reference time if a float or a list/tuple with a single
            float value is provided (reference time option 1). Otherwise a
            list or tuple with a zero-based stress period (int) and time
            step (int) and a float defining the relative time position in the
            time step is provided (reference time option 2). If referencetime
            is None, reference time is set to 0 (default is None).
        stoptimeoption : str
            String indicating how a particle tracking simulation is
            terminated based on time. If stop time option is 'total', particles
            will be stopped at the end of the final time step if 'forward'
            tracking is simulated or at the beginning of the first time step
            if backward tracking. If stop time option is 'extend', initial or
            final steady-state time steps will be extended and all particles
            will be tracked until they reach a termination location. If stop
            time option is 'specified', particles will be tracked until they
            reach a termination location or the specified stop time is reached
            (default is 'extend').
        stoptime : float
            User-specified value of tracking time at which to stop a particle
            tracking simulation. Stop time is only used if the stop time option
            is 'specified'. If stoptime is None amd the stop time option is
            'specified' particles will be terminated at the end of the last
            time step if 'forward' tracking or the beginning of the first time
            step if 'backward' tracking (default is None).
        timepointdata : list or tuple
            List or tuple with 2 items that is only used if simulationtype is
            'timeseries' or 'combined'. If the second item is a float then the
            timepoint data corresponds to time point option 1 and the first
            entry is the number of time points (timepointcount) and the second
            entry is the time point interval. If the second item is a list,
            tuple, or np.ndarray then the timepoint data corresponds to time
            point option 2 and the number of time points entries
            (timepointcount) in the second item and the second item is an
            list, tuple, or array of user-defined time points. If Timepointdata
            is None, time point option 1 is specified and the total simulation
            time is split into 100 intervals (default is None).
        zonedataoption : str
            If zonedataoption is 'off', zone array data are not read and a zone
            value of 1 is applied to all cells. If zonedataoption is 'on',
            zone array data are read (default is 'off').
        stopzone : int
            A zero-based specified integer zone value that indicates an
            automatic stopping location for particles and is only used if
            zonedataoption is 'on'. A value of -1 indicates no automatic stop
            zone is used.  Stopzone values less than -1 are not allowed. If
            stopzone is None, stopzone is set to -1 (default is None).
        zones : float or array of floats (nlay, nrow, ncol)
            Array of zero-based positive integer zones that are only used if
            zonedataoption is 'on' (default is 0).
        retardationfactoroption : str
            If retardationfactoroption is 'off', retardation array data are not
            read and a retardation factor of 1 is applied to all cells. If
            retardationfactoroption is 'on', retardation factor array data are
            read (default is 'off').
        retardation : float or array of floats (nlay, nrow, ncol)
            Array of retardation factors that are only used if
            retardationfactoroption is 'on' (default is 1).
        particlegroups : ParticleGroup or list of ParticleGroups
            ParticleGroup or list of ParticlesGroups that contain data for
            individual particle groups. If None is specified, a
            particle in the center of node 0 will be created (default is None).
        extension : string
            Filename extension (default is 'mpsim')

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow.load('mf2005.nam')
    >>> mp = flopy.modpath.Modpath7('mf2005_mp', flowmodel=m)
    >>> mpsim = flopy.modpath.Modpath7Sim(mp)

    """

    def __init__(self, model, mpnamefilename=None, listingfilename=None,
                 endpointfilename=None, pathlinefilename=None,
                 timeseriesfilename=None, tracefilename=None,
                 simulationtype='pathline', trackingdirection='forward',
                 weaksinkoption='stop_at', weaksourceoption='stop_at',
                 budgetoutputoption='no',
                 traceparticledata=None,
                 budgetcellnumbers=None, referencetime=None,
                 stoptimeoption='extend', stoptime=None,
                 timepointdata=None,
                 zonedataoption='off', stopzone=None, zones=0,
                 retardationfactoroption='off', retardation=1.,
                 particlegroups=None,
                 extension='mpsim'):
        """
        Package constructor.

        """

        unitnumber = model.next_unit()

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension, 'MPSIM', unitnumber)

        self.heading = '# {} package for'.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'

        # set file names
        if mpnamefilename is None:
            mpnamefilename = '{}.{}'.format(model.name, 'mpnam')
        self.mp_name_file = mpnamefilename
        if listingfilename is None:
            listingfilename = '{}.{}'.format(model.name, 'mplst')
        self.listingfilename = listingfilename
        if endpointfilename is None:
            endpointfilename = '{}.{}'.format(model.name, 'mpend')
        self.endpointfilename = endpointfilename
        if pathlinefilename is None:
            pathlinefilename = '{}.{}'.format(model.name, 'mppth')
        self.pathlinefilename = pathlinefilename
        if timeseriesfilename is None:
            timeseriesfilename = '{}.{}'.format(model.name, 'timeseries')
        self.timeseriesfilename = timeseriesfilename
        if tracefilename is None:
            tracefilename = '{}.{}'.format(model.name, 'trace')
        self.tracefilename = tracefilename

        try:
            self.simulationtype = simType[simulationtype.lower()].value
        except:
            self._enum_error('simulationtype', simulationtype, simType)
        try:
            self.trackingdirection = trackDir[trackingdirection.lower()].value
        except:
            self._enum_error('trackingdirection', trackingdirection,
                             trackDir)
        try:
            self.weaksinkoption = weakOpt[weaksinkoption.lower()].value
        except:
            self._enum_error('weaksinkoption', weaksinkoption,
                             weakOpt)
        try:
            self.weaksourceoption = weakOpt[weaksourceoption.lower()].value
        except:
            self._enum_error('weaksourceoption', weaksourceoption,
                             weakOpt)
        try:
            self.budgetoutputoption = \
                budgetOpt[budgetoutputoption.lower()].value
        except:
            self._enum_error('budgetoutputoption', budgetoutputoption,
                             budgetOpt)
        # tracemode
        if traceparticledata is None:
            tracemode = 0
            traceparticlegroup = None
            traceparticleid = None
        else:
            tracemode = 1
            if isinstance(traceparticledata, (list, tuple)):
                if len(traceparticledata) != 2:
                    msg = 'traceparticledata must be a list or tuple ' + \
                          'with 2 items (a integer and an integer). ' + \
                          'Passed item {}.'.format(traceparticledata)
                    raise ValueError(msg)
                try:
                    traceparticlegroup = int(traceparticledata[0])
                except:
                    msg = 'traceparticledata[0] ' + \
                          '({}) '.format(traceparticledata[0]) + \
                          'cannot be converted to a integer.'
                    raise ValueError(msg)
                try:
                    traceparticleid = int(traceparticledata[1])
                except:
                    msg = 'traceparticledata[1] ' + \
                          '({}) '.format(traceparticledata[0]) + \
                          'cannot be converted to a integer.'
                    raise ValueError(msg)
            else:
                msg = 'traceparticledata must be a list or ' + \
                      'tuple with 2 items (a integer and an integer).'
                raise ValueError(msg)

        # set tracemode, traceparticlegroup, and traceparticleid
        self.tracemode = tracemode
        self.traceparticlegroup = traceparticlegroup
        self.traceparticleid = traceparticleid

        if budgetcellnumbers is None:
            BudgetCellCount = 0
        else:
            if isinstance(budgetcellnumbers, int):
                budgetcellnumbers = [budgetcellnumbers]
            budgetcellnumbers = np.array(budgetcellnumbers, dtype=np.int32)
            # validate budget cell numbers
            ncells = np.prod(np.array(self.parent.shape))
            msg = ''
            for cell in budgetcellnumbers:
                if cell < 0 or cell >= ncells:
                    if msg == '':
                        msg = 'Specified cell number(s) exceed the ' + \
                              'number of cells in the model ' + \
                              '(Valid cells = 0-{}). '.format(ncells - 1) + \
                              'Invalid cells are: '
                    else:
                        msg += ', '
                    msg += '{}'.format(cell)
            if msg != '':
                raise ValueError(msg)
            # create Util2d object
            BudgetCellCount = budgetcellnumbers.shape[0]
            self.budgetcellnumbers = Util2d(self.parent, (BudgetCellCount,),
                                            np.int32, budgetcellnumbers,
                                            name='budgetcellnumbers',
                                            locat=self.unit_number[0])
        self.BudgetCellCount = BudgetCellCount

        if referencetime is None:
            referencetime = 0.
        if isinstance(referencetime, float):
            referencetime = [referencetime]
        elif isinstance(referencetime, np.ndarray):
            referencetime = referencetime.tolist()
        if len(referencetime) == 1:
            referencetimeOption = 1
            # validate referencetime data
            t = referencetime[0]
            if t < 0. or t > self.parent.time_end:
                msg = 'referencetime must be between 0. and ' + \
                      '{} '.format(self.parent.time_end) + \
                      '(specified value = {}).'.format(t)
                raise ValueError(msg)
        elif len(referencetime) == 3:
            referencetimeOption = 2
            # validate referencetime data
            # StressPeriod
            iper = referencetime[0]
            if iper < 0 or iper >= self.parent.nper:
                msg = 'StressPeriod must be between 0 and ' + \
                      '{} '.format(self.parent.nper - 1) + \
                      '(specified value = {}).'.format(iper)
                raise ValueError(msg)

            # TimeStep
            istp = referencetime[1]
            maxstp = self.parent.nstp[iper] + 1
            if istp < 0 or istp >= maxstp:
                msg = 'TimeStep for StressPeriod {} '.format(iper) + \
                      'must be between 0 and ' + \
                      '{} '.format(maxstp - 1) + \
                      '(specified value = {}).'.format(istp)
                raise ValueError(msg)

            # TimeFraction
            tf = referencetime[2]
            if tf < 0. or tf > 1.:
                msg = 'TimeFraction value must be between 0 and 1 ' + \
                      '(specified value={}).'.format(tf)
                raise ValueError(msg)
        else:
            msg = 'referencetime must be a float (referencetime) or ' + \
                  'a list with one item [referencetime] or three items ' + \
                  '[StressPeriod, TimeStep, TimeFraction]. ' + \
                  '{}'.format(len(referencetime)) + \
                  ' items were passed as referencetime ['
            for i, v in enumerate(referencetime):
                if i > 0:
                    msg += ', '
                msg += '{}'.format(v)
            msg += '].'
            raise ValueError(msg)
        self.referencetimeOption = referencetimeOption
        self.referencetime = referencetime

        # stoptimeoption
        try:
            self.stoptimeoption = \
                stopOpt[stoptimeoption.lower()].value
        except:
            self._enum_error('stoptimeoption', stoptimeoption,
                             stopOpt)
        # stoptime
        if self.stoptimeoption == 3:
            if stoptime is None:
                if self.trackingdirection == 1:
                    stoptime = self.parent.time_end
                else:
                    stoptime = 0.
        self.stoptime = stoptime

        # timepointdata
        if timepointdata is not None:
            if not isinstance(timepointdata, (list, tuple)):
                msg = 'timepointdata must be a list or tuple'
                raise ValueError(msg)
            else:
                if len(timepointdata) != 2:
                    msg = 'timepointdata must be a have 2 entries ' + \
                          '({} provided)'.format(len(timepointdata))
                    raise ValueError(msg)
                else:
                    if isinstance(timepointdata[1], (list, tuple)):
                        timepointdata[1] = np.array(timepointdata[1])
                    elif isinstance(timepointdata[1], float):
                        timepointdata[1] = np.array([timepointdata[1]])
                    if timepointdata[1].shape[0] == timepointdata[0]:
                        timepointoption = 2
                    elif timepointdata[1].shape[0] > 1:
                        msg = 'The number of TimePoint data ' + \
                              '({}) '.format(timepointdata[1].shape[0]) + \
                              'is not equal to TimePointCount ' + \
                              '({}).'.format(timepointdata[0])
                        raise ValueError(msg)
                    else:
                        timepointoption = 1
        else:
            timepointoption = 1
            timepointdata = [100, self.parent.time_end / 100.]
            timepointdata[1] = np.array([timepointdata[1]])
        self.timepointoption = timepointoption
        self.timepointdata = timepointdata

        # zonedataoption
        try:
            self.zonedataoption = onoffOpt[zonedataoption.lower()].value
        except:
            self._enum_error('zonedataoption', zonedataoption, onoffOpt)
        if self.zonedataoption == 2:
            if stopzone is None:
                stopzone = -1
            if stopzone < -1:
                msg = 'Specified stopzone value ({}) '.format(stopzone) + \
                      'must be greater than 0.'
                raise ValueError(msg)
            self.stopzone = stopzone
            if zones is None:
                msg = "zones must be specified if zonedataoption='on'."
                raise ValueError(msg)
            self.zones = Util3d(model, self.parent.shape, np.int32,
                                zones, name='zones', locat=self.unit_number[0])

        # retardationfactoroption
        try:
            self.retardationfactoroption = \
                onoffOpt[retardationfactoroption.lower()].value
        except:
            self._enum_error('retardationfactoroption',
                             retardationfactoroption, onoffOpt)
        if self.retardationfactoroption == 2:
            if retardation is None:
                msg = "retardation must be specified if " + \
                      "retardationfactoroption='on'."
                raise ValueError(msg)
            self.retardation = Util3d(model, self.parent.shape, np.float32,
                                      retardation, name='retardation',
                                      locat=self.unit_number[0])
        # particle group data
        if particlegroups is None:
            particlegroups = [ParticleGroup()]
        elif isinstance(particlegroups,
                        (ParticleGroup, ParticleGroupLRCTemplate,
                         ParticleGroupNodeTemplate)):
            particlegroups = [particlegroups]
        self.particlegroups = particlegroups

        self.parent.add_package(self)

    def _enum_error(self, v, s, e):
        msg = 'Invalid {} ({})'.format(v, s) + \
              '. Valid types are '
        for i, c in enumerate(e):
            if i > 0:
                msg += ', '
            msg += '"{}"'.format(c.name)
        raise ValueError(msg)

    def write_file(self, check=False):
        """
        Write the package file

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default False)

        Returns
        -------
        None

        """

        f = open(self.fn_path, 'w')
        # item 0
        f.write('{}\n'.format(self.heading))
        # item 1
        f.write('{}\n'.format(self.mp_name_file))
        # item 2
        f.write('{}\n'.format(self.listingfilename))
        # item 3
        f.write('{} {} {} {} {} {}\n'.format(self.simulationtype,
                                             self.trackingdirection,
                                             self.weaksinkoption,
                                             self.weaksourceoption,
                                             self.budgetoutputoption,
                                             self.tracemode))
        # item 4
        f.write('{}\n'.format(self.endpointfilename))
        # item 5
        if self.simulationtype == 2 or self.simulationtype == 4:
            f.write('{}\n'.format(self.pathlinefilename))
        # item 6
        if self.simulationtype == 3 or self.simulationtype == 4:
            f.write('{}\n'.format(self.timeseriesfilename))
        # item 7 and 8
        if self.tracemode == 1:
            f.write('{}\n'.format(self.tracefilename))
            f.write('{} {}\n'.format(self.traceparticlegroup + 1,
                                     self.traceparticleid + 1))
        # item 9
        f.write('{}\n'.format(self.BudgetCellCount))
        # item 10
        if self.BudgetCellCount > 0:
            v = Util2d(self.parent, (self.BudgetCellCount,),
                       np.int32, self.budgetcellnumbers.array + 1,
                       name='temp',
                       locat=self.unit_number[0])
            f.write(v.string)

        # item 11
        f.write('{}\n'.format(self.referencetimeOption))
        if self.referencetimeOption == 1:
            # item 12
            f.write('{:g}\n'.format(self.referencetime[0]))
        elif self.referencetimeOption == 2:
            # item 13
            f.write('{:d} {:d} {:g}\n'.format(self.referencetime[0] + 1,
                                              self.referencetime[1] + 1,
                                              self.referencetime[2]))
        # item 14
        f.write('{}\n'.format(self.stoptimeoption))
        if self.stoptimeoption == 3:
            # item 15
            f.write('{:g}\n'.format(self.stoptime + 1))

        # item 16
        if self.simulationtype == 3 or self.simulationtype == 4:
            f.write('{}\n'.format(self.timepointoption))
            if self.timepointoption == 1:
                # item 17
                f.write('{} {}\n'.format(self.timepointdata[0],
                                         self.timepointdata[1][0]))
            elif self.timepointoption == 2:
                # item 18
                f.write('{}\n'.format(self.timepointdata[0]))
                # item 19
                tp = self.timepointdata[1]
                v = Util2d(self.parent, (tp.shape[0],),
                           np.float32, tp,
                           name='temp',
                           locat=self.unit_number[0])
                f.write(v.string)

        # item 20
        f.write('{}\n'.format(self.zonedataoption))
        if self.zonedataoption == 2:
            # item 21
            f.write('{}\n'.format(self.stopzone))
            # item 22
            f.write(self.zones.get_file_entry())

        # item 23
        f.write('{}\n'.format(self.retardationfactoroption))
        if self.retardationfactoroption == 2:
            # item 24
            f.write(self.retardation.get_file_entry())

        # item 25
        f.write('{}\n'.format(len(self.particlegroups)))
        for pg in self.particlegroups:
            pg.write(f, ws=self.parent.model_ws)

        f.close()
