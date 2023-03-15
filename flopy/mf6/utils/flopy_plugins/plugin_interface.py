import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
import numpy as np


class FPPluginInterface:
    """
    Base class of flopy plugins for MODFLOW-6.  Your flopy
    plugins should not directly inherent from this class, inherent from
    the FPBMIPluginInterface instead.

    Attributes
    ----------
    simulation : MFSimulation
        Simulation object that this plugin is a part of. Simulation object
        acquired when simulation is run by flopy and the receive_vars
        method is called.
    model : MFModel
        Model object that this plugin is a part of.
    package : MFPackage
        Package data that the flopy plugin for MODFLOW-6 uses to determine
        the plugin's user settings.
    mg : Grid
        Model grid object for the model this plugin is a part of.

    """

    flopy_plugins = {}
    interface_type = "standard"
    abbr = None

    def __init__(self):
        self.simulation = None
        self.model = None
        self.package = None
        self.mg = None
        self._user_kwargs = None
        self._dis_type = None
        self.fd_debug = None

    def __init_subclass__(cls):
        """Register plugin type"""
        super().__init_subclass__()
        if cls.abbr is not None:
            if cls.interface_type == "standard":
                FPPluginInterface.flopy_plugins[cls.abbr] = cls

    def receive_vars(self, simulation, model, package, user_kwargs):
        """This method is called by MFSimulation.run_simulation prior to
        starting the simulation.  receive_vars is called prior to init_plugin.
        receive_vars is called to give the flopy plugin object access to the
        simulation, model, and package objects.

        Parameters
        ----------
        simulation : MFSimulation
            Simulation object that this plugin is a part of. Simulation object
            acquired when simulation is run by flopy and the receive_vars
            method is called.
        model : MFModel
            Model object that this plugin is a part of.
        package : MFPackage
            Package data that the flopy plugin for MODFLOW-6 uses to determine
            the plugin's user settings.
        user_kwargs : dict
            User defined variables passed as kwargs to run_simulation.
        """
        self.simulation = simulation
        self.model = model
        self.package = package
        self.mg = model.modelgrid
        self._user_kwargs = user_kwargs

    def init_plugin(self, fd_debug):
        """This method is called by MFSimulation.run_simulation prior to
        starting the simulation and immediately after receive_vars.  Override
        this method to execute any initialization code in your flopy plugin.

        Parameters
        ----------
        fd_debug : file descriptor or None
            Debug file descriptor

        Returns
        ----------
        bool : Success/Failure
        """
        if fd_debug is not None:
            self.fd_debug = fd_debug
        return True

    def sim_complete(self):
        """This method is called by MFSimulation.run_simulation after the
        simulation has completed.  Override this method to execute any
        post-model processing or cleanup code in your flopy plugin.

        Returns
        ----------
        bool : Success/Failure
        """
        return True

    def val_from_cellid(self, cell_id, grid_array):
        """This method retrieves a value from an array in the shape of the
        model grid for a particular cell id.  Structured, vertex and
        unstructured grids are supported.

        Parameters
        ----------
        cell_id : tuple
            Cell ID corresponding to the location in the array where a
            value should be retrieved.
        grid_array: numpy ndarray
            Array containing value to be retrieved.

        Returns
        ----------
        value : Value in a ndarray, None if grid type can not be resolved.
        """
        if self.mg.grid_type == "structured":
            assert len(cell_id) == 3
            return grid_array[cell_id[0], cell_id[1], cell_id[2]]
        elif self.mg.grid_type == "vertex":
            assert len(cell_id) == 2
            return grid_array[cell_id[0], cell_id[1]]
        elif self.mg.grid_type == "unstructured":
            assert len(cell_id) == 1
            return grid_array[cell_id[0]]
        return None

    @property
    def dis_type(self):
        """Returns a string specifying the type of dis package the model uses,
        which can be used in the BMI interface.

        Returns
        ----------
        str : The type of dis package used by the model
        """
        if self._dis_type is None:
            dis_pkg = self.model.get_package("dis")
            self._dis_type = dis_pkg._package_type
        return self._dis_type

    def get_node(self, cell_id):
        """Given a cell ID return the corresponding model node.

        Parameters
        ----------
        cell_id : tuple
            Cell ID

        Returns
        ----------
        int : Node corresponding to cell_id in the model.
        """
        if self.mg.grid_type == "structured":
            assert len(cell_id) == 3
            return self.mg.get_node([cell_id])[0]
        elif self.mg.grid_type == "vertex":
            assert len(cell_id) == 2
            return self.mg.get_node([cell_id])[0]
        else:
            assert len(cell_id) == 1
            return cell_id[0] + 1

    @staticmethod
    def sq_saturation(top, bot, x, c1=None, c2=None):
        """Nonlinear smoothing function returns value between 0-1;
            Cubic saturation function

        Parameters
        ----------
        top : float
            top elevation of the cell
        bot : float
            bottom elevation of the cell
        x : float
            head elevation
        c1 : float
            coefficient 1
        c2 : float
            coefficient 2
        Returns
        ----------
        float : Value from smoothing function.
        """
        # process optional variables
        if c1 is not None:
            cof1 = c1
        else:
            cof1 = -2.0
        if c1 is not None:
            cof2 = c2
        else:
            cof2 = 3.0
        #
        # -- calculate head diference from bottom (w),
        #    calculate range (b), and
        #    calculate normalized head difference from bottom (s)
        w = x - bot
        b = top - bot
        s = w / b
        #
        # -- divide cof1 and cof2 by range to the power 3 and 2, respectively
        cof1 = cof1 / b**3.0
        cof2 = cof2 / b**2.0
        #
        # -- calculate fraction
        if s < 0.0:
            y = 0.0
        elif s < 1.0:
            y = cof1 * w**3.0 + cof2 * w**2.0
        else:
            y = 1.0
        return y

    @staticmethod
    def sq_saturation_derivative(top, bot, x, c1=None, c2=None):
        """Nonlinear smoothing function returns value between 0-1;
            Cubic saturation derivative function

        Parameters
        ----------
        top : float
            top elevation of the cell
        bot : float
            bottom elevation of the cell
        x : float
            head elevation
        c1 : float
            coefficient 1
        c2 : float
            coefficient 2

        Returns
        ----------
        float : Value from smoothing function.
        """
        #!
        #! -- process optional variables
        if c1 is not None:
            cof1 = c1
        else:
            cof1 = -2.0
        if c2 is not None:
            cof2 = c2
        else:
            cof2 = 3.0
        #!
        #! -- calculate head diference from bottom (w),
        #!    calculate range (b), and
        #!    calculate normalized head difference from bottom (s)
        w = x - bot
        b = top - bot
        s = w / b
        #!
        #! -- multiply cof1 and cof2 by 3 and 2, respectively, and then
        #!    divide by range to the power 3 and 2, respectively
        cof1 = cof1 * 3.0 / b**3.0
        cof2 = cof2 * 2.0 / b**2.0
        #!
        #! -- calculate derivative of fraction with respect to x
        if s < 0.0:
            return 0.0
        elif s < 1.0:
            return cof1 * w**2.0 + cof2 * w
        else:
            return 0.0

    def _build_grid_array(self, val):
        if self.mg.grid_type == "structured":
            return np.full((self.mg.nlay, self.mg.nrow, self.mg.ncol), val)
        elif self.mg.grid_type == "vertex":
            return np.full((self.mg.nlay, self.mg.ncpl), val)
        else:
            return np.full((self.mg.nnodes,), val)


class FPBMIPluginInterface(FPPluginInterface):
    """
    Base class of flopy plugins that use MODFLOW-6's BMI interface.  If you
    are making a flopy plugin, use this class as your base class.

    Parameters
    ----------
    use_api_package : bool
        Support use of the api package for this plug-in

    Attributes
    ----------
    run_for_all_solution_groups : bool
        Whether this plug-in gets a call-back when MODFLOW-6 is solving for
        each solution group (True), or only gets a call-back when MODFLOW-6
        is solving for the solution group this flopy plug-in is a part of
        (False).
    api_package : MFPackage
        Generic boundary MODFLOW-6 package that is used by the flopy package
        to modify the groundwater flow equations. A unique instance of this
        package is created for each flopy package with a package name
        reflecting the flopy package.  Package budgets for flopy packages
        will appear in the MODFLOW-6 listing file under this generic API
        package.
    mf6 : mf6 bmi interface
        MF6 BMI interface object for direct access to the MF6 API.  Use
        mf6_sim, mf6_model, and mf6_dis for more user-friendly access to the
        MF6 API.
    mf6_sim : modflowapi.Simulation
        Modflowapi simulation object containing access to MF6 variables from
        the current running simulation.
    mf6_model : modflowapi.Model
        Modflowapi model object containing access to MF6 variables from
        this package's model in the current running simulation.
    mf6_dis : modflowapi.ArrayPackage
        Modflowapi package object containing access to the discretization
        package's MF6 variables.
    mf6_top : numpy ndarray
        array of cell top elevations accessed through the MODFLOW-6 BMI
    mf6_bot : numpy ndarray
        array of cell bottom elevations accessed through the MODFLOW-6 BMI
    mf6_area : numpy ndarray
        array of cell top/bottom areas accessed through the MODFLOW-6 BMI
    mf6_idomain : numpy ndarray
        array of cell idomain values accessed through the MODFLOW-6 BMI
    kper : int
        current stress period
    """

    _flopy_bmi_plugins = {}
    _external_loaded = False
    interface_type = "bmi"

    def __init__(self, use_api_package=True):
        super().__init__()
        self.run_for_all_solution_groups = False
        # modflowapi BMI interface related objects
        self.mf6 = None
        self.mf6_sim = None
        self.mf6_model = None
        self.mf6_dis = None

        # stress period and time step
        self.kper = None
        self.kstp = None
        self.total_kstp_completed = None
        self.iter = None

        # discretization information
        self._cache_state = {}
        self._top = None
        self._bot = None
        self._area = None
        self._idomain = None

        # flopy-plugin settings
        self._use_api_package = use_api_package
        self._input_column_spacing = 12
        self._allow_convergence = True
        self._default_package = None

        # BMI tags
        self._pkg_rhs_tag = None
        self._pkg_hcof_tag = None
        if use_api_package:
            self._pkg_nodelist_tag = None
        self._pkg_bound_tag = None
        self._pkg_nbound_tag = None

        # packages
        self.api_package = None

    def __init_subclass__(cls):
        """Register plugin type"""
        super().__init_subclass__()
        if cls.abbr is not None:
            if cls.interface_type == "bmi":
                FPBMIPluginInterface._flopy_bmi_plugins[cls.abbr] = cls

    @staticmethod
    def flopy_bmi_plugins():
        """Detects flopy plug-ins installed as python packages.

        Returns
        ----------
        flopy_plugins: dict
            Dictionary with flopy plug-in classes as values and plug-in names
            as keys
        """
        if not FPBMIPluginInterface._external_loaded:
            __eps = entry_points(group="mf6api.plugin")
            for _ep in __eps:
                # internal plug-in takes precedent over external plug-in with
                # the same name
                if _ep.name not in FPBMIPluginInterface._flopy_bmi_plugins:
                    _plugin_class = _ep.load()
                    FPBMIPluginInterface._flopy_bmi_plugins[
                        _ep.name
                    ] = _plugin_class
            FPBMIPluginInterface._external_loaded = True
        return FPBMIPluginInterface._flopy_bmi_plugins

    @property
    def uses_api_package(self):
        """Returns whether this plug-in uses the api package."""
        return self._use_api_package

    def receive_vars(self, simulation, model, package, user_kwargs):
        """This method is called by MFSimulation.run_simulation prior to
        starting the simulation.  receive_vars is called prior to init_plugin.
        receive_vars is called to give the flopy plugin object access to the
        simulation, model, and package objects.

        Parameters
        ----------
        simulation : MFSimulation
            Simulation object that this plugin is a part of. Simulation object
            acquired when simulation is run by flopy and the receive_vars
            method is called.
        model : MFModel
            Model object that this plugin is a part of.
        package : MFPackage
            Package data that the flopy plugin for MODFLOW-6 uses to determine
            the plugin's user settings.
        user_kwargs : dict
            User defined variables passed as kwargs to run_simulation.
        """
        super().receive_vars(simulation, model, package, user_kwargs)

    def init_plugin(self, fd_debug=None):
        """This method is called by MFSimulation.run_simulation prior to
        starting the simulation and immediately after receive_vars.  Override
        this method to execute any initialization code in your flopy plugin.
        FPBMIPluginInterface initializes any enabled listing and/or debug
        files here.

        Parameters
        ----------
        fd_debug : file descriptor or None
            Debug file descriptor

        Returns
        ----------
        bool : Success/Failure
        """
        super().init_plugin(fd_debug)

    def receive_bmi(self, mf6_sim):
        """This method is called by MFSimulation.run_simulation prior to
        starting the simulation and immediately after the BMI interface is
        initialized.

        Parameters
        ----------
        mf6_sim : modflowapi simulation interface object

        """
        assert len(self.model.name) <= 16, (
            "ERROR: The model name cannot " "exceed 16 characters."
        )
        self.total_kstp_completed = 0
        self.mf6_sim = mf6_sim
        self.mf6 = mf6_sim.mf6

        # resolve the model that this flopy plugin is in
        model_names = self.mf6_sim.model_names
        if self.model is not None:
            if self.model.name in model_names:
                self.mf6_model = self.mf6_sim.get_model(self.model.name)
        if self.mf6_model is None:
            if len(model_names) == 1:
                self.mf6_model = self.mf6_sim.get_model(model_names[0])

        if self._use_api_package and self._default_package is None:
            assert self.api_package is not None
            self.mf6_default_package = self.api_package.package_name

            # update nbound to maxbound
            nbound = self.mf6_default_package.get_advanced_var("nbound")
            try:
                nbound[0] = self.max_bound
            except Exception as ex:
                print(ex)
                raise ex
            self.mf6_default_package.set_advanced_var("nbound", nbound)

        if self.mf6_model is not None:
            # get dis/disv/disu package from modflowapi
            dis_type = self.dis_type
            for package in self.mf6_model.package_dict.values():
                if dis_type == package.pkg_type:
                    self.mf6_dis = package
                    break

    def run_for_solution_group(self, sln_group):
        """Returns true of this plug-in should run for solution group
        sln_group, returns false otherwise.

        Parameters
        ----------
        sln_group: modflowapi solution group
            The current solution group being processed

        Returns
            run_for_solution_group : bool
        """
        if self.run_for_all_solution_groups:
            return True
        if self.model is not None:
            if self.model.name in sln_group.model_names:
                return True
            else:
                return False
        return False

    def _reset_cache_state(self):
        self._cache_state = {key: False for key in self._cache_state.keys()}

    def stress_period_start(self, kper, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        before the start of each stress period.  Override this method to
        execute code immediately before the start of each stress period.

        Parameters
        ----------
        kper : int
            current stress period
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.
        Returns
        ----------
        bool : Success value
        """
        if kper != self.kper:
            self.kper = kper
        self.kstp = 0
        self._reset_cache_state()
        return True

    def stress_period_end(self, kper, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        after the end of each stress period.  Override this method to execute
        code immediately after the end of each stress period.

        Parameters
        ----------
        kper : int
            current stress period
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.

        Returns
        ----------
        bool : Success value
        """
        self._reset_cache_state()
        return True

    def time_step_start(self, kper, kstp, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        before the start of each time step.  Override this method to execute
        code immediately before the start of each time step.

        Parameters
        ----------
        kper : int
            current stress period
        kstp : int
            current time step
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.
        Returns
        ----------
        bool : Success value
        """
        if kper != self.kper:
            self.kper = kper
        if kstp != self.kstp:
            self.kstp = kstp
        self._reset_cache_state()
        return True

    def time_step_end(self, kper, kstp, converged, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        after the end of each time step.  Override this method to execute
        code immediately after the end of each time step.

        Parameters
        ----------
        kper : int
            current stress period
        kstp : int
            current time step
        converged : bool
            time step successfully converged on a solution
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.

        Returns
        ----------
        bool : Success value
        """
        self.total_kstp_completed += 1
        self._reset_cache_state()
        return True

    def iteration_start(self, kper, kstp, iter_num, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        before the start of each outer iteration.  Override this method to
        execute code immediately before the start of each outer iteration.

        Parameters
        ----------
        kper : int
            current stress period
        kstp : int
            current time step
        iter_num : int
            outer iteration number
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.

        Returns
        ----------
        bool : Success value
        """
        if iter_num != self.iter:
            self.iter = iter_num
        self._reset_cache_state()
        return True

    def iteration_end(self, kper, kstp, iter_num, sln_group):
        """This method is called by MFSimulation.run_simulation immediately
        after the end of each outer iteration.  Override this method to
        execute code immediately before the start of each time step.
        This method returns a success value, returning false will force
        MODFLOW to continue iterating.

        Parameters
        ----------
        kper : int
            current stress period
        kstp : int
            current time step
        iter_num : int
            outer iteration number
        sln_group : modflowapi.simulation.Simulation
            solution group that is currently solving.  only relevant if
            run_for_all_solution_groups flag is set to true.

        Returns
        ----------
        bool : Success value
        """
        self._reset_cache_state()
        return self._allow_convergence

    def sim_complete(self):
        """This method is called by MFSimulation.run_simulation immediately
        after the simulation completes.  Override this method to execute code
        immediately after the end of the simulation.

        Returns
        ----------
        bool : Success value
        """
        return True

    @property
    def mf6_default_package(self):
        return self._default_package

    @mf6_default_package.setter
    def mf6_default_package(self, package_name):
        """
        Set the default MODFLOW-6 package that this plug-in will be
        interfacing with.  This plug-in is given easy access to the
        default package's rhs, hcof, modelist, bound, and nbound properties
        though the "mf6_pkg_*" properties and the "mf6_pkg_set_*" methods.
        """
        if self.mf6 is None:
            print(
                "Exception: Can not set MODFLOW-6 default package before "
                "BMI is initialized."
            )
            raise Exception(
                "Can not set MODFLOW-6 default package before "
                "BMI is initialized."
            )
        if package_name in self.mf6_model.package_dict:
            self.package_name = package_name.upper()
            self._default_package = self.mf6_model.package_dict[package_name]
        else:
            print(
                f"Exception: Package {package_name} not found in model "
                f"{self.mf6_model.name}."
            )
            raise Exception(
                f"Package '{package_name} not found in model "
                f"{self.mf6_model.name}."
            )

    """
    Package data retrieval methods
    """

    def mf6_idomain_val(self, cell_id):
        """Given a cell ID return the corresponding idomain value.

        Parameters
        ----------
        cell_id : tuple
            Cell ID

        Returns
        ----------
        int : IDOMAIN value for a cell_id in the model.
        """
        if self._idomain is None or (
            "idomain" in self._cache_state and not self._cache_state["idomain"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get idomain value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get idomain value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._idomain = self.mf6_dis.get_array("idomain")
            self._cache_state["idomain"] = True
        return self.val_from_cellid(cell_id, self._idomain)

    @property
    def mf6_idomain(self):
        """Returns the idomain array.

        Returns
        ----------
        ndarray : IDOMAIN array
        """
        if self._idomain is None or (
            "idomain" in self._cache_state and not self._cache_state["idomain"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get idomain value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get idomain value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._idomain = self.mf6_dis.get_array("idomain")
            self._cache_state["idomain"] = True
        return self._idomain

    @property
    def max_bound(self):
        """Retrieves maxbound from the flopy package file.  If there is no
        maxbound, returns 0.

        Returns
        ----------
        int : Maxbound value

        """
        if self.package is not None:
            if hasattr(self.package, "maxbound"):
                maxbound = self.package.maxbound
                if maxbound is not None:
                    return maxbound.get_data()
            elif hasattr(self, "maxbound"):
                return self.maxbound
        return 0

    @property
    def mf6_top(self):
        """
        Returns the cell top elevation array

        Returns
        ----------
        ndarray : Cell top elevation array
        """
        if self._top is None or (
            "top" in self._cache_state and not self._cache_state["top"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get top value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get top value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._top = self.mf6_dis.get_array("top")
            self._cache_state["top"] = True

        return self._top

    def mf6_top_val(self, cell_id):
        """
        Returns the cell top elevation

        Parameters
        ----------
        cell_id : tuple
            The dis/disv/disu cell id of the cell whose top elevation to return

        Returns
        ----------
        float : Cell top elevation
        """
        if self._top is None or (
            "top" in self._cache_state and not self._cache_state["top"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get top value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get top value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._top = self.mf6_dis.get_array("top")
            self._cache_state["top"] = True

        return self.val_from_cellid(cell_id, self._top)

    def mf6_bot(self):
        """
        Returns the cell bottom elevation array

        Returns
        ----------
        ndarray : Cell bottom elevation array
        """
        if self._bot is None or (
            "bot" in self._cache_state and not self._cache_state["bot"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get bot value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get bot value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._bot = self.mf6_dis.get_array("bot")
            self._cache_state["bot"] = True

        return self._bot

    def mf6_bot_val(self, cell_id):
        """
        Returns the cell bottom elevation

        Parameters
        ----------
        cell_id : tuple
            The dis/disv/disu cell id of the cell whose bottom elevation
            to return

        Returns
        ----------
        float : Cell bottom elevation
        """
        if self._bot is None or (
            "bot" in self._cache_state and not self._cache_state["bot"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get bot value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get bot value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._bot = self.mf6_dis.get_array("bot")
            self._cache_state["bot"] = True

        return self.val_from_cellid(cell_id, self._bot)

    @property
    def mf6_area(self):
        """Get MODFLOW-6 AREA array containing model cell areas"""
        if self._area is None or (
            "area" in self._cache_state and not self._cache_state["area"]
        ):
            if self.mf6_dis is None:
                print(
                    "Exception: Can not get area value, no dis package "
                    f"found in model {self.model.name}"
                )
                raise Exception(
                    "Can not get area value, no dis package "
                    f"found in model {self.model.name}"
                )
            self._area = self.mf6_dis.get_array("area")
            self._cache_state["area"] = True

        return self._area
