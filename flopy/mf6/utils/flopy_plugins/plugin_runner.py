import os
from ctypes.util import find_library

from ....mbase import resolve_exe
from ...mfbase import FlopyException, VerbosityLevel
from ...modflow import mfgwfapi
from . import FPAPIPluginInterface


class PluginRunner:
    def __init__(self, sim, kwargs, fd_debug, debug_mode):
        self.sim = sim
        self.sim_path = sim.simulation_data.mfpath.get_sim_path()
        self.lib_name = sim.lib_name
        self.model_dict = sim.model_dict
        self.verbosity_level = sim.simulation_data.verbosity_level.value
        self.kwargs = kwargs
        self.fd_debug = fd_debug
        self.debug_mode = debug_mode
        self.flopy_api_plugins = None

    def _add_platform_lib_ext(self):
        import platform

        if platform.system().lower() == "windows":
            if not self.lib_name.lower().endswith(".dll"):
                return f"{self.lib_name}.dll"
        elif platform.system().lower() == "linux":
            if not self.lib_name.lower().endswith(".so"):
                return f"{self.lib_name}.so"
        else:
            if not self.lib_name.lower().endswith(".dylib"):
                return f"{self.lib_name}.dylib"
        return ""

    def get_api_dll(self):
        # Check to make sure that program and namefile exist
        tried = f"{self.lib_name}"
        lib = find_library(self.lib_name)
        if lib is None:
            lib_name = self._add_platform_lib_ext()
            if lib_name != "":
                tried = f"{tried} or {lib_name}"
                lib = find_library(lib_name)

            try:
                exe_path = resolve_exe("mf6.exe")
            except FileNotFoundError:
                exe_path = ""
            if exe_path != "":
                exe_dir = os.path.split(exe_path)[0]
                dll_full_path = os.path.join(exe_dir, self.lib_name)
                tried = f"{tried} or {dll_full_path}"
                if os.path.isfile(dll_full_path):
                    lib = dll_full_path
                if lib is None and lib_name != "":
                    dll_full_path = os.path.join(exe_dir, lib_name)
                    tried = f"{tried} or {dll_full_path}"
                    if os.path.isfile(dll_full_path):
                        lib = dll_full_path

        if lib is None:
            raise Exception(
                f"The library {self.lib_name} does not exist.\n"
                f"Tried to find library using the following: "
                f"{tried}"
            )
        else:
            if (
                    self.verbosity_level >= VerbosityLevel.normal.value
            ):
                print(
                    f"FloPy is using the following lib to run the model: {lib}"
                )
        return lib

    def get_api_plugins(self):
        self.flopy_api_plugins = {}
        for model in self.model_dict.values():
            for package in model.get_package():
                if package.structure.flopy_package_interface is not None:
                    interface = package.structure.flopy_package_interface
                    if interface in FPAPIPluginInterface.flopy_api_plugins():
                        self.flopy_api_plugins[
                            package
                        ] = FPAPIPluginInterface.flopy_api_plugins()[
                            interface
                        ]()
                        if self.debug_mode:
                            pg_type = type(self.flopy_api_plugins[package])
                            abbr = self.flopy_api_plugins[package].abbr
                            self.fd_debug.write(
                                "  Discovered FloPy plugin type "
                                f"{pg_type} with abbreviation "
                                f"{abbr} for package "
                                f"{package.package_abbr}.\n"
                            )
                            self.fd_debug.flush()
                    else:
                        raise FlopyException(
                            f"ERROR: Package "
                            f"'{package.package_abbr}' "
                            "uses FloPy plugin "
                            f"'{interface}', which could "
                            f"not be found."
                        )
        return self.flopy_api_plugins

    def initialize(self):
        # api model run with plugin callbacks
        if self.debug_mode:
            self.fd_debug.write("Initializing FloPy plugins...\n")
            self.fd_debug.flush()
        model = []
        api_plugin_num = 0
        for package, flopy_plugin in self.flopy_api_plugins.items():
            model.append(package.model_or_sim)
            package_kwargs = None
            if package.name[0] in self.kwargs:
                package_kwargs = self.kwargs[package.name[0]]
            flopy_plugin.receive_vars(
                self.sim, package.model_or_sim, package, package_kwargs
            )
            if self.debug_mode:
                self.fd_debug.write(
                    "  Initializing FloPy plugin "
                    f"{flopy_plugin.abbr}.\n"
                )
                self.fd_debug.flush()
            flopy_plugin.init_plugin(self.fd_debug)
            if flopy_plugin.uses_api_package:
                # set up api package
                if hasattr(package, "print_input"):
                    print_input = package.print_input.get_data()
                else:
                    print_input = True
                if hasattr(package, "print_flows"):
                    print_flows = package.print_flows.get_data()
                else:
                    print_flows = True
                if hasattr(package, "save_flows"):
                    save_flows = package.save_flows.get_data()
                else:
                    save_flows = True
                api_package = mfgwfapi.ModflowGwfapi(
                    model[0],
                    maxbound=flopy_plugin.max_bound,
                    pname=f"API_{flopy_plugin.abbr}_{api_plugin_num}",
                    filename=f"{model[0].name}_{flopy_plugin.abbr}.api",
                    print_input=print_input,
                    print_flows=print_flows,
                    save_flows=save_flows,
                )
                flopy_plugin.api_package = api_package
                api_plugin_num += 1
        return api_plugin_num

    def run_simulation(self):
        try:
            import modflowapi
        except Exception as ex:
            message = (
                "Failed to initialize modflowapi library with message:"
                f"{str(ex)}"
            )
            print(message)
            if self.debug_mode:
                self.fd_debug.write(f"{message}\n")
                self.fd_debug.close()
            raise ex

        # there are API packages, run model through API interface
        # set up modflow API
        dll = self.get_api_dll()

        try:
            modflowapi.run_simulation(
                dll,
                self.sim_path,
                self._api_callback,
            )
        except Exception as ex:
            message = (
                "Failed to complete model run with message: "
                f"{str(ex)}\nVersion {modflowapi.__version__} "
                f"of modflowapi used."
            )
            print(message)
            if self.debug_mode:
                self.fd_debug.write(f"{message}\n")
                self.fd_debug.close()
            raise ex
        flopy_plugins = {}

        # api model run with package calls at beginning and end
        if self.debug_mode:
            self.fd_debug.write("Calling sim_complete for plugins...\n")
        for package, flopy_plugin in self.flopy_api_plugins.items():
            if self.debug_mode:
                self.fd_debug.write(
                    f"  Calling sim_complete for {flopy_plugin.abbr}.\n"
                )
                self.fd_debug.flush()
            flopy_plugin.sim_complete()
            # build plugin dictionary for easy access
            flopy_plugins[package.name[0]] = flopy_plugin

        if self.debug_mode:
            self.fd_debug.write("run_simulation ending successfully\n")
            self.fd_debug.close()
        return flopy_plugins

    @staticmethod
    def _handle_api_callback_exception(loc, flopy_plugin, ex):
        message = (
            f"Exception occurred in {loc} " f" of plugin {flopy_plugin.abbr}"
        )
        print(message)
        print(str(ex))
        if flopy_plugin.fd_debug is not None:
            flopy_plugin.fd_debug.write(f"{message}\n")
            flopy_plugin.fd_debug.write(str(ex))
            flopy_plugin.fd_debug.close()
        raise ex

    def _api_callback(self, mf6_sim, callback_type):
        import modflowapi

        if callback_type == modflowapi.Callbacks.initialize:
            # api model run with package calls at beginning and end
            for flopy_plugin in self.flopy_api_plugins.values():
                try:
                    if flopy_plugin.fd_debug is not None:
                        flopy_plugin.fd_debug.write(
                            "Calling receive_api for "
                            "plugin "
                            f"{flopy_plugin.abbr}.\n"
                        )
                        flopy_plugin.fd_debug.flush()
                    flopy_plugin.receive_api(mf6_sim)
                except Exception as ex:
                    # print out exception information so it is not lost
                    message = (
                        "Exception occurred in receive_api of "
                        f"plugin {flopy_plugin.abbr}"
                    )
                    PluginRunner._handle_api_callback_exception(
                        "receive_api", flopy_plugin, ex
                    )
        elif callback_type == modflowapi.Callbacks.stress_period_start:
            # call api packages stress_period_start
            for flopy_plugin in self.flopy_api_plugins.values():
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling stress_period_start for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        flopy_plugin.stress_period_start(mf6_sim.kper, mf6_sim)
                    except Exception as ex:
                        # print out exception information so it is not lost
                        PluginRunner._handle_api_callback_exception(
                            "stress_period_start", flopy_plugin, ex
                        )
        elif callback_type == modflowapi.Callbacks.timestep_start:
            # call api packages time_step_start
            for flopy_plugin in self.flopy_api_plugins.values():
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling timestep_start for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        flopy_plugin.time_step_start(
                            mf6_sim.kper, mf6_sim.kstp, mf6_sim
                        )
                    except Exception as ex:
                        # print out exception information so it is not lost
                        PluginRunner._handle_api_callback_exception(
                            "timestep_start", flopy_plugin, ex
                        )
        elif callback_type == modflowapi.Callbacks.iteration_start:
            # call api packages iteration_start
            for flopy_plugin in self.flopy_api_plugins.values():
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling iteration_start for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        flopy_plugin.iteration_start(
                            mf6_sim.kper,
                            mf6_sim.kstp,
                            mf6_sim.iteration,
                            mf6_sim,
                        )
                    except Exception as ex:
                        # print out exception information so it is not lost
                        PluginRunner._handle_api_callback_exception(
                            "iteration_start", flopy_plugin, ex
                        )
        elif callback_type == modflowapi.Callbacks.iteration_end:
            # call api packages iteration_end
            for flopy_plugin in self.flopy_api_plugins.values():
                allow_cvg = True
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling iteration_end for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        allow_cvg = flopy_plugin.iteration_end(
                            mf6_sim.kper,
                            mf6_sim.kstp,
                            mf6_sim.iteration,
                            mf6_sim,
                        )
                    except Exception as ex:
                        PluginRunner._handle_api_callback_exception(
                            "iteration_end", flopy_plugin, ex
                        )
                    mf6_sim.allow_convergence = (
                        mf6_sim.allow_convergence and allow_cvg
                    )
        elif callback_type == modflowapi.Callbacks.timestep_end:
            # call api packages time_step_end
            slnobj = list(mf6_sim.solutions.values())[0]
            converged = (
                mf6_sim.allow_convergence and mf6_sim.iteration < slnobj.mxiter
            )
            for flopy_plugin in self.flopy_api_plugins.values():
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling timestep_end for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        flopy_plugin.time_step_end(
                            mf6_sim.kper, mf6_sim.kstp, converged, mf6_sim
                        )
                    except Exception as ex:
                        PluginRunner._handle_api_callback_exception(
                            "timestep_end", flopy_plugin, ex
                        )
        elif callback_type == modflowapi.Callbacks.stress_period_end:
            # call api packages stress_period_start
            for flopy_plugin in self.flopy_api_plugins.values():
                if flopy_plugin.run_for_solution_group(mf6_sim):
                    try:
                        if flopy_plugin.fd_debug is not None:
                            flopy_plugin.fd_debug.write(
                                "Calling stress_period_end for plugin "
                                f"{flopy_plugin.abbr}.\n"
                            )
                            flopy_plugin.fd_debug.flush()
                        flopy_plugin.stress_period_end(mf6_sim.kper, mf6_sim)
                    except Exception as ex:
                        PluginRunner._handle_api_callback_exception(
                            "stress_period_end", flopy_plugin, ex
                        )
