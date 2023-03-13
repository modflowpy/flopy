from flopy.mf6.utils.flopy_plugins import FPBMIPluginInterface


class ConductanceVals:
    def __init__(self, cond_up, cond_down):
        self.cond_up = cond_up
        self.cond_down = cond_down


class FlopyRvc(FPBMIPluginInterface):
    """
    FlopyRvc is a simple example of a flopy plug-in that modifies the behavior
    of an existing MODFLOW-6 package.  FlopyRvc allows for riverbed hydraulic
    conductance to change with flow direction.

    To use this plug-in add a ModflowGwffp_Rvc package
    (mf6/modflow/mfgwffp_rvt.py) to your flow model.
    """

    abbr = "rvc"

    def __init__(self, use_api_package=False):
        super().__init__(use_api_package)
        self.rv_cond = {}

    def init_plugin(self, fd_debug):
        super().init_plugin(fd_debug)
        # load dimensions data
        self.maxbound = self.package.maxbound.get_data()
        # load stress period data
        self.spd = self.package.stress_period_data
        self.current_spd = []
        self.current_sp = -1

    def stress_period_start(self, kper, sln_group):
        super().stress_period_start(kper, sln_group)

        # get data for this stress period
        current_spd = self.spd.get_data(kper)
        if current_spd is not None:
            current_spd = current_spd.tolist()
            if len(current_spd) > 0:
                self.current_spd = current_spd
            else:
                self.current_spd = []

        # build dictionary of river conductance
        self.rv_cond = {}
        for row_num, data_row in enumerate(self.current_spd):
            aux_offset = 0
            # get stress period variables
            pkg_name = data_row[0 + aux_offset]
            rvc_bound = data_row[1 + aux_offset]
            cond_up = data_row[2 + aux_offset]
            cond_down = data_row[3 + aux_offset]
            if pkg_name not in self.rv_cond:
                self.rv_cond[pkg_name] = {}
            self.rv_cond[pkg_name][rvc_bound] = ConductanceVals(
                cond_up, cond_down
            )

    def iteration_start(self, sp, ts, iter_num, sln_group):
        super().iteration_start(sp, ts, iter_num, sln_group)

        # record start of iteration
        model_x = self.mf6_model.X.flatten()
        # loop through riv packages in rvc stress period data
        for pkg_name, cond_dict in self.rv_cond.items():
            if pkg_name in self.mf6_model.package_dict:
                # load mf6 variables from riv package "pkg_name"
                self.mf6_default_package = pkg_name
                nodelist = self.mf6_default_package.get_advanced_var(
                    "nodelist"
                )
                bound_array = self.mf6_default_package.get_advanced_var(
                    "bound"
                )
                pkg_boundnames = self.mf6_default_package.get_advanced_var(
                    "boundname_cst"
                )

                # loop through river cells in current riv package
                for idx, riv_line in enumerate(bound_array):
                    if pkg_boundnames[idx] in cond_dict:
                        # compare stage to current head
                        if riv_line[0] > model_x[nodelist[idx]]:
                            # set conductance to cond_down
                            bound_array[idx][1] = cond_dict[
                                pkg_boundnames[idx]
                            ].cond_down
                        else:
                            # set conductance to cond_up
                            bound_array[idx][1] = cond_dict[
                                pkg_boundnames[idx]
                            ].cond_up
                # set boundary array
                self.mf6_default_package.bound = bound_array
