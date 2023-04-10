from flopy.mf6.utils.flopy_plugins import FPAPIPluginInterface


class FlopyRvp(FPAPIPluginInterface):
    """
    FlopyRvc is a simple example of a flopy plugin that behaves similarly to
    the  MODFLOW 6 RIV package.  FlopyRvc allows for riverbed hydraulic
    conductance to change with flow direction.

    To use this plugin add a ModflowGwffp_Rvc package
    (mf6/modflow/mfgwffp_rvt.py) to your flow model.
    """
    abbr = "rvp"

    def __init__(self, use_api_package=True):
        super().__init__(use_api_package)

    def init_plugin(self, fd_debug=None):
        super().init_plugin(fd_debug)
        # load options data
        self.print_input = self.package.print_input.get_data()
        # load options data
        self.print_flows = self.package.print_flows.get_data()
        # load options data
        self.save_flows = self.package.save_flows.get_data()
        # load options data
        self.auxiliary = self.package.auxiliary.get_data()
        # load options data
        self.boundnames = self.package.boundnames.get_data()
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

    def iteration_start(self, sp, ts, iter_num, sln_group):
        super().iteration_start(sp, ts, iter_num, sln_group)

        # record start of iteration
        # get mf6 variables
        cur_nodelist = self.mf6_default_package.get_advanced_var("nodelist")
        hcof_list = self.mf6_default_package.hcof
        rhs_list = self.mf6_default_package.rhs
        bound = self.mf6_default_package.get_advanced_var("bound")
        model_x = self.mf6_model.X.flatten()

        # loop through stress period data
        for row_num, data_row in enumerate(self.current_spd):
            aux_offset = 0
            # get stress period variables
            cellid = data_row[0 + aux_offset]
            hriv = data_row[1 + aux_offset]
            cond_up = data_row[2 + aux_offset]
            cond_down = data_row[3 + aux_offset]
            rbot = data_row[4 + aux_offset]

            # convert cellid to reduced node number
            node = self.get_node(cellid)
            node_r = self.mf6_model.usertonode[node]
            cur_nodelist[row_num] = node_r + 1

            if node_r <= 0:
                # cell inactive, set hcof and rhs to 0
                hcof_list[row_num] = 0.0
                rhs_list[row_num] = 0.0
                bound[row_num, 1] = 0.0
                continue
            if hriv > model_x[node_r]:
                # set conductance to cond_down
                criv = cond_down
            else:
                # set conductance to cond_up
                criv = cond_up
            if model_x[node_r] <= rbot:
                # head is below river bottom
                rhs_list[row_num] = -criv * (hriv - rbot)
                bound[row_num] = -criv * (hriv - rbot)
                hcof_list[row_num] = 0.0
            else:
                # head is above river bottom
                rhs_list[row_num] = -criv * hriv
                bound[row_num] = -criv * hriv
                hcof_list[row_num] = -criv

        # update mf6 variables
        self.mf6_default_package.set_advanced_var("nodelist", cur_nodelist)
        self.mf6_default_package.rhs = rhs_list
        self.mf6_default_package.hcof = hcof_list
        self.mf6_default_package.set_advanced_var("bound", bound)
