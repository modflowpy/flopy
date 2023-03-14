import os
import numpy as np
from flopy.mf6.modflow.mfgwf import ModflowGwf
from flopy.mf6.modflow.mfgwfdis import ModflowGwfdis
from flopy.mf6.modflow.mfgwfoc import ModflowGwfoc
from flopy.mf6.modflow.mfgwfsto import ModflowGwfsto
from flopy.mf6.modflow.mfgwfriv import ModflowGwfriv
from flopy.mf6.modflow.mfgwffp_rvc import ModflowGwffp_Rvc
from flopy.mf6.modflow.mfgwffp_rvp import ModflowGwffp_Rvp
from flopy.mf6.modflow.mfims import ModflowIms
from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.mf6.modflow.mftdis import ModflowTdis
from flopy.mf6.modflow.mfgwfic import ModflowGwfic
from flopy.mf6.modflow.mfgwfnpf import ModflowGwfnpf
from flopy.mf6.modflow.mfgwfchd import ModflowGwfchd
from flopy.mf6.data.mfdatastorage import DataStorageType
from flopy.utils.datautil import PyListUtil
from modflow_devtools.markers import requires_exe

import pytest

pytestmark = pytest.mark.mf6


@requires_exe("mf6")
def test_rvc_plugin(tmpdir):
    # names
    sim_name = "rvc_plugin_test"
    model_name = "rvc_plugin_test"
    exe_name = "mf6"

    # set up simulation
    sim = MFSimulation(
        sim_name=sim_name, version="mf6", exe_name=exe_name, sim_ws=str(tmpdir)
    )
    tdis_rc = [
        (100.0, 1, 1.0),
        (1000.0, 1, 1.0),
        (1000.0, 1, 1.0),
        (1000.0, 1, 1.0),
    ]
    tdis_package = ModflowTdis(
        sim,
        time_units="DAYS",
        nper=4,
        perioddata=tdis_rc,
        start_date_time="1/1/2010",
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=0.0001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=0.0001,
        rcloserecord=0.001,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    bot_data = [-100 for x in range(150)]
    idom = np.ones((3, 15, 10), float)
    idom[0, 0, 0] = 0
    idom[0, 0, 1] = 0
    dis_package = ModflowGwfdis(
        model,
        nlay=3,
        nrow=15,
        ncol=10,
        delr=100.0,
        delc=100.0,
        top=50.0,
        botm=[5.0, -10.0, {"factor": 1.0, "data": bot_data}],
        idomain=idom,
        filename=f"{model_name}.dis",
    )

    ic_package = ModflowGwfic(model, strt=50.0, filename=f"{model_name}.ic")
    k = np.ones((3, 15, 10), float)
    k_list = [
        0.01,
        1.0,
        1.0,
        1.0,
        0.01,
        1.0,
        1.0,
        1.0,
        0.01,
        1.0,
        1.0,
        1.0,
        0.01,
        1.0,
        0.01,
    ]
    for row in range(0, 15):
        for col in range(0, 10):
            k[0, row, col] = k_list[row] * 100.0
    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        icelltype=[1, 0, 0],
        k=k,
        k33=[0.05, 0.005, 0.001],
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=f"{model_name}.cbc",
        head_filerecord=f"{model_name}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "FIRST"), ("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    sy_template = ModflowGwfsto.sy.empty(model, True)
    for layer in range(0, 3):
        sy_template[layer]["data"] = 0.2
    layer_storage_types = [
        DataStorageType.internal_array,
        DataStorageType.internal_constant,
        DataStorageType.internal_array,
    ]
    ss_template = ModflowGwfsto.ss.empty(
        model, True, layer_storage_types, 0.000001
    )
    sto_package = ModflowGwfsto(
        model,
        save_flows=True,
        iconvert=1,
        ss=ss_template,
        sy=sy_template,
        transient={0: True, 1: True},
    )

    chd_period_data = []
    for column in range(2, 10):
        chd_period_data.append(((0, 2, column), 50.0))
        chd_period_data.append(((0, 14, column), 50.0))
    chd_period_data_2 = []
    for column in range(2, 10):
        chd_period_data_2.append(((0, 2, column), 40.0))
        chd_period_data_2.append(((0, 14, column), 40.0))

    chd_period = {0: chd_period_data, 1: chd_period_data_2}
    chd_package = ModflowGwfchd(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=3,
        stress_period_data=chd_period,
    )

    riv_spd = {
        0: [
            ((0, 5, 5), 45.0, 10.0, 40.0, "1"),
            ((0, 5, 6), 44.9, 10.0, 39.9, "2"),
            ((0, 5, 7), 44.8, 10.0, 39.8, "3"),
            ((0, 5, 8), 44.7, 10.0, 39.7, "4"),
        ],
    }
    riv_package = ModflowGwfriv(
        model,
        pname="riv_0",
        boundnames=True,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=4,
        stress_period_data=riv_spd,
    )

    # run model with exponential evt
    sim.write_simulation()
    sim.run_simulation()

    budget_fpriv_nor = model.output.budget().get_data(text="RIV", full3D=False)

    rvc_spd = {
        0: [
            ("riv_0", "1", 10.0, 1000.0),
            ("riv_0", "2", 10.0, 1000.0),
            ("riv_0", "3", 10.0, 1000.0),
            ("riv_0", "4", 10.0, 1000.0),
        ]
    }
    fp_rvc = ModflowGwffp_Rvc(model, maxbound=4, stress_period_data=rvc_spd)
    rvc_dir = os.path.join(tmpdir, "rvc")
    sim.set_sim_path(rvc_dir)
    sim.write_simulation()
    sim.run_simulation()

    budget_fpriv_rvc = model.output.budget().get_data(text="RIV", full3D=False)

    # compare results
    for riv in range(0, 4):
        assert budget_fpriv_rvc[0][riv][2] == budget_fpriv_rvc[0][riv][2]
    for sp in range(1, 4):
        for riv in range(0, 4):
            assert budget_fpriv_nor[sp][riv][2] < budget_fpriv_rvc[sp][riv][2]

    model.remove_package(riv_package.package_name)
    model.remove_package(fp_rvc.package_name)
    # use stand-alone river flopy plugin
    rvp_spd = {
        0: [
            ((0, 5, 5), 45.0, 10.0, 1000.0, 40.0),
            ((0, 5, 6), 44.9, 10.0, 1000.0, 39.9),
            ((0, 5, 7), 44.8, 10.0, 1000.0, 39.8),
            ((0, 5, 8), 44.7, 10.0, 1000.0, 39.7),
        ],
    }

    riv_package = ModflowGwffp_Rvp(
        model,
        pname="riv_std",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=4,
        stress_period_data=rvp_spd,
    )

    rvp_dir = os.path.join(tmpdir, "rvp")
    sim.set_sim_path(rvp_dir)
    sim.write_simulation()
    success = sim.run_simulation(debug_mode=True)
    if not success:
        with open("debug_run_sim.txt", "r") as fd:
            data = fd.read()
            raise Exception(f"{data}")

    budget_fpriv_rvp = model.output.budget().get_data(text="API", full3D=False)
    array_util = PyListUtil()
    assert array_util.riv_array_comp(budget_fpriv_rvc, budget_fpriv_rvp)
