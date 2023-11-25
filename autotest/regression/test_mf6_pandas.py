import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.mf6 import (
    ExtFileAction,
    MFModel,
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisv,
    ModflowGwfdrn,
    ModflowGwfevt,
    ModflowGwfevta,
    ModflowGwfghb,
    ModflowGwfgnc,
    ModflowGwfgwf,
    ModflowGwfgwt,
    ModflowGwfhfb,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrch,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfsfr,
    ModflowGwfsto,
    ModflowGwfwel,
    ModflowGwtadv,
    ModflowGwtdis,
    ModflowGwtic,
    ModflowGwtmst,
    ModflowGwtoc,
    ModflowGwtssm,
    ModflowIms,
    ModflowTdis,
    ModflowUtltas,
)
from flopy.mf6.data.mfdataplist import MFPandasList

pytestmark = pytest.mark.mf6


@requires_exe("mf6")
@pytest.mark.regression
def test_pandas_001(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "pd001"
    model_name = "pd001_mod"
    data_path = example_data_path / "mf6" / "create_tests" / test_ex_name
    ws = function_tmpdir / "ws"

    expected_output_folder = data_path / "expected_output"
    expected_head_file = expected_output_folder / "pd001_mod.hds"
    expected_cbc_file = expected_output_folder / "pd001_mod.cbc"

    # model tests
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=ws,
        continue_=True,
        memory_print_option="summary",
        use_pandas=True,
    )
    name = sim.name_file
    assert name.continue_.get_data()
    assert name.nocheck.get_data() is None
    assert name.memory_print_option.get_data() == "summary"
    assert sim.simulation_data.use_pandas

    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )
    # replace with real ims file
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename=f"{test_ex_name}.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )
    top = {"filename": "top.txt", "data": 100.0}
    botm = {"filename": "botm.txt", "data": 50.0}
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=top,
        botm=botm,
        filename=f"{model_name}.dis",
        pname="mydispkg",
    )
    ic_package = ModflowGwfic(model, strt=80.0, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=5.0,
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("np001_mod 1.cbc",)],
        head_filerecord=[("np001_mod 1.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("HEAD", "ALL")],
    )
    empty_sp_text = oc_package.saverecord.get_file_entry(1)
    assert empty_sp_text == ""
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    oc_package.saverecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )

    # test saving a text file with recarray data
    data_line = [((0, 0, 4), -2000.0), ((0, 0, 7), -2.0)]
    type_list = [("cellid", object), ("q", float)]
    data_rec = np.rec.array(data_line, type_list)
    well_spd = {
        0: {
            "filename": "wel_0.txt",
            "data": data_rec,
        },
        1: None,
    }
    wel_package = ModflowGwfwel(
        model,
        filename=f"{model_name}.wel",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=well_spd,
    )

    wel_package.stress_period_data.add_transient_key(1)
    # text user generated pandas dataframe without headers
    data_pd = pd.DataFrame([(0, 0, 4, -1000.0), (0, 0, 7, -20.0)])
    wel_package.stress_period_data.set_data(
        {1: {"filename": "wel_1.txt", "iprn": 1, "data": data_pd}}
    )

    # test getting data
    assert isinstance(wel_package.stress_period_data, MFPandasList)
    well_data_pd = wel_package.stress_period_data.get_dataframe(0)
    assert isinstance(well_data_pd, pd.DataFrame)
    assert well_data_pd.iloc[0, 0] == 0
    assert well_data_pd.iloc[0, 1] == 0
    assert well_data_pd.iloc[0, 2] == 4
    assert well_data_pd.iloc[0, 3] == -2000.0
    assert well_data_pd["layer"][0] == 0
    assert well_data_pd["row"][0] == 0
    assert well_data_pd["column"][0] == 4
    assert well_data_pd["q"][0] == -2000.0
    assert well_data_pd["layer"][1] == 0
    assert well_data_pd["row"][1] == 0
    assert well_data_pd["column"][1] == 7
    assert well_data_pd["q"][1] == -2.0

    well_data_rec = wel_package.stress_period_data.get_data(0)
    assert isinstance(well_data_rec, np.recarray)
    assert well_data_rec[0][0] == (0, 0, 4)
    assert well_data_rec[0][1] == -2000.0

    # test time series dat
    drn_package = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        timeseries=[(0.0, 60.0), (100000.0, 60.0)],
        stress_period_data=[((0, 0, 0), 80, "drn_1")],
    )
    drn_package.ts.time_series_namerecord = "drn_1"
    drn_package.ts.interpolation_methodrecord = "linearend"

    # test data with aux vars
    riv_spd = {
        0: {
            "filename": "riv_0.txt",
            "data": [((0, 0, 9), 110, 90.0, 100.0, 1.0, 2.0, 3.0)],
        }
    }
    riv_package = ModflowGwfriv(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        auxiliary=["var1", "var2", "var3"],
        stress_period_data=riv_spd,
    )
    riv_data = riv_package.stress_period_data.get_data(0)
    assert riv_data[0][0] == (0, 0, 9)
    assert riv_data[0][1] == 110
    assert riv_data[0][2] == 90.0
    assert riv_data[0][3] == 100.0
    assert riv_data[0][4] == 1.0
    assert riv_data[0][5] == 2.0
    assert riv_data[0][6] == 3.0

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # modify external files to resemble user generated text
    wel_file_0_pth = os.path.join(sim.sim_path, "wel_0.txt")
    wel_file_1_pth = os.path.join(sim.sim_path, "wel_1.txt")
    riv_file_0_pth = os.path.join(sim.sim_path, "riv_0.txt")
    with open(wel_file_0_pth, "w") as fd_wel_0:
        fd_wel_0.write("# comment header\n\n")
        fd_wel_0.write("1 1 5 -2000.0  # comment\n")
        fd_wel_0.write("# more comments\n")
        fd_wel_0.write("1 1 8 -2.0\n")
        fd_wel_0.write("# more comments\n")

    with open(wel_file_1_pth, "w") as fd_wel_1:
        fd_wel_1.write("# comment header\n\n")
        fd_wel_1.write("\t1\t1\t5\t-1000.0\t# comment\n")
        fd_wel_1.write("# more comments\n")
        fd_wel_1.write("1 1\t8\t-20.0\n")
        fd_wel_1.write("# more comments\n")

    with open(riv_file_0_pth, "w") as fd_riv_0:
        fd_riv_0.write("# comment header\n\n")
        fd_riv_0.write(
            "1\t1\t10\t110\t9.00000000E+01\t1.00000000E+02"
            "\t1.00000000E+00\t2.00000000E+00\t3.00000000E+00"
            "\t# comment\n"
        )

    # test loading and checking data
    test_sim = MFSimulation.load(
        test_ex_name,
        "mf6",
        "mf6",
        sim.sim_path,
        write_headers=False,
    )
    test_mod = test_sim.get_model()
    test_wel = test_mod.get_package("wel")

    well_data_pd_0 = test_wel.stress_period_data.get_dataframe(0)
    assert isinstance(well_data_pd_0, pd.DataFrame)
    assert well_data_pd_0.iloc[0, 0] == 0
    assert well_data_pd_0.iloc[0, 1] == 0
    assert well_data_pd_0.iloc[0, 2] == 4
    assert well_data_pd_0.iloc[0, 3] == -2000.0
    assert well_data_pd_0["layer"][0] == 0
    assert well_data_pd_0["row"][0] == 0
    assert well_data_pd_0["column"][0] == 4
    assert well_data_pd_0["q"][0] == -2000.0
    assert well_data_pd_0["layer"][1] == 0
    assert well_data_pd_0["row"][1] == 0
    assert well_data_pd_0["column"][1] == 7
    assert well_data_pd_0["q"][1] == -2.0
    well_data_pd = test_wel.stress_period_data.get_dataframe(1)
    assert isinstance(well_data_pd, pd.DataFrame)
    assert well_data_pd.iloc[0, 0] == 0
    assert well_data_pd.iloc[0, 1] == 0
    assert well_data_pd.iloc[0, 2] == 4
    assert well_data_pd.iloc[0, 3] == -1000.0
    assert well_data_pd["layer"][0] == 0
    assert well_data_pd["row"][0] == 0
    assert well_data_pd["column"][0] == 4
    assert well_data_pd["q"][0] == -1000.0
    assert well_data_pd["layer"][1] == 0
    assert well_data_pd["row"][1] == 0
    assert well_data_pd["column"][1] == 7
    assert well_data_pd["q"][1] == -20.0
    test_riv = test_mod.get_package("riv")
    riv_data_pd = test_riv.stress_period_data.get_dataframe(0)
    assert riv_data_pd.iloc[0, 0] == 0
    assert riv_data_pd.iloc[0, 1] == 0
    assert riv_data_pd.iloc[0, 2] == 9
    assert riv_data_pd.iloc[0, 3] == 110
    assert riv_data_pd.iloc[0, 4] == 90.0
    assert riv_data_pd.iloc[0, 5] == 100.0
    assert riv_data_pd.iloc[0, 6] == 1.0
    assert riv_data_pd.iloc[0, 7] == 2.0
    assert riv_data_pd.iloc[0, 8] == 3.0

    well_data_array = test_wel.stress_period_data.to_array()
    assert "q" in well_data_array
    array = np.array([0.0, 0.0, 0.0, 0.0, -2000.0, 0.0, 0.0, -2.0, 0.0, 0.0])
    array.reshape((1, 1, 10))
    compare = well_data_array["q"] == array
    assert compare.all()

    well_data_record = test_wel.stress_period_data.get_record()
    assert 0 in well_data_record
    assert "binary" in well_data_record[0]
    well_data_record[0]["binary"] = True
    well_data_pd_0.iloc[0, 3] = -10000.0
    well_data_record[0]["data"] = well_data_pd_0
    test_wel.stress_period_data.set_record(well_data_record)

    updated_record = test_wel.stress_period_data.get_record(data_frame=True)
    assert 0 in updated_record
    assert "binary" in updated_record[0]
    assert updated_record[0]["binary"]
    assert isinstance(updated_record[0]["data"], pandas.DataFrame)
    assert updated_record[0]["data"]["q"][0] == -10000

    record = [0, 0, 2, -111.0]
    test_wel.stress_period_data.append_list_as_record(record, 0)

    combined_data = test_wel.stress_period_data.get_dataframe(0)
    assert len(combined_data.axes[0]) == 3
    assert combined_data["q"][2] == -111.0

    test_drn = test_mod.get_package("drn")
    file_entry = test_drn.stress_period_data.get_file_entry()
    assert file_entry.strip() == "1 1 1 80 drn_1"
