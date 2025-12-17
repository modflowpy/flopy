import os

import flopy


def test_external_file_header_preservation(function_tmpdir):
    """Test GitHub issue #2233: https://github.com/modflowpy/flopy/issues/2233"""

    sim = flopy.mf6.MFSimulation(
        sim_name="header_preservation_test", sim_ws=function_tmpdir, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test_model")
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=3,
        ncol=3,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, k=1.0)

    wel_external_file = function_tmpdir / "wel_data.txt"
    drn_external_file = function_tmpdir / "drn_data.txt"
    with open(wel_external_file, "w") as f:
        f.write("# k i j flux - WEL package data\n")
        f.write("1 2 2 -100.0\n")
    with open(drn_external_file, "w") as f:
        f.write("# k i j elev cond - DRN package data\n")
        f.write("1 1 1 0.5 10.0\n")

    wel = flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data={0: {"filename": str(wel_external_file)}}
    )
    drn = flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data={0: {"filename": str(drn_external_file)}}
    )

    sim.simulation_data.comments_on = True  # test fails without this
    sim.set_all_data_external()
    sim.write_simulation()

    # the original files seem fine regardless
    with open(wel_external_file, "r") as fwel, open(drn_external_file, "r") as fdrn:
        wel_original_content = fwel.read()
        drn_original_content = fdrn.read()
        assert wel_original_content.startswith("# k i j flux")
        assert drn_original_content.startswith("# k i j elev cond")

    # with `store_as_external_file` headers were gone without comments_on
    wel.stress_period_data.store_as_external_file(str(wel_external_file))
    drn.stress_period_data.store_as_external_file(str(drn_external_file))
    with (
        open(function_tmpdir / "wel_data_1.txt", "r") as fwel,
        open(function_tmpdir / "drn_data_1.txt", "r") as fdrn,
    ):
        wel_original_content = fwel.read()
        drn_original_content = fdrn.read()
        assert wel_original_content.startswith("# k i j flux")
        assert drn_original_content.startswith("# k i j elev cond")
