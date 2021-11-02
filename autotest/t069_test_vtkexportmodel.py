"""
Test vtk export_model function without packages_names definition
"""

import flopy
from flopy.export import vtk
from ci_framework import (
    base_test_dir,
    FlopyTestSetup,
)

mf_exe_name = "mf6"


def test_vtk_export_model_without_packages_names():
    base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)
    test_setup = FlopyTestSetup(verbose=True, test_dirs=baseDir)

    name = "mymodel"
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=baseDir, exe_name="mf6")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 9, 9), 0.0]]
    )

    # Export model without specifying packages_names parameter
    vtk.export_model(sim.get_model(), baseDir)

    # If the function executes without error then test was successful
    assert True


if __name__ == "__main__":
    test_vtk_export_model_without_packages_names()
