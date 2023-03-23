from pathlib import Path
from shutil import copytree

import pytest
from autotest.regression.conftest import is_nested
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.mf6 import MFSimulation
from flopy.utils.compare import compare_heads

pytestmark = pytest.mark.mf6


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test_mf6_example_simulations(function_tmpdir, mf6_example_namfiles):
    # MF6 examples parametrized by simulation. `mf6_example_namfiles` is a list
    # of models to run in order provided. Coupled models share the same tempdir
    #
    # Parameters
    # ----------
    # function_tmpdir: function-scoped temporary directory fixture
    # mf6_example_namfiles: ordered list of namfiles for 1+ coupled models

    # make sure we have at least 1 name file
    if len(mf6_example_namfiles) == 0:
        pytest.skip("No namfiles (expected ordered collection)")
    namfile = Path(mf6_example_namfiles[0])  # pull the first model's namfile

    # coupled models have nested dirs (e.g., 'mf6gwf' and 'mf6gwt') under model directory
    # TODO: are there multiple types of couplings? e.g. besides GWF-GWT, mt3dms?
    nested = is_nested(namfile)
    function_tmpdir = Path(
        function_tmpdir / "workspace"
    )  # working directory (must not exist for copytree)
    cmpdir = function_tmpdir / "compare"  # comparison directory

    # copy model files into working directory
    copytree(
        src=namfile.parent.parent if nested else namfile.parent,
        dst=function_tmpdir,
    )

    def run_models():
        # run models in order received (should be alphabetical, so gwf precedes gwt)
        for namfile in mf6_example_namfiles:
            namfile_path = Path(namfile).resolve()
            namfile_name = namfile_path.name
            model_path = namfile_path.parent

            # working directory must be named according to the name file's parent (e.g.
            # 'mf6gwf') because coupled models refer to each other with relative paths
            wrkdir = (
                Path(function_tmpdir / model_path.name)
                if nested
                else function_tmpdir
            )

            # load simulation
            sim = MFSimulation.load(
                namfile_name, version="mf6", exe_name="mf6", sim_ws=wrkdir
            )
            assert isinstance(sim, MFSimulation)

            # run simulation
            success, buff = sim.run_simulation(report=True)
            assert success

            # change to comparison workspace
            sim.simulation_data.mfpath.set_sim_path(cmpdir)

            # write simulation files and rerun
            sim.write_simulation()
            success, _ = sim.run_simulation()
            assert success

            # get head file outputs
            headfiles1 = [p for p in wrkdir.glob("*.hds")]
            headfiles2 = [p for p in cmpdir.glob("*.hds")]

            # compare heads
            assert compare_heads(
                None,
                None,
                precision="double",
                text="head",
                files1=[str(p) for p in headfiles1],
                files2=[str(p) for p in headfiles2],
                outfile=cmpdir / "head_compare.dat",
            )

    run_models()
