from os import listdir
from os.path import join

import pytest
from flaky import flaky
from matplotlib import rcParams

from flopy.mf6 import MFSimulation
from flopy.modflow import Modflow


@pytest.mark.mf6
def test_vertex_model_dot_plot(example_data_path):
    rcParams["figure.max_open_warning"] = 36

    # load up the vertex example problem
    sim = MFSimulation.load(
        sim_ws=example_data_path / "mf6" / "test003_gwftri_disv"
    )
    disv_ml = sim.get_model("gwf_1")
    ax = disv_ml.plot()
    assert isinstance(ax, list)
    assert len(ax) == 36


# occasional _tkinter.TclError: Can't find a usable tk.tcl (or init.tcl)
# similar: https://github.com/microsoft/azure-pipelines-tasks/issues/16426
@flaky
def test_model_dot_plot(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = Modflow.load("ibs2k.nam", "mf2k", model_ws=loadpth, check=False)
    ax = ml.plot()
    assert isinstance(ax, list), "ml.plot() ax is is not a list"
    assert len(ax) == 18, f"number of axes ({len(ax)}) is not equal to 18"


def test_dataset_dot_plot(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = Modflow.load("ibs2k.nam", "mf2k", model_ws=loadpth, check=False)

    # plot specific dataset
    ax = ml.bcf6.hy.plot()
    assert isinstance(ax, list), "ml.bcf6.hy.plot() ax is is not a list"
    assert len(ax) == 2, f"number of hy axes ({len(ax)}) is not equal to 2"


def test_dataset_dot_plot_nlay_ne_plottable(
    function_tmpdir, example_data_path
):
    import matplotlib.pyplot as plt

    loadpth = example_data_path / "mf2005_test"
    ml = Modflow.load("ibs2k.nam", "mf2k", model_ws=loadpth, check=False)
    # special case where nlay != plottable
    ax = ml.bcf6.vcont.plot()
    assert isinstance(
        ax, plt.Axes
    ), "ml.bcf6.vcont.plot() ax is is not of type plt.Axes"


def test_model_dot_plot_export(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = Modflow.load("ibs2k.nam", "mf2k", model_ws=loadpth, check=False)

    fh = join(function_tmpdir, "ibs2k")
    ml.plot(mflay=0, filename_base=fh, file_extension="png")
    files = [f for f in listdir(function_tmpdir) if f.endswith(".png")]
    if len(files) < 10:
        raise AssertionError(
            "ml.plot did not properly export all supported data types"
        )

    for f in files:
        t = f.split("_")
        if len(t) < 3:
            raise AssertionError("Plot filenames not written correctly")
