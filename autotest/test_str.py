import matplotlib
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.modflow import Modflow
from flopy.utils import MfListBudget

str_items = {
    0: {
        "mfnam": "str.nam",
        "sfrfile": "str.str",
        "lstfile": "str.lst",
    }
}


@requires_exe("mf2005")
@requires_pkg("pandas")
def test_str_issue1164(function_tmpdir, example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    m = Modflow.load(
        str_items[0]["mfnam"],
        exe_name="mf2005",
        model_ws=str(mf2005_model_path),
        verbose=False,
        check=False,
    )

    m.change_model_ws(str(function_tmpdir))

    # adjust stress period data
    spd0 = m.str.stress_period_data[0]
    spd0["flow"][0] = 2.1149856e6  # 450000000000000000.0000e-17
    m.str.stress_period_data[0] = spd0

    # write model datasets and run fixed
    m.write_input()
    success = m.run_model()
    assert success, "could not run base model"

    # get the budget
    lst_pth = str(function_tmpdir / str_items[0]["lstfile"])
    base_wb = MfListBudget(lst_pth).get_dataframes()[0]

    # set the model to free format
    m.set_ifrefm()

    # write model datasets and run revised
    m.write_input()
    success = m.run_model()
    assert success, "could not run revised model"

    # get the revised budget
    revised_wb = MfListBudget(lst_pth).get_dataframes()[0]

    # test if the budgets are the same
    assert revised_wb.equals(base_wb), "water budgets do not match"


def test_str_plot(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    m = Modflow.load(
        str_items[0]["mfnam"],
        model_ws=str(mf2005_model_path),
        verbose=True,
        check=False,
    )
    assert isinstance(m.str.plot()[0], matplotlib.axes.Axes)
    matplotlib.pyplot.close()
