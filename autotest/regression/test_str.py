import pytest
from autotest.conftest import requires_exe, requires_pkg

from flopy.modflow import Modflow, ModflowOc, ModflowStr

str_items = {
    0: {
        "mfnam": "str.nam",
        "sfrfile": "str.str",
        "lstfile": "str.lst",
    }
}


@requires_exe("mf2005")
@requires_pkg("pymake")
@pytest.mark.regression
def test_str_fixed_free(tmpdir, example_data_path):
    import pymake

    mf2005_model_path = example_data_path / "mf2005_test"

    m = Modflow.load(
        str_items[0]["mfnam"],
        exe_name="mf2005",
        model_ws=str(mf2005_model_path),
        verbose=False,
        check=False,
    )
    m.change_model_ws(str(tmpdir))

    # get pointer to str package
    mstr = m.str
    mstr.istcb2 = -1

    # add aux variables to str
    aux_names = ["aux iface", "aux xyz"]
    names = ["iface", "xyz"]
    current, current_seg = ModflowStr.get_empty(23, 7, aux_names=names)

    # copy data from existing stress period data
    for name in mstr.stress_period_data[0].dtype.names:
        current[:][name] = mstr.stress_period_data[0][:][name]

    # fill aux variable data
    for idx, c in enumerate(mstr.stress_period_data[0]):
        for jdx, name in enumerate(names):
            current[idx][name] = idx + jdx * 10

    # replace str data with updated str data
    mstr = ModflowStr(
        m,
        mxacts=mstr.mxacts,
        nss=mstr.nss,
        ntrib=mstr.ntrib,
        ndiv=mstr.ndiv,
        icalc=mstr.icalc,
        const=mstr.const,
        ipakcb=mstr.ipakcb,
        istcb2=mstr.istcb2,
        iptflg=mstr.iptflg,
        irdflg=mstr.irdflg,
        stress_period_data={0: current},
        segment_data=mstr.segment_data,
        options=aux_names,
    )

    # add head output to oc file
    oclst = ["PRINT HEAD", "PRINT BUDGET", "SAVE HEAD", "SAVE BUDGET"]
    spd = {(0, 0): oclst, (0, 1): oclst, (0, 2): oclst}
    oc = ModflowOc(m, stress_period_data=spd)
    oc.reset_budgetunit()

    # reset ipakcb for str package to get ascii output in lst file
    mstr.ipakcb = -1

    m.write_input()
    success, buff = m.run_model()
    assert success, "base model run did not terminate successfully"

    # load the fixed format model with aux variables
    try:
        m2 = Modflow.load(
            str_items[0]["mfnam"],
            exe_name="mf2005",
            model_ws=str(mf2005_model_path),
            verbose=False,
            check=False,
        )
    except:
        m2 = None

    assert (
        m2 is not None
    ), "could not load the fixed format model with aux variables"

    for p in tmpdir.glob("*"):
        p.unlink()

    m.change_model_ws(str(tmpdir))
    m.set_ifrefm()
    m.write_input()

    success, buff = m.run_model()
    assert success, "free format model run did not terminate successfully"

    # load the free format model
    try:
        m2 = Modflow.load(
            str_items[0]["mfnam"],
            exe_name="mf2005",
            model_ws=str(tmpdir),
            verbose=False,
            check=False,
        )
    except:
        m2 = None

    assert (
        m2 is not None
    ), "could not load the free format model with aux variables"

    # compare the fixed and free format head files
    fn1 = str(tmpdir / "str.nam")
    fn2 = str(tmpdir / "str.nam")
    assert pymake.compare_heads(
        fn1, fn2, verbose=True
    ), "fixed and free format input output head files are different"
