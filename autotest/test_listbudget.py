import os
import warnings

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_pkg
from modflow_devtools.misc import has_pkg

from flopy.utils import Mf6ListBudget, MfListBudget, MfusgListBudget, MtListBudget


def test_mflistfile(example_data_path):
    pth = example_data_path / "freyberg"
    list_file = pth / "freyberg.gitlist"
    assert os.path.exists(list_file)
    mflist = MfListBudget(list_file)

    names = mflist.get_record_names()
    assert isinstance(names, tuple)
    assert len(names) > 0

    bud = mflist.get_data(idx=-1)
    assert isinstance(bud, np.ndarray)

    kstpkper = mflist.get_kstpkper()
    bud = mflist.get_data(kstpkper=kstpkper[-1])
    assert isinstance(bud, np.ndarray)

    times = mflist.get_times()
    bud = mflist.get_data(totim=times[0])
    # TODO: there are two return types, but only one is documented
    assert isinstance(bud, np.ndarray) or bud is None

    ts_lens = mflist.get_tslens()
    # verify time step lengths add up to time step total times, within
    # rounding error
    ts_len_total = 0.0
    for time, ts_len in zip(times, ts_lens):
        ts_len_total += ts_len
        assert abs(ts_len_total - time) < 1.0

    # plt.bar(bud['index'], bud['value'])
    # plt.xticks(bud['index'], bud['name'], rotation=45, size=6)
    # plt.show()

    inc = mflist.get_incremental()
    assert isinstance(inc, np.ndarray)

    cum = mflist.get_cumulative(names="PERCENT_DISCREPANCY")
    assert isinstance(cum, np.ndarray)

    df_flx, df_vol = mflist.get_dataframes(start_datetime=None)
    assert isinstance(df_flx, pd.DataFrame)
    assert isinstance(df_vol, pd.DataFrame)

    # test get runtime
    runtime = mflist.get_model_runtime(units="hours")
    assert isinstance(runtime, float)


def test_mflist_reducedpumping(example_data_path):
    """
    test reading reduced pumping data from list file
    """
    pth = example_data_path / "mfusg_test" / "03B_conduit_unconfined" / "output"
    list_file = pth / "ex3B.lst"
    mflist = MfusgListBudget(list_file)
    assert isinstance(mflist.get_reduced_pumping(), np.recarray)


def test_mf6listfile(example_data_path):
    pth = example_data_path / "mf6" / "test005_advgw_tidal" / "expected_output"
    list_file = pth / "AdvGW_tidal.gitlist"
    assert os.path.exists(list_file)
    mflist = Mf6ListBudget(list_file)
    names = mflist.get_record_names()
    for item in ["RCH_IN", "RCH2_IN", "RCH3_IN", "RCH_OUT", "RCH2_OUT", "RCH3_OUT"]:
        assert item in names, f"{item} not found in names"
    assert len(names) == 26
    inc = mflist.get_incremental()


def test_mflist_reducedpumping_fail(example_data_path):
    """
    test failure for reading reduced pumping data from list file
    """
    pth = example_data_path / "mfusg_test" / "03A_conduit_unconfined" / "output"
    list_file = pth / "ex3A.lst"
    # Catch before flopy to avoid masking file not found assert
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"{list_file} not found")
    mflist = MfusgListBudget(list_file)
    with pytest.raises(AssertionError):
        mflist.get_reduced_pumping()


def test_mtlist(example_data_path):
    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "mcomp.list")
    df_gw, df_sw = mt.parse(forgive=False, diff=False, start_datetime=None)

    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "mt3d_with_adv.list")
    df_gw, df_sw = mt.parse(forgive=False, diff=False, start_datetime=None)

    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "CrnkNic.mt3d.list")
    df_gw, df_sw = mt.parse(forgive=False, diff=True, start_datetime=None)

    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "mcomp.list")
    df_gw, df_sw = mt.parse(forgive=False, start_datetime=None)

    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "mcomp.list")
    df_gw, df_sw = mt.parse(forgive=False, start_datetime="1-1-1970")

    mt_dir = example_data_path / "mt3d_test"
    mt = MtListBudget(mt_dir / "mt3d_imm_sor.list")
    df_gw, df_sw = mt.parse(forgive=False, start_datetime="1-1-1970")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mt_dir = example_data_path / "mt3d_test"
        mt = MtListBudget(mt_dir / "mcomp_fail1.list")
        df_gw, df_sw = mt.parse(forgive=True, start_datetime="1-1-1970")

        assert len(w) == 1, len(w)
        assert w[0].category == UserWarning, w[0]
        assert "error parsing GW mass budget" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mt_dir = example_data_path / "mt3d_test"
        mt = MtListBudget(mt_dir / "mcomp_fail2.list")
        df_gw, df_sw = mt.parse(forgive=True, start_datetime="1-1-1970")

        assert len(w) == 1, len(w)
        assert w[0].category == UserWarning, w[0]
        assert "error parsing SW mass budget" in str(w[0].message)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("8.5159-100", 8.5159e-100),
        ("-8.5159-100", -8.5159e-100),
        ("1.0000+100", 1.0e100),
        ("-1.2000+150", -1.2e150),
        ("3.9204E-103", 3.9204e-103),
        ("60.3962", 60.3962),
    ],
)
def test_mf6listfile_parse_budget_line_missing_e(value, expected):
    line = f"        FLOW-JA-FACE = {value:>16}          FLOW-JA-FACE = {value:>16}"
    entry, flux, cumu = Mf6ListBudget._parse_budget_line(None, line)
    assert entry == "FLOW-JA-FACE"
    assert flux == expected
    assert cumu == expected


MF6_BUDGET_TEMPLATE = """
  VOLUME BUDGET FOR ENTIRE MODEL AT END OF TIME STEP    1, STRESS PERIOD   {kper}
  ------------------------------------------------------------------------------

     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T

           IN:                                      IN:
           ---                                      ---
                 STO =         100.0000                   STO =         100.0000
                 CHD =         {chd:>8}                   CHD =         {chd:>8}

            TOTAL IN =         100.0000              TOTAL IN =         100.0000

          OUT:                                     OUT:
          ----                                     ----
                 STO =         100.0000                   STO =         100.0000
                 CHD =           0.0000                   CHD =           0.0000

           TOTAL OUT =         100.0000             TOTAL OUT =         100.0000

            IN - OUT =           0.0000              IN - OUT =           0.0000

 PERCENT DISCREPANCY =           0.00     PERCENT DISCREPANCY =           0.00
"""


def test_mf6listfile_unparseable_first_entry(function_tmpdir):
    # see https://github.com/modflowpy/flopy/issues/2856
    list_file = function_tmpdir / "gwf.lst"
    list_file.write_text(
        MF6_BUDGET_TEMPLATE.format(kper=1, chd="********")
        + MF6_BUDGET_TEMPLATE.format(kper=2, chd="0.0000")
    )
    match = r"first budget.*CHD_IN at time step 1, stress period 1: .*\*{8}"
    with pytest.raises(ValueError, match=match):
        Mf6ListBudget(list_file)


def test_mf6listfile_unparseable_later_entry(function_tmpdir):
    list_file = function_tmpdir / "gwf.lst"
    list_file.write_text(
        MF6_BUDGET_TEMPLATE.format(kper=1, chd="0.0000")
        + MF6_BUDGET_TEMPLATE.format(kper=2, chd="********")
    )
    mflist = Mf6ListBudget(list_file)
    inc, cum = mflist.get_budget()
    assert inc["STO_IN"][0] == pytest.approx(100.0)
    assert np.isnan(inc["STO_IN"][1])
    assert np.isnan(cum["CHD_IN"][1])
