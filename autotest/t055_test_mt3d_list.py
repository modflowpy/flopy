

import os
import warnings
import flopy


def test_mtlist():
    try:
        import pandas as pd
    except:
        return

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp.list"))
    df_gw, df_sw = mt.parse(forgive=False, diff=False, start_datetime=None)

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "CrnkNic.mt3d.list"))
    df_gw, df_sw = mt.parse(forgive=False, diff=True, start_datetime=None)

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp.list"))
    df_gw, df_sw = mt.parse(forgive=False, start_datetime=None)

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp.list"))
    df_gw, df_sw = mt.parse(forgive=False, start_datetime="1-1-1970")

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mt3d_imm_sor.list"))
    df_gw, df_sw = mt.parse(forgive=False, start_datetime="1-1-1970")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
        mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail1.list"))
        df_gw, df_sw = mt.parse(forgive=True, start_datetime="1-1-1970")

        assert len(w) == 1, len(w)
        assert w[0].category == UserWarning, w[0]
        assert 'error parsing GW mass budget' in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
        mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail2.list"))
        df_gw, df_sw = mt.parse(forgive=True, start_datetime="1-1-1970")

        assert len(w) == 1, len(w)
        assert w[0].category == UserWarning, w[0]
        assert 'error parsing SW mass budget' in str(w[0].message)


if __name__ == '__main__':
    test_mtlist()