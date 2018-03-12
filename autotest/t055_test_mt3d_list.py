

import os
import flopy


def test_mtlist():
    try:
        import pandas as pd
    except:
        return
    mt_dir = os.path.join("..","examples","data","mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir,"mcomp.list"))
    df_gw,df_sw = mt.parse(forgive=False)

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail1.list"))
    df_gw, df_sw = mt.parse(forgive=True)

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail2.list"))
    df_gw, df_sw = mt.parse(forgive=True)

if __name__ == '__main__':
    test_mtlist()