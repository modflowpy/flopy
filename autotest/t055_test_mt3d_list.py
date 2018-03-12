

import os
import flopy


def test_mtlist():
    try:
        import pandas as pd
    except:
        return
    mt_dir = os.path.join("..","examples","data","mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir,"mcomp.list"))
    df_gw,df_sw = mt.parse(forgive=False, start_datetime="1-1-1970")

    import matplotlib.pyplot as plt
    df_sw.plot()
    plt.show()

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail1.list"))
    df_gw, df_sw = mt.parse(forgive=True,start_datetime="1-1-1970")

    mt_dir = os.path.join("..", "examples", "data", "mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir, "mcomp_fail2.list"))
    df_gw, df_sw = mt.parse(forgive=True, start_datetime="1-1-1970")

if __name__ == '__main__':
    test_mtlist()