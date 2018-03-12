

import os
import flopy


def test_mtlist():
    try:
        import pandas as pd
    except:
        return
    mt_dir = os.path.join("..","examples","data","mt3d_test")
    mt = flopy.utils.MtListBudget(os.path.join(mt_dir,"mcomp.list"))
    df_gw,df_sw = mt.parse()

if __name__ == '__main__':
    test_mtlist()