"""
Some basic tests for mflistfile.py module (not super rigorous)

"""

import sys
import os
import flopy
import numpy as np
#import matplotlib.pyplot as plt


def test_mflist():
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    list_file = os.path.join(pth, 'freyberg.gitlist')
    assert os.path.exists(list_file)
    mflist = flopy.utils.MfListBudget(list_file)

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
    assert isinstance(bud, np.ndarray)

    #plt.bar(bud['index'], bud['value'])
    #plt.xticks(bud['index'], bud['name'], rotation=45, size=6)
    #plt.show()

    inc = mflist.get_incremental()
    assert isinstance(inc, np.ndarray)

    cum = mflist.get_cumulative(names='PERCENT_DISCREPANCY')
    assert isinstance(cum, np.ndarray)

    # if pandas is installed
    try:
        import pandas
    except:
        return
    df_flx, df_vol = mflist.get_dataframes(start_datetime=None)
    assert isinstance(df_flx, pandas.DataFrame)
    assert isinstance(df_vol, pandas.DataFrame)

    return

if __name__ == '__main__':
    test_mflist()
