"""
Some basic tests for mflistfile.py module (not super rigorous)

"""

import sys
import os
import flopy


def test_mflist():
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    list_file = os.path.join(pth, 'freyberg.gitlist')
    assert os.path.exists(list_file)
    mflist = flopy.utils.MfListBudget(list_file)
    df_in, df_out = mflist.get_dataframes(start_datetime=None)
    return

if __name__ == '__main__':
    test_mflist()
