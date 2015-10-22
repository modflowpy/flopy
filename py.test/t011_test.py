"""
Some basic tests for mflistfile.py module (not super rigorous)

"""

import sys
import os
import flopy


def test_mflist():
    model_ws = os.path.join("..", "examples", "data", "freyberg")
    nam = "freyberg"
    list_file = os.path.join(model_ws,nam+".lst")
    assert os.path.exists(list_file)
    mflist = flopy.utils.MfListBudget(list_file)
    df_in,df_out = mflist.get_dataframes(start_datetime=None)
    return
