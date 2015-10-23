"""
Some basic tests for mflistfile.py module (not super rigorous)

"""

import sys
import os
import flopy


def test_mflist():
    model_ws = os.path.join('..', 'examples', 'data', 'freyberg')
    nam = 'freyberg'
    ml = flopy.modflow.Modflow.load(nam, model_ws=model_ws, 
                                    exe_name='mf2005', version='mf2005')
    new_ws = 'temp'
    ml.change_model_ws(new_ws)
    ml.write_input()
    ml.run_model(silent=True)
    
    list_file = os.path.join(new_ws,'{}.list'.format(nam))
    assert os.path.exists(list_file)
    mflist = flopy.utils.MfListBudget(list_file)
    df_in,df_out = mflist.get_dataframes(start_datetime=None)
    return

if __name__ == '__main__':
    test_mflist()
