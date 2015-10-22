"""
Some basic tests for SFR checker (not super rigorous)
need to add a test case that has elevation input by reach
"""

import sys
import os
import flopy
from flopy.modflow.mfsfr2 import check


def load_check_sfr(mfnam, model_ws, checker_output_path):

    #print('Testing {}\n'.format(mfnam) + '='*100)
    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws)
    m.change_model_ws(checker_output_path)
    checker_outfile = 'SFRcheck_{}.txt'.format(m.get_name())
    
    return m.sfr2.check(checker_outfile, level=1)


def test_sfrcheck():
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    cpth = os.path.join('data')
    m = flopy.modflow.Modflow.load('test1tr.nam', model_ws=path, verbose=False)
    
    sfr = flopy.modflow.ModflowSfr2.load(os.path.join(path, 'test1tr.sfr'), m)
    
    # run level=0 check
    m.change_model_ws(cpth)
    fpth = 'SFRchecker_results.txt'
    m.sfr2.check(fpth, level=0)
    
    # test checks without modifications
    chk = check(m.sfr2)
    chk.numbering()
    assert 'continuity in segment and reach numbering' in chk.passed
    chk.routing()
    assert 'circular routing' in chk.passed
    chk.overlapping_conductance()
    assert 'overlapping conductance' in chk.failed # this example model has overlapping conductance
    chk.elevations()
    for test in ['segment elevations', 'reach elevations', 'reach elevations vs. grid elevations']:
        assert test in chk.passed
    chk.slope()
    assert 'slope' in chk.passed
    
    # create gaps in segment numbering
    m.sfr2.segment_data[0]['nseg'][-1] += 1
    m.sfr2.reach_data['ireach'][3] += 1
    
    # create circular routing instance
    m.sfr2.segment_data[0]['outseg'][1] = 1
    m.sfr2.segment_data[0]['outseg']
    
    chk = check(m.sfr2)
    chk.numbering()
    assert 'continuity in segment and reach numbering' in chk.failed
    chk.routing()
    assert 'circular routing' in chk.failed
    
    sfr_items = {0: {'mfnam': 'test1ss.nam',
                     'sfrfile': 'test1ss.sfr'},
                 1: {'mfnam': 'test1tr.nam',
                     'sfrfile': 'test1tr.sfr'},
                 2: {'mfnam': 'testsfr2_tab.nam',
                     'sfrfile': 'testsfr2_tab_ICALC1.sfr'},
                 3: {'mfnam': 'testsfr2_tab.nam',
                     'sfrfile': 'testsfr2_tab_ICALC2.sfr'},
                 4: {'mfnam': 'testsfr2.nam',
                     'sfrfile': 'testsfr2.sfr'},
                 5: {'mfnam': 'UZFtest2.nam',
                     'sfrfile': 'UZFtest2.sfr'},
                 }
    

    passed = {}
    failed = {}
    
    for i, case in sfr_items.items():
        chk = load_check_sfr(case['mfnam'], model_ws=path, checker_output_path=cpth)
        passed[i] = chk.passed
        failed[i] = chk.failed
    assert 'overlapping conductance' in failed[1]
    assert 'segment elevations vs. model grid' in failed[2]


if __name__ == '__main__':
    test_sfrcheck()
