"""
Some basic tests for SFR checker (not super rigorous)
need to add a test case that has elevation input by reach
"""

import sys
sys.path.insert(0, '../flopy')
import os
import flopy
from flopy.modflow.mfsfr2 import check

path = '../examples/data/mf2005_test/'

m = flopy.modflow.Modflow.load('test1tr.nam', model_ws=path, verbose=True)

sfr = flopy.modflow.ModflowSfr2.load(path + 'test1tr.sfr', m)

# run level=0 check
m.sfr2.check('/Users/aleaf/Documents/GitHub/flopy3/notebooks/checker_results.txt', level=0)

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

path = 'written_sfr'
if not os.path.isdir(path):
    os.makedirs(path)
mfpath = '../examples/data/mf2005_test/'

test_cases = {0: {'mfnam': 'test1ss.nam',
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


def sfr_checker_test(mfnam, model_ws, checker_output_path):

    print('Testing {}'.format(mfnam) + '='*100)
    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws)
    checker_outfile = mfnam [:-4] + '_SFRcheck.txt'
    return m.sfr2.check(checker_output_path + checker_outfile, level=1)

passed = {}
failed = {}

for i, case in test_cases.items():
    chk = sfr_checker_test(case['mfnam'], model_ws=mfpath, checker_output_path=path)
    passed[i] = chk.passed
    failed[i] = chk.failed
assert 'overlapping conductance' in failed[1]
assert 'segment elevations vs. model grid' in failed[2]
j=2