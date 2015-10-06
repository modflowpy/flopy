__author__ = 'aleaf'

import os
import numpy as np
import flopy


path = '../data/'

def test_sfr(mfnam, sfrfile, model_ws, outfolder='written_sfr'):

    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws, verbose=True)
    sfr = flopy.modflow.ModflowSfr2.load(os.path.join(model_ws, sfrfile), m)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outpath = os.path.join(outfolder, sfrfile)
    sfr.write(outpath)

    sfr2 = flopy.modflow.ModflowSfr2.load(outpath, m)

    assert np.all(sfr2.reach_data == sfr.reach_data)
    assert np.all(sfr2.dataset_5 == sfr.dataset_5)
    for k, v in sfr2.segment_data.items():
        assert np.all(v == sfr.segment_data[k])
    for k, v in sfr2.channel_flow_data.items():
        assert np.all(v == sfr.channel_flow_data[k])
    for k, v in sfr2.channel_geometry_data.items():
        assert np.all(v == sfr.channel_geometry_data[k])

    return m, sfr

m, sfr = test_sfr('test1ss.nam', 'test1ss.sfr', path+'test-run')
'''
assert len(sfr.dataset_5) == 1
assert sfr.segment_data[0].shape == (8,)
assert sfr.reach_data.shape == (36,)
assert len(sfr.reach_data[0]) == 6
assert len(sfr.channel_flow_data) == 1
assert len(sfr.channel_flow_data[0]) == 1
assert len(sfr.channel_flow_data[0][0]) == 3
assert len(sfr.channel_flow_data[0][0][0]) == 11
# would be good to test for floats here
assert len(sfr.channel_geometry_data[0]) == 2
assert list(sfr.channel_geometry_data[0].keys()) == [6, 7]
assert sfr.channel_geometry_data[0][6][0] == [0.0,  10.,  80.,  100.,  150.,  170.,  240.,  250.]
'''
m, sfr = test_sfr('test1tr.nam', 'test1tr.sfr', path+'test-run')

#assert list(sfr.dataset_5.keys()) == [0, 1]

m, sfr = test_sfr('testsfr2_tab.nam', 'testsfr2_tab_ICALC1.sfr', path+'test-run')

assert list(sfr.dataset_5.keys()) == list(range(0, 50))

m, sfr = test_sfr('testsfr2_tab.nam', 'testsfr2_tab_ICALC2.sfr', path+'test-run')

assert sfr.channel_geometry_data[0][0] == [[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                                           [6.0, 4.5, 3.5, 0.0, 0.3, 3.5, 4.5, 6.0]]

m, sfr = test_sfr('testsfr2.nam', 'testsfr2.sfr', path+'test-run')

assert round(sum(sfr.segment_data[49][0]), 7) == 3.9700007

m, sfr = test_sfr('UZFtest2.nam', 'UZFtest2.sfr', path+'test-run')

j=2