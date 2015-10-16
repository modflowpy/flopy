__author__ = 'aleaf'

import os
import numpy as np
import flopy

# pytest changes the directory to flopy3
path = ''
if os.path.split(os.getcwd())[-1] == 'py.test':
    path += '../'
path += 'examples/data/mf2005_test/'
#path = os.path.join('..', 'examples', 'data', 'mf2005_test')

def sfr_process(mfnam, sfrfile, model_ws, outfolder='data'):

    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws, verbose=True)
    sfr = m.get_package('SFR2')

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

def test_sfr():
    m, sfr = sfr_process('test1ss.nam', 'test1ss.sfr', path)

    m, sfr = sfr_process('test1tr.nam', 'test1tr.sfr', path)
    
    #assert list(sfr.dataset_5.keys()) == [0, 1]
    
    m, sfr = sfr_process('testsfr2_tab.nam', 'testsfr2_tab_ICALC1.sfr', path)
    
    assert list(sfr.dataset_5.keys()) == list(range(0, 50))
    
    m, sfr = sfr_process('testsfr2_tab.nam', 'testsfr2_tab_ICALC2.sfr', path)
    
    assert sfr.channel_geometry_data[0][1] == [[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                                               [6.0, 4.5, 3.5, 0.0, 0.3, 3.5, 4.5, 6.0]]
    
    m, sfr = sfr_process('testsfr2.nam', 'testsfr2.sfr', path)
    
    assert round(sum(sfr.segment_data[49][0]), 7) == 3.9700007
    
    m, sfr = sfr_process('UZFtest2.nam', 'UZFtest2.sfr', path)

if __name__ == '__main__':
    test_sfr()
