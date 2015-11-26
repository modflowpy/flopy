__author__ = 'aleaf'

#import sys
#sys.path.append('/Users/aleaf/Documents/GitHub/flopy3')
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import flopy

if os.path.split(os.getcwd())[-1] == 'flopy3':
    path = os.path.join('examples', 'data', 'mf2005_test')
    path2 = os.path.join('examples', 'data', 'sfr_test')
    outpath = os.path.join('py.test/temp')
else:
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    path2 = os.path.join('..', 'examples', 'data', 'sfr_test')
    outpath = 'temp'

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
                 6: {'mfnam': 'TL2009.nam',
                     'sfrfile': 'TL2009.sfr'}
                 }

def sfr_process(mfnam, sfrfile, model_ws, outfolder=outpath):

    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws, verbose=False)
    sfr = m.get_package('SFR2')

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outpath = os.path.join(outfolder, sfrfile)
    sfr.write_file(outpath)

    m.remove_package('SFR2')
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

def load_sfr_only(sfrfile):
    m = flopy.modflow.Modflow()
    sfr = flopy.modflow.ModflowSfr2.load(sfrfile, m)
    return m, sfr

def load_all_sfr_only(path):
    for i, item in sfr_items.items():
        load_sfr_only(os.path.join(path, item['sfrfile']))

def interpolate_to_reaches(sfr):
    reach_data = sfr.reach_data
    segment_data = sfr.segment_data[0]
    for reachvar, segvars in {'strtop': ('elevup', 'elevdn'),
                              'strthick': ('thickm1', 'thickm2'),
                              'strhc1': ('hcond1', 'hcond2')}.items():
        reach_data[reachvar] = sfr._interpolate_to_reaches(*segvars)
        for seg in segment_data.nseg:
            reaches = reach_data[reach_data.iseg == seg]
            dist = np.cumsum(reaches.rchlen) - 0.5 * reaches.rchlen
            fp = [segment_data[segment_data['nseg'] == seg][segvars[0]][0],
                  segment_data[segment_data['nseg'] == seg][segvars[1]][0]]
            xp = [dist[0], dist[-1]]
            assert np.sum(np.abs(reaches[reachvar] - np.interp(dist, xp, fp).tolist())) < 0.01
    return reach_data

def test_sfr():

    load_all_sfr_only(path2)

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

    assert isinstance(sfr.plot()[0], matplotlib.axes.Axes) # test the plot() method

    # trout lake example (only sfr file is included)
    # can add tests for sfr connection with lak package
    m, sfr = load_sfr_only(os.path.join(path2, 'TL2009.sfr'))
    # convert sfr package to reach input
    sfr.reachinput = True
    sfr.isfropt = 1
    sfr.reach_data = interpolate_to_reaches(sfr)
    sfr.get_slopes()
    assert sfr.reach_data.slope[29] == (sfr.reach_data.strtop[29] - sfr.reach_data.strtop[107])\
                                       /sfr.reach_data.rchlen[29]
    chk = sfr.check()
    assert sfr.reach_data.slope.min() < 0.0001 and 'minimum slope' in chk.failed
    sfr.reach_data.slope[0] = 1.1
    chk.slope(maximum_slope=1.0)
    assert 'maximum slope' in chk.failed

def test_sfr_renumbering():
    # test segment renumbering

    r = np.zeros((9, 2), dtype=[('iseg', int), ('ireach', int)])
    r = np.core.records.fromarrays(r.transpose(), dtype=[('iseg', int), ('ireach', int)])
    r['iseg'] = range(1, 10)
    r['ireach'] = np.ones(9)

    d = np.zeros((9, 2), dtype=[('nseg', int), ('outseg', int)])
    d = np.core.records.fromarrays(d.transpose(), dtype=[('nseg', int), ('outseg', int)])
    d['nseg'] = range(1, 10)
    d['outseg'] = [4, 0, 6, 8, 3, 8, 1, 2, 8]
    m = flopy.modflow.Modflow()
    sfr = flopy.modflow.ModflowSfr2(m, reach_data=r, segment_data={0: d})
    chk = sfr.check()
    assert 'segment numbering order' in chk.failed
    sfr.renumber_segments()
    chk = sfr.check()
    assert 'continuity in segment and reach numbering' in chk.passed
    assert 'segment numbering order' in chk.passed

if __name__ == '__main__':
    test_sfr()
    test_sfr_renumbering()
