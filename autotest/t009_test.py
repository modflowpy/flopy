__author__ = 'aleaf'

import sys
sys.path.append('/Users/aleaf/Documents/GitHub/flopy3')
import os
import glob
import shutil
import numpy as np
from flopy.utils.recarray_utils import create_empty_recarray

try:
    import matplotlib
    # if os.getenv('TRAVIS'):  # are we running https://travis-ci.org/ automated tests ?
    #     matplotlib.use('Agg')  # Force matplotlib  not to use any Xwindows backend
except:
    matplotlib = None

import flopy
fm = flopy.modflow
from flopy.utils.sfroutputfile import SfrFile
from flopy.utils.reference import SpatialReference

if os.path.split(os.getcwd())[-1] == 'flopy3':
    path = os.path.join('examples', 'data', 'mf2005_test')
    path2 = os.path.join('examples', 'data', 'sfr_test')
    outpath = os.path.join('py.test/temp')
else:
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    path2 = os.path.join('..', 'examples', 'data', 'sfr_test')
    outpath = os.path.join('temp', 't009')
    # make the directory if it does not exist
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

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

def create_sfr_data():
    dtype = np.dtype([('k', int),
             ('i', int),
             ('j', int),
             ('iseg', int),
             ('ireach', int)])
    r = create_empty_recarray(27, dtype=dtype)
    r['i'] = [3, 4, 5,
              7, 8, 9,
              0, 1, 2,
              4, 4, 5,
              0, 0, 0,
              3, 4, 5,
              0, 1, 2,
              4, 5, 6,
              2, 2, 2]
    r['j'] = [0, 1, 2,
              6, 6, 6,
              6, 6, 6,
              3, 4, 5,
              9, 8, 7,
              6, 6, 6,
              0, 0, 0,
              6, 6, 6,
              9, 8, 7]
    r['iseg'] = sorted(list(range(1, 10)) * 3)
    r['ireach'] = [1, 2, 3] * 9

    d = create_empty_recarray(9, dtype=np.dtype([('nseg', int), ('outseg', int)]))
    d['nseg'] = range(1, 10)
    d['outseg'] = [4, 0, 6, 8, 3, 8, 1, 2, 8]
    return r, d

def sfr_process(mfnam, sfrfile, model_ws, outfolder=outpath):
    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws, verbose=False)
    sfr = m.get_package('SFR')

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outpath = os.path.join(outfolder, sfrfile)
    sfr.write_file(outpath)

    m.remove_package('SFR')
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
            assert np.sum(np.abs(
                reaches[reachvar] - np.interp(dist, xp, fp).tolist())) < 0.01
    return reach_data


def test_sfr():
    load_all_sfr_only(path2)

    m, sfr = sfr_process('test1ss.nam', 'test1ss.sfr', path)

    m, sfr = sfr_process('test1tr.nam', 'test1tr.sfr', path)

    # assert list(sfr.dataset_5.keys()) == [0, 1]

    m, sfr = sfr_process('testsfr2_tab.nam', 'testsfr2_tab_ICALC1.sfr', path)

    assert list(sfr.dataset_5.keys()) == list(range(0, 50))

    m, sfr = sfr_process('testsfr2_tab.nam', 'testsfr2_tab_ICALC2.sfr', path)

    assert sfr.channel_geometry_data[0][1] == [
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        [6.0, 4.5, 3.5, 0.0, 0.3, 3.5, 4.5, 6.0]]

    m, sfr = sfr_process('testsfr2.nam', 'testsfr2.sfr', path)

    assert round(sum(sfr.segment_data[49][0]), 7) == 3.9700007

    m, sfr = sfr_process('UZFtest2.nam', 'UZFtest2.sfr', path)

    if matplotlib is not None:
        assert isinstance(sfr.plot()[0],
                          matplotlib.axes.Axes)  # test the plot() method

    # trout lake example (only sfr file is included)
    # can add tests for sfr connection with lak package
    m, sfr = load_sfr_only(os.path.join(path2, 'TL2009.sfr'))
    # convert sfr package to reach input
    sfr.reachinput = True
    sfr.isfropt = 1
    sfr.reach_data = interpolate_to_reaches(sfr)
    sfr.get_slopes(minimum_slope=-100, maximum_slope=100)
    reach_inds = 29
    outreach = sfr.reach_data.outreach[reach_inds]
    out_inds = np.where(sfr.reach_data.reachID == outreach)
    assert sfr.reach_data.slope[reach_inds] == (sfr.reach_data.strtop[reach_inds] -
                                        sfr.reach_data.strtop[out_inds]) \
                                       / sfr.reach_data.rchlen[reach_inds]
    chk = sfr.check()
    assert sfr.reach_data.slope.min() < 0.0001 and 'minimum slope' in chk.warnings
    sfr.reach_data.slope[0] = 1.1
    chk.slope(maximum_slope=1.0)
    assert 'maximum slope' in chk.warnings


def test_sfr_renumbering():
    # test segment renumbering

    dtype = np.dtype([('iseg', int), ('ireach', int)])
    r = create_empty_recarray(27, dtype)
    r['iseg'] = sorted(list(range(1, 10)) * 3)
    r['ireach'] = [1, 2, 3] * 9

    dtype = np.dtype([('nseg', int), ('outseg', int)])
    d = create_empty_recarray(9, dtype)
    d['nseg'] = range(1, 10)
    d['outseg'] = [4, 0, 6, 8, 3, 8, 1, 2, 8]
    m = flopy.modflow.Modflow()
    sfr = flopy.modflow.ModflowSfr2(m, reach_data=r, segment_data={0: d})
    chk = sfr.check()
    assert 'segment numbering order' in chk.warnings
    sfr.renumber_segments()
    chk = sfr.check()
    assert 'continuity in segment and reach numbering' in chk.passed
    assert 'segment numbering order' in chk.passed

    # test renumbering non-consecutive segment numbers
    r['iseg'] *= 2
    r['ireach'] = [1, 2, 3] * 9

    dtype = np.dtype([('nseg', int), ('outseg', int)])
    d = create_empty_recarray(9, dtype)
    d['nseg'] = np.arange(1, 10) * 2
    d['outseg'] = np.array([4, 0, 6, 8, 3, 8, 1, 2, 8]) * 2
    m = flopy.modflow.Modflow()
    sfr = flopy.modflow.ModflowSfr2(m, reach_data=r, segment_data={0: d})
    chk = sfr.check()
    assert 'segment numbering order' in chk.warnings
    sfr.renumber_segments()
    chk = sfr.check()
    assert 'continuity in segment and reach numbering' in chk.passed
    assert 'segment numbering order' in chk.passed

    # test computing of outreaches
    assert np.array_equal(sfr.reach_data.outreach,
                          np.array([2, 3, 7,
                                    5, 6, 10,
                                    8, 9, 16,
                                    11, 12, 19,
                                    14, 15, 22,
                                    17, 18, 22,
                                    20, 21, 22,
                                    23, 24, 25,
                                    26, 27, 0]))
    # test slope
    sfr.reach_data['rchlen'] = [10] * 3 * 5 + [100] * 2 * 3 + [1] * 2 * 3
    strtop = np.zeros(len(sfr.reach_data))
    strtop[2] = .3
    strtop[21] = -.2
    strtop[22] = -.4
    sfr.reach_data['strtop'] = strtop
    default_slope = .0001
    sfr.get_slopes(default_slope=default_slope)
    sl1 = sfr.reach_data.slope[2]
    def isequal(v1, v2):
        return np.abs(v1-v2) < 1e-6
    assert isequal(sfr.reach_data.slope[2], 0.03)
    assert isequal(sfr.reach_data.slope[14], 0.02)
    assert isequal(sfr.reach_data.slope[20], sfr.reach_data.slope[17])
    assert isequal(sfr.reach_data.slope[21], 0.2)
    assert isequal(sfr.reach_data.slope[-1], default_slope)

def test_const():

    fm = flopy.modflow
    m = fm.Modflow()
    dis = fm.ModflowDis(m, 1, 10, 10, lenuni=2, itmuni=4)
    m.sr = SpatialReference()
    r, d = create_sfr_data()
    sfr = flopy.modflow.ModflowSfr2(m, reach_data=r, segment_data={0: d})
    assert sfr.const == 86400.
    m.dis.itmuni = 1.
    m.sfr.const = None
    assert sfr.const == 1.
    m.dis.lenuni = 1.
    m.sfr.const = None
    assert sfr.const == 1.486
    m.dis.itmuni = 4.
    m.sfr.const = None
    assert sfr.const == 1.486 * 86400.
    assert True

def test_export():
    fm = flopy.modflow
    m = fm.Modflow()
    dis = fm.ModflowDis(m, 1, 10, 10, lenuni=2, itmuni=4)
    m.sr = SpatialReference(delr=m.dis.delr.array, delc=m.dis.delc.array)
    m.sr.write_shapefile(os.path.join(outpath, 'grid.shp'))
    r, d = create_sfr_data()
    sfr = flopy.modflow.ModflowSfr2(m, reach_data=r, segment_data={0: d})
    sfr.segment_data[0]['flow'][-1] = 1e4
    sfr.stress_period_data.export(os.path.join(outpath, 'sfr.shp'), sparse=True)
    sfr.export_linkages(os.path.join(outpath, 'linkages.shp'))
    sfr.export_outlets(os.path.join(outpath, 'outlets.shp'))
    sfr.export_transient_variable(os.path.join(outpath, 'inlets.shp'),
                                  'flow')

    from flopy.export.shapefile_utils import shp2recarray
    ra = shp2recarray(os.path.join(outpath, 'inlets.shp'))
    assert ra.flow0[0] == 1e4
    ra = shp2recarray(os.path.join(outpath, 'outlets.shp'))
    assert ra.iseg[0] + ra.ireach[0] == 5
    ra = shp2recarray(os.path.join(outpath, 'linkages.shp'))
    crds = np.array(list(ra.geometry[2].coords))
    assert np.array_equal(crds, np.array([[2.5,  4.5], [3.5,  5.5]]))
    ra = shp2recarray(os.path.join(outpath, 'sfr.shp'))
    assert ra.iseg0.sum() == sfr.reach_data.iseg.sum()
    assert ra.ireach0.sum() == sfr.reach_data.ireach.sum()
    y = np.concatenate([np.array(g.exterior)[:, 1] for g in ra.geometry])
    x = np.concatenate([np.array(g.exterior)[:, 0] for g in ra.geometry])
    assert (x.min(), y.min(), x.max(), y.max()) == m.sr.bounds
    assert ra[(ra.iseg0 == 2) & (ra.ireach0 == 1)]['geometry'][0].bounds \
        == (6.0, 2.0, 7.0, 3.0)

def test_example():
    m = flopy.modflow.Modflow.load('test1ss.nam', version='mf2005',
                                   exe_name='mf2005.exe',
                                   model_ws=path,
                                   load_only=['ghb', 'evt', 'rch', 'dis',
                                              'bas6', 'oc', 'sip', 'lpf'])
    reach_data = np.genfromtxt(
        '../examples/data/sfr_examples/test1ss_reach_data.csv', delimiter=',',
        names=True)
    segment_data = np.genfromtxt(
        '../examples/data/sfr_examples/test1ss_segment_data.csv',
        delimiter=',', names=True)
    # segment_data = {0: ss_segment_data}

    channel_flow_data = {
        0: {1: [[0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0],
                [0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.1, 1.25, 1.4, 1.7, 2.6],
                [3.0, 3.5, 4.2, 5.3, 7.0, 8.5, 12.0, 14.0, 17.0, 20.0, 22.0]]}}
    channel_geometry_data = {
        0: {7: [[0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
                [20.0, 13.0, 10.0, 2.0, 0.0, 10.0, 13.0, 20.0]],
            8: [[0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
                [25.0, 17.0, 13.0, 4.0, 0.0, 10.0, 16.0, 20.0]]}}

    nstrm = len(reach_data)  # number of reaches
    nss = len(segment_data)  # number of segments
    nsfrpar = 0  # number of parameters (not supported)
    nparseg = 0
    const = 1.486  # constant for manning's equation, units of cfs
    dleak = 0.0001  # closure tolerance for stream stage computation
    ipakcb = 53  # flag for writing SFR output to cell-by-cell budget (on unit 53)
    istcb2 = 81  # flag for writing SFR output to text file
    dataset_5 = {0: [nss, 0, 0]}  # dataset 5 (see online guide)

    sfr = flopy.modflow.ModflowSfr2(m, nstrm=nstrm, nss=nss, const=const,
                                    dleak=dleak, ipakcb=ipakcb, istcb2=istcb2,
                                    reach_data=reach_data,
                                    segment_data=segment_data,
                                    channel_geometry_data=channel_geometry_data,
                                    channel_flow_data=channel_flow_data,
                                    dataset_5=dataset_5)

    #assert istcb2 in m.package_units
    assert istcb2 in m.output_units
    assert True

    # test handling of a 0-D array (produced by genfromtxt sometimes)
    segment_data = np.array(segment_data[0])
    reach_data = reach_data[reach_data['iseg'] == 1]
    nss = 1
    sfr = flopy.modflow.ModflowSfr2(m, nstrm=nstrm, nss=nss, const=const,
                                    dleak=dleak, ipakcb=ipakcb, istcb2=istcb2,
                                    reach_data=reach_data,
                                    segment_data=segment_data,
                                    channel_geometry_data=channel_geometry_data,
                                    channel_flow_data=channel_flow_data,
                                    dataset_5=dataset_5)

    # test default construction of dataset_5
    sfr2 = flopy.modflow.ModflowSfr2(m, nstrm=nstrm, nss=nss, const=const,
                                    dleak=dleak, ipakcb=ipakcb, istcb2=istcb2,
                                    reach_data=reach_data,
                                    segment_data=segment_data,
                                    channel_geometry_data=channel_geometry_data,
                                    channel_flow_data=channel_flow_data)
    assert len(sfr2.dataset_5) == 1
    assert sfr2.dataset_5[0][0] == sfr2.nss
    nper = 9
    m.dis.nper = nper
    assert len(sfr2.dataset_5) == nper
    for i in range(1, nper):
        assert sfr2.dataset_5[i][0] == -1

def test_transient_example():
    path = os.path.join('temp', 't009')
    gpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'testsfr2.*')
    for f in glob.glob(gpth):
        shutil.copy(f, path)
    mf = flopy.modflow
    m = mf.Modflow.load('testsfr2.nam', model_ws=path)

    # test handling of unformatted output file
    m.sfr.istcb2 = -49
    m.set_output_attribute(unit=abs(m.sfr.istcb2), attr={'binflag':True})
    m.write_input()
    m2 = mf.Modflow.load('testsfr2.nam', model_ws=path)
    assert m2.sfr.istcb2 == -49
    assert m2.get_output_attribute(unit=abs(m2.sfr.istcb2), attr='binflag')

def test_assign_layers():
    m = fm.Modflow()
    m.dis = fm.ModflowDis(nrow=1, ncol=6, nlay=7,
                            botm=np.array([[   50.,    49.,    42.,    27.,     6.,   -33.],
                                      [ -196.,  -246.,  -297.,  -351.,  -405.,  -462.],
                                      [ -817.,  -881.,  -951., -1032., -1141., -1278.],
                                      [-1305., -1387., -1466., -1546., -1629., -1720.],
                                      [-2882., -2965., -3032., -3121., -3226., -3341.],
                                      [-3273., -3368., -3451., -3528., -3598., -3670.],
                                      [-3962., -4080., -4188., -4292., -4392., -4496.]]),
                          model=m)
    reach_data = fm.ModflowSfr2.get_empty_reach_data(5)
    seg_data = {0: fm.ModflowSfr2.get_empty_segment_data(1)}
    seg_data[0]['outseg'] = 0
    reach_data['k'] = 0
    reach_data['i'] = 0
    reach_data['j'] = np.arange(5)
    reach_data['strtop'] = np.array([20, -250, 0., -3000., -4500.])
    reach_data['strthick'] = 1.
    sfr = fm.ModflowSfr2(reach_data=reach_data,
                         segment_data=seg_data,
                         model=m)
    sfr.assign_layers()
    assert np.array_equal(sfr.reach_data.k, np.array([1, 2, 1, 4, 6]))

    l = m.dis.get_layer(0, 0, 0.)
    assert l == 1
    l = m.dis.get_layer(0, [0, 1], 0.)
    assert np.array_equal(l, np.array([1, 1]))

    
def test_SfrFile():
    sfrout = SfrFile('../examples/data/sfr_examples/sfroutput2.txt')
    # will be None if pandas is not installed
    if sfrout.pd is not None:
        df = sfrout.get_dataframe()
        assert df.layer.values[0] == 1
        assert df.column.values[0] == 169
        assert df.Cond.values[0] == 74510.0
        assert df.col18.values[3] == 1.288E+03

    sfrout = SfrFile('../examples/data/sfr_examples/test1tr.flw')
    if sfrout.pd is not None:
        df = sfrout.get_dataframe()
        assert df.col16.values[-1] == 5.502E-02
        assert df.shape == (1080, 20)


def test_sfr_plot():
    #m = flopy.modflow.Modflow.load('test1ss.nam', model_ws=path, verbose=False)
    #sfr = m.get_package('SFR')
    #sfr.plot(key='strtop')
    #plt.show()
    #assert True
    pass

if __name__ == '__main__':
    test_sfr()
    test_sfr_renumbering()
    test_example()
    test_export()
    test_transient_example()
    test_sfr_plot()
    test_assign_layers()
    test_SfrFile()
    test_const()
    pass
