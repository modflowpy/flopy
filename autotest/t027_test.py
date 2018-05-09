"""
test MNW1 and MNW2 packages
"""
import sys
sys.path.insert(0, '..')
import shutil
import os
import flopy
import numpy as np
from flopy.utils.flopy_io import line_parse

cpth = os.path.join('temp', 't027')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)
mf2005pth = os.path.join('..', 'examples', 'data', 'mnw2_examples')
mnw1_path = os.path.join('..', 'examples', 'data', 'mf2005_test')

def test_line_parse():
    """t027 test line_parse method in MNW2 Package class"""
    # ensure that line_parse is working correctly
    # comment handling
    line = line_parse('Well-A  -1                   ; 2a. WELLID,NNODES')
    assert line == ['Well-A', '-1']


def test_load():
    """t027 test load of MNW2 Package"""
    # load in the test problem (1 well, 3 stress periods)
    m = flopy.modflow.Modflow.load('MNW2-Fig28.nam', model_ws=mf2005pth,
                                   verbose=True, forgive=False)
    m.change_model_ws(cpth)
    assert m.has_package('MNW2')
    assert m.has_package('MNWI')

    # load a real mnw2 package from a steady state model (multiple wells)
    m2 = flopy.modflow.Modflow('br', model_ws=cpth)
    path = os.path.join('..', 'examples', 'data', 'mnw2_examples')
    mnw2_2 = flopy.modflow.ModflowMnw2.load(path + '/BadRiver_cal.mnw2', m2)
    mnw2_2.write_file(os.path.join(cpth, 'brtest.mnw2'))

    m3 = flopy.modflow.Modflow('br', model_ws=cpth)
    mnw2_3 = flopy.modflow.ModflowMnw2.load(cpth + '/brtest.mnw2', m3)
    mnw2_2.node_data.sort(order='wellid')
    mnw2_3.node_data.sort(order='wellid')
    assert np.array_equal(mnw2_2.node_data, mnw2_3.node_data)
    assert (mnw2_2.stress_period_data[0].qdes - mnw2_3.stress_period_data[
        0].qdes).max() < 0.01
    assert np.abs(
        mnw2_2.stress_period_data[0].qdes - mnw2_3.stress_period_data[
            0].qdes).min() < 0.01

def test_mnw1_load_write():
    m = flopy.modflow.Modflow.load('mnw1.nam', model_ws=mnw1_path,
                                   load_only=['mnw1'],
                                   verbose=True, forgive=False)
    assert m.has_package('MNW1')
    assert m.mnw1.mxmnw == 120
    for i in range(3):
        assert len(m.mnw1.stress_period_data[i]) == 17
        assert len(np.unique(m.mnw1.stress_period_data[i]['mnw_no'])) == 15
        assert len(set(m.mnw1.stress_period_data[i]['label'])) == 4
    shutil.copy(mnw1_path + '/mnw1.nam', cpth)
    shutil.copy(mnw1_path + '/mnw1.dis', cpth)
    shutil.copy(mnw1_path + '/mnw1.bas', cpth)
    m.mnw1.fn_path = cpth + '/mnw1.mnw'
    m.mnw1.write_file()
    m2 = flopy.modflow.Modflow.load('mnw1.nam', model_ws=cpth,
                                   load_only=['mnw1'],
                                   verbose=True, forgive=False)
    assert m.stress_period_data == m2.stress_period_data

def test_make_package():
    """t027 test make MNW2 Package"""
    m4 = flopy.modflow.Modflow('mnw2example', model_ws=cpth)
    dis = flopy.modflow.ModflowDis(nrow=5, ncol=5, nlay=3, nper=3, top=10,
                                   botm=0, model=m4)

    # make the package from the tables (ztop, zbotm format)
    node_data = np.array(
        [(0, 1, 1, 9.5, 7.1, 'well1', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
         (1, 1, 1, 7.1, 5.1, 'well1', 'skin', -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
         (
         2, 3, 3, 9.1, 3.7, 'well2', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 4.1)],
        dtype=[('index', '<i8'), ('i', '<i8'), ('j', '<i8'),
               ('ztop', '<f8'), ('zbotm', '<f8'),
               ('wellid', 'O'), ('losstype', 'O'), ('pumploc', '<i8'),
               ('qlimit', '<i8'), ('ppflag', '<i8'), ('pumpcap', '<i8'),
               ('rw', '<f8'), ('rskin', '<f8'), ('kskin', '<f8'),
               ('zpump', '<f8')]).view(np.recarray)

    stress_period_data = {0: np.array([(0, 0, 'well1', 0), (1, 0, 'well2', 0)],
                                      dtype=[('index', '<i8'), ('per', '<i8'),
                                             ('wellid', 'O'),
                                             ('qdes', '<i8')]).view(
        np.recarray),
                          1: np.array(
                              [(2, 1, 'well1', 100), (3, 1, 'well2', 1000)],
                              dtype=[('index', '<i8'), ('per', '<i8'),
                                     ('wellid', 'O'), ('qdes', '<i8')]).view(
                              np.recarray)}

    mnw2_4 = flopy.modflow.ModflowMnw2(model=m4, mnwmax=2, nodtot=3,
                                       node_data=node_data,
                                       stress_period_data=stress_period_data,
                                       itmp=[2, 2, -1],
                                       # reuse second per pumping for last stress period
                                       )
    m4.write_input()

    # make the package from the tables (k, i, j format)
    node_data = np.array(
        [(0, 3, 1, 1, 'well1', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
         (1, 2, 1, 1, 'well1', 'skin', -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
         (2, 1, 3, 3, 'well2', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 4.1)],
        dtype=[('index', '<i8'), ('k', '<i8'), ('i', '<i8'), ('j', '<i8'),
               ('wellid', 'O'), ('losstype', 'O'), ('pumploc', '<i8'),
               ('qlimit', '<i8'), ('ppflag', '<i8'), ('pumpcap', '<i8'),
               ('rw', '<f8'), ('rskin', '<f8'), ('kskin', '<f8'),
               ('zpump', '<f8')]).view(np.recarray)

    stress_period_data = {0: np.array([(0, 0, 'well1', 0), (1, 0, 'well2', 0)],
                                      dtype=[('index', '<i8'), ('per', '<i8'),
                                             ('wellid', 'O'),
                                             ('qdes', '<i8')]).view(
        np.recarray),
                          1: np.array(
                              [(2, 1, 'well1', 100), (3, 1, 'well2', 1000)],
                              dtype=[('index', '<i8'), ('per', '<i8'),
                                     ('wellid', 'O'), ('qdes', '<i8')]).view(
                              np.recarray)}

    mnw2_4 = flopy.modflow.ModflowMnw2(model=m4, mnwmax=2, nodtot=3,
                                       node_data=node_data,
                                       stress_period_data=stress_period_data,
                                       itmp=[2, 2, -1],
                                       # reuse second per pumping for last stress period
                                       )
    spd = m4.mnw2.stress_period_data[0]
    inds = spd.k, spd.i, spd.j
    assert np.array_equal(np.array(inds).transpose(),
                          np.array([(2, 1, 1), (1, 3, 3)]))
    m4.write_input()

    # make the package from the objects
    mnw2fromobj = flopy.modflow.ModflowMnw2(model=m4, mnwmax=2,
                                            mnw=mnw2_4.mnw,
                                            itmp=[2, 2, -1],
                                            # reuse second per pumping for last stress period
                                            )
    # verify that the two input methods produce the same results
    assert np.array_equal(mnw2_4.stress_period_data[1],
                          mnw2fromobj.stress_period_data[1])


def test_export():
    """t027 test export of MNW2 Package to netcdf files"""
    try:
        import netCDF4
    except:
        netCDF4 = None
    m = flopy.modflow.Modflow.load('MNW2-Fig28.nam', model_ws=mf2005pth,
                                   load_only=['dis', 'bas6', 'mnwi', 'mnw2',
                                              'wel'], verbose=True,
                                   check=False)

    # netDF4 tests
    if netCDF4 is not None:
        m.wel.export(os.path.join(cpth, 'MNW2-Fig28_well.nc'))
        m.mnw2.export(os.path.join(cpth, 'MNW2-Fig28.nc'))
        fpth = os.path.join(cpth, 'MNW2-Fig28.nc')
        nc = netCDF4.Dataset(fpth)
        assert np.array_equal(nc.variables['mnw2_qdes'][:, 0, 29, 40],
                              np.array([0., -10000., -10000.], dtype='float32'))
        assert np.sum(nc.variables['mnw2_rw'][:, :, 29, 40]) - 5.1987 < 1e-4

    # need to add shapefile test


def test_checks():
    """t027 test MNW2 Package checks in FloPy"""
    m = flopy.modflow.Modflow.load('MNW2-Fig28.nam', model_ws=mf2005pth,
                                   load_only=['dis', 'bas6', 'mnwi', 'wel'],
                                   verbose=True, check=False)
    chk = m.check()
    assert 'MNWI package present without MNW2 package.' in '.'.join(
        chk.summary_array.desc)


if __name__ == '__main__':
    #test_line_parse()
    #test_load()
    #test_make_package()
    #test_export()
    #test_checks()
    test_mnw1_load_write()
    pass