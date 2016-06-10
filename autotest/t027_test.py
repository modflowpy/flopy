"""
test MNW2 package
"""
import os
import flopy
import numpy as np

cpth = os.path.join('temp')


def test_load():

    # load in the test problem (1 well, 3 stress periods)
    m = flopy.modflow.Modflow('MNW2-Fig28', model_ws=cpth)
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    dis = flopy.modflow.ModflowDis.load(path + '/MNW2-Fig28.dis', m)
    mnw2 = flopy.modflow.ModflowMnw2.load(path + '/MNW2-Fig28.mnw2', m)

    # load a real mnw2 package from a steady state model (multiple wells)
    m = flopy.modflow.Modflow('br', model_ws=cpth)
    path = os.path.join('..', 'examples', 'data', 'mnw2_examples')
    mnw2 = flopy.modflow.ModflowMnw2.load(path + '/BadRiver_cal.mnw2', m)
    mnw2.write_file(os.path.join(cpth, 'brtest.mnw2'))

    m2 = flopy.modflow.Modflow('br', model_ws=cpth)
    mnw2_2 = flopy.modflow.ModflowMnw2.load(cpth + '/brtest.mnw2', m)

    assert np.array_equal(mnw2.node_data, mnw2_2.node_data)
    assert (mnw2.stress_period_data[0].qdes - mnw2_2.stress_period_data[0].qdes).max() < 0.01
    assert np.abs(mnw2.stress_period_data[0].qdes - mnw2_2.stress_period_data[0].qdes).min() < 0.01


def test_make_package():
    m = flopy.modflow.Modflow('mnw2example', model_ws=cpth)
    dis = flopy.modflow.ModflowDis(nrow=5, ncol=5, nlay=3, nper=3, top=10, botm=0, model=m)

    # make the package from the tables
    node_data = np.array([(0, 1, 1, 9.5, 7.1, 'well1', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
                          (1, 1, 1, 7.1, 5.1, 'well1', 'skin', -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
                          (2, 3, 3, 9.1, 3.7, 'well2', 'skin', -1, 0, 0, 0, 1.0, 2.0, 5.0, 4.1)],
                          dtype=[('index', '<i8'), ('i', '<i8'), ('j', '<i8'),
                                 ('ztop', '<f8'), ('zbotm', '<f8'),
                                 ('wellid', 'O'), ('losstype', 'O'), ('pumploc', '<i8'),
                                 ('qlimit', '<i8'), ('ppflag', '<i8'), ('pumpcap', '<i8'),
                                 ('rw', '<f8'), ('rskin', '<f8'), ('kskin', '<f8'),
                                 ('zpump', '<f8')]).view(np.recarray)

    stress_period_data = {0: np.array([(0, 0, 'well1', 0), (1, 0, 'well2', 0)],
           dtype=[('index', '<i8'), ('per', '<i8'), ('wellid', 'O'), ('qdes', '<i8')]).view(np.recarray),
                          1: np.array([(2, 1, 'well1', 100), (3, 1, 'well2', 1000)],
           dtype=[('index', '<i8'), ('per', '<i8'), ('wellid', 'O'), ('qdes', '<i8')]).view(np.recarray)}
    mnw2 = flopy.modflow.ModflowMnw2(model=m, mnwmax=2, nodtot=3,
                 node_data=node_data,
                 stress_period_data=stress_period_data,
                 itmp=[2, 2, -1], # reuse second per pumping for last stress period
                 )
    '''
    # make the package from the objects
    mnw2fromobj = flopy.modflow.ModflowMnw2(model=m, mnwmax=2,
                 mnw=mnw2.mnw,
                 itmp=[2, 2, -1], # reuse second per pumping for last stress period
                 )
    # verify that they two input methods produce the same results
    assert np.array_equal(mnw2.stress_period_data[1], mnw2fromobj.stress_period_data[1])
    '''
if __name__ == '__main__':
    #test_load()
    test_make_package()