"""
test modpath functionality
"""
import sys
sys.path.insert(0, '/Users/aleaf/Documents/GitHub/flopy3')
import glob
import shutil
import os
import flopy
import numpy as np
from flopy.utils.reference import SpatialReference
from flopy.utils.modpathfile import EndpointFile, PathlineFile
from flopy.modpath.mpsim import StartingLocationsFile

mffiles = glob.glob('../examples/data/mp6/EXAMPLE*')
path = os.path.join('temp', 'mp6')

if not os.path.isdir(path):
    os.makedirs(path)
for f in mffiles:
    shutil.copy(f, os.path.join(path, os.path.split(f)[1]))

def test_mpsim():

    model_ws = path
    m = flopy.modflow.Modflow.load('EXAMPLE.nam', model_ws=model_ws)
    m.get_package_list()

    mp = flopy.modpath.Modpath(modelname='ex6',
                           exe_name='mp6',
                           modflowmodel=m,
                           model_ws=path,
                           dis_file=m.name+'.dis',
                           head_file=m.name+'.hed',
                           budget_file=m.name+'.bud')

    mpb = flopy.modpath.ModpathBas(mp, hdry=m.lpf.hdry, laytyp=m.lpf.laytyp, ibound=1, prsity=0.1)

    sim = mp.create_mpsim(trackdir='forward', simtype='endpoint', packages='RCH')
    mp.write_input()

    # replace the well with an mnw
    node_data = np.array([(3, 12, 12, 'well1', 'skin', -1, 0, 0, 0, 1., 2., 5., 6.2),
                      (4, 12, 12, 'well1', 'skin', -1, 0, 0, 0, 0.5, 2., 5., 6.2)],
                     dtype=[('k', np.int), ('i', np.int), ('j', np.int),
                            ('wellid', np.object), ('losstype', np.object),
                            ('pumploc', np.int), ('qlimit', np.int),
                            ('ppflag', np.int), ('pumpcap', np.int),
                            ('rw', np.float), ('rskin', np.float),
                            ('kskin', np.float), ('zpump', np.float)]).view(np.recarray)

    stress_period_data = {0: np.array([(0, 'well1', -150000.0)],
                                      dtype=[('per', np.int),
                                             ('wellid', np.object),
                                             ('qdes', np.float)])}
    m.remove_package('WEL')
    mnw2 = flopy.modflow.ModflowMnw2(model=m, mnwmax=1,
                                     node_data=node_data,
                                     stress_period_data=stress_period_data,
                                     itmp=[1, -1, -1])
    # test creation of modpath simulation file for MNW2
    # (not a very robust test)
    sim = mp.create_mpsim(trackdir='backward', simtype='pathline', packages='MNW2')
    mp.write_input()

    sim = flopy.modpath.ModpathSim(model=mp)
    # starting locations file
    stl = StartingLocationsFile(model=mp)
    stldata = StartingLocationsFile.get_empty_starting_locations_data(npt=2)
    stldata['label'] = ['p1', 'p2']
    stldata[1]['i0'] = 5
    stldata[1]['j0'] = 6
    stldata[1]['xloc0'] = .1
    stldata[1]['yloc0'] = .2
    stl.data = stldata
    mp.write_input()
    stllines = open(os.path.join(path, 'ex6.loc')).readlines()
    assert stllines[3].strip() == 'group1'
    assert int(stllines[4].strip()) == 2
    assert stllines[6].strip().split()[-1] == 'p2'

def test_get_destination_data():
    m = flopy.modflow.Modflow.load('EXAMPLE.nam', model_ws=path)

    m.sr = SpatialReference(delr=m.dis.delr, delc=m.dis.delc, xul=0, yul=0, rotation=30)
    sr = SpatialReference(delr=list(m.dis.delr), delc=list(m.dis.delc), xul=1000, yul=1000, rotation=30)
    sr2 = SpatialReference(xll=sr.xll, yll=sr.yll, rotation=30)
    m.dis.export(path + '/dis.shp')

    pthld = PathlineFile(os.path.join(path, 'EXAMPLE-3.pathline'))
    epd = EndpointFile(os.path.join(path, 'EXAMPLE-3.endpoint'))

    well_epd = epd.get_destination_endpoint_data(dest_cells=[(4, 12, 12)])
    well_pthld = pthld.get_destination_pathline_data(dest_cells=[(4, 12, 12)])

    # same particle IDs should be in both endpoing data and pathline data
    assert len(set(well_epd.particleid).difference(set(well_pthld.particleid))) == 0

    # check that all starting locations are included in the pathline data
    # (pathline data slice not just endpoints)
    starting_locs = well_epd[['k0', 'i0', 'j0']]
    pathline_locs = np.array(well_pthld[['k', 'i', 'j']].tolist(), dtype=starting_locs.dtype)
    assert np.all(np.in1d(starting_locs, pathline_locs))

    # test writing a shapefile of endpoints
    epd.write_shapefile(well_epd, direction='starting',
                        shpname=os.path.join(path, 'starting_locs.shp'),
                        sr=m.sr)

    # test writing shapefile of pathlines
    pthld.write_shapefile(well_pthld, one_per_particle=True,
                          direction='starting', sr=m.sr,
                          shpname='temp/mp6/pathlines_1per.shp')
    pthld.write_shapefile(well_pthld, one_per_particle=True,
                          direction='ending', sr=m.sr,
                          shpname='temp/mp6/pathlines_1per_end.shp')
    # test writing shapefile of pathlines
    pthld.write_shapefile(well_pthld, one_per_particle=True,
                          direction='starting', sr=sr,
                          shpname='temp/mp6/pathlines_1per2.shp')
    # test writing shapefile of pathlines
    pthld.write_shapefile(well_pthld, one_per_particle=True,
                          direction='starting', sr=sr2,
                          shpname='temp/mp6/pathlines_1per2_ll.shp')
    pthld.write_shapefile(well_pthld, one_per_particle=False,
                          sr=m.sr,
                          shpname='temp/mp6/pathlines.shp')

    # test that endpoints were rotated and written correctly
    from flopy.export.shapefile_utils import shp2recarray
    ra = shp2recarray(os.path.join(path, 'starting_locs.shp'))
    p3 = ra.geometry[ra.particleid == 4][0]
    xorig, yorig = m.sr.transform(well_epd.x0[0], well_epd.y0[0])
    assert p3.x - xorig + p3.y - yorig < 1e-4
    assert np.abs(p3.x - 858.845726812 + p3.y + 2112.4355653) < 1e-4 # this also checks for 1-based

    # test that particle attribute information is consistent with pathline file
    ra = shp2recarray(os.path.join(path, 'pathlines.shp'))
    inds = (ra.particleid == 8) & (ra.i == 12) & (ra.j == 12)
    assert ra.time[inds][0] - 20181.7 < .1
    assert ra.xloc[inds][0] - 0.933 < .01

    # test that k, i, j are correct for single geometry pathlines, forwards and backwards
    ra = shp2recarray(os.path.join(path, 'pathlines_1per.shp'))
    assert ra.i[0] == 4, ra.j[0] == 5
    ra = shp2recarray(os.path.join(path, 'pathlines_1per_end.shp'))
    assert ra.i[0] == 13, ra.j[0] == 13

    # test use of arbitrary spatial reference and offset
    ra = shp2recarray(os.path.join(path, 'pathlines_1per2.shp'))
    p3_2 = ra.geometry[ra.particleid == 4][0]
    assert np.abs(p3_2.x[0] - 1858.845726812 + p3_2.y[0] + 1112.4355653) < 1e-4

    # arbitrary spatial reference with ll specified instead of ul
    ra = shp2recarray(os.path.join(path, 'pathlines_1per2_ll.shp'))
    p3_2 = ra.geometry[ra.particleid == 4][0]
    assert np.abs(p3_2.x[0] - 1858.845726812 + p3_2.y[0] + 1112.4355653) < 1e-4

    xul = 3628793
    yul = 21940389

    m = flopy.modflow.Modflow.load('EXAMPLE.nam', model_ws=path)

    m.sr = flopy.utils.reference.SpatialReference(delr=m.dis.delr, delc=m.dis.delc, lenuni=1,
                                                                     xul=xul, yul=yul, rotation=0.0)
    m.dis.export(path + '/dis2.shp')
    pthobj = flopy.utils.PathlineFile(os.path.join(path, 'EXAMPLE-3.pathline'))
    pthobj.write_shapefile(shpname='temp/mp6/pathlines_1per3.shp',
                           direction='ending',
                           sr=m.sr)

if __name__ == '__main__':

    test_mpsim()
    test_get_destination_data()