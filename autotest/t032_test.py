"""
Test shapefile stuff
"""
import sys
import os

if sys.version_info[0] == 3:
    if sys.version_info[1] < 4:
        from imp import reload  # python < 3.4
    else:
        from importlib import reload
# python2 (reload in default namespace)
sys.path.insert(0, '..')
import shutil
import numpy as np
import flopy
from flopy.utils.geometry import Polygon
from flopy.utils.reference import SpatialReference
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils.reference import getprj, epsgRef

mpth = os.path.join('temp', 't032')
# make the directory if it does not exist
if not os.path.isdir(mpth):
    os.makedirs(mpth)


def test_polygon_from_ij():
    """test creation of a polygon from an i, j location using get_vertices()."""
    m = flopy.modflow.Modflow('toy_model', model_ws=mpth)
    botm = np.zeros((2, 10, 10))
    botm[0, :, :] = 1.5
    botm[1, 5, 5] = 4  # negative layer thickness!
    botm[1, 6, 6] = 4
    dis = flopy.modflow.ModflowDis(nrow=10, ncol=10,
                                   nlay=2, delr=100, delc=100,
                                   top=3, botm=botm, model=m)

    m.sr = SpatialReference(delr=m.dis.delr * .3048, delc=m.dis.delc * .3048,
                            xul=600000, yul=5170000,
                            proj4_str='EPSG:26715', rotation=-45)

    recarray = np.array([(0, 5, 5, .1, True, 's0'),
                         (1, 4, 5, .2, False, 's1'),
                         (0, 7, 8, .3, True, 's2')],
                        dtype=[('k', '<i8'), ('i', '<i8'), ('j', '<i8'),
                               ('stuff', '<f4'), ('stuf', '|b1'),
                               ('stf', np.object)]).view(np.recarray)

    get_vertices = m.sr.get_vertices  # function to get the referenced vertices for a model cell
    geoms = [Polygon(get_vertices(i, j)) for i, j in
             zip(recarray.i, recarray.j)]

    assert geoms[0].type == 'Polygon'
    assert np.abs(geoms[0].bounds[-1] - 5169784.473861726) < 1e-6
    fpth = os.path.join(mpth, 'test.shp')
    recarray2shp(recarray, geoms, fpth, epsg=26715)
    import epsgref
    reload(epsgref)
    from epsgref import prj
    assert 26715 in prj
    fpth = os.path.join(mpth, 'test.prj')
    fpth2 = os.path.join(mpth, '26715.prj')
    shutil.copy(fpth, fpth2)
    fpth = os.path.join(mpth, 'test.shp')
    recarray2shp(recarray, geoms, fpth, prj=fpth2)


def test_dtypes():
    fpth = os.path.join(mpth, 'test.shp')
    ra = shp2recarray(fpth)
    assert "int" in ra.dtype['k'].name
    assert "float" in ra.dtype['stuff'].name
    assert "bool" in ra.dtype['stuf'].name
    assert "object" in ra.dtype['stf'].name
    assert True


def test_epsgref():
    ep = epsgRef()
    ep.reset()

    import epsgref
    getprj(4326)
    reload(epsgref)
    from epsgref import prj
    assert 4326 in prj

    ep.add(9999, 'junk')
    ep._remove_pyc()  # have to do this in python 2, otherwise won't refresh
    reload(epsgref)
    from epsgref import prj
    assert 9999 in prj

    ep.remove(9999)
    ep._remove_pyc()
    reload(epsgref)
    from epsgref import prj
    assert 9999 not in prj

    ep.reset()
    ep._remove_pyc()
    reload(epsgref)
    from epsgref import prj
    assert len(prj) == 0


if __name__ == '__main__':
    test_polygon_from_ij()
    test_dtypes()
    test_epsgref()
