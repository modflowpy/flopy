"""
Test shapefile stuff
"""
import os
import shutil
import numpy as np
import flopy
from flopy.utils.geometry import Polygon
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.export.netcdf import NetCdf
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

    ncdf = NetCdf('toy.model.nc', m)
    ncdf.write()

    m.export('toy_model_two.nc')
    dis.export('toy_model_dis.nc')

    mg = m.modelgrid
    mg.set_coord_info(xoff=mg._xul_to_xll(600000.0, -45.0),
                      yoff=mg._yul_to_yll(5170000, -45.0),
                      angrot=-45.0, proj4='EPSG:26715')

    recarray = np.array([(0, 5, 5, .1, True, 's0'),
                         (1, 4, 5, .2, False, 's1'),
                         (0, 7, 8, .3, True, 's2')],
                        dtype=[('k', '<i8'), ('i', '<i8'), ('j', '<i8'),
                               ('stuff', '<f4'), ('stuf', '|b1'),
                               ('stf', np.object)]).view(np.recarray)

    # vertices for a model cell
    geoms = [Polygon(m.modelgrid.get_cell_vertices(i, j)) for i, j in
             zip(recarray.i, recarray.j)]

    assert geoms[0].type == 'Polygon'
    assert np.abs(geoms[0].bounds[-1] - 5169784.473861726) < 1e-4
    fpth = os.path.join(mpth, 'test.shp')
    recarray2shp(recarray, geoms, fpth, epsg=26715)
    ep = epsgRef()
    prj = ep.to_dict()
    assert 26715 in prj
    fpth = os.path.join(mpth, 'test.prj')
    fpth2 = os.path.join(mpth, '26715.prj')
    shutil.copy(fpth, fpth2)
    fpth = os.path.join(mpth, 'test.shp')
    recarray2shp(recarray, geoms, fpth, prj=fpth2)

    # test_dtypes
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

    getprj(4326)
    prj = ep.to_dict()
    assert 4326 in prj

    ep.add(9999, 'junk')
    prj = ep.to_dict()
    assert 9999 in prj

    ep.remove(9999)
    prj = ep.to_dict()
    assert 9999 not in prj

    ep.reset()
    prj = ep.to_dict()
    assert len(prj) == 0


if __name__ == '__main__':
    #test_polygon_from_ij()
    #test_epsgref()
    pass
