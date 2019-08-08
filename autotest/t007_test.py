# Test export module
import sys
sys.path.append('..')
import copy
import glob
import os
import shutil
import numpy as np
import warnings
import flopy

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]
# skip = ["MNW2-Fig28.nam", "testsfr2.nam", "testsfr2_tab.nam"]
skip = []

tpth = os.path.join('temp', 't007')
# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)

npth = os.path.join('temp', 't007', 'netcdf')
# delete the directory if it exists
if os.path.isdir(npth):
    shutil.rmtree(npth)
# make the directory
os.makedirs(npth)

spth = os.path.join('temp', 't007', 'shapefile')
# make the directory if it does not exist
if not os.path.isdir(spth):
    os.makedirs(spth)

def remove_shp(shpname):
    os.remove(shpname)
    for ext in ['prj', 'shx', 'dbf']:
        fname = shpname.replace('shp', ext)
        if os.path.exists(fname):
            os.remove(fname)


def export_mf6_netcdf(path):
    sim = flopy.mf6.modflow.mfsimulation.MFSimulation.load(sim_ws=path)
    for name, model in sim.get_model_itr():
        export_netcdf(model)


def export_mf2005_netcdf(namfile):
    if namfile in skip:
        return
    print(namfile)
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    if m.dis.lenuni == 0:
        m.dis.lenuni = 1
        # print('skipping...lenuni==0 (undefined)')
        # return
    # if sum(m.dis.laycbd) != 0:
    if m.dis.botm.shape[0] != m.nlay:
        print('skipping...botm.shape[0] != nlay')
        return
    assert m, 'Could not load namefile {}'.format(namfile)
    assert isinstance(m, flopy.modflow.Modflow)
    export_netcdf(m)

def export_netcdf(m):
    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return

    fnc = m.export(os.path.join(npth, m.name + '.nc'))
    fnc.write()
    fnc_name = os.path.join(npth, m.name + '.nc')
    try:
        fnc = m.export(fnc_name)
        fnc.write()
    except Exception as e:
        raise Exception(
            'ncdf export fail for namfile {0}:\n{1}  '.format(namfile, str(e)))
    try:
        nc = netCDF4.Dataset(fnc_name, 'r')
    except Exception as e:
        raise Exception('ncdf import fail for nc file {0}'.format(fnc_name))
    return


def export_shapefile(namfile):
    try:
        import shapefile as shp
    except:
        return

    print(namfile)
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)

    assert m, 'Could not load namefile {}'.format(namfile)
    assert isinstance(m, flopy.modflow.Modflow)
    fnc_name = os.path.join(spth, m.name + '.shp')
    try:
        fnc = m.export(fnc_name)
        #fnc2 = m.export(fnc_name, package_names=None)
        #fnc3 = m.export(fnc_name, package_names=['DIS'])


    except Exception as e:
        raise Exception(
            'shapefile export fail for namfile {0}:\n{1}  '.format(namfile,
                                                                   str(e)))
    try:
        s = shp.Reader(fnc_name)
    except Exception as e:
        raise Exception(
            ' shapefile import fail for {0}:{1}'.format(fnc_name, str(e)))
    assert s.numRecords == m.nrow * m.ncol, "wrong number of records in " + \
                                            "shapefile {0}:{1:d}".format(
                                                fnc_name, s.numRecords)
    return

def test_freyberg_export():
    from flopy.discretization import StructuredGrid
    namfile = 'freyberg.nam'

    # steady state
    model_ws = '../examples/data/freyberg'
    m = flopy.modflow.Modflow.load(namfile, model_ws=model_ws,
                                   check=False, verbose=False)
    # test export at model, package and object levels
    m.export('{}/model.shp'.format(spth))
    m.wel.export('{}/wel.shp'.format(spth))
    m.lpf.hk.export('{}/hk.shp'.format(spth))
    m.riv.stress_period_data.export('{}/riv_spd.shp'.format(spth))

    # transient
    # (doesn't work at model level because the total size of
    #  the attribute fields exceeds the shapefile limit)
    model_ws = '../examples/data/freyberg_multilayer_transient/'
    m = flopy.modflow.Modflow.load(namfile, model_ws=model_ws, verbose=False,
                                   load_only=['DIS', 'BAS6', 'NWT', 'OC',
                                              'RCH',
                                              'WEL',
                                              'DRN',
                                              'UPW'])
    # test export without instantiating an sr
    outshp = os.path.join(spth, namfile[:-4] + '_drn_sparse.shp')
    m.drn.stress_period_data.export(outshp, sparse=True)
    assert os.path.exists(outshp)
    remove_shp(outshp)
    m.modelgrid = StructuredGrid(delc=m.dis.delc.array,
                                 delr=m.dis.delr.array,
                                 epsg=5070)
    # test export with an sr, regardless of whether or not wkt was found
    m.drn.stress_period_data.export(outshp, sparse=True)
    assert os.path.exists(outshp)
    remove_shp(outshp)
    m.modelgrid = StructuredGrid(delc=m.dis.delc.array,
                                 delr=m.dis.delr.array,
                                 epsg=3070)
    # verify that attributes have same sr as parent
    assert m.drn.stress_period_data.mg.epsg == m.modelgrid.epsg
    assert m.drn.stress_period_data.mg.proj4 == m.modelgrid.proj4
    assert m.drn.stress_period_data.mg.xoffset == m.modelgrid.xoffset
    assert m.drn.stress_period_data.mg.yoffset == m.modelgrid.yoffset
    assert m.drn.stress_period_data.mg.angrot == m.modelgrid.angrot

    # if wkt text was fetched from spatialreference.org
    if m.sr.wkt is not None:
        # test default package export
        outshp = os.path.join(spth, namfile[:-4]+'_dis.shp')
        m.dis.export(outshp)
        prjfile = outshp.replace('.shp', '.prj')
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == m.sr.wkt
        remove_shp(outshp)

        # test default package export to higher level dir
        outshp = os.path.join(spth, namfile[:-4] + '_dis.shp')
        m.dis.export(outshp)
        prjfile = outshp.replace('.shp', '.prj')
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == m.sr.wkt
        remove_shp(outshp)

        # test sparse package export
        outshp = os.path.join(spth, namfile[:-4]+'_drn_sparse.shp')
        m.drn.stress_period_data.export(outshp,
                                        sparse=True)
        prjfile = outshp.replace('.shp', '.prj')
        assert os.path.exists(prjfile)
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == m.sr.wkt
        remove_shp(outshp)

def test_export_output():
    import os
    import numpy as np
    import flopy

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return

    model_ws = os.path.join("..", "examples", "data", "freyberg")
    ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws)
    hds_pth = os.path.join(model_ws, "freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)

    out_pth = os.path.join(npth, "freyberg.out.nc")
    nc = flopy.export.utils.output_helper(out_pth, ml,
                                          {"freyberg.githds": hds})
    var = nc.nc.variables.get("head")
    arr = var[:]
    ibound_mask = ml.bas6.ibound.array == 0
    arr_mask = arr.mask[0]
    assert np.array_equal(ibound_mask, arr_mask)

def test_write_shapefile():
    from flopy.discretization import StructuredGrid
    from flopy.export.shapefile_utils import shp2recarray
    from flopy.export.shapefile_utils import write_grid_shapefile, write_grid_shapefile2

    sg = StructuredGrid(delr=np.ones(10) *1.1,  # cell spacing along model rows
                        delc=np.ones(10) *1.1,  # cell spacing along model columns
                        epsg=26715)
    outshp1 = os.path.join(tpth, 'junk.shp')
    outshp2 = os.path.join(tpth, 'junk2.shp')
    write_grid_shapefile(outshp1, sg, array_dict={})
    write_grid_shapefile2(outshp2, sg, array_dict={})

    # test that vertices aren't getting altered by writing shapefile
    for outshp in [outshp1, outshp2]:
        # check that pyshp reads integers
        # this only check that row/column were recorded as "N"
        # not how they will be cast by python or numpy
        import shapefile as sf
        sfobj = sf.Reader(outshp)
        for f in sfobj.fields:
            if f[0] == 'row' or f[0] == 'column':
                assert f[1] == 'N'
        recs = list(sfobj.records())
        for r in recs[0]:
            assert isinstance(r, int)

        # check that row and column appear as integers in recarray
        ra = shp2recarray(outshp)
        assert np.issubdtype(ra.dtype['row'], np.integer)
        assert np.issubdtype(ra.dtype['column'], np.integer)

        try: # check that fiona reads integers
            import fiona
            with fiona.open(outshp) as src:
                meta = src.meta
                assert 'int' in meta['schema']['properties']['row']
                assert 'int' in meta['schema']['properties']['column']
        except:
            pass


def test_export_array():
    from flopy.export import utils
    try:
        from scipy.ndimage import rotate
    except:
        rotate = False
        pass

    namfile = 'freyberg.nam'
    model_ws = '../examples/data/freyberg_multilayer_transient/'
    m = flopy.modflow.Modflow.load(namfile, model_ws=model_ws, verbose=False,
                                   load_only=['DIS', 'BAS6'])
    m.modelgrid.set_coord_info(angrot=45)
    nodata = -9999
    utils.export_array(m.modelgrid, os.path.join(tpth, 'fb.asc'),
                       m.dis.top.array, nodata=nodata)
    arr = np.loadtxt(os.path.join(tpth, 'fb.asc'), skiprows=6)

    m.modelgrid.write_shapefile(os.path.join(tpth, 'grid.shp'))
    # check bounds
    with open(os.path.join(tpth, 'fb.asc')) as src:
        for line in src:
            if 'xllcorner' in line.lower():
                val = float(line.strip().split()[-1])
                # ascii grid origin will differ if it was unrotated
                if rotate:
                    assert np.abs(val - m.modelgrid.extent[0]) < 1e-6
                else:
                    assert np.abs(val - m.modelgrid.xoffset) < 1e-6
            if 'yllcorner' in line.lower():
                val = float(line.strip().split()[-1])
                if rotate:
                    assert np.abs(val - m.modelgrid.extent[1]) < 1e-6
                else:
                    assert np.abs(val - m.modelgrid.yoffset) < 1e-6
            if 'cellsize' in line.lower():
                val = float(line.strip().split()[-1])
                rot_cellsize = np.cos(np.radians(m.modelgrid.angrot)) * m.modelgrid.delr[0] # * m.sr.length_multiplier
                # assert np.abs(val - rot_cellsize) < 1e-6
                break
    rotate = False
    rasterio = None
    if rotate:
        rotated = rotate(m.dis.top.array, m.modelgrid.angrot, cval=nodata)

    if rotate:
        assert rotated.shape == arr.shape

    try:
        # test GeoTIFF export
        import rasterio
    except:
        pass
    if rasterio is not None:
        utils.export_array(m.modelgrid,
                           os.path.join(tpth, 'fb.tif'),
                           m.dis.top.array,
                           nodata=nodata)
        with rasterio.open(os.path.join(tpth, 'fb.tif')) as src:
            arr = src.read(1)
            assert src.shape == (m.nrow, m.ncol)
            assert np.abs(src.bounds[0] - m.modelgrid.extent[0]) < 1e-6
            assert np.abs(src.bounds[1] - m.modelgrid.extent[1]) < 1e-6


def test_mbase_modelgrid():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(modelname="test", xll=500.0,
                               rotation=12.5, start_datetime="1/1/2016")
    try:
        print(ml.modelgrid.xcentergrid)
    except:
        pass
    else:
        raise Exception("should have failed")

    dis = flopy.modflow.ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.proj4 is None
    ml.model_ws = tpth

    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam", model_ws=ml.model_ws)
    assert str(ml1.modelgrid) == str(ml.modelgrid)
    assert ml1.start_datetime == ml.start_datetime
    assert ml1.modelgrid.proj4 is None


def test_mt_modelgrid():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(modelname="test", xll=500.0,
                               proj4_str='epsg:2193',
                               rotation=12.5, start_datetime="1/1/2016")
    dis = flopy.modflow.ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.epsg == 2193
    assert ml.modelgrid.idomain is None
    ml.model_ws = tpth

    mt = flopy.mt3d.Mt3dms(modelname='test_mt', modflowmodel=ml,
                           model_ws=ml.model_ws, verbose=True)

    assert mt.modelgrid.xoffset == ml.modelgrid.xoffset
    assert mt.modelgrid.yoffset == ml.modelgrid.yoffset
    assert mt.modelgrid.epsg == ml.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)

    # no modflowmodel
    swt = flopy.seawat.Seawat(modelname='test_swt', modflowmodel=None,
                              mt3dmodel=None, model_ws=ml.model_ws, verbose=True)
    assert swt.modelgrid is swt.dis is swt.bas6 is None

    #passing modflowmodel
    swt = flopy.seawat.Seawat(modelname='test_swt', modflowmodel=ml,
                              mt3dmodel=mt, model_ws=ml.model_ws, verbose=True)

    assert \
        swt.modelgrid.xoffset == mt.modelgrid.xoffset == ml.modelgrid.xoffset
    assert \
        swt.modelgrid.yoffset == mt.modelgrid.yoffset == ml.modelgrid.yoffset
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)

    # bas and btn present
    ibound = np.ones(ml.dis.botm.shape)
    ibound[0][0:5] = 0
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound)
    assert ml.modelgrid.idomain is not None

    mt = flopy.mt3d.Mt3dms(modelname='test_mt', modflowmodel=ml,
                           model_ws=ml.model_ws, verbose=True)
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=ml.bas6.ibound.array)

    # reload swt
    swt = flopy.seawat.Seawat(modelname='test_swt', modflowmodel=ml,
                              mt3dmodel=mt, model_ws=ml.model_ws, verbose=True)

    assert \
        ml.modelgrid.xoffset == mt.modelgrid.xoffset == swt.modelgrid.xoffset
    assert \
        mt.modelgrid.yoffset == ml.modelgrid.yoffset == swt.modelgrid.yoffset
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)


def test_free_format_flag():
    import flopy
    Lx = 100.
    Ly = 100.
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = False
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = True
    bas.ifrefm = False
    assert ms.free_format_input == bas.ifrefm
    bas.ifrefm = True
    assert ms.free_format_input == bas.ifrefm

    ms.model_ws = tpth
    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile, model_ws=ms.model_ws)
    assert ms1.free_format_input == ms.free_format_input
    assert ms1.free_format_input == ms1.bas6.ifrefm
    ms1.free_format_input = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = True
    assert ms1.free_format_input == ms1.bas6.ifrefm


def test_mg():
    import flopy
    from flopy.utils import geometry
    Lx = 100.
    Ly = 100.
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr, delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)

    # test instantiation of an empty basic Structured Grid
    mg = flopy.discretization.StructuredGrid(dis.delc.array, dis.delr.array)

    # test instantiation of Structured grid with offsets
    mg = flopy.discretization.StructuredGrid(dis.delc.array, dis.delr.array,
                                             xoff=1, yoff=1)

    #txt = 'yul does not approximately equal 100 - ' + \
    #      '(xul, yul) = ({}, {})'.format( ms.sr.yul, ms.sr.yul)
    assert abs(ms.modelgrid.extent[-1] - Ly) < 1e-3#, txt
    ms.modelgrid.set_coord_info(xoff=111, yoff=0)
    assert ms.modelgrid.xoffset == 111
    ms.modelgrid.set_coord_info()

    xll, yll = 321., 123.
    angrot = 20
    ms.modelgrid = flopy.discretization.StructuredGrid(delc=ms.dis.delc.array,
                                                       delr=ms.dis.delr.array,
                                                       xoff=xll, yoff=xll,
                                                       angrot=angrot,
                                                       lenuni=2)

    # test that transform for arbitrary coordinates
    # is working in same as transform for model grid
    mg2 = flopy.discretization.StructuredGrid(delc=ms.dis.delc.array,
                                              delr=ms.dis.delr.array,
                                              lenuni=2)
    x = mg2.xcellcenters[0]
    y = mg2.ycellcenters[0]
    mg2.set_coord_info(xoff=xll, yoff=yll, angrot=angrot)
    xt, yt = geometry.transform(x, y, xll, yll, mg2.angrot_radians)

    assert np.sum(xt - ms.modelgrid.xcellcenters[0]) < 1e-3
    assert np.sum(yt - ms.modelgrid.ycellcenters[0]) < 1e-3

    # test inverse transform
    x0, y0 = 9.99, 2.49
    x1, y1 = geometry.transform(x0, y0, xll, yll, angrot)
    x2, y2 = geometry.transform(x1, y1, xll, yll, angrot, inverse=True)
    assert np.abs(x2-x0) < 1e-6
    assert np.abs(y2-y0) < 1e6

    ms.start_datetime = "1-1-2016"
    assert ms.start_datetime == "1-1-2016"
    assert ms.dis.start_datetime == "1-1-2016"

    ms.model_ws = tpth

    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile, model_ws=ms.model_ws)

    assert str(ms1.modelgrid) == str(ms.modelgrid)
    assert ms1.start_datetime == ms.start_datetime
    assert ms1.modelgrid.lenuni == ms.modelgrid.lenuni
    #assert ms1.sr.lenuni != sr.lenuni


def test_epsgs():
    import flopy.export.shapefile_utils as shp
    # test setting a geographic (lat/lon) coordinate reference
    # (also tests sr.crs parsing of geographic crs info)
    delr = np.ones(10)
    delc = np.ones(10)
    sr = flopy.discretization.StructuredGrid(delr=delr, delc=delc)
    sr.epsg = 102733
    assert sr.epsg == 102733
    if not "proj4_str:+init=epsg:102733" in sr.__repr__():
        raise AssertionError()


    sr.epsg = 4326  # WGS 84
    crs = shp.CRS(epsg=4326)
    assert crs.crs['proj'] == 'longlat'
    assert crs.grid_mapping_attribs['grid_mapping_name'] == 'latitude_longitude'
    if not "proj4_str:+init=epsg:4326" in sr.__repr__():
        raise AssertionError()


def test_dynamic_xll_yll():
    nlay, nrow, ncol = 1, 10, 5
    delr, delc = 250, 500
    xll, yll = 286.80, 29.03
    # test scaling of length units
    ms2 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms2, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc)
    sr1 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=2,
                                       xll=xll, yll=yll, rotation=30)
    xul, yul = sr1.xul, sr1.yul
    sr1.length_multiplier = 1.0 / 3.281
    assert sr1.xll == xll
    assert sr1.yll == yll
    sr2 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=2,
                                       xul=xul, yul=yul, rotation=30)
    sr2.length_multiplier = 1.0 / 3.281
    assert sr2.xul == xul
    assert sr2.yul == yul

    # test resetting of attributes
    sr3 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=2,
                                       xll=xll, yll=yll, rotation=30)
    # check that xul, yul and xll, yll are being recomputed
    sr3.xll += 10.
    sr3.yll += 21.
    assert np.abs(sr3.xul - (xul + 10.)) < 1e-6
    assert np.abs(sr3.yul - (yul + 21.)) < 1e-6
    sr4 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=2,
                                       xul=xul, yul=yul, rotation=30)
    assert sr4.origin_loc == 'ul'
    sr4.xul += 10.
    sr4.yul += 21.
    assert np.abs(sr4.xll - (xll + 10.)) < 1e-6
    assert np.abs(sr4.yll - (yll + 21.)) < 1e-6
    sr4.rotation = 0.
    assert np.abs(sr4.xul - (xul + 10.)) < 1e-6 # these shouldn't move because ul has priority
    assert np.abs(sr4.yul - (yul + 21.)) < 1e-6
    assert np.abs(sr4.xll - sr4.xul) < 1e-6
    assert np.abs(sr4.yll - (sr4.yul - sr4.yedge[0])) < 1e-6
    sr4.xll = 0.
    sr4.yll = 10.
    assert sr4.origin_loc == 'll'
    assert sr4.xul == 0.
    assert sr4.yul == sr4.yedge[0] + 10.
    sr4.xul = xul
    sr4.yul = yul
    assert sr4.origin_loc == 'ul'
    sr4.rotation = 30.
    assert np.abs(sr4.xll - xll) < 1e-6
    assert np.abs(sr4.yll - yll) < 1e-6

    sr5 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=2,
                                       xll=xll, yll=yll,
                                       rotation=0, epsg=26915)
    sr5.lenuni = 1
    assert sr5.length_multiplier == .3048
    assert sr5.yul == sr5.yll + sr5.yedge[0] * sr5.length_multiplier
    sr5.lenuni = 2
    assert sr5.length_multiplier == 1.
    assert sr5.yul == sr5.yll + sr5.yedge[0]
    sr5.proj4_str = '+proj=utm +zone=16 +datum=NAD83 +units=us-ft +no_defs'
    assert sr5.units == 'feet'
    assert sr5.length_multiplier == 1/.3048


def test_namfile_readwrite():
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    xll, yll = 272300, 5086000
    fm = flopy.modflow
    m = fm.Modflow(modelname='junk', model_ws=os.path.join('temp', 't007'))
    dis = fm.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                        delc=delc)
    m.modelgrid = flopy.discretization.StructuredGrid(delc=m.dis.delc.array,
                                                      delr=m.dis.delr.array,
                                                      top=m.dis.top.array,
                                                      botm=m.dis.botm.array,
                                                      # lenuni=3,
                                                      # length_multiplier=.3048,
                                                      xoff=xll, yoff=yll,
                                                      angrot=30)

    # test reading and writing of SR information to namfile
    m.write_input()
    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    assert abs(m2.modelgrid.xoffset - xll) < 1e-2
    assert abs(m2.modelgrid.yoffset - yll) < 1e-2
    assert m2.modelgrid.angrot == 30
    # assert abs(m2.sr.length_multiplier - .3048) < 1e-10

    model_ws = os.path.join("..", "examples", "data", "freyberg_multilayer_transient")
    ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, verbose=False,
                                    check=False, exe_name="mfnwt")
    assert ml.modelgrid.xoffset == ml.modelgrid._xul_to_xll(619653)
    assert ml.modelgrid.yoffset == ml.modelgrid._yul_to_yll(3353277)
    assert ml.modelgrid.angrot == 15.


def test_read_usgs_model_reference():
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    #xll, yll = 272300, 5086000
    model_ws = os.path.join('temp', 't007')
    mrf = os.path.join(model_ws, 'usgs.model.reference')
    shutil.copy('../examples/data/usgs.model.reference', mrf)
    fm = flopy.modflow
    m = fm.Modflow(modelname='junk', model_ws=model_ws)
    # feet and days
    dis = fm.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                        delc=delc, lenuni=1, itmuni=4)
    m.write_input()

    # test reading of SR information from usgs.model.reference
    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    from flopy.discretization import StructuredGrid
    mg = StructuredGrid(delr=dis.delr.array, delc=dis.delc.array)
    mg.read_usgs_model_reference_file(mrf)
    m2.modelgrid = mg

    assert m2.modelgrid.xoffset == mg.xoffset
    assert m2.modelgrid.yoffset == mg.yoffset
    assert m2.modelgrid.angrot == mg.angrot
    assert m2.modelgrid.epsg == mg.epsg

    # test reading non-default units from usgs.model.reference
    shutil.copy(mrf, mrf+'_copy')
    with open(mrf+'_copy') as src:
        with open(mrf, 'w') as dst:
            for line in src:
                if 'epsg' in line:
                    line = line.replace("102733", '4326')
                dst.write(line)

    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    m2.modelgrid.read_usgs_model_reference_file(mrf)

    assert m2.modelgrid.epsg == 4326
    # have to delete this, otherwise it will mess up other tests
    to_del = glob.glob(mrf + '*')
    for f in to_del:
        if os.path.exists(f):
            os.remove(os.path.join(f))
    assert True


def test_rotation():

    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    xul, yul = 500000, 2934000
    mg = flopy.discretization.StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array)
    mg._angrot = 45.
    mg.set_coord_info(mg._xul_to_xll(xul), mg._yul_to_yll(yul),
                      angrot=45.)

    xll, yll = mg.xoffset, mg.yoffset
    assert np.abs(mg.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg.yvertices[0, 0] - yul) < 1e-4

    mg2 = flopy.discretization.StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array)
    mg2._angrot = -45.
    mg2.set_coord_info(mg2._xul_to_xll(xul), mg2._yul_to_yll(yul),
                       angrot=-45.)

    xll2, yll2 = mg2.xoffset, mg2.yoffset
    assert np.abs(mg2.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg2.yvertices[0, 0] - yul) < 1e-4

    mg3 = flopy.discretization.StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array,
                                              xoff=xll2, yoff=yll2, angrot=-45.)

    assert np.abs(mg3.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg3.yvertices[0, 0] - yul) < 1e-4

    mg4 = flopy.discretization.StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array,
                                              xoff=xll, yoff=yll, angrot=45.)

    assert np.abs(mg4.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg4.yvertices[0, 0] - yul) < 1e-4


def test_sr_with_Map():
    # Note that most of this is either deprecated, or has pending deprecation
    import matplotlib.pyplot as plt
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    # transformation assigned by arguments
    xul, yul, rotation = 500000., 2934000., 45.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        modelmap = flopy.plot.ModelMap(model=m, xul=xul, yul=yul,
                                       rotation=rotation)
        assert len(w) == 2, len(w)
        assert w[0].category == PendingDeprecationWarning, w[0]
        assert 'ModelMap will be replaced by PlotMapView' in str(w[0].message)
        assert w[1].category == DeprecationWarning, w[1]
        assert 'xul/yul have been deprecated' in str(w[1].message)

    lc = modelmap.plot_grid()
    xll, yll = modelmap.mg.xoffset, modelmap.mg.yoffset
    plt.close()

    def check_vertices():
        xllp, yllp = lc._paths[0].vertices[0]
        xulp, yulp = lc._paths[0].vertices[1]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6
        assert np.abs(xulp - xul) < 1e-6
        assert np.abs(yulp - yul) < 1e-6

    check_vertices()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        modelmap = flopy.plot.ModelMap(model=m, xll=xll, yll=yll,
                                       rotation=rotation)
        assert len(w) == 1, len(w)
        assert w[0].category == PendingDeprecationWarning, w[0]
        assert 'ModelMap will be replaced by PlotMapView' in str(w[0].message)

    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # transformation in m.sr
    sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                      delc=m.dis.delc.array,
                                      xll=xll, yll=yll, rotation=rotation)
    m.sr = copy.deepcopy(sr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        modelmap = flopy.plot.ModelMap(model=m)

        assert len(w) == 1, len(w)
        assert w[0].category == PendingDeprecationWarning, w[0]
        assert 'ModelMap will be replaced by PlotMapView' in str(w[0].message)

    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # transformation assign from sr instance
    m.sr._reset()
    m.sr.set_spatialreference()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        modelmap = flopy.plot.ModelMap(model=m, sr=sr)

        assert len(w) == 1, len(w)
        assert w[0].category == PendingDeprecationWarning, w[0]
        assert 'ModelMap will be replaced by PlotMapView' in str(w[0].message)

    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # test plotting of line with specification of xul, yul in Dis/Model Map
    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(mf, nlay=1, nrow=10, ncol=20, delr=1., delc=1., xul=100, yul=210)
    #fig, ax = plt.subplots()
    verts = [[101., 201.], [119., 209.]]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        modelxsect = flopy.plot.ModelCrossSection(
            model=mf, line={'line': verts},
            xul=mf.dis.sr.xul, yul=mf.dis.sr.yul)

        # for wn in w:
        #    print(str(wn))
        assert len(w) in (3, 5), len(w)
        if len(w) == 5:
            assert w[0].category == DeprecationWarning, w[0]
            assert 'SpatialReference has been deprecated' in str(w[0].message)
            assert w[1].category == DeprecationWarning, w[1]
            assert 'SpatialReference has been deprecated' in str(w[1].message)
        assert w[-3].category == PendingDeprecationWarning, w[-3]
        assert 'ModelCrossSection will be replaced by' in str(w[-3].message)
        assert w[-2].category == DeprecationWarning, w[-2]
        assert 'xul/yul have been deprecated' in str(w[-2].message)
        assert w[-1].category == DeprecationWarning, w[-1]
        assert 'xul/yul have been deprecated' in str(w[-1].message)

    linecollection = modelxsect.plot_grid()
    plt.close()


def test_modelgrid_with_PlotMapView():
    import matplotlib.pyplot as plt
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    # transformation assigned by arguments
    xll, yll, rotation = 500000., 2934000., 45.

    def check_vertices():
        #vertices = modelmap.mg.xyvertices
        xllp, yllp = lc._paths[0].vertices[0]
        #xulp, yulp = lc._paths[0].vertices[1]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6
        #assert np.abs(xulp - xul) < 1e-6
        #assert np.abs(yulp - yul) < 1e-6

#    check_vertices()
    m.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)
    modelmap = flopy.plot.PlotMapView(model=m)
    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    modelmap = flopy.plot.PlotMapView(modelgrid=m.modelgrid)
    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(mf, nlay=1, nrow=10, ncol=20, delr=1., delc=1., xul=100, yul=210)
    #fig, ax = plt.subplots()
    verts = [[101., 201.], [119., 209.]]
    # modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'line': verts},
    #                                           xul=mf.dis.sr.xul, yul=mf.dis.sr.yul)
    mf.modelgrid.set_coord_info(xoff=mf.dis.sr.xll, yoff=mf.dis.sr.yll)
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'line': verts})
    linecollection = modelxsect.plot_grid()
    plt.close()


def test_tricontour_NaN():
    from flopy.plot import PlotMapView
    import numpy as np
    from flopy.discretization import StructuredGrid

    arr = np.random.rand(10, 10) * 100
    arr[-1, :] = np.nan
    delc = np.array([10] * 10, dtype=float)
    delr = np.array([8] * 10, dtype=float)
    top = np.ones((10, 10), dtype=float)
    botm = np.ones((3, 10, 10), dtype=float)
    botm[0] = 0.75
    botm[1] = 0.5
    botm[2] = 0.25
    idomain = np.ones((3, 10, 10))
    idomain[0, 0, :] = 0
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    levels = np.linspace(vmin, vmax, 7)

    grid = StructuredGrid(delc=delc,
                          delr=delr,
                          top=top,
                          botm=botm,
                          idomain=idomain,
                          lenuni=1,
                          nlay=3, nrow=10, ncol=10)

    pmv = PlotMapView(modelgrid=grid, layer=0)
    contours = pmv.contour_array(a=arr)

    for ix, lev in enumerate(contours.levels):
        if not np.allclose(lev, levels[ix]):
            raise AssertionError("TriContour NaN catch Failed")


def test_get_vertices():
    from flopy.utils.reference import SpatialReference
    from flopy.discretization import StructuredGrid
    m = flopy.modflow.Modflow(rotation=20.)
    nrow, ncol = 40, 20
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=nrow, ncol=ncol,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    xul, yul = 500000, 2934000
    sr = SpatialReference(delc=m.dis.delc.array,
                          xul=xul, yul=yul, rotation=45.)
    mg = StructuredGrid(delc=m.dis.delc.array,
                        delr=m.dis.delr.array,
                        xoff=sr.xll, yoff=sr.yll,
                        angrot=sr.rotation)

    xgrid = mg.xvertices
    ygrid = mg.yvertices
    # a1 = np.array(mg.xyvertices)
    a1 = np.array([[xgrid[0, 0], ygrid[0,0]],
                   [xgrid[0, 1], ygrid[0,1]],
                   [xgrid[1, 1], ygrid[1, 1]],
                   [xgrid[1, 0], ygrid[1, 0]]])

    a2 = np.array(mg.get_cell_vertices(0, 0))
    assert np.array_equal(a1, a2)


def test_vertex_model_dot_plot():
    # load up the vertex example problem
    sim_name = "mfsim.nam"
    sim_path = "../examples/data/mf6/test003_gwftri_disv"
    disv_sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, version="mf6",
                                           exe_name="mf6",
                                           sim_ws=sim_path)
    disv_ml = disv_sim.get_model('gwf_1')
    if sys.version_info[0] > 2:
        ax = disv_ml.plot()
        assert ax


def test_model_dot_plot():
    loadpth = os.path.join('..', 'examples', 'data', 'secp')
    ml = flopy.modflow.Modflow.load('secp.nam', model_ws=loadpth)
    if sys.version_info[0] > 2:
        ax = ml.plot()
        assert ax


def test_get_rc_from_node_coordinates():
    m = flopy.modflow.Modflow(rotation=20.)
    nrow, ncol = 10, 10
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=nrow, ncol=ncol,
                                   delr=100.,
                                   delc=100., top=10, botm=0)
    r, c = m.dis.get_rc_from_node_coordinates([50., 110.], [50., 220.])
    assert np.array_equal(r, np.array([9, 7]))
    assert np.array_equal(c, np.array([0, 1]))

    # test variable delr and delc spacing
    mf = flopy.modflow.Modflow()
    delc = [0.5] * 5 + [2.0] * 5
    delr = [0.5] * 5 + [2.0] * 5
    nrow = 10
    ncol = 10
    mfdis=flopy.modflow.ModflowDis(mf, nrow=nrow, ncol=ncol, delr=delr, delc=delc) #, xul=50, yul=1000)
    ygrid, xgrid, zgrid = mfdis.get_node_coordinates()
    for i in range(nrow):
        for j in range(ncol):
            x = xgrid[j]
            y = ygrid[i]
            r, c = mfdis.get_rc_from_node_coordinates(x, y)
            assert r == i, 'row {} not equal {} for xy ({}, {})'.format(r, i, x, y)
            assert c == j, 'col {} not equal {} for xy ({}, {})'.format(c, j, x, y)


def test_netcdf_classmethods():
    import os
    import flopy

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return

    nam_file = "freyberg.nam"
    model_ws = os.path.join('..', 'examples', 'data',
                            'freyberg_multilayer_transient')
    ml = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False,
                                    verbose=True, load_only=[])

    f = ml.export(os.path.join(npth, "freyberg.nc"))
    v1_set = set(f.nc.variables.keys())
    fnc = os.path.join(npth, "freyberg.new.nc")
    new_f = flopy.export.NetCdf.zeros_like(f, output_filename=fnc)
    v2_set = set(new_f.nc.variables.keys())
    diff = v1_set.symmetric_difference(v2_set)
    assert len(diff) == 0, str(diff)

# def test_netcdf_overloads():
#     import os
#     import flopy
#     nam_file = "freyberg.nam"
#     model_ws = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient')
#     ml = flopy.modflow.Modflow.load(nam_file,model_ws=model_ws,check=False,
#                                     verbose=False,load_only=[])
#
#     f = ml.export(os.path.join("temp","freyberg.nc"))
#     fzero = flopy.export.NetCdf.zeros_like(f)
#     assert fzero.nc.variables["model_top"][:].sum() == 0
#     print(f.nc.variables["model_top"][0,:])
#     fplus1 = f + 1
#     assert fplus1.nc.variables["model_top"][0,0] == f.nc.variables["model_top"][0,0] + 1
#     assert (f + fplus1).nc.variables["model_top"][0,0] ==\
#            f.nc.variables["model_top"][0,0] + \
#            fplus1.nc.variables["model_top"][0,0]
#
#     fminus1 = f - 1
#     assert fminus1.nc.variables["model_top"][0,0] == f.nc.variables["model_top"][0,0] - 1
#     assert (f - fminus1).nc.variables["model_top"][0,0]==\
#            f.nc.variables["model_top"][0,0] - \
#            fminus1.nc.variables["model_top"][0,0]
#
#     ftimes2 = f * 2
#     assert ftimes2.nc.variables["model_top"][0,0] == f.nc.variables["model_top"][0,0] * 2
#     assert (f * ftimes2).nc.variables["model_top"][0,0] ==\
#             f.nc.variables["model_top"][0,0] * \
#            ftimes2.nc.variables["model_top"][0,0]
#
#     fdiv2 = f / 2
#     assert fdiv2.nc.variables["model_top"][0,0] == f.nc.variables["model_top"][0,0] / 2
#     assert (f / fdiv2).nc.variables["model_top"][0,0] == \
#          f.nc.variables["model_top"][0,0] / \
#            fdiv2.nc.variables["model_top"][0,0]
#
#     assert f.nc.variables["ibound"][0,0,0] == 1
def test_wkt_parse():
    """Test parsing of Coordinate Reference System parameters
    from well-known-text in .prj files."""

    from flopy.export.shapefile_utils import CRS

    geocs_params = [
        'wktstr', 'geogcs', 'datum', 'spheroid_name', 'semi_major_axis',
        'inverse_flattening', 'primem', 'gcs_unit']

    prjs = glob.glob('../examples/data/prj_test/*')
    for prj in prjs:
        with open(prj) as src:
            wkttxt = src.read()
            wkttxt = wkttxt.replace("'", '"')
        if len(wkttxt) > 0 and 'projcs' in wkttxt.lower():
            crsobj = CRS(esri_wkt=wkttxt)
            assert isinstance(crsobj.crs, dict)
            for k in geocs_params:
                assert crsobj.__dict__[k] is not None
            projcs_params = [k for k in crsobj.__dict__
                             if k not in geocs_params]
            if crsobj.projcs is not None:
                for k in projcs_params:
                    if k in wkttxt.lower():
                        assert crsobj.__dict__[k] is not None


def test_shapefile_ibound():
    import os
    import flopy
    try:
        import shapefile
    except:
        return

    shape_name = os.path.join(spth, "test.shp")
    nam_file = "freyberg.nam"
    model_ws = os.path.join('..', 'examples', 'data',
                            'freyberg_multilayer_transient')
    ml = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False,
                                    verbose=True, load_only=['bas6'])
    ml.export(shape_name)
    shp = shapefile.Reader(shape_name)
    field_names = [item[0] for item in shp.fields][1:]
    ib_idx = field_names.index("ibound_1")
    txt = "should be int instead of {0}".format(type(shp.record(0)[ib_idx]))
    assert type(shp.record(0)[ib_idx]) == int, txt


def test_shapefile():
    for namfile in namfiles:
        yield export_shapefile, namfile
    return


def test_netcdf():
    for namfile in namfiles:
        yield export_mf2005_netcdf, namfile

    return


def build_netcdf():
    for namfile in namfiles:
        export_mf2005_netcdf(namfile)
    return


def build_sfr_netcdf():
    namfile = 'testsfr2.nam'
    export_mf2005_netcdf(namfile)
    return


def test_export_array():
    from flopy.discretization import StructuredGrid
    from flopy.export.utils import export_array
    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1)
    filename = os.path.join(spth, 'myarray1.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), 'did not create array shapefile'

    # with modelgrid epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1, epsg=epsg)
    filename = os.path.join(spth, 'myarray2.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), 'did not create array shapefile'

    # with passing in epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1)
    filename = os.path.join(spth, 'myarray3.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), 'did not create array shapefile'
    return


def test_export_array_contours():
    from flopy.discretization import StructuredGrid
    from flopy.export.utils import export_array_contours
    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1)
    filename = os.path.join(spth, 'myarraycontours1.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), 'did not create contour shapefile'

    # with modelgrid epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1, epsg=epsg)
    filename = os.path.join(spth, 'myarraycontours2.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), 'did not create contour shapefile'

    # with passing in epsg code
    modelgrid = StructuredGrid(delr=np.ones(ncol) * 1.1,
                               delc=np.ones(nrow) * 1.1)
    filename = os.path.join(spth, 'myarraycontours3.shp')
    a = np.arange(nrow*ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), 'did not create contour shapefile'
    return


def test_export_contourf():
    try:
        import shapely
    except:
        return
    import matplotlib.pyplot as plt
    from flopy.export.utils import export_contourf
    filename = os.path.join(spth, 'myfilledcontours.shp')
    a = np.random.random((10, 10))
    cs = plt.contourf(a)
    export_contourf(filename, cs)
    assert os.path.isfile(filename), 'did not create contourf shapefile'
    return


if __name__ == '__main__':
    #test_shapefile()
    # test_shapefile_ibound()
    #test_netcdf()
    # test_netcdf_overloads()
    # test_netcdf_classmethods()
    # build_netcdf()
    # build_sfr_netcdf()
    # test_mg()
    # test_mbase_modelgrid()
    # test_mt_modelgrid()
    # test_rotation()
    # test_model_dot_plot()
    # test_vertex_model_dot_plot()
    #test_sr_with_Map()
    #test_modelgrid_with_PlotMapView()
    # test_epsgs()
    # test_sr_scaling()
    # test_read_usgs_model_reference()
    #test_dynamic_xll_yll()
    # test_namfile_readwrite()
    # test_free_format_flag()
    # test_get_vertices()
    #test_export_output()
    #for namfile in namfiles:
    #test_freyberg_export()
    #test_export_array()
    #test_write_shapefile()
    #test_wkt_parse()
    #test_get_rc_from_node_coordinates()
    # test_export_array()
    #test_export_array_contours()
    #test_tricontour_NaN()
    test_export_contourf()
    pass
