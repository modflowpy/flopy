# Test export module
import sys
sys.path.append('..')
import copy
import glob
import os
import shutil
import numpy as np
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

def export_netcdf(namfile):
    if namfile in skip:
        return
    print(namfile)
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    if m.sr.lenuni == 0:
        m.sr.lenuni = 1
        # print('skipping...lenuni==0 (undefined)')
        # return
    # if sum(m.dis.laycbd) != 0:
    if m.dis.botm.shape[0] != m.nlay:
        print('skipping...botm.shape[0] != nlay')
        return
    assert m, 'Could not load namefile {}'.format(namfile)
    assert isinstance(m, flopy.modflow.Modflow)

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
    from flopy.grid.reference import SpatialReference
    namfile = 'freyberg.nam'
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
    m.sr = SpatialReference(delc=m.dis.delc.array, epsg=5070)
    # test export with an sr, regardless of whether or not wkt was found
    m.drn.stress_period_data.export(outshp, sparse=True)
    assert os.path.exists(outshp)
    remove_shp(outshp)
    m.sr = SpatialReference(delc=m.dis.delc.array, epsg=3070)
    # verify that attributes have same sr as parent
    assert m.drn.stress_period_data.sr == m.sr
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
    from flopy.utils.reference import SpatialReference
    from flopy.export.shapefile_utils import shp2recarray
    from flopy.export.shapefile_utils import write_grid_shapefile, write_grid_shapefile2

    sr = SpatialReference(delr=np.ones(10) *1.1,  # cell spacing along model rows
                          delc=np.ones(10) *1.1,  # cell spacing along model columns
                          epsg=26715,
                          lenuni=1  # MODFLOW length units
                          )
    vrts = copy.deepcopy(sr.vertices)
    outshp1 = os.path.join(tpth, 'junk.shp')
    outshp2 = os.path.join(tpth, 'junk2.shp')
    write_grid_shapefile(outshp1, sr, array_dict={})
    write_grid_shapefile2(outshp2, sr, array_dict={})

    # test that vertices aren't getting altered by writing shapefile
    assert np.array_equal(vrts, sr.vertices)
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

    try:
        from scipy.ndimage import rotate
    except:
        rotate = False
        pass

    namfile = 'freyberg.nam'
    model_ws = '../examples/data/freyberg_multilayer_transient/'
    m = flopy.modflow.Modflow.load(namfile, model_ws=model_ws, verbose=False,
                                   load_only=['DIS', 'BAS6'])
    m.sr.rotation = 45.
    nodata = -9999
    m.modelgrid.export_array(os.path.join(tpth, 'fb.asc'), m.dis.top.array, nodata=nodata)
    arr = np.loadtxt(os.path.join(tpth, 'fb.asc'), skiprows=6)

    m.modelgrid.write_shapefile(os.path.join(tpth, 'grid.shp'))
    # check bounds
    with open(os.path.join(tpth, 'fb.asc')) as src:
        for line in src:
            if 'xllcorner' in line.lower():
                val = float(line.strip().split()[-1])
                # ascii grid origin will differ if it was unrotated
                if rotate:
                    assert np.abs(val - m.modelgrid.bounds[0]) < 1e-6
                else:
                    assert np.abs(val - m.sr.xll) < 1e-6
            if 'yllcorner' in line.lower():
                val = float(line.strip().split()[-1])
                if rotate:
                    assert np.abs(val - m.modelgrid.bounds[1]) < 1e-6
                else:
                    assert np.abs(val - m.sr.yll) < 1e-6
            if 'cellsize' in line.lower():
                val = float(line.strip().split()[-1])
                rot_cellsize = np.cos(np.radians(m.sr.rotation)) * m.modelgrid.delr[0] * m.sr.length_multiplier
                #assert np.abs(val - rot_cellsize) < 1e-6
                break
    rotate = False
    rasterio = None
    if rotate:
        rotated = rotate(m.dis.top.array, m.sr.rotation, cval=nodata)

    if rotate:
        assert rotated.shape == arr.shape

    try:
        # test GeoTIFF export
        import rasterio
    except:
        pass
    if rasterio is not None:
        m.modelgrid.export_array(os.path.join(tpth, 'fb.tif'),
                                 m.dis.top.array,
                                 nodata=nodata)
        with rasterio.open(os.path.join(tpth, 'fb.tif')) as src:
            arr = src.read(1)
            assert src.shape == (m.nrow, m.ncol)
            assert np.abs(src.bounds[0] - m.modelgrid.bounds[0]) < 1e-6
            assert np.abs(src.bounds[1] - m.modelgrid.bounds[1]) < 1e-6

def test_mbase_sr():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(modelname="test", xul=1000.0,
                               rotation=12.5, start_datetime="1/1/2016")
    try:
        print(ml.sr.xcentergrid)
    except:
        pass
    else:
        raise Exception("should have failed")

    dis = flopy.modflow.ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5),
                                   xul=500)
    print(ml.sr)
    assert ml.modelgrid.sr.xul == 500
    assert ml.modelgrid.sr.yll == -10
    ml.model_ws = tpth

    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam", model_ws=ml.model_ws)
    assert ml1.sr == ml.modelgrid.sr
    assert ml1.start_datetime == ml.start_datetime

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


def test_sr():
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
                                   delr=delr, delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)

    # test instantiation of an empty sr object
    sr = flopy.grid.reference.SpatialReference()

    # test instantiation of SR with xul, yul and no grid
    sr = flopy.grid.reference.SpatialReference(xul=1, yul=1)

    #txt = 'yul does not approximately equal 100 - ' + \
    #      '(xul, yul) = ({}, {})'.format( ms.sr.yul, ms.sr.yul)
    assert abs(ms.sr.yul - Ly) < 1e-3#, txt
    ms.sr.xul = 111
    assert ms.sr.xul == 111

    xul, yul = 321., 123.
    ms.sr = flopy.grid.reference.SpatialReference(delc=ms.dis.delc.array,
                                               lenuni=3, xul=xul, yul=yul,
                                               rotation=20)
    sr = ms.sr
    # test that transform for arbitrary coordinates
    # is working in same as transform for model grid
    mg = ms.modelgrid
    x = mg.xcell_centers(flopy.grid.modelgrid.PointType.modelxyz)
    y = mg.ycell_centers(flopy.grid.modelgrid.PointType.modelxyz)[0]
    xt, yt = sr.transform(x, y)
    assert np.sum(xt - mg.xcell_centers()[0]) < 1e-3
    x, y = mg.xcell_centers(flopy.grid.modelgrid.PointType.modelxyz)[0], \
           mg.ycell_centers(flopy.grid.modelgrid.PointType.modelxyz)
    xt, yt = sr.transform(x, y)
    assert np.sum(yt - mg.ycell_centers()) < 1e-3

    # test inverse transform
    x0, y0 = 9.99, 2.49
    x1, y1 = sr.transform(x0, y0)
    x2, y2 = sr.transform(x1, y1, inverse=True)
    assert np.abs(x2-x0) < 1e-6
    assert np.abs(y2-y0) < 1e6

    # test input using ul vs ll
    xll, yll = sr.xll, sr.yll
    ms.sr = flopy.grid.reference.SpatialReference(delc=ms.dis.delc.array,
                                                  lenuni=3, xll=xll, yll=yll,
                                                  rotation=20)
    sr2 = ms.sr
    mg2 = ms.modelgrid
    assert sr2.xul == sr.xul
    assert sr2.yul == sr.yul
    assert np.array_equal(mg.xcell_centers(), mg2.xcell_centers())
    assert np.array_equal(mg.ycell_centers(), mg2.ycell_centers())

    ms.sr.lenuni = 1
    assert ms.sr.lenuni == 1

    ms.sr.units = "feet"
    assert ms.sr.units == "feet"

    ms.sr = sr
    assert ms.sr == sr
    assert ms.sr.lenuni != ms.dis.lenuni

    try:
        ms.sr.units = "junk"
    except:
        pass
    else:
        raise Exception("should have failed")

    ms.start_datetime = "1-1-2016"
    assert ms.start_datetime == "1-1-2016"
    assert ms.dis.start_datetime == "1-1-2016"

    ms.model_ws = tpth
    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile, model_ws=ms.model_ws)
    assert ms1.sr == ms.sr
    assert ms1.dis.sr == ms.dis.sr
    assert ms1.start_datetime == ms.start_datetime
    assert ms1.sr.units == ms.sr.units
    assert ms1.dis.lenuni == ms1.sr.lenuni
    #assert ms1.sr.lenuni != sr.lenuni
    ms1.sr = sr
    assert ms1.sr == ms.sr

def test_epsgs():
    # test setting a geographic (lat/lon) coordinate reference
    # (also tests sr.crs parsing of geographic crs info)
    delr = np.ones(10)
    delc = np.ones(10)
    sr = flopy.grid.reference.SpatialReference(
                                      delc=delc,
                                      )
    sr.epsg = 102733
    assert sr.epsg == 102733

    sr.epsg = 4326  # WGS 84
    assert sr.crs.crs['proj'] == 'longlat'
    assert sr.crs.grid_mapping_attribs['grid_mapping_name'] == 'latitude_longitude'

def test_sr_scaling():
    nlay, nrow, ncol = 1, 10, 5
    delr, delc = 250, 500
    top = 100
    botm = 50
    xll, yll = 286.80, 29.03

    print(np.__version__)
    # test scaling of length units
    ms2 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms2, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc)
    ms2.sr = flopy.grid.reference.SpatialReference(delc=ms2.dis.delc.array,
                                                   lenuni=3, xll=xll,
                                                   yll=yll, rotation=0)
    ms2.sr.epsg = 26715
    ms2.dis.export(os.path.join(spth, 'dis2.shp'))
    ms3 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms3, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc, top=top, botm=botm)
    ms3.sr = flopy.grid.reference.SpatialReference(delc=ms2.dis.delc.array,
                                                   lenuni=2,
                                                   length_multiplier=2.,
                                                   xll=xll, yll=yll,
                                                   rotation=0)
    ms3.dis.export(os.path.join(spth, 'dis3.shp'), epsg=26715)

    # check that the origin(s) are maintained
    mg3 = ms3.modelgrid
    assert np.array_equal(mg3.get_cell_vertices(nrow - 1, 0)[1],
                          [ms3.sr.xll, ms3.sr.yll])
    mg2 = ms2.modelgrid
    assert np.allclose(mg3.get_cell_vertices(nrow - 1, 0)[1],
                       mg2.get_cell_vertices(nrow - 1, 0)[1])

    # check that the upper left corner is computed correctly
    # in this case, length_multiplier overrides the given units
    def check_size(mg):
        xur, yur = mg.get_cell_vertices(0, ncol - 1)[3]
        assert np.abs(xur - (xll + mg.sr.length_multiplier * delr * ncol)) < \
               1e-4
        assert np.abs(yur - (yll + mg.sr.length_multiplier * delc * nrow)) < \
               1e-4
    check_size(mg3)

    # run the same tests but with units specified instead of a length multiplier
    ms2 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms2, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr, delc=delc,
                                   lenuni=1 # feet; should have no effect on SR
                                   # (model not supplied to SR)
                                   )
    ms2.sr = flopy.grid.reference.SpatialReference(delc=ms2.dis.delc.array,
                                                   lenuni=2, # meters
                                                   epsg=26715,  # meters,
                                                   # listed
                                               # on spatialreference.org
                                                   xll=xll, yll=yll,
                                                   rotation=0)
    assert ms2.sr.model_length_units == 'meters'
    assert ms2.sr.length_multiplier == 1.
    ms2.sr.lenuni = 1 # feet; test dynamic setting
    assert ms2.sr.model_length_units == 'feet'
    check_size(mg2)
    assert ms2.sr.length_multiplier == .3048
    ms2.sr.lenuni = 3 # centimeters
    assert ms2.sr.model_length_units == 'centimeters'
    check_size(mg2)
    assert ms2.sr.length_multiplier == 0.01
    ms2.sr.lenuni = 2 # meters
    check_size(mg2)
    ms2.sr.units = 'meters'
    ms2.sr.proj4_str = '+proj=utm +zone=16 +datum=NAD83 +units=us-ft +no_defs'
    assert ms2.sr.proj4_str == '+proj=utm +zone=16 +datum=NAD83 +units=us-ft +no_defs'
    assert ms2.sr.units == 'feet'
    assert ms2.sr.length_multiplier == 1/.3048
    check_size(mg2)
    ms2.sr.epsg = 6610 # meters, not listed on spatialreference.org but understood by pyproj
    assert ms2.sr.units == 'meters'
    assert ms2.sr.proj4_str is not None
    check_size(mg2)

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
    m.sr = flopy.grid.reference.SpatialReference(delc=m.dis.delc.array,
                                                 lenuni=3,
                                                 length_multiplier=.3048,
                                                 xll=xll, yll=yll,
                                                 rotation=30)

    # test reading and writing of SR information to namfile
    m.write_input()
    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    assert abs(m2.sr.xll - xll) < 1e-2
    assert abs(m2.sr.yll - yll) < 1e-2
    assert m2.sr.rotation == 30
    assert abs(m2.sr.length_multiplier - .3048) < 1e-10

    model_ws = os.path.join("..", "examples", "data", "freyberg_multilayer_transient")
    ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws, verbose=False,
                                    check=False, exe_name="mfnwt")
    assert ml.sr.xul == 619653
    assert ml.sr.yul == 3353277
    assert ml.sr.rotation == 15.

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
    from flopy.utils.reference import SpatialReference
    d = SpatialReference.read_usgs_model_reference_file(mrf)
    assert m2.sr.xul == d['xul']
    assert m2.sr.yul == d['yul']
    assert m2.sr.rotation == d['rotation']
    assert m2.sr.lenuni == d['lenuni']
    assert m2.sr.epsg == d['epsg']

    # test reading non-default units from usgs.model.reference
    shutil.copy(mrf, mrf+'_copy')
    with open(mrf+'_copy') as src:
        with open(mrf, 'w') as dst:
            for line in src:
                if 'time_unit' in line:
                    line = line.replace('days', 'seconds')
                elif 'length_units' in line:
                    line = line.replace('feet', 'meters')
                dst.write(line)
    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    assert m2.tr.itmuni == 1
    assert m2.sr.lenuni == 2
    # have to delete this, otherwise it will mess up other tests
    to_del = glob.glob(mrf + '*')
    for f in to_del:
        if os.path.exists(f):
            os.remove(os.path.join(f))
    assert True


def test_rotation():
    from flopy.grid.modelgrid import PointType
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    xul, yul = 500000, 2934000
    m.sr = flopy.grid.SpatialReference(delc=m.dis.delc.array,
                                       xul=xul, yul=yul, rotation=45.)
    xll, yll = m.sr.xll, m.sr.yll
    mg = m.modelgrid
    assert np.abs(mg.xedgegrid()[0, 0] - xul) < 1e-4
    assert np.abs(mg.yedgegrid()[0, 0] - yul) < 1e-4
    m.sr = flopy.grid.SpatialReference(delc=m.dis.delc.array,
                                        xul=xul, yul=yul, rotation=-45.)
    mg2 = m.modelgrid
    assert np.abs(mg2.xedgegrid()[0, 0] - xul) < 1e-4
    assert np.abs(mg2.yedgegrid()[0, 0] - yul) < 1e-4
    xll2, yll2 = m.sr.xll, m.sr.yll
    m.sr = flopy.grid.SpatialReference(delc=m.dis.delc.array,
                                        xll=xll2, yll=yll2, rotation=-45.)
    mg3 = m.modelgrid
    assert np.abs(mg3.xedgegrid()[0, 0] - xul) < 1e-4
    assert np.abs(mg3.yedgegrid()[0, 0] - yul) < 1e-4
    m.sr = flopy.grid.SpatialReference(delc=m.dis.delc.array,
                                        xll=xll, yll=yll, rotation=45.)
    mg4 = m.modelgrid
    assert np.abs(mg4.xedgegrid()[0, 0] - xul) < 1e-4
    assert np.abs(mg4.yedgegrid()[0, 0] - yul) < 1e-4


def test_sr_with_Map():
    import matplotlib.pyplot as plt
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    # transformation assigned by arguments
    xul, yul, rotation = 500000., 2934000., 45.
    modelmap = flopy.plot.ModelMap(model=m, xul=xul, yul=yul,
                                   rotation=rotation)
    lc = modelmap.plot_grid()
    xll, yll = modelmap.sr.xll, modelmap.sr.yll
    plt.close()

    def check_vertices():
        xllp, yllp = lc._paths[0].vertices[0]
        xulp, yulp = lc._paths[0].vertices[1]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6
        assert np.abs(xulp - xul) < 1e-6
        assert np.abs(yulp - yul) < 1e-6

    check_vertices()

    modelmap = flopy.plot.ModelMap(model=m, xll=xll, yll=yll,
                                   rotation=rotation)
    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # transformation in m.sr
    sr = flopy.grid.reference.SpatialReference(delc=m.dis.delc.array,
                                               xll=xll, yll=yll,
                                               rotation=rotation)
    m.sr = copy.deepcopy(sr)
    modelmap = flopy.plot.ModelMap(model=m)
    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # transformation assign from sr instance
    m.modelgrid.sr._reset()
    m.modelgrid.sr.set_spatialreference(delc=m.dis.delc.array,
                                        xll=xll, yll=yll,
                                        rotation=rotation)
    modelmap = flopy.plot.ModelMap(model=m, sr=sr)
    lc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    # test plotting of line with specification of xul, yul in Dis/Model Map
    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(mf, nlay=1, nrow=10, ncol=20, delr=1., delc=1., xul=100, yul=210)
    #fig, ax = plt.subplots()
    verts = [[101., 201.], [119., 209.]]
    modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'line': verts},
                                              xul=mf.dis.sr.xul, yul=mf.dis.sr.yul)
    linecollection = modelxsect.plot_grid()
    plt.close()

def test_get_vertices():
    m = flopy.modflow.Modflow(rotation=20.)
    nrow, ncol = 40, 20
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=nrow, ncol=ncol,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    xul, yul = 500000, 2934000
    m.sr = flopy.grid.SpatialReference(delc=m.dis.delc.array,
                                       xul=xul, yul=yul, rotation=45.)
    mg = m.modelgrid
    a1 = np.array(mg.xyvertices())
    j = np.array(list(range(ncol)) * nrow)
    i = np.array(sorted(list(range(nrow)) * ncol))
    a2 = np.array(mg.get_cell_vertices(i, j))
    assert np.array_equal(a1, a2)

def test_get_rc_from_node_coordinates():
    m = flopy.modflow.Modflow(rotation=20.)
    nrow, ncol = 10, 10
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=nrow, ncol=ncol,
                                   delr=100.,
                                   delc=100., top=10, botm=0)
    r, c = m.dis.get_rc_from_node_coordinates([50., 110.], [50., 220.])
    assert np.array_equal(r, np.array([9, 7]))
    assert np.array_equal(c, np.array([0, 1]))

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

    from flopy.utils.reference import crs

    prjs = glob.glob('../examples/data/prj_test/*')

    for prj in prjs:
        with open(prj) as src:
            wkttxt = src.read()
            wkttxt = wkttxt.replace("'", '"')
        if len(wkttxt) > 0 and 'projcs' in wkttxt.lower():
            crsobj = crs(esri_wkt=wkttxt)
            geocs_params = ['wktstr', 'geogcs', 'datum', 'spheriod_name',
                            'semi_major_axis', 'inverse_flattening',
                            'primem', 'gcs_unit']
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
                                    verbose=True, load_only=[])
    ml.export(shape_name)
    shp = shapefile.Reader(shape_name)
    field_names = [item[0] for item in shp.fields][1:]
    ib_idx = field_names.index("ibound_001")
    txt = "should be int instead of {0}".format(type(shp.record(0)[ib_idx]))
    assert type(shp.record(0)[ib_idx]) == int, txt


def test_shapefile():
    for namfile in namfiles:
        yield export_shapefile, namfile
    return

def test_netcdf():
    for namfile in namfiles:
        yield export_netcdf, namfile

    return


def build_netcdf():
    for namfile in namfiles:
        export_netcdf(namfile)
    return


def build_sfr_netcdf():
    namfile = 'testsfr2.nam'
    export_netcdf(namfile)
    return


if __name__ == '__main__':
    #test_shapefile()
    # test_shapefile_ibound()
    #test_netcdf()
    # test_netcdf_overloads()
    #test_netcdf_classmethods()
    # build_netcdf()
    # build_sfr_netcdf()
    #test_sr()
    #test_mbase_sr()
    #test_rotation()
    test_sr_with_Map()
    #test_epsgs()
    #test_sr_scaling()
    #test_read_usgs_model_reference()
    #test_dynamic_xll_yll()
    #test_namfile_readwrite()
    # test_free_format_flag()
    #test_get_vertices()
    #test_export_output()
    #for namfile in namfiles:
    # for namfile in ["fhb.nam"]:
    #export_netcdf(namefile)
    #test_freyberg_export()
    #test_export_array()
    test_write_shapefile()
    #test_wkt_parse()
    #test_get_rc_from_node_coordinates()
    pass
