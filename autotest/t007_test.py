# Test export module
import sys

sys.path.insert(0, '..')
import copy
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
    assert ml.sr.xul == 500
    assert ml.sr.yll == -10
    ml.model_ws = tpth

    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam", model_ws=ml.model_ws)
    assert ml1.sr == ml.sr
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
                                   delr=delr,
                                   delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)

    # test instantiation of an empty sr object
    sr = flopy.utils.reference.SpatialReference()

    # test instantiation of SR with xul, yul and no grid
    sr = flopy.utils.reference.SpatialReference(xul=1, yul=1)

    xul, yul = 321., 123.
    sr = flopy.utils.SpatialReference(delr=ms.dis.delr.array,
                                      delc=ms.dis.delc.array, lenuni=3,
                                      xul=xul, yul=yul, rotation=20)

    #txt = 'yul does not approximately equal 100 - ' + \
    #      '(xul, yul) = ({}, {})'.format( ms.sr.yul, ms.sr.yul)
    assert abs(ms.sr.yul - 0) < 1e-3#, txt
    ms.sr.xul = 111
    assert ms.sr.xul == 111

    # test that transform for arbitrary coordinates
    # is working in same as transform for model grid
    x, y = ms.sr.xcenter, ms.sr.ycenter[0]
    xt, yt = sr.transform(x, y)
    assert np.sum(xt - sr.xcentergrid[0]) < 1e-3
    x, y = ms.sr.xcenter[0], ms.sr.ycenter
    xt, yt = sr.transform(x, y)
    assert np.sum(yt - sr.ycentergrid[:, 0]) < 1e-3

    # test input using ul vs ll
    xll, yll = sr.xll, sr.yll
    sr2 = flopy.utils.SpatialReference(delr=ms.dis.delr.array,
                                       delc=ms.dis.delc.array, lenuni=3,
                                       xll=xll, yll=yll, rotation=20)
    assert sr2.xul == sr.xul
    assert sr2.yul == sr.yul
    assert np.array_equal(sr.xcentergrid, sr2.xcentergrid)
    assert np.array_equal(sr.ycentergrid, sr2.ycentergrid)

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
    assert ms1.sr.lenuni != sr.lenuni
    ms1.sr = sr
    assert ms1.sr == ms.sr


def test_sr_scaling():
    nlay, nrow, ncol = 1, 10, 5
    delr, delc = 250, 500
    xll, yll = 286.80, 29.03
    # test scaling of length units
    ms2 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms2, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc)
    ms2.sr = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                          delc=ms2.dis.delc.array, lenuni=3,
                                          xll=xll, yll=yll, rotation=0)
    ms2.sr.epsg = 26715
    ms2.dis.export(os.path.join(spth, 'dis2.shp'))
    ms3 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms3, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr,
                                   delc=delc)
    ms3.sr = flopy.utils.SpatialReference(delr=ms3.dis.delr.array,
                                          delc=ms2.dis.delc.array, lenuni=3,
                                          length_multiplier=.3048,
                                          xll=xll, yll=yll, rotation=0)
    ms3.dis.export(os.path.join(spth, 'dis3.shp'), epsg=26715)
    assert np.array_equal(ms3.sr.get_vertices(nrow - 1, 0)[1],
                          [ms3.sr.xll, ms3.sr.yll])
    one = ms3.sr.get_vertices(nrow - 1, 0)[1]
    two = ms2.sr.get_vertices(nrow - 1, 0)[1]

    # assert np.array_equal(ms3.sr.get_vertices(nrow - 1, 0)[1],
    #                      ms2.sr.get_vertices(nrow - 1, 0)[1])
    assert np.allclose(ms3.sr.get_vertices(nrow - 1, 0)[1],
                       ms2.sr.get_vertices(nrow - 1, 0)[1])

    xur, yur = ms3.sr.get_vertices(0, ncol - 1)[3]
    assert xur == xll + ms3.sr.length_multiplier * delr * ncol
    assert yur == yll + ms3.sr.length_multiplier * delc * nrow


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
                                       delc=ms2.dis.delc.array, lenuni=3,
                                       xll=xll, yll=yll, rotation=30)
    xul, yul = sr1.xul, sr1.yul
    sr1.length_multiplier = 1.0 / 3.281
    assert sr1.xll == xll
    assert sr1.yll == yll
    sr2 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=3,
                                       xul=xul, yul=yul, rotation=30)
    sr2.length_multiplier = 1.0 / 3.281
    assert sr2.xul == xul
    assert sr2.yul == yul

    # test resetting of attributes
    sr3 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=3,
                                       xll=xll, yll=yll, rotation=30)
    # check that xul, yul and xll, yll are being recomputed
    sr3.xll += 10.
    sr3.yll += 21.
    assert sr3.xul - (xul + 10.) < 1e-6
    assert sr3.yul - (yul + 21.) < 1e-6
    sr4 = flopy.utils.SpatialReference(delr=ms2.dis.delr.array,
                                       delc=ms2.dis.delc.array, lenuni=3,
                                       xul=xul, yul=yul, rotation=30)
    sr4.xul += 10.
    sr4.yul += 21.
    assert sr4.xll - (xll + 10.) < 1e-6
    assert sr4.yll - (yll + 21.) < 1e-6
    sr4.rotation = 0.
    assert sr4.xll == sr4.xul
    assert sr4.yll == sr4.yul - sr4.yedge[0]
    assert True

def test_namfile_readwrite():
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    xll, yll = 272300, 5086000
    fm = flopy.modflow
    m = fm.Modflow(modelname='junk', model_ws=os.path.join('temp', 't007'))
    dis = fm.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                        delc=delc)
    m.sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                        delc=m.dis.delc.array, lenuni=3,
                                        length_multiplier=.3048,
                                        xll=xll, yll=yll, rotation=30)

    # test reading and writing of SR information to namfile
    m.write_input()
    m2 = fm.Modflow.load('junk.nam', model_ws=os.path.join('temp', 't007'))
    assert abs(m2.sr.xll - xll) < 1e-2
    assert abs(m2.sr.yll - yll) < 1e-2
    assert m2.sr.rotation == 30
    assert abs(m2.sr.length_multiplier - .3048) < 1e-10


def test_rotation():
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    xul, yul = 500000, 2934000
    m.sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                        delc=m.dis.delc.array,
                                        xul=xul, yul=yul, rotation=45.)
    xll, yll = m.sr.xll, m.sr.yll
    assert m.dis.sr.xgrid[0, 0] == xul
    assert m.dis.sr.ygrid[0, 0] == yul
    m.sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                        delc=m.dis.delc.array,
                                        xul=xul, yul=yul, rotation=-45.)
    assert m.dis.sr.xgrid[0, 0] == xul
    assert m.dis.sr.ygrid[0, 0] == yul
    xll2, yll2 = m.sr.xll, m.sr.yll
    m.sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                        delc=m.dis.delc.array,
                                        xll=xll2, yll=yll2, rotation=-45.)
    assert m.dis.sr.xgrid[0, 0] == xul
    assert m.dis.sr.ygrid[0, 0] == yul
    m.sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                        delc=m.dis.delc.array,
                                        xll=xll, yll=yll, rotation=45.)
    assert m.dis.sr.xgrid[0, 0] == xul
    assert m.dis.sr.ygrid[0, 0] == yul


def test_map_rotation():
    m = flopy.modflow.Modflow(rotation=20.)
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=40, ncol=20,
                                   delr=250.,
                                   delc=250., top=10, botm=0)
    # transformation assigned by arguments
    xul, yul, rotation = 500000, 2934000, 45
    modelmap = flopy.plot.ModelMap(model=m, xul=xul, yul=yul,
                                   rotation=rotation)
    lc = modelmap.plot_grid()
    xll, yll = modelmap.sr.xll, modelmap.sr.yll

    def check_vertices():
        xllp, yllp = lc._paths[0].vertices[0]
        xulp, yulp = lc._paths[0].vertices[1]
        assert (xllp, yllp) == (xll, yll)
        assert (xulp, yulp) == (xul, yul)

    check_vertices()

    modelmap = flopy.plot.ModelMap(model=m, xll=xll, yll=yll,
                                   rotation=rotation)
    lc = modelmap.plot_grid()
    check_vertices()

    # transformation in m.sr
    sr = flopy.utils.SpatialReference(delr=m.dis.delr.array,
                                      delc=m.dis.delc.array,
                                      xll=xll, yll=yll, rotation=rotation)
    m.sr = copy.deepcopy(sr)
    modelmap = flopy.plot.ModelMap(model=m)
    lc = modelmap.plot_grid()
    check_vertices()

    # transformation assign from sr instance
    m.sr._reset()
    m.sr.set_spatialreference()
    modelmap = flopy.plot.ModelMap(model=m, sr=sr)
    lc = modelmap.plot_grid()
    check_vertices()


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
    test_shapefile()
    # test_shapefile_ibound()
    # test_netcdf_overloads()
    #test_netcdf_classmethods()
    # build_netcdf()
    # build_sfr_netcdf()
    #test_sr()
    #test_mbase_sr()
    #test_rotation()
    #test_map_rotation()
    #test_sr_scaling()
    #test_dynamic_xll_yll()
    test_namfile_readwrite()
    # test_free_format_flag()
    # test_export_output()
    #for namfile in namfiles:
    # for namfile in ["fhb.nam"]:
    # export_netcdf(namfile)
    #    export_shapefile(namfile)
