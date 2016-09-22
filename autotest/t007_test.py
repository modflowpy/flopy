# Test export module
import os
import numpy as np
import flopy

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]
#skip = ["MNW2-Fig28.nam", "testsfr2.nam", "testsfr2_tab.nam"]
skip = []

def export_netcdf(namfile):
    if namfile in skip:
        return
    print(namfile)
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    if m.sr.lenuni == 0:
        m.sr.lenuni = 1
        #print('skipping...lenuni==0 (undefined)')
        #return
    #if sum(m.dis.laycbd) != 0:
    if m.dis.botm.shape[0] != m.nlay:
        print('skipping...botm.shape[0] != nlay')
        return
    assert m, 'Could not load namefile {}'.format(namfile)
    assert isinstance(m, flopy.modflow.Modflow)

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
    except:
        return
    fnc = m.export(os.path.join('temp', m.name + '.nc'))
    fnc.write()
    fnc_name = os.path.join('temp', m.name + '.nc')
    try:
        fnc = m.export(fnc_name)
        fnc.write()
    except Exception as e:
        raise Exception('ncdf export fail for namfile {0}:\n{1}  '.format(namfile,str(e)))
    try:
        nc = netCDF4.Dataset(fnc_name,'r')
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
    fnc_name = os.path.join('temp', m.name + '.shp')
    try:
        fnc = m.export(fnc_name)

    except Exception as e:
        raise Exception('shapefile export fail for namfile {0}:\n{1}  '.format(namfile,str(e)))
    try:
        s = shp.Reader(fnc_name)
    except Exception as e:
        raise Exception(' shapefile import fail for {0}:{1}'.format(fnc_name,str(e)))
    assert s.numRecords == m.nrow * m.ncol,"wrong number of records in " +\
                                           "shapefile {0}:{1:d}".format(fnc_name,s.numRecords)
    return


def test_export_output():
    import os
    import numpy as np
    import flopy

    model_ws = os.path.join("..","examples","data","freyberg")
    ml = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws)
    hds_pth = os.path.join(model_ws,"freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)

    out_pth = os.path.join("temp","freyberg.out.nc")
    nc = flopy.export.utils.output_helper(out_pth,ml,{"freyberg.githds":hds})
    var = nc.nc.variables.get("head")
    arr = var[:]
    ibound_mask = ml.bas6.ibound.array == 0
    arr_mask = arr.mask[0]
    assert np.array_equal(ibound_mask,arr_mask)


def test_mbase_sr():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(modelname="test",xul=1000.0,
                               rotation=12.5,start_datetime="1/1/2016")
    try:
        print(ml.sr.xcentergrid)
    except:
        pass
    else:
        raise Exception("should have failed")

    dis = flopy.modflow.ModflowDis(ml,nrow=10,ncol=5,delr=np.arange(5),xul=500)
    print(ml.sr)
    assert ml.sr.xul == 500
    assert ml.sr.yul == 10
    ml.model_ws = "temp"

    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam",model_ws="temp")
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
    dis = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                                   delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms,ifrefm=True)
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = False
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = True
    bas.ifrefm = False
    assert ms.free_format_input == bas.ifrefm
    bas.ifrefm = True
    assert ms.free_format_input == bas.ifrefm

    ms.model_ws = "temp"
    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile,model_ws=ms.model_ws)
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
    dis = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                                   delc=delc, top=top, botm=botm)
    bas = flopy.modflow.ModflowBas(ms,ifrefm=True)

    # test instantiation of an empty sr object
    sr = flopy.utils.reference.SpatialReference()

    sr = flopy.utils.SpatialReference(delr=ms.dis.delr.array,delc=ms.dis.delc.array,lenuni=3,
                                      xul=321,yul=123,rotation=20)
    assert ms.sr.yul == 100
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
    sr2 = flopy.utils.SpatialReference(delr=ms.dis.delr.array, delc=ms.dis.delc.array, lenuni=3,
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

    ms.model_ws = "temp"
    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile,model_ws=ms.model_ws)
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
    dis = flopy.modflow.ModflowDis(ms2, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                                   delc=delc)
    ms2.sr = flopy.utils.SpatialReference(delr=ms2.dis.delr.array, delc=ms2.dis.delc.array, lenuni=3,
                                          xll=xll, yll=yll, rotation=0)
    ms2.sr.epsg = 26715
    ms2.dis.export(os.path.join('temp', 'dis2.shp'))
    ms3 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ms3, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                                   delc=delc)
    ms3.sr = flopy.utils.SpatialReference(delr=ms3.dis.delr.array, delc=ms2.dis.delc.array, lenuni=3,
                                          length_multiplier=.3048,
                                          xll=xll, yll=yll, rotation=0)
    ms3.dis.export(os.path.join('temp', 'dis3.shp'), epsg=26715)
    assert np.array_equal(ms3.sr.get_vertices(nrow-1, 0)[1], [ms3.sr.xll, ms3.sr.yll])
    assert np.array_equal(ms3.sr.get_vertices(nrow-1, 0)[1], ms2.sr.get_vertices(nrow-1, 0)[1])
    xur, yur = ms3.sr.get_vertices(0, ncol-1)[3]
    assert xur == xll + ms3.sr.length_multiplier * delr * ncol
    assert yur == yll + ms3.sr.length_multiplier * delc * nrow

def test_netcdf_classmethods():
    import os
    import flopy
    nam_file = "freyberg.nam"
    model_ws = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient')
    ml = flopy.modflow.Modflow.load(nam_file,model_ws=model_ws,check=False,
                                    verbose=True,load_only=[])

    f = ml.export(os.path.join("temp","freyberg.nc"))
    v1_set = set(f.nc.variables.keys())
    new_f = flopy.export.NetCdf.zeros_like(f)
    v2_set = set(new_f.nc.variables.keys())
    diff = v1_set.symmetric_difference(v2_set)
    assert len(diff) == 0,str(diff)

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

    shape_name = os.path.join("temp","test.shp")
    nam_file = "freyberg.nam"
    model_ws = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient')
    ml = flopy.modflow.Modflow.load(nam_file,model_ws=model_ws,check=False,
                                    verbose=True,load_only=[])
    ml.export(shape_name)
    shp = shapefile.Reader(shape_name)
    field_names = [item[0] for item in shp.fields][1:]
    ib_idx = field_names.index("ibound_001")
    assert type(shp.record(0)[ib_idx]) == int,"should be int instead of {0}".\
        format(type(shp.record(0)[ib_idx]))


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
    #test_shapefile_ibound()
    #test_netcdf_overloads()
    #test_netcdf_classmethods()
    #build_netcdf()
    #build_sfr_netcdf()
    #test_sr()
    test_sr_scaling()
    #test_free_format_flag()
    #test_export_output()
    #for namfile in namfiles:
    #for namfile in ["fhb.nam"]:
        #export_netcdf(namfile)
        #export_shapefile(namfile)