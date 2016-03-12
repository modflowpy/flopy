# Test export module
import os
import flopy

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]
skip = ["MNW2-Fig28.nam","testsfr2.nam","testsfr2_tab.nam"]

def export_netcdf(namfile):
    if namfile in skip:
        return
    print(namfile)
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    if m.dis.lenuni == 0:
        m.dis.lenuni = 1
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

    assert ms.sr.yul == 100
    ms.sr.xul = 111
    assert ms.sr.xul == 111

    ms.sr.units = "feet"
    assert ms.sr.units == "feet"

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


def test_shapefile():
    for namfile in namfiles:
        yield export_shapefile, namfile
    return

def test_netcdf():
    for namfile in namfiles:
        yield export_netcdf, namfile
    return

if __name__ == '__main__':
    test_sr()
    #test_free_format_flag()
    #test_export_output()
    #for namfile in namfiles:
    #for namfile in ["fhb.nam"]:
        #export_netcdf(namfile)
        #export_shapefile(namfile)