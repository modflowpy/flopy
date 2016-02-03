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

def test_shapefile():
    for namfile in namfiles:
        yield export_shapefile, namfile
    return

def test_netcdf():
    for namfile in namfiles:
        yield export_netcdf, namfile
    return

if __name__ == '__main__':
    #for namfile in namfiles:
    for namfile in ["fhb.nam"]:
        export_netcdf(namfile)
        #export_shapefile(namfile)