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
    if m.dis.sr.lenuni == 0:
        m.dis.sr.lenuni = 1
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

    try:
        fnc = m.export(os.path.join('temp', m.name + '.nc'))
        fnc.write()
    except Exception as e:
        raise Exception('ncdf export fail for namfile {0}:\n{1}  '.format(namfile,str(e)))
    return


def test_well_flux_extras():
    import os
    import flopy
    ml = flopy.modflow.Modflow(model_ws="temp")
    dis = flopy.modflow.ModflowDis(ml,10,10,10,10)
    sp_data = {0: [[1, 1, 1, 1.0], [1, 1, 2, 2.0], [1, 1, 3, 3.0]],1:[1,2,4,4.0]}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data)
    wel.export(os.path.join("temp","wel_test.nc"))



def test_netcdf():
    for namfile in namfiles:
        yield export_netcdf, namfile
    return

if __name__ == '__main__':
    for namfile in namfiles:
    #for namfile in ["fhb.nam"]:
        export_netcdf(namfile)
    test_well_flux_extras()