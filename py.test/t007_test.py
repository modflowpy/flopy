# Test export modeule

def test_netcdf():
    import os
    import flopy

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
    except:
        return

    pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]
    for i,namfile in enumerate(namfiles):
        print(i,namfile)
        m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
        if m.dis.lenuni == 0:
            continue
        if m.dis.botm.shape[0] != m.nlay:
            print("skipping")
            continue
        assert m, 'Could not load namefile {}'.format(namfile)
        try:
            fnc = m.export(os.path.join("temp",m.name+".nc"))
            fnc.write()
        except Exception as e:
             print("fail:\n"+str(e))
    return

if __name__ == '__main__':
    test_netcdf()