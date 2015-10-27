# Test export modeule

def test_netcdf():
    import os
    import flopy

    model_ws = os.path.join("..", "examples", "data", "freyberg")
    nam = "freyberg"

    ml = flopy.modflow.Modflow.load(nam,model_ws=model_ws)
    ml.dis.sr.xul = 1000.0
    ml.dis.sr.yul = 2000.0
    ml.dis.sr.rotation = 15.0

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
    except:
        return

    fnc = ml.export(os.path.join("temp","test.nc"))
    hk = fnc.nc.variables["hk"]
    assert fnc.nc.variables['hk'].shape == ml.lpf.hk.shape
    fnc.write()


if __name__ == '__main__':
    test_netcdf()