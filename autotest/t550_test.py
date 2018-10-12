import os, math
import numpy as np
import flopy
fm = flopy.modflow
fp6 = flopy.mf6
from flopy.discretization import StructuredGrid
from flopy.utils import SpatialReference as OGsr
from flopy.export.shapefile_utils import shp2recarray

tmpdir = 'temp/t550/'
if not os.path.isdir(tmpdir):
    os.makedirs(tmpdir)

def test_mf6_grid_shp_export():
    nlay = 2
    nrow = 10
    ncol = 10
    top = 1
    nper = 2
    perlen = 1
    nstp = 1
    tsmult = 1
    perioddata = [[perlen, nstp, tsmult]]*2
    botm=np.zeros((2, 10, 10))

    ogsr = OGsr(delc=np.ones(nrow), delr=np.ones(ncol),
                          xll=10, yll=10
                              )
    m = fm.Modflow('junk', version='mfnwt', model_ws=tmpdir)
    dis = fm.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol,
                        nper=nper, perlen=perlen, nstp=nstp,
                        tsmult=tsmult,
                        top=top, botm=botm)

    smg = StructuredGrid(delc=np.ones(nrow), delr=np.ones(ncol),
                         top=dis.top.array, botm=botm, idomain=1,
                         xoff=10, yoff=10)


    # River package (MFlist)
    spd = fm.ModflowRiv.get_empty(10)
    spd['i'] = np.arange(10)
    spd['j'] = [5, 5, 6, 6, 7, 7, 7, 8, 9, 9]
    spd['stage'] = np.linspace(1, 0.7, 10)
    spd['rbot'] = spd['stage'] - 0.1
    spd['cond'] = 50.
    riv = fm.ModflowRiv(m, stress_period_data={0: spd})

    # Recharge package (transient 2d)
    rech = {0: 0.001, 1: 0.002}
    rch = fm.ModflowRch(m, rech=rech)

    # mf6 version of same model
    mf6name = 'junk6'
    sim = fp6.MFSimulation(sim_name=mf6name, version='mf6', exe_name='mf6',
                           sim_ws=tmpdir)
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, pname='tdis', time_units='DAYS',
                                                nper=nper,
                                                perioddata=perioddata)
    gwf = fp6.ModflowGwf(sim, modelname=mf6name,
                               model_nam_file='{}.nam'.format(mf6name))
    dis6 = fp6.ModflowGwfdis(gwf, pname='dis', nlay=nlay, nrow=nrow, ncol=ncol,
                                  top=top,
                                  botm=botm)

    def cellid(k, i, j, nrow, ncol):
        return k*nrow*ncol + i*ncol + j

    # Riv6
    spd6 = fp6.ModflowGwfriv.stress_period_data.empty(gwf, maxbound=len(spd))
    #spd6[0]['cellid'] = cellid(spd.k, spd.i, spd.j, m.nrow, m.ncol)
    spd6[0]['cellid'] = list(zip(spd.k, spd.i, spd.j))
    for c in spd.dtype.names:
        if c in spd6[0].dtype.names:
            spd6[0][c] = spd[c]
    # MFTransient list apparently requires entries for additional stress periods,
    # even if they are the same
    spd6[1] = spd6[0]
    #irch = np.zeros((nrow, ncol))
    riv6 = fp6.ModflowGwfriv(gwf, stress_period_data=spd6)
    rch6 = fp6.ModflowGwfrcha(gwf, recharge=rech)
    #rch6.export('{}/mf6.shp'.format(tmpdir))
    m.export('{}/mfnwt.shp'.format(tmpdir))
    gwf.export('{}/mf6.shp'.format(tmpdir))

    riv6spdarrays = dict(riv6.stress_period_data.masked_4D_arrays_itr())
    rivspdarrays = dict(riv.stress_period_data.masked_4D_arrays_itr())
    for k, v in rivspdarrays.items():
        assert np.abs(np.nansum(v) - np.nansum(riv6spdarrays[k])) < 1e-6, "variable {} is not equal".format(k)
        pass

    # check that the two shapefiles are the same
    ra = shp2recarray('{}/mfnwt.shp'.format(tmpdir))
    ra6 = shp2recarray('{}/mf6.shp'.format(tmpdir))

    # check first and last exported cells
    assert ra.geometry[0] == ra6.geometry[0]
    assert ra.geometry[-1] == ra6.geometry[-1]
    # fields
    different_fields = list(set(ra.dtype.names).difference(ra6.dtype.names))
    different_fields = [f for f in different_fields
                        if 'thick' not in f
                        and 'rech' not in f]
    assert len(different_fields) == 0
    for l in np.arange(m.nlay)+1:
        assert np.sum(np.abs(ra['rech_{:03d}'.format(l)] - ra6['rechar{:03d}'.format(l)])) < 1e-6
    common_fields = set(ra.dtype.names).intersection(ra6.dtype.names)
    common_fields.remove('geometry')
    # array values
    for c in common_fields:
        for it, it6 in zip(ra[c], ra6[c]):
            if math.isnan(it):
                assert math.isnan(it6)
            else:
                assert np.abs(it - it6) < 1e-6
        pass


def test_huge_shapefile():
    nlay = 2
    nrow = 200
    ncol = 200
    top = 1
    nper = 2
    perlen = 1
    nstp = 1
    tsmult = 1
    perioddata = [[perlen, nstp, tsmult]] * 2
    botm = np.zeros((nlay, nrow, ncol))

    m = fm.Modflow('junk', version='mfnwt', model_ws=tmpdir)
    dis = fm.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol,
                        nper=nper, perlen=perlen, nstp=nstp,
                        tsmult=tsmult,
                        top=top, botm=botm)
    m.export('{}/huge.shp'.format(tmpdir))


if __name__ == '__main__':
    test_mf6_grid_shp_export()
    test_huge_shapefile()