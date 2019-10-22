import os
import numpy as np
import flopy

tpth = os.path.join('temp', 't067')
if not os.path.isdir(tpth):
    os.makedirs(tpth)


def test_ulstrd():

    # Create an original model and then manually modify to use
    # advanced list reader capabilities
    ws = tpth
    nlay = 1
    nrow = 10
    ncol = 10
    nper = 3

    # create the ghbs
    ghbra = flopy.modflow.ModflowGhb.get_empty(20)
    l = 0
    for i in range(nrow):
        ghbra[l] = (0, i, 0, 1., 100. + i)
        l += 1
        ghbra[l] = (0, i, ncol - 1, 1., 200. + i)
        l += 1
    ghbspd = {0: ghbra}

    # create the drains
    drnra = flopy.modflow.ModflowDrn.get_empty(2)
    drnra[0] = (0, 1, int(ncol / 2), .5, 55.)
    drnra[1] = (0, 2, int(ncol / 2), .5, 75.)
    drnspd = {0: drnra}

    # create the wells
    welra = flopy.modflow.ModflowWel.get_empty(2)
    welra[0] = (0, 1, 1, -5.)
    welra[1] = (0, nrow - 3, ncol - 3, -10.)
    welspd = {0: welra}

    m = flopy.modflow.Modflow(modelname='original', model_ws=ws,
                              exe_name='mf2005')
    dis = flopy.modflow.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol,
                                   nper=nper)
    bas = flopy.modflow.ModflowBas(m)
    lpf = flopy.modflow.ModflowLpf(m)
    ghb = flopy.modflow.ModflowGhb(m, stress_period_data=ghbspd)
    drn = flopy.modflow.ModflowDrn(m, stress_period_data=drnspd)
    wel = flopy.modflow.ModflowWel(m, stress_period_data=welspd)
    pcg = flopy.modflow.ModflowPcg(m)
    oc = flopy.modflow.ModflowOc(m)
    m.add_external('original.drn.dat', 71)
    m.add_external('original.wel.bin', 72, binflag=True, output=False)
    m.write_input()

    # rewrite ghb
    fname = os.path.join(ws, 'original.ghb')
    with open(fname, 'w') as f:
        f.write('{} {}\n'.format(ghbra.shape[0], 0))
        for kper in range(nper):
            f.write('{} {}\n'.format(ghbra.shape[0], 0))
            f.write('open/close original.ghb.dat\n')

    # write ghb list
    sfacghb = 5
    fname = os.path.join(ws, 'original.ghb.dat')
    with open(fname, 'w') as f:
        f.write('sfac {}\n'.format(sfacghb))
        for k, i, j, stage, cond in ghbra:
            f.write('{} {} {} {} {}\n'.format(k + 1, i + 1, j + 1, stage, cond))

    # rewrite drn
    fname = os.path.join(ws, 'original.drn')
    with open(fname, 'w') as f:
        f.write('{} {}\n'.format(drnra.shape[0], 0))
        for kper in range(nper):
            f.write('{} {}\n'.format(drnra.shape[0], 0))
            f.write('external 71\n')

    # write drn list
    sfacdrn = 1.5
    fname = os.path.join(ws, 'original.drn.dat')
    with open(fname, 'w') as f:
        for kper in range(nper):
            f.write('sfac {}\n'.format(sfacdrn))
            for k, i, j, stage, cond in drnra:
                f.write(
                    '{} {} {} {} {}\n'.format(k + 1, i + 1, j + 1, stage, cond))

    # rewrite wel
    fname = os.path.join(ws, 'original.wel')
    with open(fname, 'w') as f:
        f.write('{} {}\n'.format(drnra.shape[0], 0))
        for kper in range(nper):
            f.write('{} {}\n'.format(drnra.shape[0], 0))
            f.write('external 72 (binary)\n')

    # create the wells, but use an all float dtype to write a binary file
    # use one-based values
    weldt = np.dtype([('k', '<f4'), ('i', '<f4'), ('j', '<f4'), ('q', '<f4'), ])
    welra = np.recarray(2, dtype=weldt)
    welra[0] = (1, 2, 2, -5.)
    welra[1] = (1, nrow - 2, ncol - 2, -10.)
    fname = os.path.join(ws, 'original.wel.bin')
    with open(fname, 'wb') as f:
        welra.tofile(f)
        welra.tofile(f)
        welra.tofile(f)

    # no need to run the model
    #success, buff = m.run_model(silent=True)
    #assert success, 'model did not terminate successfully'

    # the m2 model will load all of these external files, possibly using sfac
    # and just create regular list input files for wel, drn, and ghb
    fname = 'original.nam'
    m2 = flopy.modflow.Modflow.load(fname, model_ws=ws, verbose=False)
    m2.name = 'new'
    m2.write_input()

    originalghbra = m.ghb.stress_period_data[0].copy()
    originalghbra['cond'] *= sfacghb
    assert np.array_equal(originalghbra, m2.ghb.stress_period_data[0])

    originaldrnra = m.drn.stress_period_data[0].copy()
    originaldrnra['cond'] *= sfacdrn
    assert np.array_equal(originaldrnra, m2.drn.stress_period_data[0])

    originalwelra = m.wel.stress_period_data[0].copy()
    assert np.array_equal(originalwelra, m2.wel.stress_period_data[0])

    return


if __name__ == '__main__':
    test_ulstrd()
