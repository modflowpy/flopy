import os
import numpy as np
import pandas as pd
from flopy.utils import ZoneBudget, CellBudgetFile
from flopy.utils.zonbud import arr2ascii


def create3D():
    print('Creating zones.')
    z = np.random.randint(8, size=(3, 40, 20))
    for i in range(3):
        np.savetxt('complex3d_{}'.format(i+1), z[i, :, :], fmt='%8i')
    arr2ascii('complex3d', z)

    x = np.zeros((3, 40, 20), dtype=np.int64)
    x[:] = z[0, :, :]
    for i in range(3):
        np.savetxt('simple3d_{}'.format(i+1), x[i, :, :], fmt='%8i')
    arr2ascii('simple3d', x)
    return


def load_complex3D():
    z = np.stack([np.loadtxt('complex3d_{}'.format(i + 1)) for i in range(3)])
    return z.astype(np.int64)


def load_simple3D():
    z = np.stack([np.loadtxt('simple3d_{}'.format(i + 1)) for i in range(3)])
    return z.astype(np.int64)


def run_simple3d():
    print('Running simple zones.')
    z = load_simple3D()
    zb = ZoneBudget('freyberg.cbc')
    zbud = zb.get_budget(z, kstpkper=zb.get_kstpkper()[10])
    zbud.to_csv('simple3d_py.csv')
    return


def run_complex3d():
    print('Running complex zones.')
    z = load_complex3D()
    zb = ZoneBudget('freyberg.cbc')
    zbud = zb.get_budget(z, kstpkper=zb.get_kstpkper()[10])
    zbud.to_csv('complex3d_py.csv')


def __initialize__():
    create3D()
    run_simple3d()
    run_complex3d()


def main():
    run_complex3d()
    return


if __name__ == '__main__':
    main()
