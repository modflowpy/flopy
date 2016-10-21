"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import ZoneBudget
from flopy.utils.zonbud import read_zbarray, write_zbarray

pth = '../examples/data/zonbud_examples'
cbc_f = 'freyberg.gitcbc'

# nrow, ncol = 40, 20
# zon = np.zeros((1, nrow, ncol), np.int)
# zon[0, :20, :10] = 1
# zon[0, :20, 10:] = 2
# zon[0, 20:, :10] = 3
# zon[0, 20:, 10:] = 4
zon = np.random.randint(1, 6, size=(1, 40, 20))


def test_zonbud_write_csv_kstpkper():
    zbud = ZoneBudget(os.path.join(pth, cbc_f), zon, kstpkper=(0, 0))
    zbud.to_csv(os.path.join(pth, 'zbud_zonbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')
    return


def test_zonbud_write_csv_totim():
    zbud = ZoneBudget(os.path.join(pth, cbc_f), zon, totim=10.)
    zbud.to_csv(os.path.join(pth, 'zbud_zonbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')
    return


def test_zonbud_budget():
    zbud = ZoneBudget(os.path.join(pth, cbc_f), zon, kstpkper=(0, 0))
    recs = zbud.get_records()
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    recordlist = [('IN', 'CONSTANT HEAD'), ('IN', 'FROM ZONE 1')]
    recs = zbud.get_records(recordlist=recordlist, zones=[1, 3])
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    return


def test_zonbud_mass_balance():
    zbud = ZoneBudget(os.path.join(pth, cbc_f), zon, kstpkper=(0, 0))
    recs = zbud.get_mass_balance()
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    recs = zbud.get_total_outflow(zones=3)
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    recs = zbud.get_total_inflow(zones=(1, 2))
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    recs = zbud.get_percent_error()
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    return


def test_zonbud_aliases():
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Page', 0: 'Carini'}
    zbud = ZoneBudget(os.path.join(pth, cbc_f), zon, kstpkper=(0, 0), aliases=aliases)
    zbud.to_csv(os.path.join(pth, 'zbud_aliases.csv'), write_format='zonbud')
    recs = zbud.get_records()
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    recordlist = [('IN', 'FROM Mike')]
    recs = zbud.get_records(recordlist=recordlist, zones=['Trey', 3])
    if recs.shape[0] == 0:
        raise Exception('No records returned.')
    return


def test_zonbud_mult():
    cfd = ZoneBudget(os.path.join(pth, cbc_f), zon, kstpkper=(0, 0))
    cfd.to_csv(os.path.join(pth, 'zbud_zonbud.csv'))
    mgd = cfd*(7.48052/1000000)
    mgd.to_csv(os.path.join(pth, 'zbud_zonbud.csv'))
    cfd2 = cfd.copy()
    cfd2.to_csv(os.path.join(pth, 'zbud_zonbud.csv'))
    return


def test_zonbud_readwrite_zbarray():
    x = np.random.randint(100, 200, size=(5, 150, 200))
    write_zbarray(os.path.join(pth, 'randint'), x)
    write_zbarray(os.path.join(pth, 'randint'), x, fmtin=35, iprn=2)
    z = read_zbarray(os.path.join(pth, 'randint'))
    if not np.array_equal(x, z):
        raise Exception('Input and output arrays do not match.')
    return


if __name__ == '__main__':
    test_zonbud_write_csv_kstpkper()
    test_zonbud_write_csv_totim()
    test_zonbud_aliases()
    test_zonbud_budget()
    test_zonbud_mass_balance()
    test_zonbud_mult()
    test_zonbud_readwrite_zbarray()
