"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import ZoneBudget

pth = '../examples/data/zonbud_examples'
cbc_f = 'freyberg.gitcbc'

nrow, ncol = 40, 20
zon = np.zeros((1, nrow, ncol), np.int)
zon[0, :20, :10] = 1
zon[0, :20, 10:] = 2
zon[0, 20:, :10] = 3
zon[0, 20:, 10:] = 4


def test_zonbud_write_csv_kstpkper():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, kstpkper=zb.get_kstpkper()[-1])
    zbud.to_csv(os.path.join(pth, 'zbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')


def test_zonbud_write_csv_totim():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, totim=zb.get_times()[-1])
    zbud.to_csv(os.path.join(pth, 'zbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')


def test_zonbud_budget():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, kstpkper=zb.get_kstpkper()[-1])
    recordlist = [('in', 'CONSTANT HEAD'), ('in', 'FROM ZONE 1')]
    zbud.get_records(recordlist=recordlist, zones=[1, 3])
    zbud.get_records()


def test_zonbud_mass_balance():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, kstpkper=zb.get_kstpkper()[-1])

    zbud.get_mass_balance()
    zbud.get_total_outflow(zones=3)
    zbud.get_total_inflow(zones=[1, 2]).sum()
    zbud.get_percent_error()


if __name__ == '__main__':

    test_zonbud_write_csv_kstpkper()
    test_zonbud_write_csv_totim()
    test_zonbud_budget()
    test_zonbud_mass_balance()


