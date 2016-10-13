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
zon[0, :nrow/2, :ncol/2] = 1
zon[0, :nrow/2, ncol/2:] = 2
zon[0, nrow/2:, :ncol/2] = 3
zon[0, nrow/2:, ncol/2:] = 4


def test_zonbud1():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, kstpkper=zb.get_kstpkper()[-1])

    zbud.get_total_outflow()
    zbud.get_total_inflow()
    zbud.get_percent_error()

    zbud.to_csv(os.path.join(pth, 'zbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')


def test_zonbud2():

    zb = ZoneBudget(os.path.join(pth, cbc_f))
    zbud = zb.get_budget(zon, totim=zb.get_times()[-1])

    zbud.get_total_outflow()
    zbud.get_total_inflow()
    zbud.get_percent_error()

    zbud.to_csv(os.path.join(pth, 'zbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(pth, 'zbud_pandas.csv'), write_format='pandas')

if __name__ == '__main__':
    test_zonbud1()
    test_zonbud2()


