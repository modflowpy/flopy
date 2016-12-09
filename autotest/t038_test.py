"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import ZoneBudget, read_zbarray, write_zbarray, MfListBudget

loadpth = os.path.join('..', 'examples', 'data', 'zonbud_examples')
outpth = os.path.join('temp', 't038')
cbc_f = 'freyberg.gitcbc'

if not os.path.isdir(outpth):
    os.makedirs(outpth)

# nrow, ncol = 40, 20
# zon = np.zeros((1, nrow, ncol), np.int)
# zon[0, :20, :10] = 1
# zon[0, :20, 10:] = 2
# zon[0, 20:, :10] = 3
# zon[0, 20:, 10:] = 4
zon = np.random.randint(1, 6, size=(1, 40, 20))


def test_compare_results_2_zonebudget(rtol=1e-5):
    """
    Compares output from zonbud.exe to the budget calculated by the zonbud utility
    using the single-layer freyberg model.
    """
    fname = os.path.join(loadpth, 'freyberg_zonebudget.csv')
    with open(fname, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        items = line.split(',')
        if "Time Step" in line:
            kstp, kper, totim = int(items[1]) - 1, int(items[3]) - 1, float(items[5])
            continue
        elif 'ZONE' in items[1]:
            zonenames = [i.strip() for i in items[1:-1]]
            zonenames = ['_'.join(z.split()) for z in zonenames]
            continue
        elif 'IN' in items[1]:
            flow_dir = 'IN'
            continue
        elif 'OUT' in items[1]:
            flow_dir = 'OUT'
            continue
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in items[0]:
            continue
        elif items[0] == '' and items[1] == '\n':
            break
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in items[0]:
            continue
        record = '_'.join(items[0].strip().split()) + '_{}'.format(flow_dir)
        vals = [float(i) for i in items[1:-1]]
        row = (flow_dir, record) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [('flow_dir', '<U3'), ('record', '<U50')] + [(z, '<f8') for z in zonenames]
    dtype = np.dtype(dtype_list)
    zonebudget_recarray = np.array(rows, dtype=dtype)

    zon = read_zbarray(os.path.join(loadpth, 'zonef'))
    cbc_fname = os.path.join(loadpth, 'freyberg.gitcbc')
    zbud = ZoneBudget(cbc_fname, zon, kstpkper=(kstp, kper))
    zbudutil_recarray = zbud.recordarray

    flow_dirs = zonebudget_recarray['flow_dir']
    records = zonebudget_recarray['record']

    for flowdir, recname in zip(flow_dirs, records):
        r1 = np.where((zbudutil_recarray['flow_dir'] == flowdir) &
                      (zbudutil_recarray['record'] == recname))
        r2 = np.where((zonebudget_recarray['flow_dir'] == flowdir) &
                      (zonebudget_recarray['record'] == recname))

        a1 = np.array([v for v in zbudutil_recarray[zonenames][r1[0]][0]])
        a2 = np.array([v for v in zonebudget_recarray[zonenames][r2[0]][0]])
        assert a1.shape == a2.shape, 'Array shapes do not match.'
        isclose = np.allclose(a1, a2, rtol)
        assert isclose, 'Zonebudget arrays do not match within a tolerance of {}.'.format(rtol)
    return


def test_compare_mlt_results_2_zonebudget(rtol=1e-5):
    """
    Compares output from zonbud.exe to the budget calculated by the zonbud utility
    using the multilayer transient freyberg model.
    """
    fname = os.path.join(loadpth, 'freyberg_mlt_zonebudget.csv')
    with open(fname, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        items = line.split(',')
        if "Time Step" in line:
            kstp, kper, totim = int(items[1])-1, int(items[3])-1, float(items[5])
            continue
        elif 'ZONE' in items[1]:
            zonenames = [i.strip() for i in items[1:-1]]
            zonenames = ['_'.join(z.split()) for z in zonenames]
            continue
        elif 'IN' in items[1]:
            flow_dir = 'IN'
            continue
        elif 'OUT' in items[1]:
            flow_dir = 'OUT'
            continue
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in items[0]:
            continue
        elif items[0] == '' and items[1] == '\n':
            break
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in items[0]:
            continue
        record = '_'.join(items[0].strip().split()) + '_{}'.format(flow_dir)
        vals = [float(i) for i in items[1:-1]]
        row = (flow_dir, record) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [('flow_dir', '<U3'), ('record', '<U50')] + [(z, '<f8') for z in zonenames]
    dtype = np.dtype(dtype_list)
    zonebudget_recarray = np.array(rows, dtype=dtype)

    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    cbc_fname = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient', 'freyberg.cbc')
    zbud = ZoneBudget(cbc_fname, zon, kstpkper=(kstp, kper))
    zbudutil_recarray = zbud.recordarray

    flow_dirs = zonebudget_recarray['flow_dir']
    records = zonebudget_recarray['record']

    for flowdir, recname in zip(flow_dirs, records):
        r1 = np.where((zbudutil_recarray['flow_dir'] == flowdir) &
                      (zbudutil_recarray['record'] == recname))
        r2 = np.where((zonebudget_recarray['flow_dir'] == flowdir) &
                      (zonebudget_recarray['record'] == recname))
        a1 = np.array([v for v in zbudutil_recarray[zonenames][r1[0]][0]])
        a2 = np.array([v for v in zonebudget_recarray[zonenames][r2[0]][0]])
        assert a1.shape == a2.shape, 'Array shapes do not match.'
        isclose = np.allclose(a1, a2, rtol)
        assert isclose, 'Zonebudget arrays do not match within a tolerance of {}.'.format(rtol)
    return


def test_zonbud_write_csv_kstpkper():
    zbud = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0))
    zbud.to_csv(os.path.join(outpth, 'zbud_zonbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(outpth, 'zbud_pandas.csv'), write_format='pandas')
    return


def test_zonbud_write_csv_totim():
    zbud = ZoneBudget(os.path.join(loadpth, cbc_f), zon, totim=10.)
    zbud.to_csv(os.path.join(outpth, 'zbud_zonbud.csv'), write_format='zonbud')
    zbud.to_csv(os.path.join(outpth, 'zbud_pandas.csv'), write_format='pandas')
    return


def test_zonbud_budget():
    zbud = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0))
    assert zbud.get_records().shape[0] > 0, 'No records returned.'
    recordlist = ['CONSTANT_HEAD_IN', 'FROM_ZONE_1_IN']
    assert zbud.get_records(recordlist=recordlist, zones=[1, 3]).shape[0] > 0, 'No records returned.'
    return


def test_zonbud_mass_balance():
    zbud = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0))
    assert zbud.get_mass_balance().shape[0] > 0, 'No records returned.'
    assert zbud.get_total_outflow(zones=3).shape[0] > 0, 'No records returned.'
    assert zbud.get_total_inflow(zones=(1, 2)).shape[0] > 0, 'No records returned.'
    assert zbud.get_percent_error().shape[0] > 0, 'No records returned.'
    return


def test_zonbud_aliases():
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Page', 0: 'Carini'}
    zbud = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0), aliases=aliases)
    zbud.to_csv(os.path.join(outpth, 'zbud_aliases.csv'), write_format='zonbud')
    assert zbud.get_records().shape[0] > 0, 'No records returned.'
    recordlist = ['FROM_Mike_IN']
    assert zbud.get_records(recordlist=recordlist, zones=['Trey', 3]).shape[0] > 0, 'No records returned.'
    return


def test_zonbud_mult():
    cfd = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0))
    cfd.to_csv(os.path.join(outpth, 'zbud_zonbud.csv'))
    mgd = cfd*(7.48052/1000000)
    mgd.to_csv(os.path.join(outpth, 'zbud_zonbud.csv'))
    cfd2 = cfd.copy()
    assert cfd.recordarray is not cfd2.recordarray, 'Copied object is a shallow copy.'
    cfd2.to_csv(os.path.join(outpth, 'zbud_zonbud.csv'))
    cfd2 / 5
    return


def test_zonbud_readwrite_zbarray():
    x = np.random.randint(100, 200, size=(5, 150, 200))
    write_zbarray(os.path.join(outpth, 'randint'), x)
    write_zbarray(os.path.join(outpth, 'randint'), x, fmtin=35, iprn=2)
    z = read_zbarray(os.path.join(outpth, 'randint'))
    assert np.array_equal(x, z), 'Input and output arrays do not match.'
    return


def junk():
    listf = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient', 'freyberg.list')
    bud = MfListBudget(listf)
    # print(bud.get_record_names())
    # print(bud.get_kstpkper())
    print(bud.get_budget(names=['CONSTANT_HEAD_IN']))
    # print(bud.get_data(kstpkper=None))
    return

if __name__ == '__main__':
    # junk()
    test_compare_results_2_zonebudget()
    test_compare_mlt_results_2_zonebudget()
    test_zonbud_write_csv_kstpkper()
    test_zonbud_write_csv_totim()
    test_zonbud_aliases()
    test_zonbud_budget()
    test_zonbud_mass_balance()
    test_zonbud_mult()
    test_zonbud_readwrite_zbarray()
