"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import CellBudgetFile, ZoneBudget, read_zbarray, write_zbarray

loadpth = os.path.join('..', 'examples', 'data', 'zonbud_examples')
outpth = os.path.join('temp', 't038')

if not os.path.isdir(outpth):
    os.makedirs(outpth)


def test_compare2zonebudget(rtol=1e-2):
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
        if record.startswith(('FROM_', 'TO_')):
            record = '_'.join(record.split('_')[1:])
        vals = [float(i) for i in items[1:-1]]
        row = (record,) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [('record', '<U50')] + [(z, '<f8') for z in zonenames]
    dtype = np.dtype(dtype_list)
    zonebudget_recarray = np.array(rows, dtype=dtype)

    zon = read_zbarray(os.path.join(loadpth, 'zonef'))
    cbc_fname = os.path.join(loadpth, 'freyberg.cbc')
    zb = ZoneBudget(cbc_fname, zon, kstpkper=(kstp, kper))
    zbudutil_recarray = zb.get_budget()[0]

    for recname in zonebudget_recarray['record']:
        r1 = np.where((zbudutil_recarray['record'] == recname))
        r2 = np.where((zonebudget_recarray['record'] == recname))
        a1 = np.array([v for v in zbudutil_recarray[zonenames][r1[0]][0]])
        a2 = np.array([v for v in zonebudget_recarray[zonenames][r2[0]][0]])
        assert a1.shape == a2.shape, 'Array shapes do not match.'
        isclose = np.allclose(a1, a2, rtol)
        assert isclose, 'Zonebudget arrays do not match within a tolerance of {}.'.format(rtol)
    return


def test_compare2zonebudget_mlt(rtol=1e-2):
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
        elif items[1].strip() == 'IN':
            flow_dir = 'IN'
            continue
        elif items[1].strip() == 'OUT':
            flow_dir = 'OUT'
            continue
        elif 'Total' in items[0] or 'IN-OUT' in items[0]:
            record = '_'.join(items[0].strip().upper().split())
        elif 'Percent Error' in items[0]:
            record = 'PERCENT_DISCREPANCY'
        elif items[0] == '' and items[1] == '\n':
            break
        else:
            record = '_'.join(items[0].strip().split()) + '_{}'.format(flow_dir)
            if record.startswith(('FROM_', 'TO_')):
                record = '_'.join(record.split('_')[1:])
        vals = [float(i) for i in items[1:-1]]
        row = (record,) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [('record', '<U50')] + [(z, '<f8') for z in zonenames]
    dtype = np.dtype(dtype_list)
    zonebudget_recarray = np.array(rows, dtype=dtype)

    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    cbc_fname = os.path.join(loadpth, 'freyberg_mlt.cbc')
    zb = ZoneBudget(cbc_fname, zon, kstpkper=(kstp, kper))
    zbudutil_recarray = zb.get_budget()[0]

    for recname in zonebudget_recarray['record']:
        if recname in ['IN-OUT', 'PERCENT_DISCREPANCY']:
            # Skip these, may not match due to presicion
            # of the zonebudget output file
            continue
        r1 = np.where((zbudutil_recarray['record'] == recname))
        r2 = np.where((zonebudget_recarray['record'] == recname))
        a1 = np.array([v for v in zbudutil_recarray[zonenames][r1[0]][0]])
        a2 = np.array([v for v in zonebudget_recarray[zonenames][r2[0]][0]])
        assert a1.shape == a2.shape, 'Array shapes do not match ({}).'.format(recname)
        isclose = np.allclose(a1, a2, rtol)
        assert isclose, 'Zonebudget arrays do not match within a tolerance of {} {}.'.format(rtol, recname)
    return


def test_zonbud_get_budget():
    cbc_f = os.path.join(loadpth, 'freyberg_mlt.cbc')
    cbc = CellBudgetFile(cbc_f)
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    zb = ZoneBudget(cbc_f, zon)
    s = 'Number of records returned does not match the number requested.'
    assert len(zb.get_budget()) == len(cbc.get_kstpkper()), s
    return


def test_zonbud_aliases():
    cbc_f = 'freyberg.cbc'
    zon = read_zbarray(os.path.join(loadpth, 'zonef'))
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Wilson', 0: 'Carini'}
    zb = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0), aliases=aliases)
    bud = zb.get_budget()[0]
    m = bud['record'] == 'Mike_IN'
    assert bud[m].shape[0] > 0, 'No records returned.'
    return


def test_zonbud_copy():
    cbc_f = 'freyberg.cbc'
    zon = read_zbarray(os.path.join(loadpth, 'zonef'))
    cfd = ZoneBudget(os.path.join(loadpth, cbc_f), zon, kstpkper=(0, 0))
    cfd2 = cfd.copy()
    assert cfd is not cfd2, 'Copied object is a shallow copy.'
    return


def test_zonbud_readwrite_zbarray():
    x = np.random.randint(100, 200, size=(5, 150, 200))
    write_zbarray(os.path.join(outpth, 'randint'), x)
    write_zbarray(os.path.join(outpth, 'randint'), x, fmtin=35, iprn=2)
    z = read_zbarray(os.path.join(outpth, 'randint'))
    assert np.array_equal(x, z), 'Input and output arrays do not match.'
    return


def test_dataframes():
    cbc_f = os.path.join(loadpth, 'freyberg_mlt.cbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    zb = ZoneBudget(cbc_f, zon, totim=[1097.])
    df = zb.get_dataframes()
    assert(len(df)) > 0, 'No records returned.'
    df = zb.get_dataframes(start_datetime='1-1-1970')
    assert (len(df)) > 0, 'No records returned.'

def junk():
    from flopy.utils import MfListBudget
    listf = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient', 'freyberg.list')
    bud = MfListBudget(listf)
    inc, cum = bud.get_dataframes(start_datetime=None)
    print(cum.head())
    # print(bud.get_kstpkper())
    inc, cum = bud.get_budget()
    print(repr(inc))
    # print(bud.get_data(kstpkper=None))

    # zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    # cbc_fname = os.path.join('..', 'examples', 'data', 'freyberg_multilayer_transient', 'freyberg.cbc')
    # zbud = ZoneBudget(cbc_fname, zon, kstpkper=(0, 1096))
    # print(zbud.get_budget(recordlist=['CONSTANT_HEAD_IN']))
    # print(zbud.get_budget(recordlist=['CONSTANT_HEAD_IN']).sum())
    return
#
# def test_zonbud2():
#     zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
#     # zon = np.ones((3, 40, 20), np.int)
#     cbc_fname = os.path.join(loadpth, 'freyberg.cbc')
#     bud = ZoneBudget(cbc_fname, zon, totim=[1095., 1096., 1097.])
#     # print(len(bud))
#     a = bud.get_budget()[0]
#     # for i in range(a.shape[0]):
#     #     print(i, list(a[:][i]))
#     bud.to_csv(os.path.join(outpth, 'text.csv'))
#     mul = bud*-1
#     # print(repr(mul.get_budget()[-1]))
#     return

if __name__ == '__main__':
    test_compare2zonebudget()
    test_compare2zonebudget_mlt()
    test_zonbud_aliases()
    test_zonbud_get_budget()
    test_zonbud_copy()
    test_zonbud_readwrite_zbarray()
    test_dataframes()
