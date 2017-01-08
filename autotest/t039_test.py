"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import CellBudgetFile, ZoneBudget, MfListBudget, read_zbarray, write_zbarray

loadpth = os.path.join('..', 'examples', 'data', 'zonbud_examples')
outpth = os.path.join('temp', 't038')

if not os.path.isdir(outpth):
    os.makedirs(outpth)


def read_zonebudget_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        items = line.split(',')

        # Read time step information for this block
        if "Time Step" in line:
            kstp, kper, totim = int(items[1]) - 1, int(items[3]) - 1, float(items[5])
            continue

        # Get names of zones
        elif 'ZONE' in items[1]:
            zonenames = [i.strip() for i in items[1:-1]]
            zonenames = ['_'.join(z.split()) for z in zonenames]
            continue

        # Set flow direction flag--inflow
        elif 'IN' in items[1]:
            flow_dir = 'IN'
            continue

        # Set flow direction flag--outflow
        elif 'OUT' in items[1]:
            flow_dir = 'OUT'
            continue

        # Get mass-balance information for this block
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in items[0]:
            continue

        # End of block
        elif items[0] == '' and items[1] == '\n':
            continue

        record = '_'.join(items[0].strip().split()) + '_{}'.format(flow_dir)
        if record.startswith(('FROM_', 'TO_')):
            record = '_'.join(record.split('_')[1:])
        vals = [float(i) for i in items[1:-1]]
        row = (totim, kstp, kper, record,) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [('totim', np.float),
                  ('time_step', np.int),
                  ('stress_period', np.int),
                  ('name', '<U50')] + [(z, '<f8') for z in zonenames]
    dtype = np.dtype(dtype_list)
    return np.array(rows, dtype=dtype)


def test_compare2zonebudget(rtol=1e-2):
    """
    Compares output from zonbud.exe to the budget calculated by zonbud utility
    using the multilayer transient freyberg model.
    """
    zonebudget_recarray = read_zonebudget_file(os.path.join(loadpth,
                                                            'zonebudget_mlt.csv'))

    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    cbc_fname = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zb = ZoneBudget(cbc_fname, zon)
    zbutil_recarray = zb.get_budget()

    times = np.unique(zonebudget_recarray['totim'])
    zonenames = [n for n in zonebudget_recarray.dtype.names if 'ZONE' in n]

    for time in times:
        zb_arr = zonebudget_recarray[zonebudget_recarray['totim'] == time]
        zbu_arr = zbutil_recarray[zbutil_recarray['totim'] == time]
        for name in zb_arr['name']:
            r1 = np.where((zb_arr['name'] == name))
            r2 = np.where((zbu_arr['name'] == name))
            a1 = np.array([v for v in zb_arr[zonenames][r1[0]][0]])
            a2 = np.array([v for v in zbu_arr[zonenames][r2[0]][0]])
            allclose = np.allclose(a1, a2, rtol)
            s = 'Zonebudget arrays do not match at time {} ({}).'.format(time, name)
            assert allclose, s
    return


# def test_comare2mflist_mlt(rtol=1e-2):
#
#     loadpth = os.path.join('..', 'examples', 'data', 'zonbud_examples', 'freyberg_mlt')
#
#     list_f = os.path.join(loadpth, 'freyberg.list')
#     mflistbud = MfListBudget(list_f)
#     print(help(mflistbud))
#     mflistrecs = mflistbud.get_data(idx=-1, incremental=True)
#     print(repr(mflistrecs))
#
#     zon = np.ones((3, 40, 20), dtype=np.int)
#     cbc_fname = os.path.join(loadpth, 'freyberg.cbc')
#     kstp, kper = CellBudgetFile(cbc_fname).get_kstpkper()[-1]
#     zb = ZoneBudget(cbc_fname, zon, kstpkper=(kstp, kper))
#     zbrecs = zb.get_budget()
#     print(repr(zbrecs))
#     return


def test_zonbud_get_record_names():
    """
    Test zonbud get_record_names method
    """
    cbc_f = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 0))
    recnames = zb.get_record_names()
    assert len(recnames) > 0, 'No record names returned.'
    return


def test_zonbud_aliases():
    """
    Test zonbud aliases
    """
    cbc_f = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Wilson', 0: 'Carini'}
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096), aliases=aliases)
    bud = zb.get_budget()
    m = bud['name'] == 'Mike_IN'
    assert bud[m].shape[0] > 0, 'No records returned.'
    return


def test_zonbud_to_csv():
    """
    Test zonbud export to csv file method
    """
    cbc_f = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    zb = ZoneBudget(cbc_f, zon, kstpkper=[(0, 1094), (0, 1096)])
    zb.to_csv(os.path.join(outpth, 'test.csv'))
    with open(os.path.join(outpth, 'test.csv'), 'r') as f:
        lines = f.readlines()
    assert len(lines) > 0, 'No data written to csv file.'
    return


def test_zonbud_math():
    """
    Test zonbud math methods
    """
    cbc_f = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    cmd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cmd / 35.3147
    cmd * 12.
    return


def test_zonbud_copy():
    """
    Test zonbud copy
    """
    cbc_f = os.path.join(loadpth, 'freyberg_mlt', 'freyberg.gitcbc')
    zon = read_zbarray(os.path.join(loadpth, 'zonef_mlt'))
    cfd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cfd2 = cfd.copy()
    assert cfd is not cfd2, 'Copied object is a shallow copy.'
    return


def test_zonbud_readwrite_zbarray():
    """
    Test zonbud read write
    """
    x = np.random.randint(100, 200, size=(5, 150, 200))
    write_zbarray(os.path.join(outpth, 'randint'), x)
    write_zbarray(os.path.join(outpth, 'randint'), x, fmtin=35, iprn=2)
    z = read_zbarray(os.path.join(outpth, 'randint'))
    assert np.array_equal(x, z), 'Input and output arrays do not match.'
    return


if __name__ == '__main__':
    # test_comare2mflist_mlt()
    test_compare2zonebudget()
    test_zonbud_aliases()
    test_zonbud_to_csv()
    test_zonbud_math()
    test_zonbud_copy()
    test_zonbud_readwrite_zbarray()
    test_zonbud_get_record_names()
