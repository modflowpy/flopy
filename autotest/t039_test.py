"""
Test zonbud utility
"""
import os
import numpy as np
from flopy.utils import CellBudgetFile, ZoneBudget, \
    MfListBudget, read_zbarray, write_zbarray

loadpth = os.path.join('..', 'examples', 'data', 'zonbud_examples')
outpth = os.path.join('temp', 't039')
cbc_f = os.path.join(loadpth, 'freyberg.gitcbc')
zon_f = os.path.join(loadpth, 'zonef_mlt.zbr')
zbud_f = os.path.join(loadpth, 'freyberg_mlt.csv')

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
            kstp, kper, totim = int(items[1]) - 1, int(items[3]) - 1, \
                                float(items[5])
            continue

        # Get names of zones
        elif 'ZONE' in items[1]:
            zonenames = [i.strip() for i in items[1:-1]]
            zonenames = ['_'.join(z.split()) for z in zonenames]
            continue

        # Set flow direction flag--inflow
        elif 'IN' in items[1]:
            flow_dir = 'FROM'
            continue

        # Set flow direction flag--outflow
        elif 'OUT' in items[1]:
            flow_dir = 'TO'
            continue

        # Get mass-balance information for this block
        elif 'Total' in items[0] or 'IN-OUT' in items[0] or 'Percent Error' in \
                items[0]:
            continue

        # End of block
        elif items[0] == '' and items[1] == '\n':
            continue

        record = '{}_'.format(flow_dir) + '_'.join(items[0].strip().split())
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
    t039 Compare output from zonbud.exe to the budget calculated by zonbud
    utility using the multilayer transient freyberg model.
    """
    zba = read_zonebudget_file(zbud_f)
    zonenames = [n for n in zba.dtype.names if 'ZONE' in n]
    times = np.unique(zba['totim'])

    zon = read_zbarray(zon_f)
    zb = ZoneBudget(cbc_f, zon, totim=times, verbose=False)
    fpa = zb.get_budget()

    for time in times:
        zb_arr = zba[zba['totim'] == time]
        fp_arr = fpa[fpa['totim'] == time]
        for name in fp_arr['name']:
            r1 = np.where((zb_arr['name'] == name))
            r2 = np.where((fp_arr['name'] == name))
            if r1[0].shape[0] < 1 or r2[0].shape[0] < 1:
                continue
            if r1[0].shape[0] != r2[0].shape[0]:
                continue
            a1 = np.array([v for v in zb_arr[zonenames][r1[0]][0]])
            a2 = np.array([v for v in fp_arr[zonenames][r2[0]][0]])
            allclose = np.allclose(a1, a2, rtol)

            mxdiff = np.abs(a1 - a2).max()
            idxloc = np.argmax(np.abs(a1 - a2))
            # txt = '{}: {} - Max: {}  a1: {}  a2: {}'.format(time,
            #                                                 name,
            #                                                 mxdiff,
            #                                                 a1[idxloc],
            #                                                 a2[idxloc])
            # print(txt)
            s = 'Zonebudget arrays do not match at time {0} ({1}): {2}.' \
                .format(time, name, mxdiff)
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
    t039 Test zonbud get_record_names method
    """
    zon = read_zbarray(zon_f)
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 0))
    recnames = zb.get_record_names()
    assert len(recnames) > 0, 'No record names returned.'
    recnames = zb.get_record_names(stripped=True)
    assert len(recnames) > 0, 'No record names returned.'
    return


def test_zonbud_aliases():
    """
    t039 Test zonbud aliases
    """
    zon = read_zbarray(zon_f)
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Wilson', 0: 'Carini'}
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096), aliases=aliases, verbose=True)
    bud = zb.get_budget()
    assert bud[bud['name'] == 'FROM_Mike'].shape[0] > 0, 'No records returned.'
    return


def test_zonbud_to_csv():
    """
    t039 Test zonbud export to csv file method
    """
    zon = read_zbarray(zon_f)
    zb = ZoneBudget(cbc_f, zon, kstpkper=[(0, 1094), (0, 1096)])
    f_out = os.path.join(outpth, 'test.csv')
    zb.to_csv(f_out)
    with open(f_out, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 0, 'No data written to csv file.'
    return


def test_zonbud_math():
    """
    t039 Test zonbud math methods
    """
    zon = read_zbarray(zon_f)
    cmd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cmd / 35.3147
    cmd * 12.
    cmd + 1e6
    cmd - 1e6
    return


def test_zonbud_copy():
    """
    t039 Test zonbud copy
    """
    zon = read_zbarray(zon_f)
    cfd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cfd2 = cfd.copy()
    assert cfd is not cfd2, 'Copied object is a shallow copy.'
    return


def test_zonbud_readwrite_zbarray():
    """
    t039 Test zonbud read write
    """
    x = np.random.randint(100, 200, size=(5, 150, 200))
    write_zbarray(os.path.join(outpth, 'randint'), x)
    write_zbarray(os.path.join(outpth, 'randint'), x, fmtin=35, iprn=2)
    z = read_zbarray(os.path.join(outpth, 'randint'))
    assert np.array_equal(x, z), 'Input and output arrays do not match.'
    return


def test_dataframes():
    try:
        import pandas
        zon = read_zbarray(zon_f)
        cmd = ZoneBudget(cbc_f, zon, totim=1095.)
        df = cmd.get_dataframes()
        assert len(df) > 0, 'Output DataFrames empty.'
    except ImportError as e:
        print('Skipping DataFrames test, pandas not installed.')
        print(e)
    return


def test_get_budget():
    zon = read_zbarray(zon_f)
    aliases = {1: 'Trey', 2: 'Mike', 4: 'Wilson', 0: 'Carini'}
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 0), aliases=aliases)
    zb.get_budget(names='FROM_CONSTANT_HEAD', zones=1)
    zb.get_budget(names=['FROM_CONSTANT_HEAD'], zones=[1, 2])
    zb.get_budget(net=True)
    return


def test_get_model_shape():
    ZoneBudget(cbc_f, read_zbarray(zon_f), kstpkper=(0, 0), verbose=True).get_model_shape()
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
    test_dataframes()
    test_get_budget()
    test_get_model_shape()
