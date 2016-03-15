from __future__ import print_function
from collections import OrderedDict
import os
import numpy as np
import subprocess as sp


def run_zonbud(zonarray, cbcfile='modflowtest.cbc', listingfile_prefix='zbud', zonbud_ws='.',
               zonbud_exe='zonbud.exe', title='ZoneBudget Test', iprn=-1):
    """

    Parameters
    ----------
    zonarray : array of ints (nlay, nrow, ncol)
        integer-array of zone numbers
    cbcfile : str
        name of the cell-by-cell budget file
    listingfile_prefix : str
        name of the listingfile
    zonbud_ws : str
        directory where ZoneBudget output will be stored
    zonbud_exe : str
        name of the ZoneBudget executable
    title : str
        title to be printed in the listing file
    iprn : integer
        specifies whether or not the zone values are printed in the output file
        if less than zero, zone values will not be printed

    Returns
    -------
    zbud, an ordered dictionary of recarrays.

    Examples
    -------
    >>> import flopy
    >>> zbud = flopy.utils.run_zonbud(zonarray, cbcfile='modflowtest.cbc')
    """
    budget_option = 'A'
    listingfile_prefix = listingfile_prefix.split('.')[0]
    zonfile = os.path.join(zonbud_ws, listingfile_prefix + '.zon')
    listingfile = os.path.join(zonbud_ws, listingfile_prefix + ' csv')
    zbud_file = os.path.join(zonbud_ws, listingfile_prefix + '.csv')
    args = [listingfile, cbcfile, title, zonfile, budget_option]

    _write_zonfile(zonarray, zonfile, iprn)
    _call(zonbud_exe, args)
    zbud = _parse_zbud_file(zbud_file)
    return zbud


def _parse_zbud_file(zf):
    assert os.path.isfile(zf), 'Output zonebudget file {} does not exist or cannot be read.'.format(zf)
    kstpkper = []
    ins = OrderedDict()
    outs = OrderedDict()
    ins_flag = False
    outs_flag = False
    with open(zf) as f:
        for line in f:
            line_items = [i.strip() for i in line.split(',')]
            if line_items[0] == 'Time Step':
                kk = (int(line_items[1])-1, int(line_items[3])-1)
                kstpkper.append(kk)
                ins[kk] = []
                outs[kk] = []
            elif 'ZONE' in line_items[1]:
                zones = [z for z in line_items if z != '']
                col_header = ['Record Name'] + zones
                dtype = [('flow_dir', '|S3'), ('record', '|S20')] + \
                        [(col_name, np.float32) for col_name in col_header[1:]]
            elif line_items[1] == 'IN':
                ins_flag = True
                continue
            elif line_items[0] == 'Total IN':
                ins_flag = False
            elif line_items[1] == 'OUT':
                outs_flag = True
                continue
            elif line_items[0] == 'Total OUT':
                outs_flag = False
            if ins_flag:
                z = [x for x in line_items if x != '']
                z.insert(0, 'in')
                z[2:] = [float(zz) for zz in z[2:]]
                ins[kk].append(tuple(z))
            elif outs_flag:
                z = [x for x in line_items if x != '']
                z.insert(0, 'out')
                z[2:] = [float(zz) for zz in z[2:]]
                outs[kk].append(tuple(z))
    zbud = OrderedDict()
    for kk in kstpkper:
        try:
            dat = ins[kk] + outs[kk]
            zbud[kk] = np.array(dat, dtype=dtype)
        except Exception as e:
            print(e)
            return None
    return zbud


def _write_zonfile(izone, zonfile, iprn):
    assert 'int' in str(izone.dtype), 'Input zone array (dtype={}) must be an integer array.'.format(izone.dtype)
    if len(izone.shape) == 2:
        nlay = 1
        nrow, ncol = izone.shape
        z = np.zeros((nlay, nrow, ncol))
        z[0, :, :] = izone
        izone = z
    elif len(izone.shape) == 3:
        nlay, nrow, ncol = izone.shape

    with open(zonfile, 'w') as f:
        f.write('{} {} {}\n'.format(nlay, nrow, ncol))

        for lay in range(nlay):
            f.write('INTERNAL ({ncol}I4) {iprn}\n'.format(ncol=ncol, iprn=iprn))
            for row in range(nrow):
                f.write(''.join(['{:4d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')

        #     f.write('INTERNAL (free) {iprn}\n'.format(iprn=iprn))
        #     for row in range(nrow):
        #         f.write(' '.join(['{:d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')
        # f.write('ALLZONES ' + ' '.join([str(int(z)) for z in np.unique(izone)])+'\n')
    return


def is_exe(fpath):
    """
    Taken from flopy.mbase

    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """
    Taken from flopy.mbase

    """
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        # test for exe in current working directory
        if is_exe(program):
            return program
        # test for exe in path statement
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def _call(exe_name, args, zonbud_ws='./'):

    # Check to make sure that program and namefile exist
    exe = which(exe_name)
    if exe is None:
        import platform

        if platform.system() in 'Windows':
            if not exe_name.lower().endswith('.exe'):
                exe = which(exe_name + '.exe')
    if exe is None:
        s = 'The program {} does not exist or is not executable.'.format(
            exe_name)
        raise Exception(s)

    proc = sp.Popen([exe_name], stdin=sp.PIPE, stdout=sp.PIPE, cwd=zonbud_ws)
    proc.communicate(input=os.linesep.join(args))
    return
