from __future__ import print_function
import os
import numpy as np
import subprocess
import shlex


def _call(command, args):
    if type(command) == str:
        command = shlex.split(command)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process.communicate(os.linesep.join(args))
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip('\n'))
    rc = process.poll()
    return rc


def write_zonefile(izone, outfile, iprn=-1):
    """
    izone : array of ints (nlay, nrow, ncol)
        integer-array of zone numbers
    outfile: str
        Name of the output ZoneBudget array file

    Examples
    --------

    >>> import flopy
    >>> import numpy as np
    >>> nlay = 9
    >>> zon = np.array([np.loadtxt('GWBasins.zon') for lay in range(nlay)], dtype=np.int)
    >>> flopy.utils.write_zonefile(zon, 'GWBasins_zb2')

    """
    assert 'int' in str(izone.dtype), 'Input zone array (dtype={}) must be an integer array.'.format(izone.dtype)
    if len(izone.shape) == 2:
        nlay = 1
        nrow, ncol = izone.shape
    elif len(izone.shape) == 3:
        nlay, nrow, ncol = izone.shape

    with open(outfile, 'w') as f:
        f.write('{} {} {}\n'.format(nlay, nrow, ncol))
        for lay in range(nlay):
            f.write('INTERNAL      ({ncol}I8) {iprn}\n'.format(ncol=ncol, iprn=iprn))
            # f.write('INTERNAL (free) {iprn}\n'.format(iprn))
            for row in range(nrow):
                f.write(''.join(['{:8d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')
    return


def run_zonbud(zonbud_exe='zonbud.exe', listingfile='zonbudtest csv',
               cbcfile='zonbudtest.cbc', title='ZoneBudget Test',
               zonefile='izone', budget_option='A'):
    """
    zonbud_exe : str
        name of the ZoneBudget executable
    listingfile : str
        name of the listing file/output format
    cbcfile : str
        name of the cell-by-cell budget file
    title : str
        title to be printed in the listing file
    zonefile : str
        name of the input zone array file
    budget_option : str
        option for specifying when budgets are calculated

    >>> import flopy
    >>> import numpy as np
    >>> nlay = 9
    >>> zon = np.array([np.loadtxt('GWBasins.zon') for lay in range(nlay)], dtype=np.int)
    >>> flopy.utils.write_zbarray_file(zon, 'GWBasins_zb2')
    >>> flopy.utils.run_zonbud(zonefile='GWBasins_zb2', cbcfile=r'model\fas.cbc')
    """
    assert type(budget_option) == str, 'budget_option must be a string.'
    assert budget_option.upper() in ['A', 'P', 'L'], 'budget_option must be one of: "A", "P", or "L".'
    args = [listingfile, cbcfile, title, zonefile, budget_option]
    _call([zonbud_exe], args)
    return
