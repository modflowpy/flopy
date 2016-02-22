"""
Module for input/output utilities
"""
import numpy as np


def line_parse(line):
    """
    Convert a line of text into to a list of values.  This handles the
    case where a free formatted MODFLOW input file may have commas in
    it.

    """
    line = line.replace(',', ' ')
    return line.strip().split()

def write_fixed_var(v, length=10, ipos=None, free=False, comment=None):
    """

    Parameters
    ----------
    v : list, int, float, bool, or numpy array
        list, int, float, bool, or numpy array containing the data to be
        written to a string.
    length : int
        length of each column for fixed column widths. (default is 10)
    ipos : list, int, or numpy array
        user-provided column widths. (default is None)
    free : bool
        boolean indicating if a free format string should be generated.
        length and ipos are not used if free is True. (default is False)
    comment : str
        comment string to add to the end of the string

    Returns
    -------
    out : str
        fixed or free format string generated using user-provided data

    """
    if isinstance(v, np.ndarray):
        v = v.aslist()
    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
        v = [v]
    ncol = len(v)
    # construct ipos if it was not passed
    if ipos is None:
        ipos = []
        for i in range(ncol):
            ipos.append(length)
    else:
        if isinstance(ipos, np.ndarray):
            ipos = ipos.flatten().aslist()
        elif isinstance(ipos, int):
            ipos = [ipos]
        if len(ipos) < ncol:
            err = 'user provided ipos length ({})'.format(len(ipos)) + \
                  'should be greater than or equal ' + \
                  'to the length of v ({})'.format(ncol)
            raise Exception(err)
    out = ''
    for n in range(ncol):
        if free:
            write_fmt = '{} '
        else:
            write_fmt = '{{:>{}}}'.format(ipos[n])
        out += write_fmt.format(v[n])
    if comment is not None:
        out += '  # {}'.format(comment)
    out += '\n'
    return out

def read_fixed_var(line, ncol=1, length=10, ipos=None, free=False):
    """
    Parse a fixed format line using user provided data

    Parameters
    ----------
    line : str
        text string to parse.
    ncol : int
        number of columns to parse from line. (default is 1)
    length : int
        length of each column for fixed column widths. (default is 10)
    ipos : list, int, or numpy array
        user-provided column widths. (default is None)
    free : bool
        boolean indicating if sting is free format. ncol, length, and
        ipos are not used if free is True. (default is False)

    Returns
    -------
    out : list
        padded list containing data parsed from the passed text string

    """
    if free:
        out = line.rstrip().split()
    else:
        # construct ipos if it was not passed
        if ipos is None:
            ipos = []
            for i in range(ncol):
                ipos.append(length)
        else:
            if isinstance(ipos, np.ndarray):
                ipos = ipos.flatten().aslist()
            elif isinstance(ipos, int):
                ipos = [ipos]
            ncol = len(ipos)
        line = line.rstrip()
        out = []
        istart = 0
        for ivar in range(ncol):
            istop = istart + ipos[ivar]
            try:
                txt = line[istart:istop]
                if len(txt.strip()) > 0:
                    out.append(txt)
                else:
                    out.append(0)
            except:
                break
            istart = istop
    return out

def flux_to_wel(cbc_file,text,precision="single",model=None,verbose=False):
    """
    Convert flux in a binary cell budget file to a wel instance

    Parameters
    ----------
    cbc_file : (str) cell budget file name
    text : (str) text string of the desired flux type (e.g. "drains")
    precision : (optional str) precision of the cell budget file
    model : (optional) BaseModel instance.  If passed, a new ModflowWel
        instance will be added to model
    verbose : bool flag passed to CellBudgetFile

    Returns
    -------
    flopy.modflow.ModflowWel instance

    """
    from . import CellBudgetFile as CBF
    from .util_list import MfList
    from ..modflow import Modflow, ModflowWel
    cbf = CBF(cbc_file,precision=precision,verbose=verbose)

    # create a empty numpy array of shape (time,layer,row,col)
    m4d = np.zeros((cbf.nper,cbf.nlay,cbf.nrow,cbf.ncol),dtype=np.float32)
    m4d[:] = np.NaN

    # process the records in the cell budget file
    iper = -1
    for kstpkper in cbf.kstpkper:

        kstpkper = (kstpkper[0]-1,kstpkper[1]-1)
        kper = kstpkper[1]
        #if we haven't visited this kper yet
        if kper != iper:
            arr = cbf.get_data(kstpkper=kstpkper,text=text,full3D=True)
            if len(arr) > 0:
                arr = arr[0]
                print(arr.max(),arr.min(),arr.sum())
                # masked where zero
                arr[np.where(arr==0.0)] = np.NaN
                m4d[iper+1] = arr
            iper += 1



    # model wasn't passed, then create a generic model
    if model is None:
        model = Modflow("test")
    # if model doesn't have a wel package, then make a generic one...
    # need this for the from_m4d method
    if model.wel is None:
        ModflowWel(model)

    # get the stress_period_data dict {kper:np recarray}
    sp_data = MfList.from_4d(model,"WEL",{"flux":m4d})

    wel = ModflowWel(model,stress_period_data=sp_data)
    return wel
