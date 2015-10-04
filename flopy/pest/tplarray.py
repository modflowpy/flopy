from __future__ import print_function
import numpy as np


class Util3dTpl(object):
    """
    Class to define a three-dimensional template array for use with parameter
    estimation.

    Parameters
    ----------
    basearray : A Numpy ndarray.
    partype : The parameter type.  This will be written to the control record
        as a comment.
    """
    def __init__(self, basearray, partype):
        self.chararray = np.array(basearray, dtype='str')
        self.partype = partype
        return

    def __getitem__(self, k):
        return Util2dTpl(self.chararray[k], self.partype)


class Util2dTpl(object):
    """
    Class to define a two-dimensional template array for use with parameter
    estimation.

    Parameters
    ----------
    chararray : A Numpy ndarray of dtype 'str'.
    partype : The parameter type.  This will be written to the control record
        as a comment.
    """
    def __init__(self, chararray, partype):
        self.chararray = chararray
        self.partype = partype

        return

    def get_file_entry(self):
        """
        Convert the array into a string.

        Returns
        -------
        file_entry : str
        """
        au = np.unique(self.chararray)
        if au.shape[0] == 1:
            file_entry = 'CONSTANT {0}    #{1}\n'.format(au[0], self.partype)
        else:
            cr = 'INTERNAL {0} (FREE) -1      #{1}\n'.format(1.0, self.partype)
            astring = ''
            icount = 0
            for i in range(self.chararray.shape[0]):
                for j in range(self.chararray.shape[1]):
                    icount += 1
                    astring += ' {0:>15s}'.format(self.chararray[i, j])
                    if icount == 10:
                        astring += '\n'
                        icount = 0
            file_entry = cr + astring
        return file_entry

