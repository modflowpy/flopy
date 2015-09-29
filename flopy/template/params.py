from __future__ import print_function
import numpy as np

def Refarray2Params(mfpackage, partype, parname, parzone,
                 startvalue, lbound, ubound, transform, refarr):
    """

    """

    idx = np.where(refarr == parzone)

    curr_params = Params(mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform)

    return curr_params





class Params(object):

    def __init__(self, mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform='log'):
        """
        Class to hold parameter definition information

        parameters
        __________
        parstyle = either 'constant', 'listrc', 'refarray'

        :return:
        """
        self.name = parname
        self.type = partype
        self.mfpackage = mfpackage
        self.startvalue = startvalue
        self.lbound = lbound
        self.ubound = ubound
        self.transform = transform
        self.idx = idx


if __name__ == '__main__':
    mfpackage = 'lpf'
    partype = 'hk'
    parname = 'hk2'
    parzone = 2
    startvalue = 120
    lbound = 5
    ubound = 500
    transform = 'log'
    refarr = np.ones((2,5,9), dtype=int)
    refarr[0,1:3,3:5] = 2
    refarr[1,2:4,3:5] = 2

    print (refarr)


    par1 = Refarray2Params(mfpackage, partype, parname, parzone, startvalue, lbound, ubound, transform, refarr)

