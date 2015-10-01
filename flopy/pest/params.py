from __future__ import print_function
import numpy as np


class Params(object):

    def __init__(self, mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform='log'):
        """
        Class to hold parameter definition information

        parameters
        __________
        parstyle = either 'constant', 'listrc', 'refarray'
        mfpackage = 'LPF', 'BAS6', ...

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


