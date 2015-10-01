from __future__ import print_function
import numpy as np


class Params(object):

    def __init__(self, mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform='log'):
        """
        Class to hold parameter definition information

        parameters
        __________
        partype = 'hk', 'vkz', ... must be the name of an array within a package
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


