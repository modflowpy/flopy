from __future__ import print_function
import numpy as np


class Util3dTpl(object):
    def __init__(self, basearray, partype):
        self.chararray = np.array(basearray, dtype='str')
        self.partype = partype

        return

    def __getitem__(self, k):
        return Util2dTpl(self.chararray[k], self.partype)

class Util2dTpl(object):

    def __init__(self, chararray, partype):
        self.chararray = chararray
        self.partype = partype

        return

    def get_file_entry(self):

        au = np.unique(self.chararray)
        if au.shape[0] == 1:
            file_entry = 'CONSTANT {0:>15s}\n'.format(au[0])
            return 'CONSTANT {0}    #{1}\n'.format(au[0], self.partype)
        else:
            cr = 'INTERNAL 1.0 (FREE) -1      #{1}\n'.format(au[0], self.partype)
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


if __name__ == '__main__':

    hk = np.ones((10, 10, 10), dtype=np.float32)
    u3dtpl = Util3dTpl(hk, 'hk')
    print(u3dtpl[1].get_file_entry())
