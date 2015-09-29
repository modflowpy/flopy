from __future__ import print_function
import numpy as np


class Util3dTpl(object):
    def __init__(self, basearray):
        self.chararray = np.array(basearray, dtype='|S16')
        return

    def __getitem__(self, k):
        return Util2dTpl(self.chararray[k])

class Util2dTpl(object):

    def __init__(self, chararray):
        self.chararray = chararray
        return

    def get_file_entry(self):
        cr = 'INTERNAL 1.0 (FREE) -1\n'
        astring = ''
        icount = 0
        for i in range(self.chararray.shape[0]):
            for j in range(self.chararray.shape[1]):
                icount += 1
                astring += ' {0:>15s}'.format(self.chararray[i, j])
                if icount == 10:
                    astring += '\n'
                    icount = 0
        return cr + astring


if __name__ == '__main__':

    hk = np.ones((10, 10, 10), dtype=np.float32)
    u3dtpl = Util3dTpl(hk)
    print(u3dtpl[1].get_file_entry())
