import numpy as np
from flopy.mbase import Package
from flopy.utils import util_3d


class SeawatVdf(Package):
    '''
    Variable density flow package class\n

    In swt_4 mtdnconc became mt3drhoflag. If the latter one is defined in
    kwargs, it will overwrite mtdnconc. Same goes for denseslp, which has
    become drhodc.
    '''
    def __init__(self, model, mtdnconc=1, mfnadvfd=1, nswtcpl=1, iwtable=1,
                 densemin=1.000, densemax=1.025, dnscrit=1e-2, denseref=1.000,
                 denseslp=.025, crhoref=0, firstdt=0.001, indense=0, dense=1.000,
                 nsrhoeos=1, drhodprhd=4.46e-3, prhdref=0., extension='vdf',
                 **kwargs):

        Package.__init__(self, model, extension, 'VDF', 37)
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        self.mtdnconc = kwargs.pop('mt3drhoflag', mtdnconc)
        self.mfnadvfd = mfnadvfd
        self.nswtcpl = nswtcpl
        self.iwtable = iwtable
        self.densemin = densemin
        self.densemax = densemax
        self.dnscrit = dnscrit
        self.nsrhoeos = nsrhoeos
        self.denseref = denseref
        self.denseslp = kwargs.pop('drhodc', denseslp)
        self.crhoref = crhoref
        self.drhodprhd = drhodprhd
        self.prhdref = prhdref
        self.firstdt = firstdt
        self.indense = indense
        self.dense = util_3d(model, (nlay, nrow, ncol), np.float32, dense,
                             name='dense')
        self.parent.add_package(self)
        return

    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        f_vdf = open(self.fn_path, 'w')

        # item 1
        f_vdf.write('%10i%10i%10i%10i\n' % (self.mtdnconc, self.mfnadvfd,
                                            self.nswtcpl, self.iwtable))

        # item 2
        f_vdf.write('%10.4f%10.4f\n' % (self.densemin, self.densemax))

        # item 3
        if (self.nswtcpl > 1 or self.nswtcpl == -1):
            f_vdf.write('%10f\n' % (self.dnscrit))

        # item 4
        if self.mtdnconc >= 0:
            if self.nsrhoeos is 1:
                f_vdf.write('%10.4f%10.4f\n' % (self.denseref, self.denseslp))
            else:
                f_vdf.write('%10.4f%10.4f\n' % (self.denseref,
                                                self.denseslp[0]))

        elif self.mtdnconc == -1:
            f_vdf.write('%10.4f%10.4f%10.4f\n' % (self.denseref,
                                                  self.drhodprhd,
                                                  self.prhdref))
            f_vdf.write('%10i\n' % self.nsrhoeos)
            if self.nsrhoeos is 1:
                f_vdf.write('%10i%10.4f%10.4f\n' % (1, self.denseslp,
                                                    self.crhoref))
            else:
                for i in xrange(self.nsrhoeos-1):
                    mtrhospec = 2 + i
                    f_vdf.write('%10i%10.4f%10.4f\n' % (mtrhospec,
                                                        self.denseslp[i+1],
                                                        self.crhoref[i+1]))

        # item 5
        f_vdf.write('%10f\n' % (self.firstdt))

        # item 6
        if (self.mtdnconc == 0):
            f_vdf.write('%10i\n' % (self.indense))

        # item 7
            if (self.indense > 0):
                f_vdf.write(self.dense.get_file_entry())

        f_vdf.close()
        return
