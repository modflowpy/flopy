import numpy as np
from flopy.mbase import Package
from flopy.utils import util_3d


class SeawatVsc(Package):

    def __init__(self, model, mt3dmuflg=-1, viscmin=0, viscmax=0,
                 viscref=8.904e-4, nsmueos=0, mutempopt=2, mtmuspec=1,
                 dmudc=1.923e-06, cmuref=0, mtmutempspec=1,
                 amucoeff=[0.001, 1, 0.015512, -20, -1.572], invisc=-1,
                 visc=-1, extension='vsc', **kwargs):

        if len(list(kwargs.keys())) > 0:
            raise Exception("VSC error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))

        Package.__init__(self, model, extension, 'VSC', 38)
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        self.mt3dmuflg = mt3dmuflg
        self.viscmin = viscmin
        self.viscmax = viscmax
        self.viscref = viscref
        self.nsmueos = nsmueos
        self.mutempopt = mutempopt
        self.mtmuspec = mtmuspec
        self.dmudc = dmudc
        self.cmuref = cmuref
        self.mtmutempspec = mtmutempspec
        self.amucoeff = amucoeff
        self.invisc = invisc
        self.visc = util_3d(model, (nlay, nrow, ncol), np.float32, visc,
                            name='visc')
        self.parent.add_package(self)
        return

    def write_file(self):
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        f_vsc = open(self.fn_path, 'w')

        # item 1
        f_vsc.write('%10i\n' % self.mt3dmuflg)

        # item 2
        if isinstance(self.viscmin, int) and self.viscmin is 0:
            f_vsc.write('%10i' % self.viscmin)
        else:
            f_vsc.write('%10.3E' % self.viscmin)

        if isinstance(self.viscmax, int) and self.viscmax is 0:
            f_vsc.write('%10i\n' % self.viscmax)
        else:
            f_vsc.write('%10.3E\n' % self.viscmax)

        # item 3
        if self.mt3dmuflg >= 0:
            f_vsc.write('%10.3E%10.2E%10.2E\n' % (self.viscref, self.dmudc,
                                                  self.cmuref))
        if self.mt3dmuflg == -1:
            f_vsc.write('%10.3E\n' % self.viscref)
            f_vsc.write('%10i%10i\n' % (self.nsmueos, self.mutempopt))

            for iwr in range(self.nsmueos):
                f_vsc.write('%10i%10.2E%10.2E\n' % ([self.mtmuspec][iwr],
                                                    [self.dmudc][iwr],
                                                    [self.cmuref][iwr]))

        if self.mutempopt > 0:
            f_vsc.write('%10i' % self.mtmutempspec)

            if self.mutempopt == 1:
                string = '%10.3E%10f%10f%10f\n'
            elif self.mutempopt == 2:
                string = '%10.3E%10f%10f %9f %9f\n'
            elif self.mutempopt == 3:
                string = '%10f %9f\n'

            f_vsc.write(string % tuple(self.amucoeff))

        # item 4
        if self.mt3dmuflg == 0:
            f_vsc.write('%10i\n' % self.invisc)

        # item 5
        if self.mt3dmuflg == 0 and self.invisc > 0:
            f_vsc.write(self.visc.get_file_entry())

        f_vsc.close()
        return
