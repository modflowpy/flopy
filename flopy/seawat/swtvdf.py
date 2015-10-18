import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d, util_3d


class SeawatVdf(Package):
    """
    Variable density flow package class

    In swt_4 mtdnconc became mt3drhoflag. If the latter one is defined in
    kwargs, it will overwrite mtdnconc. Same goes for denseslp, which has
    become drhodc.
    """
    def __init__(self, model, mtdnconc=1, mfnadvfd=1, nswtcpl=1, iwtable=1,
                 densemin=1.000, densemax=1.025, dnscrit=1e-2, denseref=1.000,
                 denseslp=.025, crhoref=0, firstdt=0.001, indense=0,
                 dense=1.000, nsrhoeos=1, drhodprhd=4.46e-3, prhdref=0.,
                 extension='vdf', **kwargs):

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
        if dense is not None:
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

    @staticmethod
    def load(f, model, nper=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.seawat.swt.Seawat`) to
            which this package will be added.
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        vdf : SeawatVdf object
            SeawatVdf object.

        Examples
        --------

        >>> import flopy
        >>> mf = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis(mf)
        >>> mt = flopy.mt3d.Mt3dms()
        >>> swt = flopy.seawat.Seawat(modflowmodel=mf, mt3dmsmodel=mt)
        >>> vdf = flopy.seawat.SeawatVdf.load('test.vdf', m)

        """

        if model.verbose:
            sys.stdout.write('loading vdf package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # Determine problem dimensions
        nrow, ncol, nlay, nper = model.mf.get_nrow_ncol_nlay_nper()

        # Item 1: MT3DRHOFLG MFNADVFD NSWTCPL IWTABLE - line already read above
        if model.verbose:
            print('   loading MT3DRHOFLG MFNADVFD NSWTCPL IWTABLE...')
        t = line.strip().split()
        mt3drhoflg = int(t[0])
        mfnadvfd = int(t[1])
        nswtcpl = int(t[2])
        iwtable = int(t[3])
        if model.verbose:
            print('   MT3DRHOFLG {}'.format(mt3drhoflg))
            print('   MFNADVFD {}'.format(mfnadvfd))
            print('   NSWTCPL {}'.format(nswtcpl))
            print('   IWTABLE {}'.format(iwtable))

        # Item 2 -- DENSEMIN DENSEMAX
        if model.verbose:
            print('   loading DENSEMIN DENSEMAX...')
        line = f.readline()
        t = line.strip().split()
        densemin = float(t[0])
        densemax = float(t[1])

        # Item 3 -- DNSCRIT
        if model.verbose:
            print('   loading DNSCRIT...')
        dnscrit = None
        if nswtcpl > 1 or nswtcpl == -1:
            line = f.readline()
            t = line.strip().split()
            dnscrit = float(t[0])

        # Item 4 -- DENSEREF DRHODC
        drhodprhd = None
        prhdref = None
        nsrhoeos = None
        mtrhospec = None
        crhoref = None
        if mt3drhoflg >= 0:
            if model.verbose:
                print('   loading DENSEREF DRHODC(1)...')
            line = f.readline()
            t = line.strip().split()
            denseref = float(t[0])
            drhodc = float(t[1])
        else:
            if model.verbose:
                print('   loading DENSEREF DRHODPRHD PRHDREF...')
            line = f.readline()
            t = line.strip().split()
            denseref = float(t[0])
            drhodprhd = float(t[1])
            prhdref = float(t[2])

            if model.verbose:
                print('   loading NSRHOEOS...')
            line = f.readline()
            t = line.strip().split()
            nsrhoeos = int(t[0])

            if model.verbose:
                print('    loading MTRHOSPEC DRHODC CRHOREF...')
            mtrhospec = []
            drhodc = []
            crhoref = []
            for i in range(nsrhoeos):
                line = f.readline()
                t = line.strip().split()
                mtrhospec.append(int(t[0]))
                drhodc.append(float(t[1]))
                crhoref.append(float(t[2]))

        # Item 5 -- FIRSTDT
        if model.verbose:
            print('   loading FIRSTDT...')
        line = f.readline()
        t = line.strip().split()
        firstdt = float(t[0])

        # Items 6 and 7 -- INDENSE DENSE
        indense = None
        dense = None
        if mt3drhoflg == 0:
            if model.verbose:
                print('   loading INDENSE...')
            line = f.readline()
            t = line.strip().split()
            indense = int(t[0])

            if indense > 0:
                dense = [0] * nlay
                for k in range(nlay):
                    if model.verbose:
                        print('   loading DENSE layer {0:3d}...'.format(k + 1))
                    t = util_2d.load(f, model.mf, (nrow, ncol), np.float32,
                                     'dense', ext_unit_dict)
                    dense[k] = t

        # Construct and return vdf package
        vdf = SeawatVdf(model, mt3drhoflg=mt3drhoflg, mfnadvfd=mfnadvfd,
                        nswtcpl=nswtcpl, iwtable=iwtable,
                        densemin=densemin, densemax=densemax,
                        dnscrit=dnscrit, denseref=denseref, drhodc=drhodc,
                        drhodprhd=drhodprhd, prhdref=prhdref,
                        nsrhoeos=nsrhoeos, mtrhospec=mtrhospec,
                        crhoref=crhoref, indense=indense, dense=dense)
        return vdf
