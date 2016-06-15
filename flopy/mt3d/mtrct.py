import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util3d


class Mt3dRct(Package):
    """
    Chemical reaction package class

    """
    unitnumber = 36

    def __init__(self, model, isothm=0, ireact=0, igetsc=1, rhob=None,
                 prsity2=None, srconc=None, sp1=None, sp2=None, rc1=None,
                 rc2=None, extension='rct', unitnumber=None, **kwargs):
        if unitnumber is None:
            unitnumber = self.unitnumber
        Package.__init__(self, model, extension, 'RCT', unitnumber)
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp

        # Item E1: ISOTHM, IREACT, IRCTOP, IGETSC
        self.isothm = isothm
        self.ireact = ireact
        self.irctop = 2  # All RCT vars are specified as 3D arrays
        self.igetsc = igetsc

        # Item E2A: RHOB
        if rhob is None:
            rhob = 1.8e3
        self.rhob = Util3d(model, (nlay, nrow, ncol), np.float32, rhob,
                           name='rhob', locat=self.unit_number[0],
                           array_free_format=False)

        # Item E2B: PRSITY
        if prsity2 is None:
            prsity2 = 0.1
        self.prsity2 = Util3d(model, (nlay, nrow, ncol), np.float32, prsity2,
                              name='prsity2', locat=self.unit_number[0],
                              array_free_format=False)

        # Item E2C: SRCONC
        if srconc is None:
            srconc = 0.0
        self.srconc = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, srconc,
                     name='srconc1', locat=self.unit_number[0],
                     array_free_format=False)
        self.srconc.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "srconc" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("BTN: setting srconc for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.srconc.append(u3d)

        # Item E3: SP1
        if sp1 is None:
            sp1 = 0.0
        self.sp1 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, sp1, name='sp11',
                     locat=self.unit_number[0], array_free_format=False)
        self.sp1.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "sp1" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("BTN: setting sp1 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.sp1.append(u3d)

        # Item E4: SP2
        if sp2 is None:
            sp2 = 0.0
        self.sp2 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, sp2, name='sp21',
                     locat=self.unit_number[0], array_free_format=False)
        self.sp2.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "sp2" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("BTN: setting sp2 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.sp2.append(u3d)

        # Item E5: RC1
        if rc1 is None:
            rc1 = 0.0
        self.rc1 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, rc1, name='rc11',
                     locat=self.unit_number[0], array_free_format=False)
        self.rc1.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "rc1" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("BTN: setting rc1 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.rc1.append(u3d)

        # Item E4: RC2
        if rc2 is None:
            rc2 = 0.0
        self.rc2 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, rc2, name='rc21',
                     locat=self.unit_number[0], array_free_format=False)
        self.rc2.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "rc2" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("BTN: setting rc2 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.rc2.append(u3d)

        # Check to make sure that all kwargs have been consumed
        if len(list(kwargs.keys())) > 0:
            raise Exception("RCT error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))

        self.parent.add_package(self)
        return

    def __repr__(self):
        return 'Chemical reaction package class'

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_rct = open(self.fn_path, 'w')
        f_rct.write('%10i%10i%10i%10i\n' % (self.isothm, self.ireact,
                                            self.irctop, self.igetsc))
        if (self.isothm in [1, 2, 3, 4, 6]):
            f_rct.write(self.rhob.get_file_entry())
        if (self.isothm in [5, 6]):
            f_rct.write(self.prsity2.get_file_entry())
        if (self.igetsc > 0):
            for icomp in range(len(self.srconc)):
                f_rct.write(self.srconc[icomp].get_file_entry())
        if (self.isothm > 0):
            for icomp in range(len(self.sp1)):
                f_rct.write(self.sp1[icomp].get_file_entry())
        if (self.isothm > 0):
            for icomp in range(len(self.sp2)):
                f_rct.write(self.sp2[icomp].get_file_entry())
        if (self.ireact > 0):
            for icomp in range(len(self.rc1)):
                f_rct.write(self.rc1[icomp].get_file_entry())
        if (self.ireact > 0):
            for icomp in range(len(self.rc2)):
                f_rct.write(self.rc2[icomp].get_file_entry())
        f_rct.close()
        return

    @staticmethod
    def load(f, model, nlay=None, nrow=None, ncol=None, ncomp=None,
             ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        rct :  Mt3dRct object
            Mt3dRct object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> rct = flopy.mt3d.Mt3dRct.load('test.rct', mt)

        """

        if model.verbose:
            sys.stdout.write('loading rct package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set dimensions if necessary
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol
        if ncomp is None:
            ncomp = model.ncomp

        # Setup kwargs to store multispecies information
        kwargs = {}

        # Item E1
        line = f.readline()
        if model.verbose:
            print('   loading ISOTHM, IREACT, IRCTOP, IGETSC...')
        isothm = int(line[0:10])
        ireact = int(line[11:20])
        try:
            irctop = int(line[21:30])
        except:
            irctop = 0
        try:
            igetsc = int(line[31:40])
        except:
            igetsc = 0
        if model.verbose:
            print('   ISOTHM {}'.format(isothm))
            print('   IREACT {}'.format(ireact))
            print('   IRCTOP {}'.format(irctop))
            print('   IGETSC {}'.format(igetsc))

        # Item E2A: RHOB
        rhob = None
        if model.verbose:
            print('   loading RHOB...')
        if isothm in [1, 2, 3, 4, 6]:
            rhob = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                               'rhob', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   RHOB {}'.format(rhob))

        # Item E2A: PRSITY2
        prsity2 = None
        if model.verbose:
            print('   loading PRSITY2...')
        if isothm in [5, 6]:
            prsity2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                               'prsity2', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   PRSITY2 {}'.format(prsity2))

        # Item E2C: SRCONC
        srconc = None
        if model.verbose:
            print('   loading SRCONC...')
        if igetsc > 0:
            srconc = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                 'srconc1', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SRCONC {}'.format(srconc))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "srconc" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SRCONC{} {}'.format(icomp, u3d))

        # Item E3: SP1
        sp1 = None
        if model.verbose:
            print('   loading SP1...')
        if isothm > 0:
            sp1 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'sp11', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SP1 {}'.format(sp1))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "sp1" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SP1{} {}'.format(icomp, u3d))

        # Item E4: SP2
        sp2 = None
        if model.verbose:
            print('   loading SP2...')
        if isothm > 0:
            sp2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'sp21', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SP2 {}'.format(sp2))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "sp2" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SP2{} {}'.format(icomp, u3d))

        # Item E5: RC1
        rc1 = None
        if model.verbose:
            print('   loading RC1...')
        if ireact > 0:
            rc1 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'rc11', ext_unit_dict,
                              array_format="mt3d")
            if model.verbose:
                print('   RC1 {}'.format(rc1))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "rc1" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   RC1{} {}'.format(icomp, u3d))

        # Item E6: RC2
        rc2 = None
        if model.verbose:
            print('   loading RC2...')
        if ireact > 0:
            rc2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'rc21', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   RC2 {}'.format(rc2))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "rc2" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   RC2{} {}'.format(icomp, u3d))

        # Close the file
        f.close()

        # Construct and return rct package
        rct = Mt3dRct(model, isothm=isothm, ireact=ireact, igetsc=igetsc,
                      rhob=rhob, prsity2=prsity2, srconc=srconc, sp1=sp1,
                      sp2=sp2, rc1=rc1, rc2=rc2)
        return rct


