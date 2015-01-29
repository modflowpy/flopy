from flopy.mbase import Package

class ModflowSip(Package):
    'Strongly Implicit Procedure package class\n'
    def __init__(self, model, mxiter=200, nparm=5, \
                 accl=1, hclose=1e-5, ipcalc=1, wseed=0, iprsip=0, extension='sip', unitnumber=25):
        """
        Package constructor.

        """
        Package.__init__(self, model, extension, 'SIP', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.url = 'sip.htm'
        self.mxiter = mxiter
        self.nparm = nparm
        self.accl = accl
        self.hclose = hclose
        self.ipcalc = ipcalc
        self.wseed = wseed
        self.iprsip = iprsip
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package input file.

        """
        # Open file for writing
        f_sip = open(self.fn_path, 'w')
        f_sip.write('%10i%10i\n' % (self.mxiter,self.nparm))
        f_sip.write('%10f%10f%10i%10f%10i\n' % (self.accl, self.hclose, self.ipcalc, self.wseed, self.iprsip))
        f_sip.close()

    @staticmethod
    def load(f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        sip : ModflowSip object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> sip = flopy.modflow.ModflowSip.load('test.sip', m)

        """

        if model.verbose:
            print 'loading sip package file...'
        if type(f) is not file:
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header

        print '   Warning: load method not completed. default sip object created.'

        #--close the open file
        f.close()

        sip = ModflowSip(model)
        return sip
