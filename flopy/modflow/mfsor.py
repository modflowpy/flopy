import sys
from flopy.mbase import Package

class ModflowSor(Package):
    'Slice-successive overrelaxation package class\n'
    def __init__(self, model, mxiter=200, \
                 accl=1, hclose=1e-5, iprsor=0, extension='sor', unitnumber=26):
        """
        Package constructor.

        """
        Package.__init__(self, model, extension, 'sor', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.url = 'sor.htm'
        self.mxiter = mxiter
        self.accl= accl
        self.hclose = hclose
        self.iprsor = iprsor
        self.parent.add_package(self)


    def write_file(self):
        """
        Write the package input file.

        """
        # Open file for writing
        f_sor = open(self.fn_path, 'w')
        f_sor.write('%10i\n' % (self.mxiter))
        f_sor.write('%10f%10f%10i\n' % (self.accl, self.hclose, self.iprsor))
        f_sor.close()


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
        sor : ModflowSor object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> sor = flopy.modflow.ModflowSor.load('test.sor', m)

        """

        if model.verbose:
            sys.stdout.write('loading sor package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header

        print('   Warning: load method not completed. default sor object created.')

        # close the open file
        f.close()

        # create sor object
        sor = ModflowSor(model)

        # return sor object
        return sor
