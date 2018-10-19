import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Util3d

class Mt3dDsp(Package):
    """
    MT3DMS Dispersion Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to which
        this package will be added.
    al : float or array of floats (nlay, nrow, ncol)
        AL is the longitudinal dispersivity, for every cell of the model grid
        (unit, L).
        (default is 0.01)
    trpt : float or array of floats (nlay)
        s a 1D real array defining the ratio of the horizontal transverse
        dispersivity to the longitudinal dispersivity. Each value
        in the array corresponds to one model layer. Some recent field
        studies suggest that TRPT is generally not greater than 0.1.
        (default is 0.1)
    trpv : float or array of floats (nlay)
        is the ratio of the vertical transverse dispersivity to the
        longitudinal dispersivity. Each value in the array corresponds to one
        model layer. Some recent field studies suggest that TRPT is generally
        not greater than 0.01.  Set TRPV equal to TRPT to use the standard
        isotropic dispersion model (Equation 10 in Chapter 2). Otherwise,
        the modified isotropic dispersion model is used (Equation 11 in
        Chapter 2).
        (default is 0.01)
    dmcoef : float or array of floats (nlay) or (nlay, nrow, ncol) if the
        multiDiff option is used.
        DMCOEF is the effective molecular diffusion coefficient (unit, L2T-1).
        Set DMCOEF = 0 if the effect of molecular diffusion is considered
        unimportant. Each value in the array corresponds to one model layer.
        The value for dmcoef applies only to species 1.  See kwargs for
        entering dmcoef for other species.
        (default is 1.e-9).
    multiDiff : boolean
        To activate the component-dependent diffusion option, a keyword
        input record must be inserted to the beginning of the Dispersion
        (DSP) input file. The symbol $ in the first column of an input line
        signifies a keyword input record containing one or more predefined
        keywords. Above the keyword input record, comment lines marked by the
        symbol # in the first column are allowed. Comment lines are processed
        but have no effect on the simulation. Furthermore, blank lines are
        also acceptable above the keyword input record. Below the keyword
        input record, the format of the DSP input file must remain unchanged
        from the previous versions except for the diffusion coefficient as
        explained below. If no keyword input record is specified, the input
        file remains backward compatible with all previous versions of MT3DMS.
        The predefined keyword for the component-dependent diffusion option
        is MultiDiffusion. The keyword is case insensitive so
        ''MultiDiffusion'' is equivalent to either ''Multidiffusion'' or
        ''multidiffusion''. If this keyword is specified in the keyword input
        record that has been inserted into the beginning of the DSP input
        file, the component-dependent diffusion option has been activated and
        the user needs to specify one diffusion coefficient for each mobile
        solute component and at each model cell. This is done by specifying
        one mobile component at a time, from the first component to the last
        component (MCOMP). For each mobile component, the real array reader
        utility (RARRAY) is used to input the 3-D diffusion coefficient
        array, one model layer at a time.
        (default is False)
    extension : string
        Filename extension (default is 'dsp')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.
    kwargs : dictionary
        If a multi-species simulation, then dmcoef values can be specified for
        other species as dmcoef2, dmcoef3, etc.  For example:
        dmcoef1=1.e-10, dmcoef2=4.e-10, ...  If a value is not specifed, then
        dmcoef is set to 0.0.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mt3d.Mt3dms()
    >>> dsp = flopy.mt3d.Mt3dDsp(m)

    """
    def __init__(self, model, al=0.01, trpt=0.1, trpv=0.01, dmcoef=1e-9,
                 extension='dsp', multiDiff=False, unitnumber=None,
                 filenames=None, **kwargs):

        if unitnumber is None:
            unitnumber = Mt3dDsp.defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dDsp.reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dDsp.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp
        mcomp = model.mcomp
        self.multiDiff = multiDiff
        self.al = Util3d(model, (nlay,nrow,ncol), np.float32, al, name='al',
                         locat=self.unit_number[0],
                         array_free_format=False)
        self.trpt = Util2d(model, (nlay,), np.float32, trpt, name='trpt',
                           locat=self.unit_number[0],
                           array_free_format=False)
        self.trpv = Util2d(model, (nlay,), np.float32, trpv, name='trpv',
                           locat=self.unit_number[0],
                           array_free_format=False)

        # Multi-species and multi-diffusion, hence the complexity
        self.dmcoef = []
        shape = (nlay, 1)
        utype = Util2d
        nmcomp = ncomp
        if multiDiff:
            shape = (nlay, nrow, ncol)
            utype = Util3d
            nmcomp = mcomp
        u2or3 = utype(model, shape, np.float32, dmcoef,
                      name='dmcoef1', locat=self.unit_number[0],
                      array_free_format=False)
        self.dmcoef.append(u2or3)
        for icomp in range(2, nmcomp + 1):
            name = "dmcoef" + str(icomp)
            val = 0.0
            if name in list(kwargs.keys()):
                val = kwargs.pop(name)
            else:
                print("DSP: setting dmcoef for component " +
                      str(icomp) + " to zero, kwarg name " +
                      name)
            u2or3 = utype(model, shape, np.float32, val,
                          name=name, locat=self.unit_number[0],
                          array_free_format=False)
            self.dmcoef.append(u2or3)

        if len(list(kwargs.keys())) > 0:
            raise Exception("DSP error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Get size
        nrow = self.parent.nrow
        ncol = self.parent.ncol
        nlay = self.parent.nlay

        # Open file for writing
        f_dsp = open(self.fn_path, 'w')

        # Write multidiffusion keyword
        if self.multiDiff:
            f_dsp.write('$ MultiDiffusion\n')

        # Write arrays
        f_dsp.write(self.al.get_file_entry())
        f_dsp.write(self.trpt.get_file_entry())
        f_dsp.write(self.trpv.get_file_entry())
        f_dsp.write(self.dmcoef[0].get_file_entry())
        if self.multiDiff:
            for i in range(1, len(self.dmcoef)):
                f_dsp.write(self.dmcoef[i].get_file_entry())
        f_dsp.close()
        return

    @staticmethod
    def load(f, model, nlay=None, nrow=None, ncol=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        nlay : int
            number of model layers.  If None it will be retrieved from the
            model.
        nrow : int
            number of model rows.  If None it will be retrieved from the
            model.
        ncol : int
            number of model columns.  If None it will be retrieved from the
            model.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        adv :  Mt3dDsp object
            Mt3dDsp object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> dsp = flopy.mt3d.Mt3dAdv.load('test.dsp', m)

        """

        if model.verbose:
            sys.stdout.write('loading dsp package file...\n')

        # Set dimensions if necessary
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Dataset 0 -- comment line
        imsd = 0
        while True:
            line = f.readline()
            if line.strip() == '':
                continue
            elif line[0] == '#':
                continue
            elif line[0] == '$':
                imsd = 1
                break
            else:
                break

        # Check for keywords (multidiffusion)
        multiDiff = False
        if imsd == 1:
            keywords = line[1:].strip().split()
            for k in keywords:
                if k.lower() == 'multidiffusion':
                    multiDiff = True
        else:
            # go back to beginning of file
            f.seek(0, 0)

        # Read arrays
        if model.verbose:
            print('   loading AL...')
        al = Util3d.load(f, model, (nlay, nrow, ncol), np.float32, 'al',
                         ext_unit_dict, array_format="mt3d")

        if model.verbose:
            print('   loading TRPT...')
        trpt = Util2d.load(f, model, (nlay,), np.float32, 'trpt',
                           ext_unit_dict, array_format="mt3d",
                           array_free_format=False)

        if model.verbose:
            print('   loading TRPV...')
        trpv = Util2d.load(f, model, (nlay,), np.float32, 'trpv',
                           ext_unit_dict, array_format="mt3d",
                           array_free_format=False)

        if model.verbose:
            print('   loading DMCOEFF...')
        kwargs = {}
        dmcoef = []
        if multiDiff:
            dmcoef = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                 'dmcoef1', ext_unit_dict, array_format="mt3d")
            if model.mcomp > 1:
                for icomp in range(2, model.mcomp + 1):
                    name = "dmcoef" + str(icomp)
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                       name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d


        else:
            dmcoef = Util2d.load(f, model, (nlay,), np.float32,
                               'dmcoef1', ext_unit_dict, array_format="mt3d")
            # if model.mcomp > 1:
            #     for icomp in range(2, model.mcomp + 1):
            #         name = "dmcoef" + str(icomp + 1)
            #         u2d = Util2d.load(f, model, (nlay,), np.float32, name,
            #                     ext_unit_dict, array_format="mt3d")
            #         kwargs[name] = u2d

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=Mt3dDsp.ftype())

        dsp = Mt3dDsp(model, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef,
                      multiDiff=multiDiff, unitnumber=unitnumber,
                      filenames=filenames, **kwargs)
        return dsp

    @staticmethod
    def ftype():
        return 'DSP'

    @staticmethod
    def defaultunit():
        return 33

    @staticmethod
    def reservedunit():
        return 3
