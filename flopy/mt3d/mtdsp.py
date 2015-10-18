import sys
import numpy as np
from flopy.mbase import Package
from flopy.utils import util_2d,util_3d

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
    dmcoef : float or array of floats (nlay)
        DMCOEF is the effective molecular diffusion coefficient (unit, L2T-1).
        Set DMCOEF = 0 if the effect of molecular diffusion is considered
        unimportant. Each value in the array corresponds to one model layer.
        (default is 1.e-9)
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
        File unit number (default is 33).

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
                 extension='dsp', multiDiff=False, unitnumber=33, **kwargs):
        '''
        if dmcoef is passed as a list of (nlay, nrow, ncol) arrays,
        then the multicomponent diffusion is activated
        '''
        Package.__init__(self, model, extension, 'DSP', unitnumber)
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper
        ncomp = self.parent.get_ncomp()        
        self.multiDiff = multiDiff
        self.al = util_3d(model,(nlay,nrow,ncol),np.float32,al,name='al',
                          locat=self.unit_number[0])
        self.trpt = util_2d(model,(nlay,),np.float32,trpt,name='trpt',
                            locat=self.unit_number[0])
        self.trpv = util_2d(model,(nlay,),np.float32,trpv,name='trpv',
                            locat=self.unit_number[0])
        self.dmcoef = []
        a = util_3d(model, (nlay, nrow, ncol), np.float32, dmcoef,
                    name='dmcoef1', locat=self.unit_number[0])
        self.dmcoef.append(a)
        if self.multiDiff:
            for icomp in range(2, ncomp+1):
                name = "dmcoef" + str(icomp)
                val = 0.0
                if name in list(kwargs.keys()):
                    val = kwargs[name]
                    kwargs.pop(name)
                else:
                    print("DSP: setting dmcoef for component " +\
                          str(icomp) + " to zero, kwarg name " +\
                          name)
                a = util_3d(model, (nlay, nrow, ncol), np.float32, val,
                            name=name, locat=self.unit_number[0])
                self.dmcoef.append(a)
        if len(list(kwargs.keys())) > 0:
            raise Exception("DSP error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the file.
        """

        # Get size
        nrow, ncol, nlay, nper = self.parent.mf.nrow_ncol_nlay_nper

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
            dum, dum, nlay, dum = model.mf.nrow_ncol_nlay_nper
        if nrow is None:
            nrow, dum, dum, dum = model.mf.nrow_ncol_nlay_nper
        if ncol is None:
            dum, ncol, dum, dum = model.mf.nrow_ncol_nlay_nper

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # Check for keywords (multidiffusion)
        multiDiff = False
        if line[0] == '$':
            keywords = line[0:].strip().split()
            for k in keywords:
                if k.lower() == 'multidiffusion':
                    multiDiff = True
            line = f.readline()

        # Read arrays
        if model.verbose:
            print('   loading AL...')
        al = util_3d.load(f, model, (nlay, nrow, ncol), np.float32, 'al',
                          ext_unit_dict)

        if model.verbose:
            print('   loading TRPT...')
        trpt = util_2d.load(f, model, (nlay, 1), np.float32, 'trpt',
                            ext_unit_dict)

        if model.verbose:
            print('   loading TRPV...')
        trpv = util_2d.load(f, model, (nlay, 1), np.float32, 'trpv',
                            ext_unit_dict)

        if model.verbose:
            print('   loading DMCOEFF...')
        if multiDiff:
            raise NotImplementedError("dsp.load() doesn't support multidiffusion yet.")
        else:
            dmcoef = util_2d.load(f, model, (nlay, 1), np.float32, 'dmcoef',
                            ext_unit_dict)

        dsp = Mt3dDsp(model, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef,
                      multiDiff=multiDiff)
        return dsp
