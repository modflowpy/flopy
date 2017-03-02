import os
from ..mbase import BaseModel
from ..pakbase import Package
from ..modflow import Modflow
from ..mt3d import Mt3dms
from .swtvdf import SeawatVdf
from .swtvsc import SeawatVsc

class SeawatList(Package):
    """
    List Package class
    """
    def __init__(self, model, extension='list', listunit=7):
        Package.__init__(self, model, extension, 'LIST', listunit)
        return

    def __repr__( self ):
        return 'List package class'

    def write_file(self):
        # Not implemented for list class
        return


class Seawat(BaseModel):
    """
    SEAWAT Model Class.

    Parameters
    ----------
    modelname : string, optional
        Name of model.  This string will be used to name the SEAWAT input
        that are created with write_model. (the default is 'swttest')
    namefile_ext : string, optional
        Extension for the namefile (the default is 'nam')
    version : string, optional
        Version of SEAWAT to use (the default is 'seawat').
    exe_name : string, optional
        The name of the executable to use (the default is
        'swt_v4.exe').
    listunit : integer, optional
        Unit number for the list file (the default is 2).
    model_ws : string, optional
        model workspace.  Directory name to create model data sets.
        (default is the present working directory).
    external_path : string
        Location for external files (default is None).
    verbose : boolean, optional
        Print additional information to the screen (default is False).
    load : boolean, optional
         (default is True).
    silent : integer
        (default is 0)

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
    >>> m = flopy.seawat.swt.Seawat()

    """

    def __init__(self, modelname='swttest', namefile_ext='nam',
                 modflowmodel=None, mt3dmodel=None,
                 version='seawat', exe_name='swt_v4',
                 structured=True, listunit=2, model_ws='.', external_path=None,
                 verbose=False, load=True, silent=0):

        # Call constructor for parent object
        BaseModel.__init__(self, modelname, namefile_ext, exe_name, model_ws,
                           structured=structured)

        # Set attributes
        self.version_types = {'seawat': 'SEAWAT'}
        self.set_version(version)
        self.lst = SeawatList(self, listunit=listunit)

        # If a MODFLOW model was passed in, then add its packages
        self.mf = self
        if modflowmodel is not None:
            for p in modflowmodel.packagelist:
                self.packagelist.append(p)
        else:
            modflowmodel = Modflow()

        # If a MT3D model was passed in, then add its packages
        if mt3dmodel is not None:
            for p in mt3dmodel.packagelist:
                self.packagelist.append(p)
        else:
            mt3dmodel = Mt3dms()

        # external option stuff
        self.array_free_format = False
        self.array_format = 'mt3d'
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.external = False
        self.verbose = verbose
        self.load = load
        # the starting external data unit number
        self._next_ext_unit = 3000
        if external_path is not None:
            assert model_ws == '.', "ERROR: external cannot be used " + \
                                    "with model_ws"

            # external_path = os.path.join(model_ws, external_path)
            if os.path.exists(external_path):
                print("Note: external_path " + str(external_path) +
                      " already exists")
            # assert os.path.exists(external_path),'external_path does not exist'
            else:
                os.mkdir(external_path)
            self.external = True
        self.external_path = external_path
        self.verbose = verbose
        self.silent = silent

        # Create a dictionary to map package with package object.
        # This is used for loading models.
        self.mfnam_packages = {}
        for k, v in modflowmodel.mfnam_packages.items():
            self.mfnam_packages[k] = v
        for k, v in mt3dmodel.mfnam_packages.items():
            self.mfnam_packages[k] = v
        self.mfnam_packages['vdf'] = SeawatVdf
        self.mfnam_packages['vsc'] = SeawatVsc
        return

    @property
    def nlay(self):
        if (self.dis):
            return self.dis.nlay
        else:
            return 0

    @property
    def nrow(self):
        if (self.dis):
            return self.dis.nrow
        else:
            return 0

    @property
    def ncol(self):
        if (self.dis):
            return self.dis.ncol
        else:
            return 0

    @property
    def nper(self):
        if (self.dis):
            return self.dis.nper
        else:
            return 0

    @property
    def nrow_ncol_nlay_nper(self):
        dis = self.get_package('DIS')
        if (dis):
            return dis.nrow, dis.ncol, dis.nlay, dis.nper
        else:
            return 0, 0, 0, 0

    def get_nrow_ncol_nlay_nper(self):
        return self.nrow_ncol_nlay_nper

    def get_ifrefm(self):
        bas = self.get_package('BAS6')
        if (bas):
            return bas.ifrefm
        else:
            return False

    @property
    def ncomp(self):
        if (self.btn):
            return self.btn.ncomp
        else:
            return 1

    @property
    def mcomp(self):
        if (self.btn):
            return self.btn.mcomp
        else:
            return 1

    def _set_name(self, value):
        # Overrides BaseModel's setter for name property
        BaseModel._set_name(self, value)

        #for i in range(len(self.lst.extension)):
        #    self.lst.file_name[i] = self.name + '.' + self.lst.extension[i]
        #return

    def change_model_ws(self, new_pth=None, reset_external=False):
        #if hasattr(self,"_mf"):
        if self._mf is not None:
            self._mf.change_model_ws(new_pth=new_pth,
                                     reset_external=reset_external)
        #if hasattr(self,"_mt"):
        if self._mt is not None:
            self._mt.change_model_ws(new_pth=new_pth,
                                     reset_external=reset_external)
        super(Seawat,self).change_model_ws(new_pth=new_pth,
                                           reset_external=reset_external)
    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        # open and write header
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, 'w')
        f_nam.write('{}\n'.format(self.heading))

        # Write global file entry
        if self.glo is not None:
            if self.glo.unit_number[0] > 0:
                f_nam.write('{:14s} {:5d}  {}\n'.format(self.glo.name[0],
                                                        self.glo.unit_number[0],
                                                        self.glo.file_name[0]))
        # Write list file entry
        f_nam.write('{:14s} {:5d}  {}\n'.format(self.lst.name[0],
                                                self.lst.unit_number[0],
                                                self.lst.file_name[0]))

        # Write SEAWAT entries and close
        f_nam.write('{}'.format(self.get_name_file_entries()))

        if self._mf is not None:
            # write the external files
            for b, u, f in zip(self._mf.external_binflag,
                               self._mf.external_units, \
                               self._mf.external_fnames):
                tag = "DATA"
                if b:
                    tag = "DATA(BINARY)"
                f_nam.write('{0:14s} {1:5d}  {2}\n'.format(tag, u, f))

            # write the output files
            for u, f, b in zip(self._mf.output_units, self._mf.output_fnames,
                               self._mf.output_binflag):
                if u == 0:
                    continue
                if b:
                    f_nam.write(
                        'DATA(BINARY)   {0:5d}  '.format(u) + f + ' REPLACE\n')
                else:
                    f_nam.write('DATA           {0:5d}  '.format(u) + f + '\n')

        if self._mt is not None:
            # write the external files
            for b, u, f in zip(self._mt.external_binflag,
                               self._mt.external_units, \
                               self._mt.external_fnames):
                tag = "DATA"
                if b:
                    tag = "DATA(BINARY)"
                f_nam.write('{0:14s} {1:5d}  {2}\n'.format(tag, u, f))

            # write the output files
            for u, f, b in zip(self._mt.output_units, self._mt.output_fnames,
                               self._mt.output_binflag):
                if u == 0:
                    continue
                if b:
                    f_nam.write(
                        'DATA(BINARY)   {0:5d}  '.format(u) + f + ' REPLACE\n')
                else:
                    f_nam.write('DATA           {0:5d}  '.format(u) + f + '\n')

        # write the external files
        for b, u, f in zip(self.external_binflag, self.external_units, \
                           self.external_fnames):
            tag = "DATA"
            if b:
                tag = "DATA(BINARY)"
            f_nam.write('{0:14s} {1:5d}  {2}\n'.format(tag, u, f))


        # write the output files
        for u, f, b in zip(self.output_units, self.output_fnames,
                           self.output_binflag):
            if u == 0:
                continue
            if b:
                f_nam.write(
                    'DATA(BINARY)   {0:5d}  '.format(u) + f + ' REPLACE\n')
            else:
                f_nam.write('DATA           {0:5d}  '.format(u) + f + '\n')

        f_nam.close()
        return

    @staticmethod
    def load(f, version='seawat', exe_name='swt_v4', verbose=False,
             model_ws='.', load_only=None):
        """
        Load an existing model.

        Parameters
        ----------
        f : string
            Full path and name of SEAWAT name file.

        version : string
            The version of SEAWAT (seawat)
            (default is seawat)

        exe_name : string
            The name of the executable to use if this loaded model is run.
            (default is swt_v4.exe)

        verbose : bool
            Write information on the load process if True.
            (default is False)

        model_ws : string
            The path for the model workspace.
            (default is the current working directory '.')

        load_only : list of strings
            Filetype(s) to load (e.g. ['lpf', 'adv'])
            (default is None, which means that all will be loaded)

        Returns
        -------
        m : flopy.seawat.swt.Seawat
            flopy Seawat model object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.seawat.swt.Seawat.load(f)

        """
        # test if name file is passed with extension (i.e., is a valid file)
        if os.path.isfile(os.path.join(model_ws, f)):
            modelname = f.rpartition('.')[0]
        else:
            modelname = f

        # create instance of a seawat model and load modflow and mt3dms models
        ms = Seawat(modelname=modelname, namefile_ext='nam',
                    modflowmodel=None, mt3dmodel=None,
                    version=version, exe_name=exe_name, model_ws=model_ws,
                    verbose=verbose)

        mf = Modflow.load(f, version='mf2k', exe_name=None, verbose=verbose,
                          model_ws=model_ws, load_only=load_only, forgive=True,
                          check=False)

        mt = Mt3dms.load(f, version='mt3dms', exe_name=None, verbose=verbose,
                         model_ws=model_ws, forgive=True)

        # set listing and global files using mf objects
        ms.lst = mf.lst
        ms.glo = mf.glo

        for p in mf.packagelist:
            p.parent = ms
            ms.add_package(p)
        ms._mt = None
        if mt is not None:
            for p in mt.packagelist:
                p.parent = ms
                ms.add_package(p)
            mt.external_units = []
            mt.external_binflag = []
            mt.external_fnames = []
            ms._mt = mt
        ms._mf = mf



        # return model object
        return ms
