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


class Seawat0(BaseModel):
    """
    SEAWAT base class

    """
    def __init__(self, modelname='swttest', namefile_ext='nam',
                 modflowmodel=None, mt3dmsmodel=None, 
                 version='seawat', exe_name='swt_v4.exe', model_ws=None,
                 verbose=False, external_path=None):
        BaseModel.__init__(self, modelname, namefile_ext, exe_name=exe_name, 
                           model_ws=model_ws)

        self.version_types = {'seawat': 'SEAWAT'}
        self.set_version(version)

        self.__mf = modflowmodel
        self.__mt = mt3dmsmodel
        self.lst = SeawatList(self)
        self.__vdf = None
        self.verbose = verbose
        self.external_path = external_path
        return
        
    def __repr__( self ):
        return 'SEAWAT model'

    def getvdf(self):
        if (self.__vdf == None):
            for p in (self.packagelist):
                if isinstance(p, SeawatVdf):
                    self.__vdf = p
        return self.__vdf

    def getmf(self):
        return self.__mf

    def getmt(self):
        return self.__mt

    mf = property(getmf) # Property has no setter, so read-only
    mt = property(getmt) # Property has no setter, so read-only
    vdf = property(getvdf) # Property has no setter, so read-only

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        fn_path = os.path.join(self.model_ws,self.namefile)
        f_nam = open(fn_path, 'w')
        f_nam.write('%s\n' % (self.heading) )
        f_nam.write('%s\t%3i\t%s\n' % (self.lst.name[0], 
                                       self.lst.unit_number[0], 
                                       self.lst.file_name[0]))
        f_nam.write('%s\n' % ('# Flow') )
        f_nam.write('%s' % self.__mf.get_name_file_entries())
        for u,f in zip(self.mf.external_units,self.mf.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u)+f+'\n'	)
        f_nam.write('%s\n' % ('# Transport') )
        f_nam.write('%s' % self.__mt.get_name_file_entries())
        for u,f in zip(self.mt.external_units,self.mt.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u)+f+'\n'	)
        f_nam.write('%s\n' % ('# Variable density flow') )
        f_nam.write('%s' % self.get_name_file_entries())
        f_nam.close()
        return


class Seawat2(BaseModel):
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
    >>> m = flopy.seawat.swt.Seawat2()

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

        # If a MODFLOW model was passed in, then use it, otherwise create
        # one.
        if modflowmodel is not None:
            self.mf = modflowmodel
            self.packagelist.extend(self.mf.packagelist)
        else:
            self.mf = Modflow(modelname=modelname, version='mf2k',
                              exe_name='mf2k.exe', structured=structured,
                              listunit=listunit, model_ws=model_ws,
                              external_path=external_path, verbose=verbose,
                              namefile_ext='nam_mf')

        # If a MT3D model was passed in, then use it, otherwise create
        # one.
        if mt3dmodel is not None:
            self.mt = mt3dmodel
            self.packagelist.extend(self.mt.packagelist)
        else:
            self.mt = Mt3dms(modelname=modelname, version='mt3dms',
                             exe_name='mt3dms.exe', structured=structured,
                             listunit=listunit, model_ws=model_ws,
                             external_path=external_path, verbose=verbose,
                             namefile_ext='nam_mt3d')

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
        self.mfnam_packages = {
            'vdf': SeawatVdf,
            'vsc': SeawatVsc,
        }
        return

    def __getattr__(self, item):
        """
        __getattr__ - syntactic sugar (overriding for SEAWAT)

        Parameters
        ----------
        item : str
            3 character package name (case insensitive) or "sr" to access
            the SpatialReference instance of the ModflowDis object


        Returns
        -------
        sr : SpatialReference instance
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        Note
        ----
        if self.dis is not None, then the spatial reference instance is updated
        using self.dis.delr, self.dis.delc, and self.dis.lenuni before being
        returned
        """
        item = item.lower()
        if item == 'sr':
            if self.mf.dis is not None:
                return self.mf.dis.sr
            else:
                return None
        if item == "start_datetime":
            if self.mf.dis is not None:
                return self.mf.dis.start_datetime
            else:
                return None

        # find package in one of three models and return
        if item in self.mfnam_packages:
            return self.get_package(item)
        elif item in self.mf.mfnam_packages:
            return self.mf.get_package(item)
        elif item in self.mt.mfnam_packages:
            return self.mt.get_package(item)
        return None

    @property
    def nlay(self):
        if (self.mf.dis):
            return self.mf.dis.nlay
        else:
            return 0

    @property
    def nrow(self):
        if (self.mf.dis):
            return self.mf.dis.nrow
        else:
            return 0

    @property
    def ncol(self):
        if (self.mf.dis):
            return self.mf.dis.ncol
        else:
            return 0

    @property
    def nper(self):
        if (self.mf.dis):
            return self.dis.nper
        else:
            return 0

    @property
    def nrow_ncol_nlay_nper(self):
        dis = self.mf.get_package('DIS')
        if (dis):
            return dis.nrow, dis.ncol, dis.nlay, dis.nper
        else:
            return 0, 0, 0, 0

    def get_nrow_ncol_nlay_nper(self):
        return self.nrow_ncol_nlay_nper

    @property
    def ncomp(self):
        if (self.mt.btn):
            return self.mt.btn.ncomp
        else:
            return 1

    @property
    def mcomp(self):
        if (self.mt.btn):
            return self.mt.btn.mcomp
        else:
            return 1

    def get_ifrefm(self):
        bas = self.get_package('BAS6')
        if (bas):
            return bas.ifrefm
        else:
            return False

    def add_package(self, p):
        """
        Add a package.

        Parameters
        ----------
        p : Package object

        """
        pname = p.name[0].lower()
        if pname in self.mfnam_packages:
            super(Seawat2, self).add_package(p)
        elif pname in self.mf.mfnam_packages:
            self.mf.add_package(p)
        elif pname in self.mt.mfnam_packages:
            self.mt.add_package(p)
        return

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
        f_nam.write('%s\n' % (self.heading))

        # Write list file entry
        f_nam.write('%s\t%3i\t%s\n' % (self.lst.name[0],
                                       self.lst.unit_number[0],
                                       self.lst.file_name[0]))

        # Write MODFLOW entries
        f_nam.write('%s\n' % ('# Flow'))
        f_nam.write('%s' % self.mf.get_name_file_entries())
        for u, f in zip(self.mf.external_units, self.mf.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u) + f + '\n')

        # Write MT3DMS entries
        f_nam.write('%s\n' % ('# Transport'))
        f_nam.write('%s' % self.mt.get_name_file_entries())
        for u, f in zip(self.mt.external_units, self.mt.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u) + f + '\n')

        # Write SEAWAT entries and close
        f_nam.write('%s\n' % ('# Variable density flow'))
        f_nam.write('%s' % self.get_name_file_entries())
        f_nam.close()
        return

    def write_input(self, SelPackList=False, check=False):
        """
        Write the input.

        Parameters
        ----------
        SelPackList : False or list of packages

        """
        self.mf.write_input(SelPackList, check)
        self.mt.write_input(SelPackList, check)
        super(Seawat2, self).write_input(SelPackList, check)
        return

    def get_package(self, name):
        """
        Get a package.

        Parameters
        ----------
        name : str
            Name of the package, 'RIV', 'LPF', etc.

        Returns
        -------
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        """
        name = name.lower()
        if name in self.mfnam_packages:
            return super(Seawat2, self).get_package(name)
        elif name in self.mf.mfnam_packages:
            return self.mf.get_package(name)
        elif name in self.mt.mfnam_packages:
            return self.mt.get_package(name)
        return None


    @staticmethod
    def load(f, version='swt_v4', exe_name='swt_v4.exe', verbose=False,
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

        mf = Modflow.load(f, version='mf2k', exe_name=None, verbose=verbose,
                          model_ws=model_ws, load_only=load_only, forgive=True,
                          check=True)

        mt = Mt3dms.load(f, version='mt3dms', exe_name=None, verbose=verbose,
                         model_ws=model_ws)

        ms = Seawat2(modelname='swttest', namefile_ext='nam',
                     modflowmodel=mf, mt3dmodel=mt,
                     version='seawat', exe_name='swt_v4', model_ws=model_ws,
                     verbose=verbose)

        # return model object
        return ms


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
        self.array_free_format = True
        self.array_format = 'modflow'
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
        f_nam.write('%s\n' % (self.heading))

        # Write list file entry
        f_nam.write('%s\t%3i\t%s\n' % (self.lst.name[0],
                                       self.lst.unit_number[0],
                                       self.lst.file_name[0]))

        # Write SEAWAT entries and close
        f_nam.write('%s' % self.get_name_file_entries())
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

        mf = Modflow.load(f, version='mf2k', exe_name=None, verbose=verbose,
                          model_ws=model_ws, load_only=load_only, forgive=True,
                          check=False)

        mt = Mt3dms.load(f, version='mt3dms', exe_name=None, verbose=verbose,
                         model_ws=model_ws, forgive=True)

        ms = Seawat(modelname=modelname, namefile_ext='nam',
                    modflowmodel=None, mt3dmodel=None,
                    version=version, exe_name=exe_name, model_ws=model_ws,
                    verbose=verbose)

        for p in mf.packagelist:
            p.parent = ms
            ms.add_package(p)

        if mt is not None:
            for p in mt.packagelist:
                p.parent = ms
                ms.add_package(p)

        # return model object
        return ms
