"""
mbase module
  This module contains the base model class from which
  all of the other models inherit from.

"""

from __future__ import print_function
import sys
import os
import subprocess as sp
import copy
import numpy as np
from flopy import utils

# Global variables
iconst = 1  # Multiplier for individual array elements in integer and real arrays read by MODFLOW's U2DREL, U1DREL and U2DINT.
iprn = -1  # Printout flag. If >= 0 then array values read are printed in listing file.


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        # test for exe in current working directory
        if is_exe(program):
            return program
        # test for exe in path statement
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


class BaseModel(object):
    """
    MODFLOW based models base class

    Parameters
    ----------

    modelname : string
        Name of the model.  Model files will be given this name. (default is
        'modflowtest'

    namefile_ext : string
        name file extension (default is 'nam')

    exe_name : string
        name of the modflow executable

    model_ws : string
        Path to the model workspace.  Model files will be created in this
        directory.  Default is None, in which case model_ws is assigned
        to the current working directory.

    """

    def __init__(self, modelname='modflowtest', namefile_ext='nam',
                 exe_name='mf2k.exe', model_ws=None,
                 structured=True):
        """
        BaseModel init
        """
        self.__name = modelname
        self.namefile_ext = namefile_ext
        self.namefile = self.__name + '.' + self.namefile_ext
        self.packagelist = []
        self.heading = ''
        self.exe_name = exe_name
        self.external_extension = 'ref'
        if model_ws is None: model_ws = os.getcwd()
        if not os.path.exists(model_ws):
            try:
                os.makedirs(model_ws)
            except:
                print(
                    '\n{0:s} not valid, workspace-folder was changed to {1:s}\n'.format(
                        model_ws, os.getcwd()))
                model_ws = os.getcwd()
        self._model_ws = model_ws
        self.structured = structured
        self.pop_key_list = []
        self.cl_params = ''

        # Model file information
        # external option stuff
        self.free_format = True
        self.array_format = None
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.package_units = []

        return

    def next_ext_unit(self):
        """
        Function to encapsulate next_ext_unit attribute

        """
        next_unit = self._next_ext_unit + 1
        self._next_ext_unit += 1
        return next_unit

    def export(self, f, **kwargs):
        # for pak in self.packagelist:
        #    f = pak.export(f)
        # return f
        from .export import utils
        return utils.model_helper(f, self, **kwargs)

    def add_package(self, p):
        """
        Add a package.

        Parameters
        ----------
        p : Package object

        """
        for u in p.unit_number:
            if u in self.package_units or u in self.external_units:
                print("WARNING: unit {0} of package {1} already in use".format(
                      u,p.name))
            self.package_units.append(u)
        for i, pp in enumerate(self.packagelist):
            if pp.allowDuplicates:
                continue
            elif isinstance(p, type(pp)):
                print('****Warning -- two packages of the same type: ',
                      type(p), type(pp))
                print('replacing existing Package...')
                self.packagelist[i] = p
                return
        if self.verbose:
            print('adding Package: ', p.name[0])
        self.packagelist.append(p)

    def remove_package(self, pname):
        """
        Remove a package from this model

        Parameters
        ----------
        pname : string
            Name of the package, such as 'RIV', 'BAS6', etc.

        """
        for i, pp in enumerate(self.packagelist):
            if pname in pp.name:
                if self.verbose:
                    print('removing Package: ', pp.name)
                self.packagelist.pop(i)
                return
        raise StopIteration(
            'Package name ' + pname + ' not found in Package list')

    def __getattr__(self, item):
        """
        __getattr__ - syntactic sugar

        Parameters
        ----------
        item : str
            3 character package name (case insensitive)

        Returns
        -------
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        """
        return self.get_package(item)

    def add_external(self, fname, unit, binflag=False):
        """
        Assign an external array so that it will be listed as a DATA or
        DATA(BINARY) entry in the name file.  This will allow an outside
        file package to refer to it.

        Parameters
        ----------
        fname : str
            filename of external array
        unit : int
            unit number of external array
        binflag : boolean
            binary or not. (default is False)

        """
        if fname in self.external_fnames:
            print("BaseModel.add_external() warning: " +
                  "replacing existing filename {0}".format(fname))
            idx = self.external_fnames.index(fname)
            self.external_fnames.pop(idx)
            self.external_units.pop(idx)
            self.external_binflag.pop(idx)

        self.external_fnames.append(fname)
        self.external_units.append(unit)
        self.external_binflag.append(binflag)
        return

    def remove_external(self, fname=None, unit=None):
        """
        Remove an external file from the model by specifying either the
        file name or the unit number.

        Parameters
        ----------
        fname : str
            filename of external array
        unit : int
            unit number of external array

        """
        if fname is not None:
            for i, e in enumerate(self.external_fnames):
                if fname in e:
                    self.external_fnames.pop(i)
                    self.external_units.pop(i)
                    self.external_binflag.pop(i)
        elif unit is not None:
            for i, u in enumerate(self.external_units):
                if u == unit:
                    self.external_fnames.pop(i)
                    self.external_units.pop(i)
                    self.external_binflag.pop(i)
        else:
            raise Exception(
                ' either fname or unit must be passed to remove_external()')
        return

    def get_name_file_entries(self):
        """
        Get a string representation of the name file.

        Parameters
        ----------

        """
        s = ''
        for p in self.packagelist:
            for i in range(len(p.name)):
                if p.unit_number[i] == 0:
                    continue
                s = s + ('{0:12s} {1:3d} {2:s} {3:s}\n'.format(p.name[i],
                                                               p.unit_number[i],
                                                               p.file_name[i],
                                                               p.extra[i]))
        return s

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
        for pp in (self.packagelist):
            if (pp.name[0].upper() == name.upper()):
                return pp
        return None

    def get_package_list(self):
        """
        Get a list of all the package names.

        Parameters
        ----------

        Returns
        -------
        val : list of strings
            Can be used to see what packages are in the model, and can then
            be used with get_package to pull out individual packages.

        """
        val = []
        for pp in (self.packagelist):
            val.append(pp.name[0].upper())
        return val

    def change_model_ws(self, new_pth=None,reset_external=False):
        """
        Change the model work space.

        Parameters
        ----------
        new_pth : str
            Location of new model workspace.  If this path does not exist,
            it will be created. (default is None, which will be assigned to
            the present working directory).

        Returns
        -------
        val : list of strings
            Can be used to see what packages are in the model, and can then
            be used with get_package to pull out individual packages.

        """
        if new_pth is None:
            new_pth = os.getcwd()
        if not os.path.exists(new_pth):
            try:
                sys.stdout.write(
                    '\ncreating model workspace...\n   {}\n'.format(new_pth))
                os.makedirs(new_pth)
            except:
                # print '\n%s not valid, workspace-folder was changed to %s\n' % (new_pth, os.getcwd())
                print(
                    '\n{0:s} not valid, workspace-folder was changed to {1:s}\n'.format(
                        new_pth, os.getcwd()))
                new_pth = os.getcwd()
        # --reset the model workspace
        self._model_ws = new_pth
        sys.stdout.write(
            '\nchanging model workspace...\n   {}\n'.format(new_pth))
        # reset the paths for each package
        for pp in (self.packagelist):
            pp.fn_path = os.path.join(self.model_ws, pp.file_name[0])

        # create the external path (if needed)
        if hasattr(self, "external_path") and self.external_path is not None \
                and not os.path.exists(os.path.join(self._model_ws,
                                                    self.external_path)):
            pth = os.path.join(self._model_ws, self.external_path)
            os.makedirs(pth)
            if reset_external:
                self._reset_external(pth)
        elif reset_external:
            self._reset_external(self._model_ws)
        return None

    def _reset_external(self,pth):
        new_ext_fnames = []
        for ext_file in self.external_fnames:
            new_ext_file = os.path.join(pth,os.path.split(ext_file)[-1])
            new_ext_fnames.append(new_ext_file)
        self.external_fnames = new_ext_fnames

    @property
    def model_ws(self):
        return copy.deepcopy(self._model_ws)

    def _set_name(self, value):
        """
        Set model name

        Parameters
        ----------
        value : str
            Name to assign to model.

        """
        self.__name = str(value)
        self.namefile = self.__name + '.' + self.namefile_ext
        for p in self.packagelist:
            for i in range(len(p.extension)):
                p.file_name[i] = self.__name + '.' + p.extension[i]
            p.fn_path = os.path.join(self.model_ws, p.file_name[0])

    def __setattr__(self, key, value):

        if key == "name":
            self._set_name(value)
        elif key == "model_ws":
            self.change_model_ws(value)
        else:
            super(BaseModel, self).__setattr__(key, value)

    def run_model(self, silent=False, pause=False, report=False,
                  normal_msg='normal termination'):
        """
        This method will run the model using subprocess.Popen.

        Parameters
        ----------
        silent : boolean
            Echo run information to screen (default is True).
        pause : boolean, optional
            Pause upon completion (default is False).
        report : boolean, optional
            Save stdout lines to a list (buff) which is returned
            by the method . (default is False).
        normal_msg : str
            Normal termination message used to determine if the
            run terminated normally. (default is 'normal termination')

        Returns
        -------
        (success, buff)
        success : boolean
        buff : list of lines of stdout

        """

        return run_model(self.exe_name, self.namefile, model_ws=self.model_ws,
                         silent=silent, pause=pause, report=report,
                         normal_msg=normal_msg)

    def load_results(self):

        print('load_results not implemented')

        return None

    def write_input(self, SelPackList=False, check=False):
        """
        Write the input.

        Parameters
        ----------
        SelPackList : False or list of packages

        """
        if check:
            # run check prior to writing input
            self.check(f='{}.chk'.format(self.name), verbose=self.verbose, level=1)

        # org_dir = os.getcwd()
        # os.chdir(self.model_ws)
        if self.verbose:
            print('\nWriting packages:')
        if SelPackList == False:
            for p in self.packagelist:
                if self.verbose:
                    print('   Package: ', p.name[0])
                # prevent individual package checks from running after model-level package check above
                # otherwise checks are run twice
                # or the model level check proceedure would have to be split up
                # or each package would need a check arguemnt,
                # or default for package level check would have to be False
                try:
                    p.write_file(check=False)
                except TypeError:
                    p.write_file()
        else:
            for pon in SelPackList:
                for i, p in enumerate(self.packagelist):
                    if pon in p.name:
                        if self.verbose:
                            print('   Package: ', p.name[0])
                    try:
                        p.write_file(check=False)
                    except TypeError:
                        p.write_file()
                        break
        if self.verbose:
            print(' ')
        # write name file
        self.write_name_file()
        # os.chdir(org_dir)
        return

    def write_name_file(self):
        """
        Every Package needs its own writenamefile function

        """
        raise Exception(
            'IMPLEMENTATION ERROR: writenamefile must be overloaded')

    @property
    def name(self):
        """
        Get model name

        Returns
        -------
        name : str
            name of model

        """
        return copy.deepcopy(self.__name)

    def add_pop_key_list(self, key):
        """
        Add a external file unit number to a list that will be used to remove
        model output (typically binary) files from ext_unit_dict.

        Parameters
        ----------
        key : int
            file unit number

        Returns
        -------

        Examples
        --------

        """
        if key not in self.pop_key_list:
            self.pop_key_list.append(key)

    def check(self, f=None, verbose=True, level=1):
        """
        Check model data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.check()
        """

        results = {}
        for p in self.packagelist:
            results[p.name[0]] = p.check(f=None, verbose=False, level=level-1)

        # check instance for model-level check
        chk = utils.check(self, f=f, verbose=verbose, level=level)

        # model level checks
        # solver check
        if self.version in chk.solver_packages.keys():
            solvers = set(chk.solver_packages[self.version]).intersection(set(self.get_package_list()))
            if not solvers:
                chk._add_to_summary('Error', desc='\r    No solver package',
                                    package='model')
            elif len(list(solvers)) > 1:
                for s in solvers:
                    chk._add_to_summary('Error', desc='\r    Multiple solver packages',
                                        package=s)
            else:
                chk.passed.append('Compatible solver package')

        # check for unit number conflicts
        package_units = {}
        duplicate_units = {}
        for p in self.packagelist:
            for i in range(len(p.name)):
                if p.unit_number[i] != 0:
                    if p.unit_number[i] in package_units.values():
                        duplicate_units[p.name[i]] = p.unit_number[i]
                        otherpackage = [k for k, v in package_units.items()
                                        if v == p.unit_number[i]][0]
                        duplicate_units[otherpackage] = p.unit_number[i]
        if len(duplicate_units) > 0:
            for k, v in duplicate_units.items():
                chk._add_to_summary('Error', package=k, value=v, desc='unit number conflict')
        else:
            chk.passed.append('Unit number conflicts')

        # add package check results to model level check summary
        for k, r in results.items():
            if r is not None and r.summary_array is not None: # currently SFR doesn't have one
                chk.summary_array = np.append(chk.summary_array, r.summary_array).view(np.recarray)
                chk.passed += ['{} package: {}'.format(r.package.name[0], psd) for psd in r.passed]
        chk.summarize()
        return chk

    def plot(self, SelPackList=None, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        model input data

        Parameters
        ----------
        SelPackList : bool or list
            List of of packages to plot. If SelPackList=None all packages
            are plotted. (default is None)
        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return.
                (default is zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        ----------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.plot()

        """
        # valid keyword arguments
        if 'kper' in kwargs:
            kper = int(kwargs.pop('kper'))
        else:
            kper = 0

        if 'mflay' in kwargs:
            mflay = kwargs.pop('mflay')
        else:
            mflay = None

        if 'filename_base' in kwargs:
            fileb = kwargs.pop('filename_base')
        else:
            fileb = None

        if 'file_extension' in kwargs:
            fext = kwargs.pop('file_extension')
            fext = fext.replace('.', '')
        else:
            fext = 'png'

        if 'key' in kwargs:
            key = kwargs.pop('key')
        else:
            key = None

        if self.verbose:
            print('\nPlotting Packages')

        axes = []
        ifig = 0
        if SelPackList is None:
            for p in self.packagelist:
                caxs = p.plot(initial_fig=ifig,
                              filename_base=fileb, file_extension=fext,
                              kper=kper, mflay=mflay, key=key)
                # unroll nested lists of axes into a single list of axes
                if isinstance(caxs, list):
                    for c in caxs:
                        axes.append(c)
                else:
                    axes.append(caxs)
                # update next active figure number
                ifig = len(axes) + 1
        else:
            for pon in SelPackList:
                for i, p in enumerate(self.packagelist):
                    if pon in p.name:
                        if self.verbose:
                            print('   Plotting Package: ', p.name[0])
                        caxs = p.plot(initial_fig=ifig,
                                      filename_base=fileb, file_extension=fext,
                                      kper=kper, mflay=mflay, key=key)
                        # unroll nested lists of axes into a single list of axes
                        if isinstance(caxs, list):
                            for c in caxs:
                                axes.append(c)
                        else:
                            axes.append(caxs)
                        # update next active figure number
                        ifig = len(axes) + 1
                        break
        if self.verbose:
            print(' ')
        return axes

    def to_shapefile(self, filename, package_names=None, **kwargs):
        """
        Wrapper function for writing a shapefile for the model grid.  If
        package_names is not None, then search through the requested packages
        looking for arrays that can be added to the shapefile as attributes

        Parameters
        ----------
        filename : string
            name of the shapefile to write
        package_names : list of package names (e.g. ["dis","lpf"])
            Packages to export data arrays to shapefile. (default is None)

        Returns
        -------
        None

        Examples
        --------
        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> m.to_shapefile('model.shp', SelPackList)

        """
        import warnings
        warnings.warn("to_shapefile() is deprecated. use .export()")
        self.export(filename, package_names=package_names)
        return


def run_model(exe_name, namefile, model_ws='./',
              silent=False, pause=False, report=False,
              normal_msg='normal termination'):
    """
    This function will run the model using subprocess.Popen.

    Parameters
    ----------
    exe_name : str
        Executable name (with path, if necessary) to run.
    namefile : str
        Namefile of model to run. The namefile must be the
        filename of the namefile without the path.
    model_ws : str
        Path to the location of the namefile. (default is the
        current working directory - './')
    silent : boolean
        Echo run information to screen (default is True).
    pause : boolean, optional
        Pause upon completion (default is False).
    report : boolean, optional
        Save stdout lines to a list (buff) which is returned
        by the method . (default is False).
    normal_msg : str
        Normal termination message used to determine if the
        run terminated normally. (default is 'normal termination')

    Returns
    -------
    (success, buff)
    success : boolean
    buff : list of lines of stdout

    """
    success = False
    buff = []

    # Check to make sure that program and namefile exist
    exe = which(exe_name)
    if exe is None:
        import platform

        if platform.system() in 'Windows':
            if not exe_name.lower().endswith('.exe'):
                exe = which(exe_name + '.exe')
    if exe is None:
        s = 'The program {} does not exist or is not executable.'.format(
            exe_name)
        raise Exception(s)
    else:
        if not silent:
            s = 'FloPy is using the following executable to run the model: {}'.format(
                exe)
            print(s)

    if not os.path.isfile(os.path.join(model_ws, namefile)):
        s = 'The namefile for this model does not exists: {}'.format(namefile)
        raise Exception(s)

    proc = sp.Popen([exe_name, namefile],
                    stdout=sp.PIPE, cwd=model_ws)
    while True:
        line = proc.stdout.readline()
        c = line.decode('utf-8')
        if c != '':
            if normal_msg in c.lower():
                success = True
            c = c.rstrip('\r\n')
            if not silent:
                print('{}'.format(c))
            if report == True:
                buff.append(c)
        else:
            break
    if pause:
        input('Press Enter to continue...')
    return success, buff
