"""
mbase module
  This module contains the base model class from which
  all of the other models inherit from.

"""
import abc
import copy
import os
import queue as Queue
import shutil
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path
from shutil import which
from subprocess import PIPE, STDOUT, Popen
from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np

from . import discretization, utils
from .discretization.grid import Grid
from .utils import flopy_io
from .version import __version__

# Prepend flopy appdir bin directory to PATH to work with "get-modflow :flopy"
if sys.platform.startswith("win"):
    flopy_bin = os.path.expandvars(r"%LOCALAPPDATA%\flopy\bin")
else:
    flopy_bin = os.path.join(os.path.expanduser("~"), ".local/share/flopy/bin")
os.environ["PATH"] = flopy_bin + os.path.pathsep + os.environ.get("PATH", "")

## Global variables
# Multiplier for individual array elements in integer and real arrays read by
# MODFLOW's U2DREL, U1DREL and U2DINT.
iconst = 1
# Printout flag. If >= 0 then array values read are printed in listing file.
iprn = -1


def resolve_exe(exe_name: Union[str, os.PathLike]) -> str:
    """
    Resolves the absolute path of the executable.

    Parameters
    ----------
    exe_name : str or PathLike
        The executable's name or path. If only the name is provided,
        the executable must be on the system path.

    Returns
    -------
        str: absolute path to the executable
    """

    exe_name = str(exe_name)
    exe = which(exe_name)
    if exe is not None:
        # in case which() returned a relative path, resolve it
        exe = which(str(Path(exe).resolve()))
    else:
        if exe_name.lower().endswith(".exe"):
            # try removing .exe suffix
            exe = which(exe_name[:-4])
        if exe is not None:
            # in case which() returned a relative path, resolve it
            exe = which(str(Path(exe).resolve()))
        else:
            # try tilde-expanded abspath
            exe = which(Path(exe_name).expanduser().absolute())
        if exe is None and exe_name.lower().endswith(".exe"):
            # try tilde-expanded abspath without .exe suffix
            exe = which(Path(exe_name[:-4]).expanduser().absolute())
    if exe is None:
        raise FileNotFoundError(
            f"The program {exe_name} does not exist or is not executable."
        )
    return exe


# external exceptions for users
class PackageLoadException(Exception):
    """
    FloPy package load exception.
    """

    def __init__(self, error, location=""):
        """Initialize exception."""
        self.message = error
        super().__init__(f"{error} ({location})")


class FileDataEntry:
    def __init__(self, fname, unit, binflag=False, output=False, package=None):
        self.fname = fname
        self.unit = unit
        self.binflag = binflag
        self.output = output
        self.package = package


class FileData:
    def __init__(self):
        self.file_data = []
        return

    def add_file(self, fname, unit, binflag=False, output=False, package=None):
        ipop = []
        for idx, file_data in enumerate(self.file_data):
            if file_data.fname == fname or file_data.unit == unit:
                ipop.append(idx)

        self.file_data.append(
            FileDataEntry(
                fname, unit, binflag=binflag, output=output, package=package
            )
        )
        return


class ModelInterface:
    def __init__(self):
        self._mg_resync = True
        self._modelgrid = None

    def update_modelgrid(self):
        if self._modelgrid is not None:
            self._modelgrid = Grid(
                crs=self._modelgrid.crs,
                xoff=self._modelgrid.xoffset,
                yoff=self._modelgrid.yoffset,
                angrot=self._modelgrid.angrot,
            )
        self._mg_resync = True

    @property
    @abc.abstractmethod
    def modelgrid(self):
        raise NotImplementedError(
            "must define modelgrid in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def packagelist(self):
        raise NotImplementedError(
            "must define packagelist in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def namefile(self):
        raise NotImplementedError(
            "must define namefile in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def model_ws(self):
        raise NotImplementedError(
            "must define model_ws in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def exename(self):
        raise NotImplementedError(
            "must define exename in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def version(self):
        raise NotImplementedError(
            "must define version in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def solver_tols(self):
        raise NotImplementedError(
            "must define version in child class to use this base class"
        )

    @abc.abstractmethod
    def export(self, f, **kwargs):
        raise NotImplementedError(
            "must define export in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def laytyp(self):
        raise NotImplementedError(
            "must define laytyp in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def hdry(self):
        raise NotImplementedError(
            "must define hdry in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def hnoflo(self):
        raise NotImplementedError(
            "must define hnoflo in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def laycbd(self):
        raise NotImplementedError(
            "must define laycbd in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def verbose(self):
        raise NotImplementedError(
            "must define verbose in child class to use this base class"
        )

    @abc.abstractmethod
    def check(self, f=None, verbose=True, level=1):
        raise NotImplementedError(
            "must define check in child class to use this base class"
        )

    def get_package_list(self, ftype=None):
        """
        Get a list of all the package names.

        Parameters
        ----------
        ftype : str
            Type of package, 'RIV', 'LPF', etc.

        Returns
        -------
        val : list of strings
            Can be used to see what packages are in the model, and can then
            be used with get_package to pull out individual packages.

        """
        val = []
        for pp in self.packagelist:
            if ftype is None:
                val.append(pp.name[0].upper())
            elif pp.package_type.lower() == ftype:
                val.append(pp.name[0].upper())
        return val

    def _check(self, chk, level=1):
        """
        Check model data for common errors.

        Parameters
        ----------
        chk : the check object
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.
        summarize : bool
            Boolean flag used to determine if summary of results is written
            to the screen

        Returns
        -------
        Check object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.check()
        """

        # check instance for model-level check
        results = {}

        for p in self.packagelist:
            if chk.package_check_levels.get(p.name[0].lower(), 0) <= level:
                results[p.name[0]] = p.check(
                    f=None,
                    verbose=False,
                    level=level - 1,
                    checktype=chk.__class__,
                )

        # model level checks
        # solver check
        if self.version in chk.solver_packages.keys():
            solvers = set(chk.solver_packages[self.version]).intersection(
                set(self.get_package_list())
            )
            if not solvers:
                chk._add_to_summary(
                    "Error", desc="\r    No solver package", package="model"
                )
            elif len(list(solvers)) > 1:
                for s in solvers:
                    chk._add_to_summary(
                        "Error",
                        desc="\r    Multiple solver packages",
                        package=s,
                    )
            else:
                chk.passed.append("Compatible solver package")

        # add package check results to model level check summary
        for r in results.values():
            if (
                r is not None and r.summary_array is not None
            ):  # currently SFR doesn't have one
                chk.summary_array = np.append(
                    chk.summary_array, r.summary_array
                ).view(np.recarray)
                chk.passed += [
                    f"{r.package.name[0]} package: {psd}" for psd in r.passed
                ]
        chk.summarize()
        return chk


class BaseModel(ModelInterface):
    """
    MODFLOW-based models base class.

    Parameters
    ----------
    modelname : str, default "modflowtest"
        Name of the model, which is also used for model file names.
    namefile_ext : str, default "nam"
        Name file extension, without "."
    exe_name : str or PathLike, default "mf2005"
        Name or path of the modflow executable. If a name is provided,
        the executable must be on the system path.
    model_ws : str or PathLike, optional, default "."
        Path to the model workspace.  Model files will be created in this
        directory.  Default is the current working directory.
    structured : bool, default True
        Specify if model grid is structured (default) or unstructured.
    verbose : bool, default False
        Print additional information to the screen.
    **kwargs : dict, optional
        Used to define: ``xll``/``yll`` for the x- and y-coordinates of
        the lower-left corner of the grid, ``xul``/``yul`` for the
        x- and y-coordinates of the upper-left corner of the grid
        (deprecated), ``rotation`` for the grid rotation (default 0.0),
        ``crs`` for the coordinate reference system, and ``start_datetime`` for
        model start date (default "1-1-1970").

    """

    def __init__(
        self,
        modelname="modflowtest",
        namefile_ext="nam",
        exe_name: Union[str, os.PathLike] = "mf2005",
        model_ws: Union[str, os.PathLike] = os.curdir,
        structured=True,
        verbose=False,
        **kwargs,
    ):
        """Initialize BaseModel."""
        super().__init__()
        self.__name = modelname
        self.namefile_ext = namefile_ext or ""
        self._namefile = self.__name + "." + self.namefile_ext
        self._packagelist = []
        self.heading = ""
        try:
            self.exe_name = resolve_exe(exe_name)
        except:
            self.exe_name = "mf2005"
        self._verbose = verbose
        self.external_path = None
        self.external_extension = "ref"
        if model_ws is None:
            model_ws = os.getcwd()
        model_ws = Path(model_ws).expanduser().absolute()
        try:
            model_ws.mkdir(parents=True, exist_ok=True)
        except:
            warn(
                f"\n{model_ws} not valid, "
                f"workspace-folder was changed to {os.getcwd()}\n"
            )
            model_ws = os.getcwd()
        self._model_ws = str(model_ws)
        self.structured = structured
        self.pop_key_list = []
        self.cl_params = ""

        # check for reference info in kwargs
        # we are just carrying these until a dis package is added
        xll = kwargs.pop("xll", None)
        yll = kwargs.pop("yll", None)
        self._xul = kwargs.pop("xul", None)
        self._yul = kwargs.pop("yul", None)

        self._rotation = kwargs.pop("rotation", 0.0)
        self._crs = kwargs.pop("crs", None)
        self._start_datetime = kwargs.pop("start_datetime", "1-1-1970")

        # build model discretization objects
        self._modelgrid = Grid(
            crs=self._crs,
            xoff=xll,
            yoff=yll,
            angrot=self._rotation,
        )
        self._modeltime = None

        # Model file information
        self.__onunit__ = 10
        # external option stuff
        self.array_free_format = True
        self.free_format_input = True
        self.parameter_load = False
        self.array_format = None
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.external_output = []
        self.package_units = []
        self._next_ext_unit = None

        # output files
        self.output_fnames = []
        self.output_units = []
        self.output_binflag = []
        self.output_packages = []

    @property
    def modeltime(self):
        raise NotImplementedError(
            "must define modeltime in child class to use this base class"
        )

    @property
    def modelgrid(self):
        raise NotImplementedError(
            "must define modelgrid in child class to use this base class"
        )

    @property
    def packagelist(self):
        return self._packagelist

    @packagelist.setter
    def packagelist(self, packagelist):
        self._packagelist = packagelist

    @property
    def namefile(self) -> str:
        return self._namefile

    @namefile.setter
    def namefile(self, namefile: str):
        self._namefile = namefile

    @property
    def model_ws(self) -> str:
        return self._model_ws

    @model_ws.setter
    def model_ws(self, model_ws: Union[str, os.PathLike]):
        self._model_ws = str(Path(model_ws).expanduser().absolute())

    @property
    def exename(self) -> str:
        return self._exename

    @exename.setter
    def exename(self, exename):
        self._exename = resolve_exe(exename)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, version):
        self._version = version

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def laytyp(self):
        if self.get_package("LPF") is not None:
            return self.get_package("LPF").laytyp.array
        if self.get_package("BCF6") is not None:
            return self.get_package("BCF6").laycon.array
        if self.get_package("UPW") is not None:
            return self.get_package("UPW").laytyp.array

        return None

    @property
    def hdry(self):
        if self.get_package("LPF") is not None:
            return self.get_package("LPF").hdry
        if self.get_package("BCF6") is not None:
            return self.get_package("BCF6").hdry
        if self.get_package("UPW") is not None:
            return self.get_package("UPW").hdry
        return -1e30

    @property
    def hnoflo(self):
        try:
            bas6 = self.get_package("BAS6")
            return bas6.hnoflo
        except AttributeError:
            return 1e30

    @property
    def laycbd(self):
        try:
            dis = self.get_package("DIS")
            return dis.laycbd.array
        except AttributeError:
            return None

    # we don't need these - no need for controlled access to array_free_format
    # def set_free_format(self, value=True):
    #     """
    #     Set the free format flag for the model instance
    #
    #     Parameters
    #     ----------
    #     value : bool
    #         Boolean value to set free format flag for model. (default is True)
    #
    #     Returns
    #     -------
    #
    #     """
    #     if not isinstance(value, bool):
    #         print('Error: set_free_format passed value must be a boolean')
    #         return False
    #     self.array_free_format = value
    #
    # def get_free_format(self):
    #     """
    #     Return the free format flag for the model
    #
    #     Returns
    #     -------
    #     out : bool
    #         Free format flag for the model
    #
    #     """
    #     return self.array_free_format

    def next_unit(self, i=None):
        if i is not None:
            self.__onunit__ = i - 1
        else:
            self.__onunit__ += 1
        return self.__onunit__

    def next_ext_unit(self):
        """
        Function to encapsulate next_ext_unit attribute

        """
        next_unit = self._next_ext_unit + 1
        self._next_ext_unit += 1
        return next_unit

    def export(self, f: Union[str, os.PathLike], **kwargs):
        """
        Method to export a model to netcdf or shapefile based on the
        extension of the file name (.shp for shapefile, .nc for netcdf)

        Parameters
        ----------
        f : str or PathLike
            The file path
        kwargs : keyword arguments
            modelgrid : flopy.discretization.Grid instance
                user supplied modelgrid which can be used for exporting
                in lieu of the modelgrid associated with the model object

        Returns
        -------
            None or Netcdf object

        """
        from .export import utils

        return utils.model_export(f, self, **kwargs)

    def add_package(self, p):
        """
        Add a package.

        Parameters
        ----------
        p : Package object

        """
        for idx, u in enumerate(p.unit_number):
            if u != 0:
                if u in self.package_units or u in self.external_units:
                    try:
                        pn = p.name[idx]
                    except:
                        pn = p.name
                    if self.verbose:
                        print(
                            f"\nWARNING:\n    unit {u} of package {pn} already in use."
                        )
            self.package_units.append(u)
        for i, pp in enumerate(self.packagelist):
            if pp.allowDuplicates:
                continue
            elif isinstance(p, type(pp)):
                if self.verbose:
                    print(
                        "\nWARNING:\n    Two packages of the same type, "
                        f"Replacing existing '{p.name[0]}' package."
                    )
                self.packagelist[i] = p
                return
        if self.verbose:
            print("adding Package: ", p.name[0])
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
            if pname.upper() in pp.name:
                if self.verbose:
                    print("removing Package: ", pp.name)

                # Remove the package object from the model's packagelist
                p = self.packagelist.pop(i)

                # Remove the package unit number from the list of package
                # units stored with the model
                for iu in p.unit_number:
                    if iu in self.package_units:
                        self.package_units.remove(iu)
                return
        raise StopIteration(
            "Package name " + pname + " not found in Package list"
        )

    def __getattr__(self, item):
        """
        __getattr__ - syntactic sugar

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
        if item == "output_packages" or not hasattr(self, "output_packages"):
            raise AttributeError(item)

        if item == "tr":
            if self.dis is not None:
                return self.dis.tr
            else:
                return None

        if item == "nper":
            if self.dis is not None:
                return self.dis.nper
            else:
                return 0

        if item == "start_datetime":
            if self.dis is not None:
                return self.dis.start_datetime
            else:
                return None

        # return self.get_package(item)
        # to avoid infinite recursion
        if (
            item == "_packagelist"
            or item == "packagelist"
            or item == "mfnam_packages"
        ):
            raise AttributeError(item)
        pckg = self.get_package(item)
        if pckg is not None or item in self.mfnam_packages:
            return pckg
        if item == "modelgrid":
            return
        raise AttributeError(item)

    def get_ext_dict_attr(
        self,
        ext_unit_dict=None,
        unit=None,
        filetype=None,
        pop_key=True,
    ):
        iu = None
        fname = None
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if key == unit:
                    iu = key
                    fname = os.path.basename(value.filename)
                    break
                elif value.filetype == filetype:
                    iu = key
                    fname = os.path.basename(value.filename)
                    if pop_key:
                        self.add_pop_key_list(iu)
                    break
        return iu, fname

    def _output_msg(self, i, add=True):
        if add:
            txt1 = "Adding"
            txt2 = "to"
        else:
            txt1 = "Removing"
            txt2 = "from"
        print(
            f"{txt1} {self.output_fnames[i]} (unit={self.output_units[i]}) "
            f"{txt2} the output list."
        )

    def add_output_file(
        self,
        unit,
        fname: Optional[Union[str, os.PathLike]] = None,
        extension="cbc",
        binflag=True,
        package=None,
    ):
        """
        Add an ascii or binary output file for a package

        Parameters
        ----------
        unit : int
            Unit number of external array
        fname : str or PathLike, optional
            Path of external array, default is None
        extension : str
            Extension to use for the cell-by-cell file. Only used if fname
            is None, default is cbc
        binflag : bool
            Whether the output file is a binary file, efault is True
        package : str
            The package the output file is attached to, default is None

        """
        add_cbc = False
        if unit > 0:
            add_cbc = True
            # determine if the file is in external_units
            if abs(unit) in self.external_units:
                idx = self.external_units.index(abs(unit))
                if fname is None:
                    fname = os.path.basename(self.external_fnames[idx])
                binflag = self.external_binflag[idx]
                self.remove_external(unit=abs(unit))
            # determine if the unit exists in the output data
            if abs(unit) in self.output_units:
                add_cbc = False
                idx = self.output_units.index(abs(unit))
                # determine if binflag has changed
                if binflag is not self.output_binflag[idx]:
                    add_cbc = True
                if add_cbc:
                    self.remove_output(unit=abs(unit))
                else:
                    if package is not None:
                        self.output_packages[idx].append(package)

        if add_cbc:
            if fname is None:
                fname = f"{self.name}.{extension}"
                # check if this file name exists for a different unit number
                if fname in self.output_fnames:
                    idx = self.output_fnames.index(fname)
                    iut = self.output_units[idx]
                    if iut != unit:
                        # include unit number in fname if package has
                        # not been passed
                        if package is None:
                            fname = f"{self.name}.{unit}.{extension}"
                        # include package name in fname
                        else:
                            fname = f"{self.name}.{package}.{extension}"
            else:
                fname = os.path.basename(fname)
            self.add_output(fname, unit, binflag=binflag, package=package)

    def add_output(
        self, fname: Union[str, os.PathLike], unit, binflag=False, package=None
    ):
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
        fname = str(fname)
        if fname in self.output_fnames:
            if self.verbose:
                print(
                    "BaseModel.add_output() warning: "
                    f"replacing existing filename {fname}"
                )
            idx = self.output_fnames.index(fname)
            if self.verbose:
                self._output_msg(idx, add=False)
            self.output_fnames.pop(idx)
            self.output_units.pop(idx)
            self.output_binflag.pop(idx)
            self.output_packages.pop(idx)

        self.output_fnames.append(fname)
        self.output_units.append(unit)
        self.output_binflag.append(binflag)
        if package is not None:
            self.output_packages.append([package])
        else:
            self.output_packages.append([])

        if self.verbose:
            self._output_msg(-1, add=True)

    def remove_output(
        self, fname: Optional[Union[str, os.PathLike]] = None, unit=None
    ):
        """
        Remove an output file from the model by specifying either the
        file name or the unit number.

        Parameters
        ----------
        fname : str or PathLike, optional
            Path of output array
        unit : int, optional
            Unit number of output array
        """
        if fname is not None:
            fname = str(fname)
            for i, e in enumerate(self.output_fnames):
                if fname in e:
                    if self.verbose:
                        self._output_msg(i, add=False)
                    self.output_fnames.pop(i)
                    self.output_units.pop(i)
                    self.output_binflag.pop(i)
                    self.output_packages.pop(i)
        elif unit is not None:
            for i, u in enumerate(self.output_units):
                if u == unit:
                    if self.verbose:
                        self._output_msg(i, add=False)
                    self.output_fnames.pop(i)
                    self.output_units.pop(i)
                    self.output_binflag.pop(i)
                    self.output_packages.pop(i)
        else:
            msg = " either fname or unit must be passed to remove_output()"
            raise Exception(msg)

    def get_output(
        self, fname: Optional[Union[str, os.PathLike]] = None, unit=None
    ):
        """
        Get an output file from the model by specifying either the
        file name or the unit number.

        Parameters
        ----------
        fname : str or PathLike, optional
            Path of output array
        unit : int, optional
            Unit number of output array
        """
        if fname is not None:
            fname = str(fname)
            for i, e in enumerate(self.output_fnames):
                if fname in e:
                    return self.output_units[i]
            return None
        elif unit is not None:
            for i, u in enumerate(self.output_units):
                if u == unit:
                    return self.output_fnames[i]
            return None
        else:
            msg = " either fname or unit must be passed to get_output()"
            raise Exception(msg)

    def set_output_attribute(
        self,
        fname: Optional[Union[str, os.PathLike]] = None,
        unit=None,
        attr=None,
    ):
        """
        Set a variable in an output file from the model by specifying either
        the file name or the unit number and a dictionary with attributes
        to change.

        Parameters
        ----------
        fname : str or PathLike, optional
            Path of output array
        unit : int, optional
            Unit number of output array
        """
        idx = None
        if fname is not None:
            fname = str(fname)
            for i, e in enumerate(self.output_fnames):
                if fname in e:
                    idx = i
                    break
            return None
        elif unit is not None:
            for i, u in enumerate(self.output_units):
                if u == unit:
                    idx = i
                    break
        else:
            msg = (
                " either fname or unit must be passed "
                "to set_output_attribute()"
            )
            raise Exception(msg)
        if attr is not None:
            if idx is not None:
                for key, value in attr.items:
                    if key == "binflag":
                        self.output_binflag[idx] = value
                    elif key == "fname":
                        self.output_fnames[idx] = value
                    elif key == "unit":
                        self.output_units[idx] = value

    def get_output_attribute(
        self,
        fname: Optional[Union[str, os.PathLike]] = None,
        unit=None,
        attr=None,
    ):
        """
        Get an attribute of a model output file by specifying either
        the file name or the unit number.

        Parameters
        ----------
        fname : str or PathLike, optional
            path of output array
        unit : int, optional
            Unit number of output array
        """
        idx = None
        if fname is not None:
            for i, e in enumerate(self.output_fnames):
                if fname in e:
                    idx = i
                    break
            return None
        elif unit is not None:
            for i, u in enumerate(self.output_units):
                if u == unit:
                    idx = i
                    break
        else:
            raise Exception(
                " either fname or unit must be passed "
                "to set_output_attribute()"
            )
        v = None
        if attr is not None:
            if idx is not None:
                if attr == "binflag":
                    v = self.output_binflag[idx]
                elif attr == "fname":
                    v = self.output_fnames[idx]
                elif attr == "unit":
                    v = self.output_units[idx]
        return v

    def add_external(
        self, fname: Union[str, os.PathLike], unit, binflag=False, output=False
    ):
        """
        Assign an external array so that it will be listed as a DATA or
        DATA(BINARY) entry in the name file. This will allow an outside
        file package to refer to it.

        Parameters
        ----------
        fname : str or PathLike
            Path of external array
        unit : int
            Unit number of external array
        binflag : boolean, optional
            Binary or not, default is False
        """
        if fname in self.external_fnames:
            if self.verbose:
                print(
                    "BaseModel.add_external() warning: "
                    f"replacing existing filename {fname}"
                )
            idx = self.external_fnames.index(fname)
            self.external_fnames.pop(idx)
            self.external_units.pop(idx)
            self.external_binflag.pop(idx)
            self.external_output.pop(idx)
        if unit in self.external_units:
            if self.verbose:
                msg = f"BaseModel.add_external() warning: replacing existing unit {unit}"
                print(msg)
            idx = self.external_units.index(unit)
            self.external_fnames.pop(idx)
            self.external_units.pop(idx)
            self.external_binflag.pop(idx)
            self.external_output.pop(idx)

        self.external_fnames.append(fname)
        self.external_units.append(unit)
        self.external_binflag.append(binflag)
        self.external_output.append(output)

    def remove_external(
        self, fname: Optional[Union[str, os.PathLike]] = None, unit=None
    ):
        """
        Remove an external file from the model by specifying either the
        file name or the unit number.

        Parameters
        ----------
        fname : str or PathLike, optional
            Path of external array
        unit : int, optional
            Unit number of external array
        """
        plist = []
        if fname is not None:
            for i, e in enumerate(self.external_fnames):
                if fname in e:
                    plist.append(i)
        elif unit is not None:
            for i, u in enumerate(self.external_units):
                if u == unit:
                    plist.append(i)
        else:
            msg = " either fname or unit must be passed to remove_external()"
            raise Exception(msg)
        # remove external file
        j = 0
        for i in plist:
            ipos = i - j
            self.external_fnames.pop(ipos)
            self.external_units.pop(ipos)
            self.external_binflag.pop(ipos)
            self.external_output.pop(ipos)
            j += 1

    def add_existing_package(
        self,
        filename: Union[str, os.PathLike],
        ptype=None,
        copy_to_model_ws=True,
    ):
        """
        Add an existing package to a model instance.

        Parameters
        ----------

        filename : str or PathLike
            Path of the file to add as a package
        ptype : optional
            Model package type (e.g. "lpf", "wel", etc). If None
            then the file extension of the filename arg is used
        copy_to_model_ws : bool
            Copy the package file into the model workspace.

        Returns
        -------
        None

        """
        if ptype is None:
            ptype = filename.split(".")[-1]
        ptype = str(ptype).upper()

        # for pak in self.packagelist:
        #     if ptype in pak.name:
        #         print("BaseModel.add_existing_package() warning: " +\
        #               "replacing existing package {0}".format(ptype))
        class Obj:
            pass

        fake_package = Obj()
        fake_package.write_file = lambda: None
        fake_package.name = [ptype]
        fake_package.extension = [filename.split(".")[-1]]
        fake_package.unit_number = [self.next_ext_unit()]
        if copy_to_model_ws:
            base_filename = os.path.split(filename)[-1]
            fake_package.file_name = [base_filename]
            shutil.copy2(filename, os.path.join(self.model_ws, base_filename))
        else:
            fake_package.file_name = [filename]
        fake_package.allowDuplicates = True
        self.add_package(fake_package)

    def get_name_file_entries(self):
        """
        Get a string representation of the name file.

        Parameters
        ----------

        """
        lines = []
        for p in self.packagelist:
            for i in range(len(p.name)):
                if p.unit_number[i] == 0:
                    continue
                s = f"{p.name[i]:14s} {p.unit_number[i]:5d}  {p.file_name[i]}"
                lines.append(s)
        return "\n".join(lines) + "\n"

    def has_package(self, name):
        """
        Check if package name is in package list.

        Parameters
        ----------
        name : str
            Name of the package, 'DIS', 'BAS6', etc. (case-insensitive).

        Returns
        -------
        bool
            True if package name exists, otherwise False if not found.

        """
        if not name:
            raise ValueError("invalid package name")
        name = name.upper()
        for p in self.packagelist:
            for pn in p.name:
                if pn.upper() == name:
                    return True
        return False

    def get_package(self, name):
        """
        Get a package.

        Parameters
        ----------
        name : str
            Name of the package, 'RIV', 'LPF', etc. (case-insensitive).

        Returns
        -------
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        """
        if not name:
            raise ValueError("invalid package name")
        name = name.upper()
        for pp in self.packagelist:
            if pp.name[0].upper() == name:
                return pp
        return None

    def set_version(self, version):
        self.version = version.lower()

        # check that this is a valid model version
        if self.version not in list(self.version_types.keys()):
            err = (
                f"Error: Unsupported model version ({self.version}). "
                "Valid model versions are:"
            )
            for v in list(self.version_types.keys()):
                err += f" {v}"
            raise Exception(err)

        # set namefile heading
        self.heading = (
            f"# Name file for {self.version_types[self.version]}, "
            f"generated by Flopy version {__version__}."
        )

        # set heading for each package
        for p in self.get_package_list():
            pak = self.get_package(p)
            if hasattr(pak, "heading"):
                pak._generate_heading()

        return None

    def change_model_ws(
        self,
        new_pth: Optional[Union[str, os.PathLike]] = os.curdir,
        reset_external=False,
    ):
        """
        Change the model work space.

        Parameters
        ----------
        new_pth : str or PathLike
            Path of the new model workspace. If this path does not exist,
            it will be created. If no value (None) is given, the default
            is the present working directory.

        Returns
        -------
        val : list of strings
            Can be used to see what packages are in the model, and can then
            be used with get_package to pull out individual packages.

        """
        if new_pth is None:
            new_pth = os.curdir
        if not os.path.exists(new_pth):
            try:
                print(
                    f"\ncreating model workspace...\n   {flopy_io.relpath_safe(new_pth)}"
                )
                os.makedirs(new_pth)
            except:
                raise OSError(f"{new_pth} not valid, workspace-folder")
                # line = '\n{} not valid, workspace-folder '.format(new_pth) + \
                #        'was changed to {}\n'.format(os.getcwd())
                # print(line)
                # new_pth = os.getcwd()

        # --reset the model workspace
        old_pth = self._model_ws
        self._model_ws = new_pth
        if self.verbose:
            print(
                f"\nchanging model workspace...\n   {flopy_io.relpath_safe(new_pth)}"
            )
        # reset the paths for each package
        for pp in self.packagelist:
            pp.fn_path = os.path.join(self.model_ws, pp.file_name[0])

        # create the external path (if needed)
        if (
            hasattr(self, "external_path")
            and self.external_path is not None
            and not os.path.exists(
                os.path.join(self._model_ws, self.external_path)
            )
        ):
            pth = os.path.join(self._model_ws, self.external_path)
            os.makedirs(pth)
            if reset_external:
                self._reset_external(pth, old_pth)
        elif reset_external:
            self._reset_external(self._model_ws, old_pth)
        return None

    def _reset_external(self, pth, old_pth):
        new_ext_fnames = []
        for ext_file, output in zip(
            self.external_fnames, self.external_output
        ):
            # new_ext_file = os.path.join(pth, os.path.split(ext_file)[-1])
            # this is a wicked mess
            if output:
                # new_ext_file = os.path.join(pth, os.path.split(ext_file)[-1])
                new_ext_file = ext_file
            else:
                # fpth = os.path.abspath(os.path.join(old_pth, ext_file))
                # new_ext_file = os.path.relpath(fpth, os.path.abspath(pth))
                fdir = os.path.dirname(ext_file)
                if fdir == "":
                    fpth = os.path.abspath(os.path.join(old_pth, ext_file))
                else:
                    fpth = ext_file
                ao = os.path.abspath(os.path.dirname(fpth))
                ep = os.path.abspath(pth)
                relp = os.path.relpath(ao, ep)
                new_ext_file = os.path.join(relp, os.path.basename(ext_file))
            new_ext_fnames.append(new_ext_file)
        self.external_fnames = new_ext_fnames

    def _set_name(self, value):
        """
        Set model name

        Parameters
        ----------
        value : str
            Name to assign to model.

        """
        self.__name = str(value)
        self.namefile = self.__name + "." + self.namefile_ext
        for p in self.packagelist:
            for i in range(len(p.extension)):
                p.file_name[i] = self.__name + "." + p.extension[i]
            p.fn_path = os.path.join(self.model_ws, p.file_name[0])

    def __setattr__(self, key, value):
        if key == "free_format_input":
            # if self.bas6 is not None:
            #    self.bas6.ifrefm = value
            super().__setattr__(key, value)
        elif key == "name":
            self._set_name(value)
        elif key == "model_ws":
            self.change_model_ws(value)
        elif key == "sr" and value.__class__.__name__ == "SpatialReference":
            warnings.warn(
                "SpatialReference has been deprecated.",
                category=DeprecationWarning,
            )
            if self.dis is not None:
                self.dis.sr = value
            else:
                raise Exception(
                    "cannot set SpatialReference - ModflowDis not found"
                )
        elif key == "tr":
            assert isinstance(
                value, discretization.reference.TemporalReference
            )
            if self.dis is not None:
                self.dis.tr = value
            else:
                raise Exception(
                    "cannot set TemporalReference - ModflowDis not found"
                )
        elif key == "start_datetime":
            if self.dis is not None:
                self.dis.start_datetime = value
                self.tr.start_datetime = value
            else:
                raise Exception(
                    "cannot set start_datetime - ModflowDis not found"
                )
        else:
            super().__setattr__(key, value)

    def run_model(
        self,
        silent=False,
        pause=False,
        report=False,
        normal_msg="normal termination",
    ) -> Tuple[bool, List[str]]:
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
        success : boolean
        buff : list of lines of stdout

        """

        return run_model(
            self.exe_name,
            self.namefile,
            model_ws=self.model_ws,
            silent=silent,
            pause=pause,
            report=report,
            normal_msg=normal_msg,
        )

    def load_results(self):
        print("load_results not implemented")

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
            self.check(f=f"{self.name}.chk", verbose=self.verbose, level=1)

        # reset the model to free_format if parameter substitution was
        # performed on a model load
        if self.parameter_load and not self.free_format_input:
            if self.verbose:
                print(
                    "\nResetting free_format_input to True to "
                    "preserve the precision of the parameter data."
                )
            self.free_format_input = True

        if self.verbose:
            print("\nWriting packages:")

        if SelPackList == False:
            for p in self.packagelist:
                if self.verbose:
                    print("   Package: ", p.name[0])
                # prevent individual package checks from running after
                # model-level package check above
                # otherwise checks are run twice
                # or the model level check procedure would have to be split up
                # or each package would need a check argument,
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
                            print("   Package: ", p.name[0])
                        try:
                            p.write_file(check=False)
                        except TypeError:
                            p.write_file()
                            break
        if self.verbose:
            print(" ")
        # write name file
        self.write_name_file()
        # os.chdir(org_dir)

    def write_name_file(self):
        """
        Every Package needs its own writenamefile function

        """
        raise Exception(
            "IMPLEMENTATION ERROR: writenamefile must be overloaded"
        )

    def set_model_units(self):
        """
        Every model needs its own set_model_units method

        """
        raise Exception(
            "IMPLEMENTATION ERROR: set_model_units must be overloaded"
        )

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

    def check(
        self,
        f: Optional[Union[str, os.PathLike]] = None,
        verbose=True,
        level=1,
    ):
        """
        Check model data for common errors.

        Parameters
        ----------
        f : str or PathLike, optional, default None
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

        # check instance for model-level check
        chk = utils.check(self, f=f, verbose=verbose, level=level)
        # check for unit number conflicts
        package_units = {}
        duplicate_units = {}
        for p in self.packagelist:
            for i in range(len(p.name)):
                if p.unit_number[i] != 0:
                    if p.unit_number[i] in package_units.values():
                        duplicate_units[p.name[i]] = p.unit_number[i]
                        otherpackage = [
                            k
                            for k, v in package_units.items()
                            if v == p.unit_number[i]
                        ][0]
                        duplicate_units[otherpackage] = p.unit_number[i]
        if len(duplicate_units) > 0:
            for k, v in duplicate_units.items():
                chk._add_to_summary(
                    "Error", package=k, value=v, desc="unit number conflict"
                )
        else:
            chk.passed.append("Unit number conflicts")

        return self._check(chk, level)

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
        from .plot import PlotUtilities

        axes = PlotUtilities._plot_model_helper(
            self, SelPackList=SelPackList, **kwargs
        )
        return axes

    def to_shapefile(
        self, filename: Union[str, os.PathLike], package_names=None, **kwargs
    ):
        """
        Wrapper function for writing a shapefile for the model grid.  If
        package_names is not None, then search through the requested packages
        looking for arrays that can be added to the shapefile as attributes

        Parameters
        ----------
        filename : str or PathLike
            Path of the shapefile to write
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
        warnings.warn("to_shapefile() is deprecated. use .export()")
        self.export(filename, package_names=package_names)


def run_model(
    exe_name: Union[str, os.PathLike],
    namefile: Optional[str],
    model_ws: Union[str, os.PathLike] = os.curdir,
    silent=False,
    pause=False,
    report=False,
    processors=None,
    normal_msg="normal termination",
    use_async=False,
    cargs=None,
) -> Tuple[bool, List[str]]:
    """
    Run the model using subprocess.Popen, optionally collecting stdout and printing
    timestamped progress. Model workspace, namefile, executable to use, and several
    other options may be configured, and additional command line arguments may also
    be provided.

    Parameters
    ----------
    exe_name : str or PathLike
        Executable name or path. If the executable name is provided,
        the executable must be on the system path. Alternatively, a
        full path to the executable may be provided.
    namefile : str, optional
        Name of the name file of model to run. The name may be None
        to run models that don't require a control file (name file)
    model_ws : str or PathLike, optional, default '.'
        Path to the parent directory of the namefile. (default is the
        current working directory '.')
    silent : boolean, default True
        Whether to suppress model output. (Default is True)
    pause : boolean, optional, default False
        Pause and wait for keystroke upon completion. (Default is False)
    report : boolean, optional, default False
        Save stdout lines to a list (buff) returned by the method. (Default is False)
    processors: int
        Number of processors. Parallel simulations are only supported for
        MODFLOW 6 simulations. (default is None)
    normal_msg : str or list
        Termination message used to determine if the model terminated normally.
        More than one message can be provided using a list.
        (Default is 'normal termination')
    use_async : boolean
        Asynchronously read model stdout and report with timestamps. Good for
        models taking a long time to run, not good for models that run quickly.
    cargs : str or list, optional, default None
        Additional command line arguments to pass to the executable.
        (Default is None)
    Returns
    -------
    success : boolean
    buff : list of lines of stdout (empty if report is False)

    """
    success = False
    buff = []

    # convert normal_msg to a list of lower case str for comparison
    if isinstance(normal_msg, str):
        normal_msg = [normal_msg]
    for idx, s in enumerate(normal_msg):
        normal_msg[idx] = s.lower()

    # make sure executable exists
    if exe_name is None:
        raise ValueError(f"An executable name or path must be provided")
    exe_path = resolve_exe(exe_name)
    if not silent:
        print(
            f"FloPy is using the following executable to run the model: {flopy_io.relpath_safe(exe_path, model_ws)}"
        )

    # make sure namefile exists
    if namefile is not None and not os.path.isfile(
        os.path.join(model_ws, namefile)
    ):
        raise FileNotFoundError(
            f"The namefile for this model does not exist: {namefile}"
        )

    # simple little function for the thread to target
    def q_output(output, q):
        for line in iter(output.readline, b""):
            q.put(line)
            # time.sleep(1)
            # output.close()

    # create a list of arguments to pass to Popen
    if processors is not None:
        if "mf6" not in exe_path:
            raise ValueError("processors kwarg only supported for MODFLOW 6")
        mpiexec_path = resolve_exe("mpiexec")
        if not silent:
            print(f"FloPy is using {mpiexec_path} to run {exe_path}.")
        argv = [mpiexec_path, "-np", f"{processors}", exe_path, "-p"]
    else:
        argv = [exe_path]

    if namefile is not None:
        argv.append(Path(namefile).name)

    # add additional arguments to Popen arguments
    if cargs is not None:
        if isinstance(cargs, str):
            cargs = [cargs]
        for t in cargs:
            argv.append(t)

    # run the model with Popen
    proc = Popen(argv, stdout=PIPE, stderr=STDOUT, cwd=model_ws)

    if not use_async:
        while True:
            line = proc.stdout.readline().decode("utf-8")
            if line == "" and proc.poll() is not None:
                break
            if line:
                for msg in normal_msg:
                    if msg in line.lower():
                        success = True
                        break
                line = line.rstrip("\r\n")
                if not silent:
                    print(line)
                if report:
                    buff.append(line)
            else:
                break
        return success, buff

    # some tricks for the async stdout reading
    q = Queue.Queue()
    thread = threading.Thread(target=q_output, args=(proc.stdout, q))
    thread.daemon = True
    thread.start()

    failed_words = ["fail", "error"]
    last = datetime.now()
    lastsec = 0.0
    while True:
        try:
            line = q.get_nowait()
        except Queue.Empty:
            pass
        else:
            if line == "":
                break
            line = line.decode().lower().strip()
            if line != "":
                now = datetime.now()
                dt = now - last
                tsecs = dt.total_seconds() - lastsec
                line = f"(elapsed:{tsecs})-->{line}"
                lastsec = tsecs + lastsec
                buff.append(line)
                if not silent:
                    print(line)
                for fword in failed_words:
                    if fword in line:
                        success = False
                        break
        if proc.poll() is not None:
            break
    proc.wait()
    thread.join(timeout=1)
    buff.extend(proc.stdout.readlines())
    proc.stdout.close()

    for line in buff:
        for msg in normal_msg:
            if msg in line.lower():
                print("success")
                success = True
                break

    if pause:
        input("Press Enter to continue...")
    return success, buff
