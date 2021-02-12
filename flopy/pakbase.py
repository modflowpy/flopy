"""
pakbase module
  This module contains the base package class from which
  all of the other packages inherit from.

"""

from __future__ import print_function

import abc
import os
import webbrowser as wb

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from .modflow.mfparbc import ModflowParBc as mfparbc
from .utils import Util2d, Util3d, Transient2d, MfList, check
from .utils import OptionBlock
from .utils.flopy_io import ulstrd


class PackageInterface(object):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError(
            "must define name in child " "class to use this base class"
        )

    @name.setter
    @abc.abstractmethod
    def name(self, name):
        raise NotImplementedError(
            "must define name in child " "class to use this base class"
        )

    @property
    @abc.abstractmethod
    def parent(self):
        raise NotImplementedError(
            "must define parent in child " "class to use this base class"
        )

    @parent.setter
    @abc.abstractmethod
    def parent(self, name):
        raise NotImplementedError(
            "must define parent in child " "class to use this base class"
        )

    @property
    @abc.abstractmethod
    def package_type(self):
        raise NotImplementedError(
            "must define package_type in child " "class to use this base class"
        )

    @property
    @abc.abstractmethod
    def data_list(self):
        # [data_object, data_object, ...]
        raise NotImplementedError(
            "must define data_list in child " "class to use this base class"
        )

    @abc.abstractmethod
    def export(self, f, **kwargs):
        raise NotImplementedError(
            "must define export in child " "class to use this base class"
        )

    @property
    @abc.abstractmethod
    def plottable(self):
        raise NotImplementedError(
            "must define plottable in child " "class to use this base class"
        )

    @property
    def has_stress_period_data(self):
        return self.__dict__.get("stress_period_data", None) is not None

    @staticmethod
    def _check_thresholds(chk, array, active, thresholds, name):
        """Checks array against min and max threshold values."""
        mn, mx = thresholds
        chk.values(
            array,
            active & (array < mn),
            "{} values below checker threshold of {}".format(name, mn),
            "Warning",
        )
        chk.values(
            array,
            active & (array > mx),
            "{} values above checker threshold of {}".format(name, mx),
            "Warning",
        )

    @staticmethod
    def _confined_layer_check(chk):
        return

    def _other_xpf_checks(self, chk, active):
        # check for negative hani
        chk.values(
            self.__dict__["hani"].array,
            active & (self.__dict__["hani"].array < 0),
            "negative horizontal anisotropy values",
            "Error",
        )

        # check vkcb if there are any quasi-3D layers
        if self.parent.dis.laycbd.sum() > 0:
            # pad non-quasi-3D layers in vkcb array with ones so
            # they won't fail checker
            vkcb = self.vkcb.array.copy()
            for l in range(self.vkcb.shape[0]):
                if self.parent.dis.laycbd[l] == 0:
                    # assign 1 instead of zero as default value that
                    # won't violate checker
                    # (allows for same structure as other checks)
                    vkcb[l, :, :] = 1
            chk.values(
                vkcb,
                active & (vkcb <= 0),
                "zero or negative quasi-3D confining bed Kv values",
                "Error",
            )
            self._check_thresholds(
                chk,
                vkcb,
                active,
                chk.property_threshold_values["vkcb"],
                "quasi-3D confining bed Kv",
            )

    @staticmethod
    def _get_nan_exclusion_list():
        return []

    def _get_check(self, f, verbose, level, checktype):
        if checktype is not None:
            return checktype(self, f=f, verbose=verbose, level=level)
        else:
            return check(self, f=f, verbose=verbose, level=level)

    def _check_oc(self, f=None, verbose=True, level=1, checktype=None):
        spd_inds_valid = True
        chk = self._get_check(f, verbose, level, checktype)
        spd = getattr(self, "stress_period_data")
        nan_exclusion_list = self._get_nan_exclusion_list()
        for per in spd.data.keys():
            if isinstance(spd.data[per], np.recarray):
                spdata = self.stress_period_data.data[per]
                inds = chk._get_cell_inds(spdata)

                # General BC checks
                # check for valid cell indices
                spd_inds_valid = chk._stress_period_data_valid_indices(spdata)

                # first check for and list nan values
                chk._stress_period_data_nans(spdata, nan_exclusion_list)

                if spd_inds_valid:
                    # next check for BCs in inactive cells
                    chk._stress_period_data_inactivecells(spdata)

                    # More specific BC checks
                    # check elevations in the ghb, drain, and riv packages
                    if self.name[0] in check.bc_stage_names.keys():
                        # check that bc elevations are above model
                        # cell bottoms -- also checks for nan values
                        elev_name = chk.bc_stage_names[self.name[0]]
                        mg = self.parent.modelgrid
                        botms = mg.botm[inds]
                        test = spdata[elev_name] < botms
                        en = "BC elevation below cell bottom"
                        chk.stress_period_data_values(
                            spdata,
                            test,
                            col=elev_name,
                            error_name=en,
                            error_type="Error",
                        )

        chk.summarize()
        return chk

    def _get_kparams(self):
        # build model specific parameter lists
        kparams_all = {
            "hk": "horizontal hydraulic conductivity",
            "vka": "vertical hydraulic conductivity",
            "k": "horizontal hydraulic conductivity",
            "k22": "hydraulic conductivity second axis",
            "k33": "vertical hydraulic conductivity",
        }
        kparams = {}
        vka_param = None
        for kp, name in kparams_all.items():
            if kp in self.__dict__:
                kparams[kp] = name
        if "hk" in self.__dict__:
            hk = self.hk.array.copy()
        else:
            hk = self.k.array.copy()
        if "vka" in self.__dict__ and self.layvka.sum() > 0:
            vka = self.vka.array
            vka_param = kparams.pop("vka")
        elif "k33" in self.__dict__:
            vka = self.k33.array
            vka_param = kparams.pop("k33")
        else:
            vka = None
        if vka is not None:
            vka = vka.copy()
        return kparams, hk, vka, vka_param

    def _check_flowp(self, f=None, verbose=True, level=1, checktype=None):
        chk = self._get_check(f, verbose, level, checktype)
        active = chk.get_active()

        # build model specific parameter lists
        kparams, hk, vka, vka_param = self._get_kparams()

        # check for zero or negative values of hydraulic conductivity,
        # anisotropy, and quasi-3D confining beds
        for kp, name in kparams.items():
            if self.__dict__[kp].array is not None:
                chk.values(
                    self.__dict__[kp].array,
                    active & (self.__dict__[kp].array <= 0),
                    "zero or negative {} values".format(name),
                    "Error",
                )

        if "hani" in self.__dict__:
            self._other_xpf_checks(chk, active)

        # check for unusually high or low values of hydraulic conductivity
        # convert vertical anisotropy to Kv for checking
        if vka is not None:
            if "layvka" in self.__dict__:
                for l in range(vka.shape[0]):
                    vka[l] *= hk[l] if self.layvka.array[l] != 0 else 1
            self._check_thresholds(
                chk,
                vka,
                active,
                chk.property_threshold_values["vka"],
                vka_param,
            )

        for kp, name in kparams.items():
            if self.__dict__[kp].array is not None:
                self._check_thresholds(
                    chk,
                    self.__dict__[kp].array,
                    active,
                    chk.property_threshold_values[kp],
                    name,
                )
        if self.name[0] in ["UPW", "LPF"]:
            storage_coeff = "STORAGECOEFFICIENT" in self.options or (
                "storagecoefficient" in self.__dict__
                and self.storagecoefficient.get_data()
            )
            self._check_storage(chk, storage_coeff)
        chk.summarize()
        return chk

    def check(self, f=None, verbose=True, level=1, checktype=None):
        """
        Check package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a sting is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.
        checktype : check
            Checker type to be used. By default class check is used from
            check.py.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.dis.check()

        """
        chk = None

        if (
            self.has_stress_period_data
            and self.name[0] != "OC"
            and self.package_type.upper() != "OC"
        ):
            chk = self._check_oc(f, verbose, level, checktype)
        # check property values in upw and lpf packages
        elif self.name[0] in ["UPW", "LPF"] or self.package_type.upper() in [
            "NPF"
        ]:
            chk = self._check_flowp(f, verbose, level, checktype)
        elif self.package_type.upper() in ["STO"]:
            chk = self._get_check(f, verbose, level, checktype)
            storage_coeff = self.storagecoefficient.get_data()
            if storage_coeff is None:
                storage_coeff = False
            self._check_storage(chk, storage_coeff)
        else:
            txt = "check method not implemented for " + "{} Package.".format(
                self.name[0]
            )
            if f is not None:
                if isinstance(f, str):
                    pth = os.path.join(self.parent.model_ws, f)
                    f = open(pth, "w")
                    f.write(txt)
                    f.close()
            if verbose:
                print(txt)
        return chk

    def _check_storage(self, chk, storage_coeff):
        # only check storage if model is transient
        if not np.all(self.parent.modeltime.steady_state):
            active = chk.get_active()
            # do the same for storage if the model is transient
            sarrays = {"ss": self.ss.array, "sy": self.sy.array}
            # convert to specific for checking
            if storage_coeff:
                desc = (
                    "\r    STORAGECOEFFICIENT option is "
                    + "activated, storage values are read "
                    + "storage coefficients"
                )
                chk._add_to_summary(type="Warning", desc=desc)

            chk.values(
                sarrays["ss"],
                active & (sarrays["ss"] < 0),
                "zero or negative specific storage values",
                "Error",
            )
            self._check_thresholds(
                chk,
                sarrays["ss"],
                active,
                chk.property_threshold_values["ss"],
                "specific storage",
            )

            # only check specific yield for convertible layers
            if "laytyp" in self.__dict__:
                inds = np.array(
                    [
                        True
                        if l > 0 or l < 0 and "THICKSTRT" in self.options
                        else False
                        for l in self.laytyp
                    ]
                )
                sarrays["sy"] = sarrays["sy"][inds, :, :]
                active = active[inds, :, :]
            else:
                iconvert = self.iconvert.array
                for ishape in np.ndindex(active.shape):
                    if active[ishape]:
                        active[ishape] = (
                            iconvert[ishape] > 0 or iconvert[ishape] < 0
                        )
            chk.values(
                sarrays["sy"],
                active & (sarrays["sy"] < 0),
                "zero or negative specific yield values",
                "Error",
            )
            self._check_thresholds(
                chk,
                sarrays["sy"],
                active,
                chk.property_threshold_values["sy"],
                "specific yield",
            )


class Package(PackageInterface):
    """
    Base package class from which most other packages are derived.

    """

    def __init__(
        self,
        parent,
        extension="glo",
        name="GLOBAL",
        unit_number=1,
        extra="",
        filenames=None,
        allowDuplicates=False,
    ):
        """
        Package init

        """
        # To be able to access the parent model object's attributes
        self.parent = parent
        if not isinstance(extension, list):
            extension = [extension]
        self.extension = []
        self.file_name = []
        for idx, e in enumerate(extension):
            self.extension.append(e)
            file_name = self.parent.name + "." + e
            if filenames is not None:
                if idx < len(filenames):
                    if filenames[idx] is not None:
                        file_name = filenames[idx]
            self.file_name.append(file_name)

        self.fn_path = os.path.join(self.parent.model_ws, self.file_name[0])
        if not isinstance(name, list):
            name = [name]
        self._name = name
        if not isinstance(unit_number, list):
            unit_number = [unit_number]
        self.unit_number = unit_number
        if not isinstance(extra, list):
            self.extra = len(self.unit_number) * [extra]
        else:
            self.extra = extra
        self.url = "index.html"
        self.allowDuplicates = allowDuplicates

        self.acceptable_dtypes = [int, np.float32, str]

        return

    def __repr__(self):
        s = self.__doc__
        exclude_attributes = ["extension", "heading", "name", "parent", "url"]
        for attr, value in sorted(self.__dict__.items()):
            if not (attr in exclude_attributes):
                if isinstance(value, list):
                    if len(value) == 1:
                        s += " {:s} = {:s}\n".format(attr, str(value[0]))
                    else:
                        s += " {:s} ".format(
                            attr
                        ) + "(list, items = {:d})\n".format(len(value))
                elif isinstance(value, np.ndarray):
                    s += " {:s} (array, shape = ".format(
                        attr
                    ) + "{:s})\n".format(value.shape.__str__()[1:-1])
                else:
                    s += (
                        " {:s} = ".format(attr)
                        + "{:s} ".format(str(value))
                        + "({:s})\n".format(str(type(value))[7:-2])
                    )
        return s

    def __getitem__(self, item):
        if hasattr(self, "stress_period_data"):
            # added this check because stress_period_data also used in Oc and
            # Oc88 but is not a MfList
            spd = getattr(self, "stress_period_data")
            if isinstance(item, MfList):
                if not isinstance(item, list) and not isinstance(item, tuple):
                    msg = (
                        "package.__getitem__() kper "
                        + str(item)
                        + " not in data.keys()"
                    )
                    assert item in list(spd.data.keys()), msg
                    return spd[item]

                if item[1] not in self.dtype.names:
                    msg = (
                        "package.__getitem(): item "
                        + str(item)
                        + " not in dtype names "
                        + str(self.dtype.names)
                    )
                    raise Exception(msg)

                msg = (
                    "package.__getitem__() kper "
                    + str(item[0])
                    + " not in data.keys()"
                )
                assert item[0] in list(spd.data.keys()), msg

                if spd.vtype[item[0]] == np.recarray:
                    return spd[item[0]][item[1]]

    def __setitem__(self, key, value):
        raise NotImplementedError("package.__setitem__() not implemented")

    def __setattr__(self, key, value):
        var_dict = vars(self)
        if key in list(var_dict.keys()):
            old_value = var_dict[key]
            if isinstance(old_value, Util2d):
                value = Util2d(
                    self.parent,
                    old_value.shape,
                    old_value.dtype,
                    value,
                    name=old_value.name,
                    fmtin=old_value.format.fortran,
                    locat=old_value.locat,
                    array_free_format=old_value.format.array_free_format,
                )
            elif isinstance(old_value, Util3d):
                value = Util3d(
                    self.parent,
                    old_value.shape,
                    old_value.dtype,
                    value,
                    name=old_value.name_base,
                    fmtin=old_value.fmtin,
                    locat=old_value.locat,
                    array_free_format=old_value.array_free_format,
                )
            elif isinstance(old_value, Transient2d):
                value = Transient2d(
                    self.parent,
                    old_value.shape,
                    old_value.dtype,
                    value,
                    name=old_value.name_base,
                    fmtin=old_value.fmtin,
                    locat=old_value.locat,
                )
            elif isinstance(old_value, MfList):
                value = MfList(self, dtype=old_value.dtype, data=value)
            elif isinstance(old_value, list):
                if len(old_value) > 0:
                    if isinstance(old_value[0], Util3d):
                        new_list = []
                        for vo, v in zip(old_value, value):
                            new_list.append(
                                Util3d(
                                    self.parent,
                                    vo.shape,
                                    vo.dtype,
                                    v,
                                    name=vo.name_base,
                                    fmtin=vo.fmtin,
                                    locat=vo.locat,
                                )
                            )
                        value = new_list
                    elif isinstance(old_value[0], Util2d):
                        new_list = []
                        for vo, v in zip(old_value, value):
                            new_list.append(
                                Util2d(
                                    self.parent,
                                    vo.shape,
                                    vo.dtype,
                                    v,
                                    name=vo.name,
                                    fmtin=vo.fmtin,
                                    locat=vo.locat,
                                )
                            )
                        value = new_list

        super(Package, self).__setattr__(key, value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def package_type(self):
        if len(self.name) > 0:
            return self.name[0].lower()

    @property
    def plottable(self):
        return True

    @property
    def data_list(self):
        # return [data_object, data_object, ...]
        dl = []
        attrs = dir(self)
        if "sr" in attrs:
            attrs.remove("sr")
        if "start_datetime" in attrs:
            attrs.remove("start_datetime")
        for attr in attrs:
            if "__" in attr or "data_list" in attr:
                continue
            dl.append(self.__getattribute__(attr))
        return dl

    def export(self, f, **kwargs):
        """
        Method to export a package to netcdf or shapefile based on the
        extension of the file name (.shp for shapefile, .nc for netcdf)

        Parameters
        ----------
        f : str
            filename
        kwargs : keyword arguments
            modelgrid : flopy.discretization.Grid instance
                user supplied modelgrid which can be used for exporting
                in lieu of the modelgrid associated with the model object

        Returns
        -------
            None or Netcdf object

        """
        from flopy import export

        return export.utils.package_export(f, self, **kwargs)

    @staticmethod
    def add_to_dtype(dtype, field_names, field_types):
        """
        Add one or more fields to a structured array data type

        Parameters
        ----------
        dtype : numpy.dtype
            Input structured array datatype to add to.
        field_names : str or list
            One or more field names.
        field_types : numpy.dtype or list
            One or more data types. If one data type is supplied, it is
            repeated for each field name.
        """
        if not isinstance(field_names, list):
            field_names = [field_names]
        if not isinstance(field_types, list):
            field_types = [field_types] * len(field_names)
        newdtypes = dtype.descr
        for field_name, field_type in zip(field_names, field_types):
            newdtypes.append((str(field_name), field_type))
        return np.dtype(newdtypes)

    @staticmethod
    def _get_sfac_columns():
        """
        This should be overriden for individual packages that support an
        sfac multiplier for individual list columns

        """
        return []

    def _confined_layer_check(self, chk):
        # check for confined layers above convertible layers
        confined = False
        thickstrt = False
        for option in self.options:
            if option.lower() == "thickstrt":
                thickstrt = True
        for i, l in enumerate(self.laytyp.array.tolist()):
            if l == 0 or l < 0 and thickstrt:
                confined = True
                continue
            if confined and l > 0:
                desc = (
                    "\r    LAYTYP: unconfined (convertible) "
                    + "layer below confined layer"
                )
                chk._add_to_summary(type="Warning", desc=desc)

    def level1_arraylist(self, idx, v, name, txt):
        ndim = v.ndim
        if ndim == 3:
            kon = -1
            for [k, i, j] in idx:
                if k > kon:
                    kon = k
                    tag = name[k].lower().replace(" layer ", "")
                    txt += (
                        "    {:>10s}".format("layer")
                        + "{:>10s}".format("row")
                        + "{:>10s}".format("column")
                        + "{:>15s}\n".format(tag)
                    )
                txt += "    {:10d}{:10d}{:10d}{:15.7g}\n".format(
                    k + 1, i + 1, j + 1, v[k, i, j]
                )
        elif ndim == 2:
            tag = name[0].lower().replace(" layer ", "")
            txt += (
                "    {:>10s}".format("row")
                + "{:>10s}".format("column")
                + "{:>15s}\n".format(tag)
            )
            for [i, j] in idx:
                txt += "    {:10d}{:10d}{:15.7g}\n".format(
                    i + 1, j + 1, v[i, j]
                )
        elif ndim == 1:
            txt += "    {:>10s}{:>15s}\n".format("number", name[0])
            for i in idx:
                txt += "    {:10d}{:15.7g}\n".format(i + 1, v[i])
        return txt

    def plot(self, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        package input data

        Parameters
        ----------
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
                MODFLOW zero-based stress period number to return. (default is
                zero)
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
        >>> ml.dis.plot()

        """
        from flopy.plot import PlotUtilities

        if not self.plottable:
            raise TypeError("Package {} is not plottable".format(self.name))

        axes = PlotUtilities._plot_package_helper(self, **kwargs)
        return axes

    def to_shapefile(self, filename, **kwargs):
        """
        Export 2-D, 3-D, and transient 2-D model data to shapefile (polygons).
        Adds an attribute for each layer in each data array

        Parameters
        ----------
        filename : str
            Shapefile name to write

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.lpf.to_shapefile('test_hk.shp')

        """
        import warnings

        warnings.warn("to_shapefile() is deprecated. use .export()")
        self.export(filename)

    def webdoc(self):
        if self.parent.version == "mf2k":
            wa = (
                "http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/"
                + self.url
            )
        elif self.parent.version == "mf2005":
            wa = (
                "http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/"
                + self.url
            )
        elif self.parent.version == "ModflowNwt":
            wa = (
                "http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/"
                + self.url
            )
        else:
            wa = None

        # open the web address
        if wa is not None:
            wb.open(wa)

    def write_file(self, check=False):
        """
        Every Package needs its own write_file function

        """
        print("IMPLEMENTATION ERROR: write_file must be overloaded")
        return

    @staticmethod
    def load(f, model, pak_type, ext_unit_dict=None, **kwargs):
        """
        Default load method for standard boundary packages.

        """

        # parse keywords
        if "nper" in kwargs:
            nper = kwargs.pop("nper")
        else:
            nper = None
        if "unitnumber" in kwargs:
            unitnumber = kwargs.pop("unitnumber")
        else:
            unitnumber = None
        if "check" in kwargs:
            check = kwargs.pop("check")
        else:
            check = True

        # open the file if not already open
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        elif hasattr(f, "name"):
            filename = f.name
        else:
            filename = "?"

        # set string from pak_type
        pak_type_str = str(pak_type).lower()

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # check for mfnwt version 11 option block
        nwt_options = None
        if model.version == "mfnwt" and "options" in line.lower():
            nwt_options = OptionBlock.load_options(f, pak_type)
            line = f.readline()

        # check for parameters
        nppak = 0
        if "parameter" in line.lower():
            t = line.strip().split()
            nppak = int(t[1])
            mxl = 0
            if nppak > 0:
                mxl = int(t[2])
                if model.verbose:
                    msg = (
                        3 * " "
                        + "Parameters detected. Number of "
                        + "parameters = {}".format(nppak)
                    )
                    print(msg)
            line = f.readline()

        # dataset 2a
        t = line.strip().split()
        imax = 2
        ipakcb = 0
        try:
            ipakcb = int(t[1])
        except:
            if model.verbose:
                msg = 3 * " " + "implicit ipakcb in {}".format(filename)
                print(msg)
        if "modflowdrt" in pak_type_str:
            try:
                nppak = int(t[2])
                imax += 1
            except:
                if model.verbose:
                    msg = 3 * " " + "implicit nppak in {}".format(filename)
                    print(msg)
            if nppak > 0:
                mxl = int(t[3])
                imax += 1
                if model.verbose:
                    msg = (
                        3 * " "
                        + "Parameters detected. Number of "
                        + "parameters = {}".format(nppak)
                    )
                    print(msg)

        options = []
        aux_names = []
        if len(t) > imax:
            it = imax
            while it < len(t):
                toption = t[it]
                if toption.lower() == "noprint":
                    options.append(toption.lower())
                elif "aux" in toption.lower():
                    options.append(" ".join(t[it : it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1

        # add auxillary information to nwt options
        if nwt_options is not None and options:
            if options[0] == "noprint":
                nwt_options.noprint = True
                if len(options) > 1:
                    nwt_options.auxillary = options[1:]
            else:
                nwt_options.auxillary = options

            options = nwt_options

        # set partype
        #  and read phiramp for modflow-nwt well package
        partype = ["cond"]
        if "modflowwel" in pak_type_str:
            partype = ["flux"]

        # check for "standard" single line options from mfnwt
        if "nwt" in model.version.lower():
            if "flopy.modflow.mfwel.modflowwel".lower() in pak_type_str:
                ipos = f.tell()
                line = f.readline()
                # test for specify keyword if a NWT well file
                if "specify" in line.lower():
                    nwt_options = OptionBlock(
                        line.lower().strip(), pak_type, block=False
                    )
                    if options:
                        if options[0] == "noprint":
                            nwt_options.noprint = True
                            if len(options) > 1:
                                nwt_options.auxillary = options[1:]
                        else:
                            nwt_options.auxillary = options

                    options = nwt_options
                else:
                    f.seek(ipos)
        elif "flopy.modflow.mfchd.modflowchd".lower() in pak_type_str:
            partype = ["shead", "ehead"]

        # get the list columns that should be scaled with sfac
        sfac_columns = pak_type._get_sfac_columns()

        # read parameter data
        if nppak > 0:
            dt = pak_type.get_empty(
                1, aux_names=aux_names, structured=model.structured
            ).dtype
            pak_parms = mfparbc.load(
                f, nppak, dt, model, ext_unit_dict, model.verbose
            )

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # read data for every stress period
        bnd_output = None
        stress_period_data = {}
        current = None
        for iper in range(nper):
            if model.verbose:
                msg = (
                    "   loading "
                    + str(pak_type)
                    + " for kper {:5d}".format(iper + 1)
                )
                print(msg)
            line = f.readline()
            if line == "":
                break
            t = line.strip().split()
            itmp = int(t[0])
            itmpp = 0
            try:
                itmpp = int(t[1])
            except:
                if model.verbose:
                    print("   implicit itmpp in {}".format(filename))

            if itmp == 0:
                bnd_output = None
                current = pak_type.get_empty(
                    itmp, aux_names=aux_names, structured=model.structured
                )
            elif itmp > 0:
                current = pak_type.get_empty(
                    itmp, aux_names=aux_names, structured=model.structured
                )
                current = ulstrd(
                    f, itmp, current, model, sfac_columns, ext_unit_dict
                )
                if model.structured:
                    current["k"] -= 1
                    current["i"] -= 1
                    current["j"] -= 1
                else:
                    current["node"] -= 1
                bnd_output = np.recarray.copy(current)
            else:
                if current is None:
                    bnd_output = None
                else:
                    bnd_output = np.recarray.copy(current)

            for iparm in range(itmpp):
                line = f.readline()
                t = line.strip().split()
                pname = t[0].lower()
                iname = "static"
                try:
                    tn = t[1]
                    c = tn.lower()
                    instance_dict = pak_parms.bc_parms[pname][1]
                    if c in instance_dict:
                        iname = c
                    else:
                        iname = "static"
                except:
                    if model.verbose:
                        print(
                            "  implicit static instance for "
                            + "parameter {}".format(pname)
                        )

                par_dict, current_dict = pak_parms.get(pname)
                data_dict = current_dict[iname]

                par_current = pak_type.get_empty(
                    par_dict["nlst"], aux_names=aux_names
                )

                #  get appropriate parval
                if model.mfpar.pval is None:
                    parval = float(par_dict["parval"])
                else:
                    try:
                        parval = float(model.mfpar.pval.pval_dict[pname])
                    except:
                        parval = float(par_dict["parval"])

                # fill current parameter data (par_current)
                for ibnd, t in enumerate(data_dict):
                    t = tuple(t)
                    par_current[ibnd] = tuple(
                        t[: len(par_current.dtype.names)]
                    )

                if model.structured:
                    par_current["k"] -= 1
                    par_current["i"] -= 1
                    par_current["j"] -= 1
                else:
                    par_current["node"] -= 1

                for ptype in partype:
                    par_current[ptype] *= parval

                if bnd_output is None:
                    bnd_output = np.recarray.copy(par_current)
                else:
                    bnd_output = stack_arrays(
                        (bnd_output, par_current),
                        asrecarray=True,
                        usemask=False,
                    )

            if bnd_output is None:
                stress_period_data[iper] = itmp
            else:
                stress_period_data[iper] = bnd_output

        dtype = pak_type.get_empty(
            0, aux_names=aux_names, structured=model.structured
        ).dtype

        if openfile:
            f.close()

        # set package unit number
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=pak_type._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)

        pak = pak_type(
            model,
            ipakcb=ipakcb,
            stress_period_data=stress_period_data,
            dtype=dtype,
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )
        if check:
            pak.check(
                f="{}.chk".format(pak.name[0]),
                verbose=pak.parent.verbose,
                level=0,
            )
        return pak
