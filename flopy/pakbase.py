"""
pakbase module
  This module contains the base package class from which
  all of the other packages inherit from.

"""
import abc
import os
import webbrowser as wb
from typing import Union

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from .utils import MfList, OptionBlock, Transient2d, Util2d, Util3d, check
from .utils.flopy_io import ulstrd


class PackageInterface:
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError(
            "must define name in child class to use this base class"
        )

    @name.setter
    @abc.abstractmethod
    def name(self, name):
        raise NotImplementedError(
            "must define name in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def parent(self):
        raise NotImplementedError(
            "must define parent in child class to use this base class"
        )

    @parent.setter
    @abc.abstractmethod
    def parent(self, name):
        raise NotImplementedError(
            "must define parent in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def package_type(self):
        raise NotImplementedError(
            "must define package_type in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def data_list(self):
        # [data_object, data_object, ...]
        raise NotImplementedError(
            "must define data_list in child class to use this base class"
        )

    @abc.abstractmethod
    def export(self, f, **kwargs):
        raise NotImplementedError(
            "must define export in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def plottable(self):
        raise NotImplementedError(
            "must define plottable in child class to use this base class"
        )

    @property
    def has_stress_period_data(self):
        return self.__dict__.get("stress_period_data", None) is not None

    @property
    def _mg_resync(self):
        if self.package_type.lower()[:4] in ("dis", "bas"):
            return True
        return False

    @staticmethod
    def _check_thresholds(chk, array, active, thresholds, name):
        """Checks array against min and max threshold values."""
        mn, mx = thresholds
        chk.values(
            array,
            active & (array < mn),
            f"{name} values below checker threshold of {mn}",
            "Warning",
        )
        chk.values(
            array,
            active & (array > mx),
            f"{name} values above checker threshold of {mx}",
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
        if "DIS" in self.parent.get_package_list():
            dis = self.parent.dis
        else:
            dis = self.parent.disu
        if dis.laycbd.sum() > 0:
            # pad non-quasi-3D layers in vkcb array with ones so
            # they won't fail checker
            vkcb = self.vkcb.array.copy()
            for l in range(self.vkcb.shape[0]):
                if dis.laycbd[l] == 0:
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
            if self.hk.shape[1] == None:
                hk = np.asarray(
                    [a.array.flatten() for a in self.hk], dtype=object
                )
            else:
                hk = self.hk.array.copy()
        else:
            hk = self.k.array.copy()
        if "vka" in self.__dict__ and self.layvka.sum() > 0:
            if self.vka.shape[1] == None:
                vka = np.asarray(
                    [a.array.flatten() for a in self.vka], dtype=object
                )
            else:
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
                    f"zero or negative {name} values",
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
            txt = f"check method not implemented for {self.name[0]} Package."
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
                    "\r    STORAGECOEFFICIENT option is activated, "
                    "storage values are read storage coefficients"
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
            skip_sy_check = False
            if "laytyp" in self.__dict__:
                inds = np.array(
                    [
                        True
                        if l > 0 or l < 0 and "THICKSTRT" in self.options
                        else False
                        for l in self.laytyp
                    ]
                )
                if inds.any():
                    if self.sy.shape[1] is None:
                        # unstructured; build flat nodal property array slicers (by layer)
                        node_to = np.cumsum([s.array.size for s in self.ss])
                        node_from = np.array([0] + list(node_to[:-1]))
                        node_k_slices = np.array(
                            [
                                np.s_[n_from:n_to]
                                for n_from, n_to in zip(node_from, node_to)
                            ]
                        )[inds]
                        sarrays["sy"] = np.concatenate(
                            [sarrays["sy"][sl] for sl in node_k_slices]
                        ).flatten()
                        active = np.concatenate(
                            [active[sl] for sl in node_k_slices]
                        ).flatten()
                    else:
                        sarrays["sy"] = sarrays["sy"][inds, :, :]
                        active = active[inds, :, :]
                else:
                    skip_sy_check = True
            else:
                iconvert = self.iconvert.array
                for ishape in np.ndindex(active.shape):
                    if active[ishape]:
                        active[ishape] = (
                            iconvert[ishape] > 0 or iconvert[ishape] < 0
                        )
            if not skip_sy_check:
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

    Parameters
    ----------
    parent : object
        Parent model object.
    extension : str or list, default "glo"
        File extension, without ".", use list to describe more than one.
    name : str or list, default "GLOBAL"
        Package name, use list to describe more than one.
    unit_number : int or list, default 1
        Unit number, use list to describe more than one.
    filenames : str or list, default None
    allowDuplicates : bool, default False
        Allow more than one instance of package in parent.
    """

    def __init__(
        self,
        parent,
        extension="glo",
        name="GLOBAL",
        unit_number=1,
        filenames=None,
        allowDuplicates=False,
    ):
        # To be able to access the parent model object's attributes
        self.parent = parent
        if not isinstance(extension, list):
            extension = [extension]
        self.extension = []
        self.file_name = []
        if isinstance(filenames, str):
            filenames = [filenames]
        for idx, e in enumerate(extension):
            self.extension.append(e)
            file_name = f"{self.parent.name}.{e}"
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
                        s += f" {attr} = {value[0]!s}\n"
                    else:
                        s += f" {attr} (list, items = {len(value)})\n"
                elif isinstance(value, np.ndarray):
                    s += f" {attr} (array, shape = {str(value.shape)[1:-1]})\n"
                else:
                    s += f" {attr} = {value!s} ({str(type(value))[7:-2]})\n"
        return s

    def __getitem__(self, item):
        if hasattr(self, "stress_period_data"):
            # added this check because stress_period_data also used in Oc and
            # Oc88 but is not a MfList
            spd = getattr(self, "stress_period_data")
            if isinstance(item, MfList):
                if not isinstance(item, list) and not isinstance(item, tuple):
                    msg = (
                        f"package.__getitem__() kper {item} not in data.keys()"
                    )
                    assert item in list(spd.data.keys()), msg
                    return spd[item]

                if item[1] not in self.dtype.names:
                    raise Exception(
                        "package.__getitem(): item {} not in dtype names "
                        "{}".format(item, self.dtype.names)
                    )

                msg = (
                    f"package.__getitem__() kper {item[0]} not in data.keys()"
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
                                    fmtin=vo.format.fortran,
                                    locat=vo.locat,
                                )
                            )
                        value = new_list

        if all(hasattr(self, attr) for attr in ["parent", "_name"]):
            if not self.parent._mg_resync:
                self.parent._mg_resync = self._mg_resync

        super().__setattr__(key, value)

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
        from . import export

        return export.utils.package_export(f, self, **kwargs)

    def _generate_heading(self):
        """Generate heading."""
        from . import __version__

        parent = self.parent
        self.heading = (
            f"# {self.name[0]} package for "
            f"{parent.version_types[parent.version]} "
            f"generated by Flopy {__version__}"
        )

    @staticmethod
    def _prepare_filenames(filenames, num=1):
        """Prepare filenames parameter."""
        if filenames is None:
            return [None] * num
        elif isinstance(filenames, str):
            filenames = [filenames]
        if isinstance(filenames, list):
            if len(filenames) < num:
                filenames += [None] * (num - len(filenames))
            elif len(filenames) > num:
                filenames = filenames[:num]
            return filenames
        raise ValueError(f"unexpected filenames: {filenames}")

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
                    "layer below confined layer"
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
                    txt += f"    {'layer':>10s}{'row':>10s}{'column':>10s}{tag:>15s}\n"
                txt += f"    {k + 1:10d}{i + 1:10d}{j + 1:10d}{v[k, i, j]:15.7g}\n"
        elif ndim == 2:
            tag = name[0].lower().replace(" layer ", "")
            txt += f"    {'row':>10s}{'column':>10s}{tag:>15s}\n"
            for [i, j] in idx:
                txt += f"    {i + 1:10d}{j + 1:10d}{v[i, j]:15.7g}\n"
        elif ndim == 1:
            txt += f"    {'number':>10s}{name[0]:>15s}\n"
            for i in idx:
                txt += f"    {i + 1:10d}{v[i]:15.7g}\n"
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
        from .plot import PlotUtilities

        if not self.plottable:
            raise TypeError(f"Package {self.name} is not plottable")

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
        """Open the web documentation."""
        if self.parent.version == "mf2k":
            wa = f"https://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/{self.url}"
        elif self.parent.version == "mf2005":
            wa = f"https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/{self.url}"
        elif self.parent.version == "ModflowNwt":
            wa = f"https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/{self.url}"
        else:
            return

        wb.open(wa)

    def write_file(self, f=None, check=False):
        """
        Every Package needs its own write_file function

        """
        print("IMPLEMENTATION ERROR: write_file must be overloaded")
        return

    @staticmethod
    def load(
        f: Union[str, bytes, os.PathLike],
        model,
        pak_type,
        ext_unit_dict=None,
        **kwargs,
    ):
        """
        Default load method for standard boundary packages.

        """
        from .modflow.mfparbc import ModflowParBc as mfparbc

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
                    print(
                        f"   Parameters detected. Number of parameters = {nppak}"
                    )
            line = f.readline()

        # dataset 2a
        t = line.strip().split()
        imax = 2
        ipakcb = 0
        try:
            ipakcb = int(t[1])
        except:
            if model.verbose:
                print(f"   implicit ipakcb in {filename}")
        if "modflowdrt" in pak_type_str:
            try:
                nppak = int(t[2])
                imax += 1
            except:
                if model.verbose:
                    print(f"   implicit nppak in {filename}")
            if nppak > 0:
                mxl = int(t[3])
                imax += 1
                if model.verbose:
                    print(
                        f"   Parameters detected. Number of parameters = {nppak}"
                    )

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
                if "mfusgwel" in pak_type_str:
                    if toption.lower() == "autoflowreduce":
                        options.append(toption.lower())
                    elif toption.lower() == "iunitafr":
                        options.append(f"{toption.lower()} {t[it+1]}")
                        it += 1
                it += 1

        # add auxillary information to nwt options
        if nwt_options is not None:
            if options:
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
        bnd_output_cln = None
        stress_period_data_cln = {}
        current_cln = None
        for iper in range(nper):
            if model.verbose:
                msg = f"   loading {pak_type} for kper {iper + 1:5d}"
                print(msg)
            line = f.readline()
            if line == "":
                break
            t = line.strip().split()
            itmp = int(t[0])
            itmpp = 0
            if nppak > 0:
                itmpp = int(t[1])

            if len(t) > 1:
                t = t[:2]  # trap cases with text followed by digits (eg SP 5)
            itmp_cln = 0
            if "mfusgwel" in pak_type_str:
                try:
                    itmp_cln = int(t[2])
                except:
                    if model.verbose:
                        print(f"   implicit itmp_cln of 0 in {filename}")

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

            if itmp_cln == 0:
                bnd_output_cln = None
                current_cln = pak_type.get_empty(
                    itmp_cln, aux_names=aux_names, structured=False
                )
            elif itmp_cln > 0:
                current_cln = pak_type.get_empty(
                    itmp_cln, aux_names=aux_names, structured=False
                )
                current_cln = ulstrd(
                    f,
                    itmp_cln,
                    current_cln,
                    model,
                    sfac_columns,
                    ext_unit_dict,
                )
                current_cln["node"] -= 1
                bnd_output_cln = np.recarray.copy(current_cln)
            else:
                if current_cln is None:
                    bnd_output_cln = None
                else:
                    bnd_output_cln = np.recarray.copy(current_cln)

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
                            f"  implicit static instance for parameter {pname}"
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

            if bnd_output_cln is None:
                stress_period_data_cln[iper] = itmp_cln
            else:
                stress_period_data_cln[iper] = bnd_output_cln

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

        if "mfusgwel" in pak_type_str:
            cln_dtype = pak_type.get_empty(
                0, aux_names=aux_names, structured=False
            ).dtype
            pak = pak_type(
                model,
                ipakcb=ipakcb,
                stress_period_data=stress_period_data,
                cln_stress_period_data=stress_period_data_cln,
                dtype=dtype,
                cln_dtype=cln_dtype,
                options=options,
                unitnumber=unitnumber,
                filenames=filenames,
            )
        else:
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
                f=f"{pak.name[0]}.chk",
                verbose=pak.parent.verbose,
                level=0,
            )
        return pak
