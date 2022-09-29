"""Mfusg module."""
import os
from inspect import getfullargspec

import flopy

from ..mbase import PackageLoadException
from ..modflow import Modflow
from ..utils import mfreadnam


class MfUsg(Modflow):
    """MODFLOW-USG Model Class.

    Parameters
    ----------
    modelname : str, default "modflowusgtest".
        Name of model.  This string will be used to name the MODFLOW input
        that are created with write_model.
    namefile_ext : str, default "nam"
        Extension for the namefile.
    exe_name : str, default "mfusg"
        The name of the executable to use.
    structured : bool, default True
        Specify if model grid is structured (default) or unstructured.
    listunit : int, default 2
        Unit number for the list file.
    model_ws : str, default "."
        Model workspace.  Directory name to create model data sets.
        Default is the present working directory.
    external_path : str, optional
        Location for external files.
    verbose : bool, default False
        Print additional information to the screen.


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
    >>> usg = flopy.mfusg.MfUsg()
    """

    def __init__(
        self,
        modelname="modflowusgtest",
        structured=True,
        model_ws=".",
        **kwargs,
    ):
        """Constructs the MfUsg object. Overrides the parent Modflow object."""
        valid_args_defaults = {
            "namefile_ext": "nam",
            "exe_name": "mfusg",
            "listunit": 2,
            "external_path": None,
            "verbose": False,
        }

        for arg, default_value in valid_args_defaults.items():
            setattr(self, arg, kwargs.pop(arg, default_value))

        # remove "version" from kwarg if inadvertently provided
        try:
            kwargs.pop("version")
        except KeyError:
            pass

        super().__init__(
            modelname,
            self.namefile_ext,
            version="mfusg",
            exe_name=self.exe_name,
            structured=structured,
            listunit=self.listunit,
            model_ws=model_ws,
            external_path=self.external_path,
            verbose=self.verbose,
            **kwargs,
        )
        # Create a dictionary to map package with package object.
        # This is used for loading models.
        self.mfnam_packages = {
            "zone": flopy.modflow.ModflowZon,
            "mult": flopy.modflow.ModflowMlt,
            "pval": flopy.modflow.ModflowPval,
            "bas6": flopy.modflow.ModflowBas,
            "dis": flopy.modflow.ModflowDis,
            "hfb6": flopy.modflow.ModflowHfb,
            "chd": flopy.modflow.ModflowChd,
            "fhb": flopy.modflow.ModflowFhb,
            "drn": flopy.modflow.ModflowDrn,
            "drt": flopy.modflow.ModflowDrt,
            "rch": flopy.modflow.ModflowRch,
            "evt": flopy.modflow.ModflowEvt,
            "ghb": flopy.modflow.ModflowGhb,
            "riv": flopy.modflow.ModflowRiv,
            "str": flopy.modflow.ModflowStr,
            "sfr": flopy.modflow.ModflowSfr2,
            "lak": flopy.modflow.ModflowLak,
            "gage": flopy.modflow.ModflowGage,
            "oc": flopy.modflow.ModflowOc,
            "sub": flopy.modflow.ModflowSub,
            "swt": flopy.modflow.ModflowSwt,
            "disu": flopy.mfusg.MfUsgDisU,
            "sms": flopy.mfusg.MfUsgSms,
            "wel": flopy.mfusg.MfUsgWel,
            "bcf6": flopy.mfusg.MfUsgBcf,
            "lpf": flopy.mfusg.MfUsgLpf,
            "cln": flopy.mfusg.MfUsgCln,
            "gnc": flopy.mfusg.MfUsgGnc,
        }

    def __repr__(self):
        """Returns a representation of the MfUsg object."""
        nrow, ncol, nlay, nper = self.get_nrow_ncol_nlay_nper()
        if nrow is not None:
            # structured case
            msg = (
                f"MODFLOW {nlay} layer(s) {nrow} row(s) {ncol} column(s) "
                f"{nper} stress period(s)"
            )
        else:
            # unstructured case
            msg = (
                "MODFLOW unstructured\n"
                f"  nodes = {ncol.sum()}\n"
                f"  layers = {nlay}\n"
                f"  stress periods = {nper}\n"
                f"  nodelay = {ncol}\n"
            )
        return msg

    @classmethod
    def load(
        cls,
        f,
        version="mfusg",
        exe_name="mfusg",
        verbose=False,
        model_ws=".",
        load_only=None,
        forgive=False,
        check=True,
    ):
        """Load an existing MODFLOW-USG model.

        Parameters
        ----------
        f : str
            Path to MODFLOW name file to load.
        version : str, default "mfusg"
            MODFLOW version. Must be "mfusg".
        exe_name : str, default "mfusg"
            MODFLOW executable name.
        verbose : bool, default False
            Show messages that can be useful for debugging.
        model_ws : str, default "."
            Model workspace path. Default is the current directory.
        load_only : list, str or None
            List of case insensitive packages to load, e.g. ["bas6", "lpf"].
            One package can also be specified, e.g. "rch". Default is None,
            which attempts to load all files. An empty list [] will not load
            any additional packages than is necessary. At a minimum, "dis" or
            "disu" is always loaded.
        forgive : bool, optional
            Option to raise exceptions on package load failure, which can be
            useful for debugging. Default False.
        check : boolean, optional
            Check model input for common errors. Default True.

        Returns
        -------
        flopy.mfusg.MfUsg

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg.load('model.nam')
        """
        if version != "mfusg":
            version = "mfusg"

        # similar to modflow command: if file does not exist , try file.nam
        namefile_path = os.path.join(model_ws, f)
        if not os.path.isfile(namefile_path) and os.path.isfile(
            f"{namefile_path}.nam"
        ):
            namefile_path += ".nam"
        if not os.path.isfile(namefile_path):
            raise OSError(f"cannot find name file: {namefile_path}")

        # Determine model name from 'f', without any extension or path
        modelname = os.path.splitext(os.path.basename(f))[0]

        if verbose:
            print(f"\nCreating new model with name: {modelname}\n{50 * '-'}\n")

        attribs = mfreadnam.attribs_from_namfile_header(
            os.path.join(model_ws, f)
        )

        model = cls(
            modelname,
            exe_name=exe_name,
            verbose=verbose,
            model_ws=model_ws,
            **attribs,
        )

        # read name file
        ext_unit_dict = mfreadnam.parsenamefile(
            namefile_path, model.mfnam_packages, verbose=verbose
        )
        if model.verbose:
            print(
                f"\n{50 * '-'}\nExternal unit dictionary:\n"
                f"{ext_unit_dict}\n{50 * '-'}\n"
            )

        # create a dict where key is the package name, value is unitnumber
        ext_pkg_d = {v.filetype: k for (k, v) in ext_unit_dict.items()}

        # reset version based on packages in the name file
        if "DISU" in ext_pkg_d:
            model.structured = False

        # reset unit number for list file
        if "LIST" in ext_pkg_d:
            unitnumber = ext_pkg_d["LIST"]
            filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
            model.lst.unit_number = [unitnumber]
            model.lst.file_name = [filepth]

        # look for the free format flag in bas6
        bas_key = ext_pkg_d.get("BAS6")
        if bas_key is not None:
            bas = ext_unit_dict[bas_key]
            start = bas.filehandle.tell()
            line = bas.filehandle.readline()
            while line.startswith("#"):
                line = bas.filehandle.readline()
            if "FREE" in line.upper():
                model.free_format_input = True
            bas.filehandle.seek(start)
        if verbose:
            print(f"ModflowBas6 free format:{model.free_format_input}\n")

        # set mfpar / pval
        cls._set_mfpar_pval(model, ext_unit_dict, ext_pkg_d)

        files_successfully_loaded, files_not_loaded = cls._load_packages(
            model, ext_unit_dict, ext_pkg_d, load_only, forgive
        )

        # set up binary output / external file units
        cls._set_output_external(model, ext_unit_dict)

        # send messages re: success/failure of loading
        cls._send_load_messages(
            model, files_successfully_loaded, files_not_loaded
        )

        if check:
            model.check(f=f"{model.name}.chk", verbose=model.verbose, level=0)

        # return model object
        return model

    @classmethod
    def _load_packages(
        cls, model, ext_unit_dict, ext_pkg_d, load_only, forgive
    ):
        """
        Method to load packages into the MODFLOW-USG Model Class.
        For internal class use - should not be called by the user.

        Parameters
        ----------
        model : MfUsg model object
        ext_unit_dict : dict
            For each file listed in the name file, a
            :class:`flopy.utils.mfreadnam.NamData` instance.
            Keyed by unit number.
        ext_pkg_d : dict
            key is package name, value is unitnumber
        load_only : list, str or None
            List of case insensitive packages to load, e.g. ["bas6", "lpf"].
            One package can also be specified, e.g. "rch". Default is None,
            which attempts to load all files. An empty list [] will not load
            any additional packages than is necessary. At a minimum, "dis" or
            "disu" is always loaded.
        forgive : bool
            Option to raise exceptions on package load failure.

        Returns
        ----------
        files_successfully_loaded : list of loaded files
        files_not_loaded : list of files that were not loaded
        """
        files_successfully_loaded = []
        files_not_loaded = []

        # load dis
        dis_key = ext_pkg_d.get("DIS") or ext_pkg_d.get("DISU")
        if dis_key is None:
            raise KeyError("discretization entry not found in nam file")
        disnamdata = ext_unit_dict[dis_key]
        dis = disnamdata.package.load(
            disnamdata.filehandle,
            model,
            ext_unit_dict=ext_unit_dict,
            check=False,
        )
        files_successfully_loaded.append(disnamdata.filename)
        if model.verbose:
            print(f"   {dis.name[0]:4s} package load...success")
        assert model.pop_key_list.pop() == dis_key
        ext_unit_dict.pop(dis_key).filehandle.close()

        dis.start_datetime = model.start_datetime

        load_only = cls._prepare_load_only(load_only, ext_pkg_d)

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                (
                    files_successfully_loaded,
                    files_not_loaded,
                ) = cls._load_ext_unit_dict_paks(
                    model,
                    ext_unit_dict,
                    load_only,
                    item,
                    forgive,
                    files_successfully_loaded,
                    files_not_loaded,
                )
            elif "data" not in item.filetype.lower():
                files_not_loaded.append(item.filename)
                if model.verbose:
                    print(f"   {item.filetype:4s} package load...skipped")
            elif "data" in item.filetype.lower():
                cls._prepare_external_files(model, key, item)
            else:
                raise KeyError(f"unhandled case: {key}, {item}")

        return files_successfully_loaded, files_not_loaded

    @staticmethod
    def _prepare_load_only(load_only, ext_pkg_d):
        """Prepare load_only list."""
        if load_only is None:
            # load all packages/files
            load_only = ext_pkg_d.keys()
        else:  # check items in list
            if not isinstance(load_only, list):
                load_only = [load_only]
            not_found = []
            for idx, filetype in enumerate(load_only):
                load_only[idx] = filetype = filetype.upper()
                if filetype not in ext_pkg_d:
                    not_found.append(filetype)
            if not_found:
                raise KeyError(
                    "the following load_only entries were not found "
                    "in the ext_unit_dict: " + str(not_found)
                )
        return load_only

    @classmethod
    def _load_ext_unit_dict_paks(
        cls,
        model,
        ext_unit_dict,
        load_only,
        item,
        forgive,
        files_successfully_loaded,
        files_not_loaded,
    ):
        """Load packages from ext_unit_dict."""
        if item.filetype in load_only:
            if forgive:
                try:
                    cls._ext_unit_d_load(model, ext_unit_dict, item)
                    files_successfully_loaded.append(item.filename)
                    if model.verbose:
                        print(
                            f"   {item.filetype:4s} package \
                            load...success"
                        )
                except PackageLoadException as err:
                    model.load_fail = True
                    if model.verbose:
                        raise PackageLoadException(
                            error=f"   {item.filetype:4s} package \
                            load...failed"
                        ) from err
                    files_not_loaded.append(item.filename)
            else:
                cls._ext_unit_d_load(model, ext_unit_dict, item)
                files_successfully_loaded.append(item.filename)
                if model.verbose:
                    print(f"   {item.filetype:4s} package load...success")
        else:
            if model.verbose:
                print(f"   {item.filetype:4s} package load...skipped")
            files_not_loaded.append(item.filename)

        return files_successfully_loaded, files_not_loaded

    @staticmethod
    def _prepare_external_files(model, key, item):
        """Prepare external files for ext_unit_dict item."""
        if model.verbose:
            print(f"   {item.filetype} package load...skipped")
            print(f"      {os.path.basename(item.filename)}")
        if key not in model.pop_key_list:
            # do not add unit number (key) if it already exists
            if key not in model.external_units:
                model.external_fnames.append(item.filename)
                model.external_units.append(key)
                model.external_binflag.append(
                    "binary" in item.filetype.lower()
                )
                model.external_output.append(False)

    @staticmethod
    def _ext_unit_d_load(model, ext_unit_dict, ext_unit_d_item):
        """
        Method to load a package from an ext_unit_dict item into model

        Parameters
        ----------
        model : MfUsg model object for which package in ext_unit_d_item will
            be loaded
        ext_unit_dict : dict
            For each file listed in the name file, a
            :class:`flopy.utils.mfreadnam.NamData` instance.
            Keyed by unit number.
        ext_unit_d_item : :class:`flopy.utils.mfreadnam.NamData` instance.
            Must be an item of ext_unit_dict.
        """
        package_load_args = getfullargspec(ext_unit_d_item.package.load)[0]
        if "check" in package_load_args:
            ext_unit_d_item.package.load(
                ext_unit_d_item.filehandle,
                model,
                ext_unit_dict=ext_unit_dict,
                check=False,
            )
        else:
            ext_unit_d_item.package.load(
                ext_unit_d_item.filehandle,
                model,
                ext_unit_dict=ext_unit_dict,
            )

    @staticmethod
    def _set_mfpar_pval(model, ext_unit_dict, ext_pkg_d):
        """Set mfpar/pval items."""
        # zone, mult, pval
        if "PVAL" in ext_pkg_d:
            model.mfpar.set_pval(model, ext_unit_dict)
            assert model.pop_key_list.pop() == ext_pkg_d.get("PVAL")
        if "ZONE" in ext_pkg_d:
            model.mfpar.set_zone(model, ext_unit_dict)
            assert model.pop_key_list.pop() == ext_pkg_d.get("ZONE")
        if "MULT" in ext_pkg_d:
            model.mfpar.set_mult(model, ext_unit_dict)
            assert model.pop_key_list.pop() == ext_pkg_d.get("MULT")

    @staticmethod
    def _set_output_external(model, ext_unit_dict):
        """Set up binary output / external file units."""
        # pop binary output keys and any external file units that are now
        # internal
        for key in model.pop_key_list:
            try:
                model.remove_external(unit=key)
                item = ext_unit_dict.pop(key)
                if hasattr(item.filehandle, "close"):
                    item.filehandle.close()
            except KeyError:
                if model.verbose:
                    print(
                        f"\nWARNING:\n    External file unit {key} does not "
                        "exist in ext_unit_dict."
                    )

    @staticmethod
    def _send_load_messages(
        model, files_successfully_loaded, files_not_loaded
    ):
        """Send messages re: success/failure of loading."""
        # write message indicating packages that were successfully loaded
        if model.verbose:
            print("")
            print(
                f"   The following {len(files_successfully_loaded)} packages "
                "were successfully loaded."
            )
            for fname in files_successfully_loaded:
                print(f"      {os.path.basename(fname)}")
            if len(files_not_loaded) > 0:
                print(
                    f"   The following {len(files_not_loaded)} packages "
                    "were not loaded."
                )
                for fname in files_not_loaded:
                    print(f"      {os.path.basename(fname)}")


def fmt_string(array):
    """
    Returns a C-style fmt string for numpy savetxt that corresponds to
    the dtype.

    Parameters
    ----------
    array : numpy array
    """
    fmts = []
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if vtype in ("i", "b"):
            fmts.append("%10d")
        elif vtype == "f":
            fmts.append("%14.6g")
        elif vtype == "o":
            fmts.append("%10s")
        elif vtype == "s":
            msg = (
                "mfusg.fmt_string error: 'str' type found in dtype."
                "This gives unpredictable results when"
                "recarray to file - change to 'object' type"
            )
            raise TypeError(msg)
        else:
            raise TypeError(
                "mfusg.fmt_string error: unknown vtype in" f"field: {field}"
            )
    return "".join(fmts)
