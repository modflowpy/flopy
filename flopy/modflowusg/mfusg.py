"""mfusg module"""
import os
import flopy
import warnings
from inspect import getfullargspec

from ..utils import mfreadnam
from ..modflow import Modflow
from ..mbase import PackageLoadException


class ModflowUsg(Modflow):
    """
    MODFLOW-USG Model Class.

    Parameters
    ----------
    modelname : str, default "modflowusgtest".
        Name of model.  This string will be used to name the MODFLOW input
        that are created with write_model.
    namefile_ext : str, default "nam"
        Extension for the namefile.
    exe_name : str, default "mfusg.exe"
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
    >>> usg = flopy.modflowusg.ModflowUsg()
    """

    def __init__(
        self,
        modelname="modflowusgtest",
        structured=True,
        model_ws=".",
        **kwargs,
    ):
        """
        Constructs the ModflowUsg object.

        Overrides the parent Modflow object.
        """
        valid_args_defaults = {
            "namefile_ext": "nam",
            "exe_name": "mfusg.exe",
            "listunit": 2,
            "external_path": None,
            "verbose": False,
        }
        for arg, default_value in valid_args_defaults.items():
            setattr(self, arg, kwargs.pop(arg, default_value))

        setattr(self, "version", "mfusg")

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
            "disu": flopy.modflowusg.ModflowUsgDisU,
            "sms": flopy.modflowusg.ModflowUsgSms,
            "wel": flopy.modflowusg.ModflowUsgWel,
            "bcf6": flopy.modflowusg.ModflowUsgBcf,
            "lpf": flopy.modflowusg.ModflowUsgLpf,
            "cln": flopy.modflowusg.ModflowUsgCln,
            "gnc": flopy.modflowusg.ModflowUsgGnc,
            "bct": flopy.modflowusg.ModflowUsgBct,
        }
        return

    def __repr__(self):
        """Returns a representation of the ModflowUsg object."""
        nrow, ncol, nlay, nper = self.get_nrow_ncol_nlay_nper()
        if nrow is not None:
            # structured case
            s = (
                "MODFLOW {} layer(s) {} row(s) {} column(s) "
                "{} stress period(s)".format(nlay, nrow, ncol, nper)
            )
        else:
            # unstructured case
            nodes = ncol.sum()
            nodelay = " ".join(str(i) for i in ncol)
            print(nodelay, nlay, nper)
            s = (
                "MODFLOW unstructured\n"
                "  nodes = {}\n"
                "  layers = {}\n"
                "  periods = {}\n"
                "  nodelay = {}\n".format(nodes, nlay, nper, ncol)
            )
        return s

    @classmethod
    def load(
        cls,
        f,
        exe_name="mfusg.exe",
        verbose=False,
        model_ws=".",
        load_only=None,
        forgive=False,
        check=True,
    ):
        """
        Load an existing MODFLOW-USG model.

        Parameters
        ----------
        f : str
            Path to MODFLOW name file to load.
        exe_name : str, default "mfusg.exe"
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
        flopy.modflowusg.ModflowUsg

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflowusg.ModflowUsg.load('model.nam')
        """
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

        ml = cls(
            modelname,
            exe_name=exe_name,
            verbose=verbose,
            model_ws=model_ws,
            **attribs,
        )

        # read name file
        ext_unit_dict = mfreadnam.parsenamefile(
            namefile_path, ml.mfnam_packages, verbose=verbose
        )
        if ml.verbose:
            print(
                "\n{}\nExternal unit dictionary:\n{}\n{}\n".format(
                    50 * "-", ext_unit_dict, 50 * "-"
                )
            )

        # create a dict where key is the package name, value is unitnumber
        ext_pkg_d = {v.filetype: k for (k, v) in ext_unit_dict.items()}

        # reset version based on packages in the name file
        if "DISU" in ext_pkg_d:
            ml.structured = False

        # reset unit number for list file
        if "LIST" in ext_pkg_d:
            unitnumber = ext_pkg_d["LIST"]
            filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
            ml.lst.unit_number = [unitnumber]
            ml.lst.file_name = [filepth]

        # look for the free format flag in bas6
        bas_key = ext_pkg_d.get("BAS6")
        if bas_key is not None:
            bas = ext_unit_dict[bas_key]
            start = bas.filehandle.tell()
            line = bas.filehandle.readline()
            while line.startswith("#"):
                line = bas.filehandle.readline()
            if "FREE" in line.upper():
                ml.free_format_input = True
            bas.filehandle.seek(start)
        if verbose:
            print(f"ModflowBas6 free format:{ml.free_format_input}\n")

        # zone, mult, pval
        if "PVAL" in ext_pkg_d:
            ml.mfpar.set_pval(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("PVAL")
        if "ZONE" in ext_pkg_d:
            ml.mfpar.set_zone(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("ZONE")
        if "MULT" in ext_pkg_d:
            ml.mfpar.set_mult(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("MULT")

        files_successfully_loaded, files_not_loaded = cls._load_packages(
            ml, ext_unit_dict, ext_pkg_d, load_only, forgive
        )

        # pop binary output keys and any external file units that are now
        # internal
        for key in ml.pop_key_list:
            try:
                ml.remove_external(unit=key)
                item = ext_unit_dict.pop(key)
                if hasattr(item.filehandle, "close"):
                    item.filehandle.close()
            except KeyError:
                if ml.verbose:
                    print(
                        f"\nWARNING:\n    External file unit {key} does not "
                        "exist in ext_unit_dict."
                    )

        # write message indicating packages that were successfully loaded
        if ml.verbose:
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
        if check:
            ml.check(f=f"{ml.name}.chk", verbose=ml.verbose, level=0)

        # return model object
        return ml

    @staticmethod
    def _ext_unit_d_load(ml, ext_unit_dict, ext_unit_d_item):
        """
        Method to load a package from an ext_unit_dict item into model

        Parameters
        ----------
        ml : ModflowUsg model object for which package in ext_unit_d_item will
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
                ml,
                ext_unit_dict=ext_unit_dict,
                check=False,
            )
        else:
            ext_unit_d_item.package.load(
                ext_unit_d_item.filehandle,
                ml,
                ext_unit_dict=ext_unit_dict,
            )

    @classmethod
    def _load_packages(cls, ml, ext_unit_dict, ext_pkg_d, load_only, forgive):
        """
        Method to load packages into the MODFLOW-USG Model Class.
        For internal class use - should not be called by the user.

        Parameters
        ----------
        ml : ModflowUsg model object
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
            disnamdata.filehandle, ml, ext_unit_dict=ext_unit_dict, check=False
        )
        files_successfully_loaded.append(disnamdata.filename)
        if ml.verbose:
            print(f"   {dis.name[0]:4s} package load...success")
        assert ml.pop_key_list.pop() == dis_key
        ext_unit_dict.pop(dis_key).filehandle.close()

        dis.start_datetime = ml._start_datetime

        if load_only is None:
            # load all packages/files
            load_only = ext_pkg_d.keys()
        else:  # check items in list
            if not isinstance(load_only, list):
                load_only = [load_only]
            not_found = []
            for i, filetype in enumerate(load_only):
                load_only[i] = filetype = filetype.upper()
                if filetype not in ext_pkg_d:
                    not_found.append(filetype)
            if not_found:
                raise KeyError(
                    "the following load_only entries were not found "
                    "in the ext_unit_dict: " + str(not_found)
                )

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                if item.filetype in load_only:
                    if forgive:
                        try:
                            cls._ext_unit_d_load(ml, ext_unit_dict, item)
                            files_successfully_loaded.append(item.filename)
                            if ml.verbose:
                                print(
                                    f"   {item.filetype:4s} package \
                                    load...success"
                                )
                        except PackageLoadException as e:
                            ml.load_fail = True
                            if ml.verbose:
                                raise PackageLoadException(
                                    error=f"   {item.filetype:4s} package \
                                    load...failed\n   {e!s}"
                                )
                            files_not_loaded.append(item.filename)
                    else:
                        cls._ext_unit_d_load(ml, ext_unit_dict, item)
                        files_successfully_loaded.append(item.filename)
                        if ml.verbose:
                            print(
                                f"   {item.filetype:4s} package load...success"
                            )
                else:
                    if ml.verbose:
                        print(f"   {item.filetype:4s} package load...skipped")
                    files_not_loaded.append(item.filename)
            elif "data" not in item.filetype.lower():
                files_not_loaded.append(item.filename)
                if ml.verbose:
                    print(f"   {item.filetype:4s} package load...skipped")
            elif "data" in item.filetype.lower():
                if ml.verbose:
                    print(f"   {item.filetype} package load...skipped")
                    print(f"      {os.path.basename(item.filename)}")
                if key not in ml.pop_key_list:
                    # do not add unit number (key) if it already exists
                    if key not in ml.external_units:
                        ml.external_fnames.append(item.filename)
                        ml.external_units.append(key)
                        ml.external_binflag.append(
                            "binary" in item.filetype.lower()
                        )
                        ml.external_output.append(False)
            else:
                raise KeyError(f"unhandled case: {key}, {item}")

        return files_successfully_loaded, files_not_loaded


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
            fmts.append("%10.2E")
        elif vtype == "o":
            fmts.append("%10s")
        elif vtype == "s":
            msg = (
                "mfusg.fmt_string error: 'str' type found in dtype. "
                "This gives unpredictable results when "
                "recarray to file - change to 'object' type"
            )
            raise TypeError(msg)
        else:
            raise TypeError(
                "mfusg.fmt_string error: unknown vtype in "
                "field: {}".format(field)
            )
    return "".join(fmts)
