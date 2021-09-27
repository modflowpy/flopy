"""
mfusg module.


"""

import os
import flopy
from inspect import getfullargspec
from ..utils import mfreadnam

from ..modflow import Modflow


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
        namefile_ext="nam",
        exe_name="mfusg.exe",
        structured=True,
        listunit=2,
        model_ws=".",
        external_path=None,
        verbose=False,
        version="mfusg",
        **kwargs,
    ):
        super().__init__(
            modelname,
            namefile_ext,
            version="mfusg",
            exe_name=exe_name,
            structured=structured,
            listunit=listunit,
            model_ws=model_ws,
            external_path=external_path,
            verbose=verbose,
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

        files_successfully_loaded = []
        files_not_loaded = []

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

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                if item.filetype in load_only:
                    package_load_args = getfullargspec(item.package.load)[0]
                    if forgive:
                        try:
                            if "check" in package_load_args:
                                item.package.load(
                                    item.filehandle,
                                    ml,
                                    ext_unit_dict=ext_unit_dict,
                                    check=False,
                                )
                            else:
                                item.package.load(
                                    item.filehandle,
                                    ml,
                                    ext_unit_dict=ext_unit_dict,
                                )
                            files_successfully_loaded.append(item.filename)
                            if ml.verbose:
                                print(
                                    f"   {item.filetype:4s} package load...success"
                                )
                        except Exception as e:
                            ml.load_fail = True
                            if ml.verbose:
                                print(
                                    f"   {item.filetype:4s} package load...failed"
                                )
                                print(f"   {e!s}")
                            files_not_loaded.append(item.filename)
                    else:
                        if "check" in package_load_args:
                            item.package.load(
                                item.filehandle,
                                ml,
                                ext_unit_dict=ext_unit_dict,
                                check=False,
                            )
                        else:
                            item.package.load(
                                item.filehandle,
                                ml,
                                ext_unit_dict=ext_unit_dict,
                            )
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


def fmt_string(array):
    """
    Returns a C-style fmt string for numpy savetxt that corresponds to
    the dtype.

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
                "mfcln.fmt_string error: 'str' type found in dtype. "
                "This gives unpredictable results when "
                "recarray to file - change to 'object' type"
            )
            raise TypeError(msg)
        else:
            raise TypeError(
                "mfcln.fmt_string error: unknown vtype in "
                "field: {}".format(field)
            )
    return "".join(fmts)
