"""
mfreadnam module.  Contains the NamData class. Note that the user can access
the NamData class as `flopy.modflow.NamData`.

Additional information about the MODFLOW name file can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/name_file.htm>`_.

"""
import os
import sys

if sys.version_info < (3, 6):
    from collections import OrderedDict

    dict = OrderedDict


class NamData(object):
    """
    MODFLOW Namefile Class.

    Parameters
    ----------
    pkgtype : string
        String identifying the type of MODFLOW package. See the
        mfnam_packages dictionary keys in the model object for a list
        of supported packages. This dictionary is also passed in as packages.
    name : string
        Filename of the package file identified in the name file
    handle : file handle
        File handle referring to the file identified by `name`
    packages : dictionary
        Dictionary of package objects as defined in the
        `mfnam_packages` attribute of :class:`flopy.modflow.mf.Modflow`.

    Attributes
    ----------
    filehandle : file handle
        File handle to the package file. Read from `handle`.
    filename : string
        Filename of the package file identified in the name file.
        Read from `name`.
    filetype : string
        String identifying the type of MODFLOW package. Read from
        `pkgtype`.
    package : string
        Package type. Only assigned if `pkgtype` is found in the keys
        of `packages`

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, pkgtype, name, handle, packages):
        self.filehandle = handle
        self.filename = name
        self.filetype = pkgtype
        self.package = None
        if self.filetype.lower() in packages:
            self.package = packages[self.filetype.lower()]

    def __repr__(self):
        return "filename:{0}, filetype:{1}".format(
            self.filename, self.filetype
        )


def getfiletypeunit(nf, filetype):
    """
    Method to return unit number of a package from a NamData instance

    Parameters
    ----------
    nf : NamData instance
    filetype : string, name of package seeking information for

    Returns
    -------
    cunit : int, unit number corresponding to the package type

    """
    for cunit, cvals in nf.items():
        if cvals.filetype.lower() == filetype.lower():
            return cunit
    print('Name file does not contain file of type "{0}"'.format(filetype))
    return None


def parsenamefile(namfilename, packages, verbose=True):
    """
    Returns dict from the nam file with NamData keyed by unit number

    Parameters
    ----------
    namefilename : str
        Name of the MODFLOW namefile to parse.
    packages : dict
        Dictionary of package objects as defined in the `mfnam_packages`
        attribute of :class:`flopy.modflow.mf.Modflow`.
    verbose : bool
        Print messages to screen.  Default is True.

    Returns
    -------
    dict or OrderedDict
        For each file listed in the name file, a
        :class:`flopy.utils.mfreadnam.NamData` instance
        is stored in the returned dict keyed by unit number. Prior to Python
        version 3.6 the return object is an OrderedDict to retain the order
        of items in the nam file.

    Raises
    ------
    IOError:
        If namfilename does not exist in the directory.
    ValueError:
        For lines that cannot be parsed.
    """
    # initiate the ext_unit_dict ordered dictionary
    ext_unit_dict = dict()

    if verbose:
        print("Parsing the namefile --> {0:s}".format(namfilename))

    if not os.path.isfile(namfilename):
        # help diagnose the namfile and directory
        e = "Could not find {} ".format(
            namfilename
        ) + "in directory {}".format(os.path.dirname(namfilename))
        raise IOError(e)
    with open(namfilename, "r") as fp:
        lines = fp.readlines()

    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            # skip blank lines or comments
            continue
        items = line.split()
        # ensure we have at least three items
        if len(items) < 3:
            e = "line number {} has fewer than 3 items: {}".format(ln, line)
            raise ValueError(e)
        ftype, key, fpath = items[0:3]
        ftype = ftype.upper()

        # remove quotes in file path
        if '"' in fpath:
            fpath = fpath.replace('"', "")
        if "'" in fpath:
            fpath = fpath.replace("'", "")

        # need make filenames with paths system agnostic
        if "/" in fpath:
            raw = fpath.split("/")
        elif "\\" in fpath:
            raw = fpath.split("\\")
        else:
            raw = [fpath]
        fpath = os.path.join(*raw)

        fname = os.path.join(os.path.dirname(namfilename), fpath)
        if not os.path.isfile(fname) or not os.path.exists(fname):
            # change to lower and make comparison (required for linux)
            dn = os.path.dirname(fname)
            fls = os.listdir(dn)
            lownams = [f.lower() for f in fls]
            bname = os.path.basename(fname)
            if bname.lower() in lownams:
                idx = lownams.index(bname.lower())
                fname = os.path.join(dn, fls[idx])
        # open the file
        kwargs = {}
        if ftype == "DATA(BINARY)":
            openmode = "rb"
        else:
            openmode = "r"
            kwargs["errors"] = "replace"
        try:
            filehandle = open(fname, openmode, **kwargs)
        except IOError:
            if verbose:
                print("could not set filehandle to {0:s}".format(fpath))
            filehandle = None
        # be sure the second value is an integer
        try:
            key = int(key)
        except ValueError:
            raise ValueError(
                "line number {}: the unit number (second item) "
                "is not an integer: {}".format(ln, line)
            )
        # Trap for the case where unit numbers are specified as zero
        # In this case, the package must have a variable called
        # unit number attached to it.  If not, then the key is set
        # to fname
        if key == 0:
            ftype_lower = ftype.lower()
            if ftype_lower in packages:
                key = packages[ftype_lower]._reservedunit()
            else:
                key = ftype
        ext_unit_dict[key] = NamData(ftype, fname, filehandle, packages)
    return ext_unit_dict


def attribs_from_namfile_header(namefile):
    # check for reference info in the nam file header
    defaults = {
        "xll": None,
        "yll": None,
        "xul": None,
        "yul": None,
        "rotation": 0.0,
        "proj4_str": None,
    }
    if namefile is None:
        return defaults
    header = []
    with open(namefile, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            header.extend(line.strip().replace("#", "").split(";"))

    for item in header:
        if "xll" in item.lower():
            try:
                xll = float(item.split(":")[1])
                defaults["xll"] = xll
            except:
                print("   could not parse xll " + "in {}".format(namefile))
        elif "yll" in item.lower():
            try:
                yll = float(item.split(":")[1])
                defaults["yll"] = yll
            except:
                print("   could not parse yll " + "in {}".format(namefile))
        elif "xul" in item.lower():
            try:
                xul = float(item.split(":")[1])
                defaults["xul"] = xul
            except:
                print("   could not parse xul " + "in {}".format(namefile))
        elif "yul" in item.lower():
            try:
                yul = float(item.split(":")[1])
                defaults["yul"] = yul
            except:
                print("   could not parse yul " + "in {}".format(namefile))
        elif "rotation" in item.lower():
            try:
                angrot = float(item.split(":")[1])
                defaults["rotation"] = angrot
            except:
                print(
                    "   could not parse rotation " + "in {}".format(namefile)
                )
        elif "proj4_str" in item.lower():
            try:
                proj4 = ":".join(item.split(":")[1:]).strip()
                if proj4.lower() == "none":
                    proj4 = None
                defaults["proj4_str"] = proj4
            except:
                print(
                    "   could not parse proj4_str " + "in {}".format(namefile)
                )
        elif "start" in item.lower():
            try:
                start_datetime = item.split(":")[1].strip()
                defaults["start_datetime"] = start_datetime
            except:
                print("   could not parse start " + "in {}".format(namefile))
    return defaults
