"""
mfreadnam module.  Contains the NamData class. Note that the user can access
the NamData class as `flopy.modflow.NamData`.

Additional information about the MODFLOW name file can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/name_file.html>`_.

"""
from pathlib import Path, PurePosixPath, PureWindowsPath


class NamData:
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
        return f"filename:{self.filename}, filetype:{self.filetype}"


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
    print(f'Name file does not contain file of type "{filetype}"')
    return None


def parsenamefile(namfilename, packages, verbose=True):
    """
    Returns dict from the nam file with NamData keyed by unit number

    Parameters
    ----------
    namefilename : str or Path
        Name of the MODFLOW namefile to parse.
    packages : dict
        Dictionary of package objects as defined in the `mfnam_packages`
        attribute of :class:`flopy.modflow.mf.Modflow`.
    verbose : bool
        Print messages to screen.  Default is True.

    Returns
    -------
    dict
        For each file listed in the name file, a
        :class:`flopy.utils.mfreadnam.NamData` instance
        is stored in the returned dict keyed by unit number.

    Raises
    ------
    FileNotFoundError
        If namfilename does not exist in the directory.
    ValueError
        For lines that cannot be parsed.
    """
    # initiate the ext_unit_dict dictionary
    ext_unit_dict = {}

    if verbose:
        print(f"Parsing the namefile --> {namfilename}")

    namfilename = Path(namfilename)
    if not namfilename.is_file():
        # help diagnose the namfile and directory
        raise FileNotFoundError(
            f"Could not find {namfilename} in directory {namfilename.parent}"
        )
    lines = namfilename.read_text().rstrip().split("\n")

    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            # skip blank lines or comments
            continue
        items = line.split()
        # ensure we have at least three items
        if len(items) < 3:
            e = f"line number {ln} has fewer than 3 items: {line}"
            raise ValueError(e)
        ftype, key, fpath = items[0:3]
        ftype = ftype.upper()

        # remove quotes in file path
        if '"' in fpath:
            fpath = fpath.replace('"', "")
        if "'" in fpath:
            fpath = fpath.replace("'", "")

        # need make filenames with paths system agnostic
        if "\\" in fpath:
            fpath = PureWindowsPath(fpath)
        elif "/" in fpath:
            fpath = PurePosixPath(fpath)
        fpath = Path(fpath)

        fname = namfilename.parent / fpath
        if not fname.is_file():
            if fname.parent.is_dir():
                # change to lower and make comparison (required for linux)
                lname = fname.name.lower()
                for pth in fname.parent.iterdir():
                    if pth.is_file() and pth.name.lower() == lname:
                        if verbose:
                            print(f"correcting {fname.name} to {pth.name}")
                        fname = pth
                        break

        # open the file
        try:
            if ftype == "DATA(BINARY)":
                filehandle = fname.open("rb")
            else:
                filehandle = fname.open("r", errors="replace")
        except OSError:
            if verbose:
                print(f"could not set filehandle to {fpath}")
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
        ext_unit_dict[key] = NamData(ftype, str(fname), filehandle, packages)
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
                print(f"   could not parse xll in {namefile}")
        elif "yll" in item.lower():
            try:
                yll = float(item.split(":")[1])
                defaults["yll"] = yll
            except:
                print(f"   could not parse yll in {namefile}")
        elif "xul" in item.lower():
            try:
                xul = float(item.split(":")[1])
                defaults["xul"] = xul
            except:
                print(f"   could not parse xul in {namefile}")
        elif "yul" in item.lower():
            try:
                yul = float(item.split(":")[1])
                defaults["yul"] = yul
            except:
                print(f"   could not parse yul in {namefile}")
        elif "rotation" in item.lower():
            try:
                angrot = float(item.split(":")[1])
                defaults["rotation"] = angrot
            except:
                print(f"   could not parse rotation in {namefile}")
        elif "proj4_str" in item.lower():
            try:
                proj4 = ":".join(item.split(":")[1:]).strip()
                if proj4.lower() == "none":
                    proj4 = None
                defaults["proj4_str"] = proj4
            except:
                print(f"   could not parse proj4_str in {namefile}")
        elif "start" in item.lower():
            try:
                start_datetime = item.split(":")[1].strip()
                defaults["start_datetime"] = start_datetime
            except:
                print(f"   could not parse start in {namefile}")
    return defaults
