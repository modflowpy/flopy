"""
mfreadnam module.  Contains the NamData class. Note that the user can access
the NamData class as `flopy.modflow.NamData`.

Additional information about the MODFLOW name file can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/name_file.html>`_.

"""
import os
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Tuple, Union


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
        "crs": None,
        "proj4_str": None,
    }
    if namefile is None:
        return defaults
    header = []
    with open(namefile) as f:
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
                defaults["crs"] = proj4
            except:
                print(f"   could not parse proj4_str in {namefile}")
        elif "crs" in item.lower():
            try:
                crs = ":".join(item.split(":")[1:]).strip()
                if crs.lower() == "none":
                    proj4 = None
                defaults["crs"] = crs
            except:
                print(f"   could not parse proj4_str in {namefile}")
        elif "start" in item.lower():
            try:
                start_datetime = item.split(":")[1].strip()
                defaults["start_datetime"] = start_datetime
            except:
                print(f"   could not parse start in {namefile}")
    return defaults


def get_entries_from_namefile(
    path: Union[str, PathLike],
    ftype: str = None,
    unit: int = None,
    extension: str = None,
) -> List[Tuple]:
    """Get entries from an MF6 namefile. Can select using FTYPE, UNIT, or file extension.
    This function only supports MF6 namefiles.

    Parameters
    ----------
    path : str or PathLike
        path to a MODFLOW-based model name file
    ftype : str
        package type
    unit : int
        file unit number
    extension : str
        file extension

    Returns
    -------
    entries : list of tuples
        list of tuples containing FTYPE, UNIT, FNAME, STATUS for each
        namefile entry that meets a user-specified value.
    """
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip() == "":
                continue
            if line[0] == "#":
                continue
            ll = line.strip().split()
            if len(ll) < 3:
                continue
            status = "UNKNOWN"
            if len(ll) > 3:
                status = ll[3].upper()
            if ftype is not None:
                if ftype.upper() in ll[0].upper():
                    filename = os.path.join(os.path.split(path)[0], ll[2])
                    entries.append((filename, ll[0], ll[1], status))
            elif unit is not None:
                if int(unit) == int(ll[1]):
                    filename = os.path.join(os.path.split(path)[0], ll[2])
                    entries.append((filename, ll[0], ll[1], status))
            elif extension is not None:
                filename = os.path.join(os.path.split(path)[0], ll[2])
                ext = os.path.splitext(filename)[1]
                if len(ext) > 0:
                    if ext[0] == ".":
                        ext = ext[1:]
                    if extension.lower() == ext.lower():
                        entries.append((filename, ll[0], ll[1], status))

    return entries


def get_input_files(namefile):
    """Return a list of all the input files in this model.
    Parameters
    ----------
    namefile : str
        path to a MODFLOW-based model name file
    Returns
    -------
    filelist : list
        list of MODFLOW-based model input files
    """
    ignore_ext = (
        ".hds",
        ".hed",
        ".bud",
        ".cbb",
        ".cbc",
        ".ddn",
        ".ucn",
        ".glo",
        ".lst",
        ".list",
        ".gwv",
        ".mv",
        ".out",
    )

    srcdir = os.path.dirname(namefile)
    filelist = []
    fname = os.path.join(srcdir, namefile)
    with open(fname) as f:
        lines = f.readlines()

    for line in lines:
        ll = line.strip().split()
        if len(ll) < 2:
            continue
        if line.strip()[0] in ["#", "!"]:
            continue
        ext = os.path.splitext(ll[2])[1]
        if ext.lower() not in ignore_ext:
            if len(ll) > 3:
                if "replace" in ll[3].lower():
                    continue
            filelist.append(ll[2])

    # Now go through every file and look for other files to copy,
    # such as 'OPEN/CLOSE'.  If found, then add that file to the
    # list of files to copy.
    otherfiles = []
    for fname in filelist:
        fname = os.path.join(srcdir, fname)
        try:
            f = open(fname, "r")
            for line in f:
                # Skip invalid lines
                ll = line.strip().split()
                if len(ll) < 2:
                    continue
                if line.strip()[0] in ["#", "!"]:
                    continue

                if "OPEN/CLOSE" in line.upper():
                    for i, s in enumerate(ll):
                        if "OPEN/CLOSE" in s.upper():
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            otherfiles.append(stmp)
                            break
        except:
            print(fname + " does not exist")

    filelist = filelist + otherfiles

    return filelist


def get_sim_name(namefiles, rootpth=None):
    """Get simulation name.
    Parameters
    ----------
    namefiles : str or list of strings
        path(s) to MODFLOW-based model name files
    rootpth : str
        optional root directory path (default is None)
    Returns
    -------
    simname : list
        list of namefiles without the file extension
    """
    if isinstance(namefiles, str):
        namefiles = [namefiles]
    sim_name = []
    for namefile in namefiles:
        t = namefile.split(os.sep)
        if rootpth is None:
            idx = -1
        else:
            idx = t.index(os.path.split(rootpth)[1])

        # build dst with everything after the rootpth and before
        # the namefile file name.
        dst = ""
        if idx < len(t):
            for d in t[idx + 1 : -1]:
                dst += f"{d}_"

        # add namefile basename without extension
        dst += t[-1].replace(".nam", "")
        sim_name.append(dst)

    return sim_name


def get_mf6_nper(tdisfile):
    """Return the number of stress periods in the MODFLOW 6 model.
    Parameters
    ----------
    tdisfile : str
        path to the TDIS file
    Returns
    -------
    nper : int
        number of stress periods in the simulation
    """
    with open(tdisfile) as f:
        lines = f.readlines()
    line = [line for line in lines if "NPER" in line.upper()][0]
    nper = line.strip().split()[1]
    return nper


def get_mf6_mshape(disfile):
    """Return the shape of the MODFLOW 6 model.
    Parameters
    ----------
    disfile : str
        path to a MODFLOW 6 discretization file
    Returns
    -------
    mshape : tuple
        tuple with the shape of the MODFLOW 6 model.
    """
    with open(disfile) as f:
        lines = f.readlines()

    d = {}
    for line in lines:
        # Skip over blank and commented lines
        ll = line.strip().split()
        if len(ll) < 2:
            continue
        if line.strip()[0] in ["#", "!"]:
            continue

        for key in ["NODES", "NCPL", "NLAY", "NROW", "NCOL"]:
            if ll[0].upper() in key:
                d[key] = int(ll[1])

    if "NODES" in d:
        mshape = (d["NODES"],)
    elif "NCPL" in d:
        mshape = (d["NLAY"], d["NCPL"])
    elif "NLAY" in d:
        mshape = (d["NLAY"], d["NROW"], d["NCOL"])
    else:
        print(d)
        raise Exception("Could not determine model shape")
    return mshape


def get_mf6_files(mfnamefile):
    """Return a list of all the MODFLOW 6 input and output files in this model.
    Parameters
    ----------
    mfnamefile : str
        path to the MODFLOW 6 simulation name file
    Returns
    -------
    filelist : list
        list of MODFLOW 6 input files in a simulation
    outplist : list
        list of MODFLOW 6 output files in a simulation
    """

    srcdir = os.path.dirname(mfnamefile)
    filelist = []
    outplist = []

    filekeys = ["TDIS6", "GWF6", "GWT", "GWF6-GWF6", "GWF-GWT", "IMS6"]
    namefilekeys = ["GWF6", "GWT"]
    namefiles = []

    with open(mfnamefile) as f:
        # Read line and skip comments
        lines = f.readlines()

    for line in lines:
        # Skip over blank and commented lines
        ll = line.strip().split()
        if len(ll) < 2:
            continue
        if line.strip()[0] in ["#", "!"]:
            continue

        for key in filekeys:
            if key in ll[0].upper():
                fname = ll[1]
                filelist.append(fname)

        for key in namefilekeys:
            if key in ll[0].upper():
                fname = ll[1]
                namefiles.append(fname)

    # Go through name files and get files
    for namefile in namefiles:
        fname = os.path.join(srcdir, namefile)
        with open(fname) as f:
            lines = f.readlines()
        insideblock = False

        for line in lines:
            ll = line.upper().strip().split()
            if len(ll) < 2:
                continue
            if ll[0] in "BEGIN" and ll[1] in "PACKAGES":
                insideblock = True
                continue
            if ll[0] in "END" and ll[1] in "PACKAGES":
                insideblock = False

            if insideblock:
                ll = line.strip().split()
                if len(ll) < 2:
                    continue
                if line.strip()[0] in ["#", "!"]:
                    continue
                filelist.append(ll[1])

    # Recursively go through every file and look for other files to copy,
    # such as 'OPEN/CLOSE' and 'TIMESERIESFILE'.  If found, then
    # add that file to the list of files to copy.
    flist = filelist
    # olist = outplist
    while True:
        olist = []
        flist, olist = _get_mf6_external_files(srcdir, olist, flist)
        # add to filelist
        if len(flist) > 0:
            filelist = filelist + flist
        # add to outplist
        if len(olist) > 0:
            outplist = outplist + olist
        # terminate loop if no additional files
        # if len(flist) < 1 and len(olist) < 1:
        if len(flist) < 1:
            break

    return filelist, outplist


def _get_mf6_external_files(srcdir, outplist, files):
    """Get list of external files in a MODFLOW 6 simulation.
    Parameters
    ----------
    srcdir : str
        path to a directory containing a MODFLOW 6 simulation
    outplist : list
        list of output files in a MODFLOW 6 simulation
    files : list
        list of MODFLOW 6 name files
    Returns
    -------
    """
    extfiles = []

    for fname in files:
        fname = os.path.join(srcdir, fname)
        try:
            f = open(fname, "r")
            for line in f:
                # Skip invalid lines
                ll = line.strip().split()
                if len(ll) < 2:
                    continue
                if line.strip()[0] in ["#", "!"]:
                    continue

                if "OPEN/CLOSE" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "OPEN/CLOSE":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "TS6" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "FILEIN":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "TAS6" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "FILEIN":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "OBS6" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "FILEIN":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "EXTERNAL" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "EXTERNAL":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "FILE" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "FILEIN":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            extfiles.append(stmp)
                            break

                if "FILE" in line.upper():
                    for i, s in enumerate(ll):
                        if s.upper() == "FILEOUT":
                            stmp = ll[i + 1]
                            stmp = stmp.replace('"', "")
                            stmp = stmp.replace("'", "")
                            outplist.append(stmp)
                            break

        except:
            print("could not get a list of external mf6 files")

    return extfiles, outplist


def get_mf6_ftypes(namefile, ftypekeys):
    """Return a list of FTYPES that are in the name file and in ftypekeys.
    Parameters
    ----------
    namefile : str
        path to a MODFLOW 6 name file
    ftypekeys : list
        list of desired FTYPEs
    Returns
    -------
    ftypes : list
        list of FTYPES that match ftypekeys in namefile
    """
    with open(namefile) as f:
        lines = f.readlines()

    ftypes = []
    for line in lines:
        # Skip over blank and commented lines
        ll = line.strip().split()
        if len(ll) < 2:
            continue
        if line.strip()[0] in ["#", "!"]:
            continue

        for key in ftypekeys:
            if ll[0].upper() in key:
                ftypes.append(ll[0])

    return ftypes


def get_mf6_blockdata(f, blockstr):
    """Return list with all non comments between start and end of block
    specified by blockstr.
    Parameters
    ----------
    f : file object
        open file object
    blockstr : str
        name of block to search
    Returns
    -------
    data : list
        list of data in specified block
    """
    data = []

    # find beginning of block
    for line in f:
        if line[0] != "#":
            t = line.split()
            if t[0].lower() == "begin" and t[1].lower() == blockstr.lower():
                break
    for line in f:
        if line[0] != "#":
            t = line.split()
            if t[0].lower() == "end" and t[1].lower() == blockstr.lower():
                break
            else:
                data.append(line.rstrip())
    return data
