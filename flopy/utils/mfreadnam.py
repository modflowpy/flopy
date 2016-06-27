"""
mfreadnam module.  Contains the NamData class. Note that the user can access
the NamData class as `flopy.modflow.NamData`.

Additional information about the MODFLOW name file can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/name_file.htm>`_.

"""
import os

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
        return "filename:{0}, filetype:{1}".format(self.filename,self.filetype)

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

# function to test if a string is an integer
def testint(cval):
    try:
        int(cval)
        return True
    except:
        return False
    
# function to parse the name file
def parsenamefile(namfilename, packages, verbose=True):
    """
    Function to parse the nam file and return a dictionary with types,
    names, units and handles

    Parameters
    ----------
    namefilename : string
        Name of the MODFLOW namefile to parse.
    packages : dictionary
        Dictionary of package objects as defined in the `mfnam_packages`
        attribute of :class:`flopy.modflow.mf.Modflow`.
    verbose : logical
        Print messages to screen.  Default is True.

    Returns
    ----------
    ext_unit_dict : dictionary
        For each file listed in the name file, a
        :class:`flopy.utils.mfreadnam.NamData` instance
        is stored in the ext_unit_dict dictionary keyed by unit number
    """
    # add the .nam extension to namfilename if missing
    if not os.path.isfile(namfilename):
        if namfilename[-4:].lower() != '.nam':
            namfilename += '.nam'
    
    # initiate the ext_unit_dict dictionary
    ext_unit_dict = dict()

    if verbose:
        print('Parsing the namefile --> {0:s}'.format(namfilename))
        print('Setting filehandles:')

    if not os.path.isfile(namfilename):
        dn = os.path.dirname(namfilename)
        s = 'Could not find {} in path {} with files \n {}'.format(namfilename, dn, os.listdir(dn))
        raise Exception(s)
    indata = open(namfilename, 'r').readlines()
    for line in indata:
        tmp = line.strip().split()
        if len(tmp) == 0:
            continue
        # be sure the line is not a comment
        if '#' not in tmp[0]:
            # be sure the second value is an integer
            if testint(tmp[1]):

                # need make filenames with paths system agnostic
                if '/' in tmp[2]:
                    raw = tmp[2].split('/')
                elif '\\' in tmp[2]:
                    raw = tmp[2].split('\\')
                else:
                    raw = [tmp[2]]
                tmp[2] = os.path.join(*raw)

                fname = os.path.join(os.path.dirname(namfilename), tmp[2])
                if not os.path.isfile(fname) or not os.path.exists(fname):
                    # change to lower and make comparison (required for linux)
                    dn = os.path.dirname(fname)
                    fls = os.listdir(dn)
                    lownams = [f.lower() for f in fls]
                    bname = os.path.basename(fname)
                    if bname.lower() in lownams:
                        idx = lownams.index(bname.lower())
                        fname = os.path.join(dn, fls[idx])
                # parse the line
                openmode = 'r'
                if tmp[0].upper() == 'DATA(BINARY)':
                    openmode = 'rb'
                try:
                    filehandle = open(fname, openmode)
                except:
                    if verbose:
                        print('could not set filehandle for {0:s}'\
                            .format(tmp[2]))
                    filehandle = None
                # populate the dictionary
                key = int(tmp[1])
                #
                # Trap for the case where unit numbers are specified as zero
                # In this case, the package must have a variable called
                # unit number attached to it.  If not, then the key is set
                # to fname
                if key == 0:
                    ftype = tmp[0].lower()
                    if ftype in packages:
                        key = packages[ftype].unitnumber
                    else:
                        key = tmp[0]
                ext_unit_dict[key] = NamData(tmp[0].upper(), fname, filehandle,
                                             packages)
    return ext_unit_dict

