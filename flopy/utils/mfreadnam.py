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
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    pkgtype : string
        String identifying the type of MODFLOW package. See the
        mfnam_packages dictionary keys in the model object for a list
        of supported packages. This dictionary is also passed in as packages.
    packages : dictionary
        Dictionary of package objects as defined in the mfnam_packages
        attribute of the model object
    name : string
        Filename of the package file identified in the name file
    handle : file handle
        File handle refering to the file identified by "name" above

    Attributes
    ----------
    mxactd : int
        Maximum number of drains for a stress period.  This is calculated
        automatically by FloPy based on the information in
        layer_row_column_data.
    filehandle : file handle
        File handle to the package file. Read in from handle.
    filename : string
        Filename of the package file identified in the name file. Read from
        the name parameter.
    filetype : string
        String identifying the type of MODFLOW package. Read from the
        pkgtype parameter.
    package : string
        Package type. Only assigned if pkgtype is found in the keys of packages

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
        if self.filetype.lower() in packages.keys():
            self.package = packages[self.filetype.lower()]

# function to test if a string is an integer
def testint(cval):
    try:
        int(cval)
        return True
    except:
        return False
    
# function to parse the name file
def parsenamefile(namfilename, packages):
    """
    Function to parse the nam file and return a dictionary with types,
    names, units and handles

    Parameters
    ----------
    namefilename : string
        Name of the MODFLOW namefile to parse.
    packages : dictionary
        Dictionary of package objects as defined in the mfnam_packages
        attribute of the model object

    Returns
    ----------
    ext_unit_dict : dictionary
        For each file listed in the namefile, a NamData instance
        is stored in the ext_unit_dict dictionary keyed by unit number
    """
    # add the .nam extension to namfilename if missing
    if namfilename[-4:].lower() != '.nam':
        namfilename += '.nam'
    
    # initiate the ext_unit_dict dictionary
    ext_unit_dict = dict()
    
    print 'Parsing the namefile --> {0:s}'.format(namfilename)
    print 'Setting filehandles:'
    indata = open(namfilename, 'r').readlines()
    for line in indata:
        tmp = line.strip().split()
        if len(tmp) == 0:
            continue
        # be sure the line is not a comment
        if '#' not in tmp[0]:
            # be sure the second value is an integer
            if testint(tmp[1]):
                fname = os.path.join(os.path.dirname(namfilename), tmp[2])
                # parse the line
                openmode = 'r'
                if tmp[0].upper() == 'DATA(BINARY)':
                    openmode = 'rb'
                try:
                    filehandle = open(fname, openmode)
                except:
                    print 'could not set filehandle for {0:s}'.format(tmp[2])
                    filehandle = None
                # populate the dictionary
                ext_unit_dict[int(tmp[1])] = NamData(tmp[0], fname, filehandle, packages)
    return ext_unit_dict

