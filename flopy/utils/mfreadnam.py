"""
mfreadnam module.  Contains the NamData class. Note that the user can access
the NamData class as `flopy.modflow.NamData`.

Additional information about the MODFLOW name file can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/name_file.htm>`_.

"""
import os

class NamData:
    """
    MODFLOW Namefile Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.


    Attributes
    ----------
    mxactd : int
        Maximum number of drains for a stress period.  This is calculated
        automatically by FloPy based on the information in
        layer_row_column_data.

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> lrcd = [[[2, 3, 4, 10., 100.]]]  #this drain will be applied to all
    >>>                                  #stress periods
    >>> drn = flopy.modflow.ModflowDrn(m, layer_row_column_data=lrcd)

    """
    def __init__(self, pkgtype, name, handle, packages):
        self.filehandle = handle
        self.filename = name
        self.filetype = pkgtype
#         self.packages = {"bas6": flopy.modflow.ModflowBas, "dis": flopy.modflow.ModflowDis,
#              "lpf": flopy.modflow.ModflowLpf, "wel": flopy.modflow.ModflowWel,
#              "drn": flopy.modflow.ModflowDrn, "rch": flopy.modflow.ModflowRch,
#              "riv": flopy.modflow.ModflowRiv, "pcg": flopy.modflow.ModflowPcg}

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
    '''
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

    '''
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

