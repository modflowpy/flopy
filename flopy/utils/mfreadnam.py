import os
import sys
import flopy

class NamData:
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
    function to parse the nam file and return a dictionary with types, names, units and handles
    '''
    # add the .nam extension to namfilename if missing
    if namfilename[-4:].lower() != '.nam':
        namfilename += '.nam'
    
    # inititate the ext_unit_dict dictionary
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
                    #filehandle = open(tmp[2], openmode)
                    filehandle = open(fname, openmode)
                except:
                    print 'could not set filehandle for {0:s}'.format(tmp[2])
                    filehandle = None
                # populate the dictionary
                #ext_unit_dict[int(tmp[1])] = NamData(tmp[0], tmp[2], filehandle, packages)
                ext_unit_dict[int(tmp[1])] = NamData(tmp[0], fname, filehandle, packages)
    return ext_unit_dict

