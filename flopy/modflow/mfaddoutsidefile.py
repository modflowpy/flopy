from numpy import atleast_2d
from ..pakbase import Package


class mfaddoutsidefile(Package):
    '''Add a file for which you have a MODFLOW input file'''

    def __init__(self, model, name, extension, unitnumber):
        Package.__init__(self, model, extension, name, unitnumber,
                         allowDuplicates=True)  # Call ancestor's init to set self.parent, extension, name and unit number
        self.parent.add_package(self)

    def __repr__(self):
        return 'Outside Package class'

    def write_file(self):
        pass
