from ..pakbase import Package


class mfaddoutsidefile(Package):
    """
    Add a file for which you have a MODFLOW input file
    """

    def __init__(self, model, name, extension, unitnumber):
        # call base package constructor
        super().__init__(
            model, extension, name, unitnumber, allowDuplicates=True
        )
        self.parent.add_package(self)

    def __repr__(self):
        return "Outside Package class"

    def write_file(self):
        pass
