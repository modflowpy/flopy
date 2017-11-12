from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfoc(mfpackage.MFPackage):
    package_abbr = "gwfoc"
    budget_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'oc', 'options', 'budget_filerecord'))
    head_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'oc', 'options', 'head_filerecord'))
    headprintrecord = mfdatautil.ListTemplateGenerator(('gwf6', 'oc', 'options', 'headprintrecord'))
    saverecord = mfdatautil.ListTemplateGenerator(('gwf6', 'oc', 'period', 'saverecord'))
    printrecord = mfdatautil.ListTemplateGenerator(('gwf6', 'oc', 'period', 'printrecord'))
    """
    ModflowGwfoc defines a oc package within a gwf6 model.

    Attributes
    ----------
    budget_filerecord : [(budget : keyword), (fileout : keyword), (budgetfile : string)]
        budget : keyword to specify that record corresponds to the budget.
        fileout : keyword to specify that an output filename is expected next.
        budgetfile : name of the output file to write budget information.
    head_filerecord : [(head : keyword), (fileout : keyword), (headfile : string)]
        fileout : keyword to specify that an output filename is expected next.
        head : keyword to specify that record corresponds to head.
        headfile : name of the output file to write head information.
    headprintrecord : [(head : keyword), (print_format : keyword), (columns : integer), (width : integer), (digits : integer), (format : string)]
        head : keyword to specify that record corresponds to head.
        print_format : keyword to specify format for printing to the listing file.
        formatrecord : 
    saverecord : [(save : keyword), (rtype : string), (ocsetting : keystring)]
        save : keyword to indicate that information will be saved this stress period.
        rtype : type of information to save or print. Can be BUDGET or HEAD.
        ocsetting : specifies the steps for which the data will be saved.
    printrecord : [(print : keyword), (rtype : string), (ocsetting : keystring)]
        print : keyword to indicate that information will be printed this stress period.
        rtype : type of information to save or print. Can be BUDGET or HEAD.
        ocsetting : specifies the steps for which the data will be saved.

    """
    def __init__(self, model, add_to_package_list=True, budget_filerecord=None, head_filerecord=None,
                 headprintrecord=None, saverecord=None, printrecord=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfoc, self).__init__(model, "oc", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.budget_filerecord = self.build_mfdata("budget_filerecord", budget_filerecord)

        self.head_filerecord = self.build_mfdata("head_filerecord", head_filerecord)

        self.headprintrecord = self.build_mfdata("headprintrecord", headprintrecord)

        self.saverecord = self.build_mfdata("saverecord", saverecord)

        self.printrecord = self.build_mfdata("printrecord", printrecord)


