from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfoc(mfpackage.MFPackage):
    """
    ModflowGwfoc defines a oc package within a gwf6 model.

    Attributes
    ----------
    budget_filerecord : [(budgetfile : string)]
        budgetfile : name of the output file to write budget information.
    head_filerecord : [(headfile : string)]
        headfile : name of the output file to write head information.
    headprintrecord : [(columns : integer), (width : integer), (digits : integer),
      (format : string)]
        columns : number of columns for writing data.
        width : width for writing each number.
        digits : number of digits to use for writing a number.
        format : write format can be EXPONENTIAL, FIXED, GENERAL, or
          SCIENTIFIC.
    saverecord : [(rtype : string), (ocsetting : keystring)]
        rtype : type of information to save or print. Can be BUDGET or HEAD.
        ocsetting : specifies the steps for which the data will be saved.
    printrecord : [(rtype : string), (ocsetting : keystring)]
        rtype : type of information to save or print. Can be BUDGET or HEAD.
        ocsetting : specifies the steps for which the data will be saved.

    """
    budget_filerecord = ListTemplateGenerator(('gwf6', 'oc', 'options', 
                                               'budget_filerecord'))
    head_filerecord = ListTemplateGenerator(('gwf6', 'oc', 'options', 
                                             'head_filerecord'))
    headprintrecord = ListTemplateGenerator(('gwf6', 'oc', 'options', 
                                             'headprintrecord'))
    saverecord = ListTemplateGenerator(('gwf6', 'oc', 'period', 
                                        'saverecord'))
    printrecord = ListTemplateGenerator(('gwf6', 'oc', 'period', 
                                         'printrecord'))
    package_abbr = "gwfoc"

    def __init__(self, model, add_to_package_list=True, budget_filerecord=None,
                 head_filerecord=None, headprintrecord=None, saverecord=None,
                 printrecord=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfoc, self).__init__(model, "oc", fname, pname,
                                           add_to_package_list, parent_file)        

        # set up variables
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.head_filerecord = self.build_mfdata("head_filerecord", 
                                                 head_filerecord)
        self.headprintrecord = self.build_mfdata("headprintrecord", 
                                                 headprintrecord)
        self.saverecord = self.build_mfdata("saverecord",  saverecord)
        self.printrecord = self.build_mfdata("printrecord",  printrecord)
