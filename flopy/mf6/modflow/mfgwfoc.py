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
    package_type = "oc"
    dfn = [["block options", "name budget_filerecord", 
            "type record budget fileout budgetfile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name budget", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name fileout", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name budgetfile", "type string", 
            "preserve_case true", "shape", "in_record true", "reader urword", 
            "tagged false", "optional false"],
           ["block options", "name head_filerecord", 
            "type record head fileout headfile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name head", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name headfile", "type string", 
            "preserve_case true", "shape", "in_record true", "reader urword", 
            "tagged false", "optional false"],
           ["block options", "name headprintrecord", 
            "type record head print_format formatrecord", "shape", 
            "reader urword", "optional true"],
           ["block options", "name print_format", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name formatrecord", 
            "type record columns width digits format", "shape", 
            "in_record true", "reader urword", "tagged", "optional false"],
           ["block options", "name columns", "type integer", "shape", 
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name width", "type integer", "shape", 
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name digits", "type integer", "shape", 
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name format", "type string", "shape", 
            "in_record true", "reader urword", "tagged false", 
            "optional false"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name saverecord", 
            "type record save rtype ocsetting", "shape", "reader urword", 
            "tagged false", "optional true"],
           ["block period", "name save", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block period", "name printrecord", 
            "type record print rtype ocsetting", "shape", "reader urword", 
            "tagged false", "optional true"],
           ["block period", "name print", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block period", "name rtype", "type string", "shape", 
            "in_record true", "reader urword", "tagged false", 
            "optional false"],
           ["block period", "name ocsetting", 
            "type keystring all first last frequency steps", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name all", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name first", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name last", "type keyword", "shape", 
            "in_record true", "reader urword"],
           ["block period", "name frequency", "type integer", "shape", 
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name steps", "type integer", "shape (<nstp)", 
            "tagged true", "in_record true", "reader urword"]]

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
