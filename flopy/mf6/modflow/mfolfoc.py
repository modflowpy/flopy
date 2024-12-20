# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowOlfoc(mfpackage.MFPackage):
    """
    ModflowOlfoc defines a oc package within a olf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
    qoutflow_filerecord : [qoutflowfile]
        * qoutflowfile (string) name of the output file to write conc
          information.
    stage_filerecord : [stagefile]
        * stagefile (string) name of the output file to write stage
          information.
    qoutflowprintrecord : [columns, width, digits, format]
        * columns (integer) number of columns for writing data.
        * width (integer) width for writing each number.
        * digits (integer) number of digits to use for writing a number.
        * format (string) write format can be EXPONENTIAL, FIXED, GENERAL, or
          SCIENTIFIC.
    saverecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can be BUDGET.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    printrecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can be BUDGET.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    budget_filerecord = ListTemplateGenerator(('olf6', 'oc', 'options',
                                               'budget_filerecord'))
    budgetcsv_filerecord = ListTemplateGenerator(('olf6', 'oc',
                                                  'options',
                                                  'budgetcsv_filerecord'))
    qoutflow_filerecord = ListTemplateGenerator(('olf6', 'oc', 'options',
                                                 'qoutflow_filerecord'))
    stage_filerecord = ListTemplateGenerator(('olf6', 'oc', 'options',
                                              'stage_filerecord'))
    qoutflowprintrecord = ListTemplateGenerator(('olf6', 'oc', 'options',
                                                 'qoutflowprintrecord'))
    saverecord = ListTemplateGenerator(('olf6', 'oc', 'period',
                                        'saverecord'))
    printrecord = ListTemplateGenerator(('olf6', 'oc', 'period',
                                         'printrecord'))
    package_abbr = "olfoc"
    _package_type = "oc"
    dfn_file_name = "olf-oc.dfn"

    dfn = [
           ["header", ],
           ["block options", "name budget_filerecord",
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
           ["block options", "name budgetcsv_filerecord",
            "type record budgetcsv fileout budgetcsvfile", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name budgetcsv", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name budgetcsvfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name qoutflow_filerecord",
            "type record qoutflow fileout qoutflowfile", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name qoutflow", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name qoutflowfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name stage_filerecord",
            "type record stage fileout stagefile", "shape", "reader urword",
            "tagged true", "optional true"],
           ["block options", "name stage", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name stagefile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name qoutflowprintrecord",
            "type record qoutflow print_format formatrecord", "shape",
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

    def __init__(self, model, loading_package=False, budget_filerecord=None,
                 budgetcsv_filerecord=None, qoutflow_filerecord=None,
                 stage_filerecord=None, qoutflowprintrecord=None,
                 saverecord=None, printrecord=None, filename=None, pname=None,
                 **kwargs):
        super().__init__(model, "oc", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.budget_filerecord = self.build_mfdata("budget_filerecord",
                                                   budget_filerecord)
        self.budgetcsv_filerecord = self.build_mfdata("budgetcsv_filerecord",
                                                      budgetcsv_filerecord)
        self.qoutflow_filerecord = self.build_mfdata("qoutflow_filerecord",
                                                     qoutflow_filerecord)
        self.stage_filerecord = self.build_mfdata("stage_filerecord",
                                                  stage_filerecord)
        self.qoutflowprintrecord = self.build_mfdata("qoutflowprintrecord",
                                                     qoutflowprintrecord)
        self.saverecord = self.build_mfdata("saverecord", saverecord)
        self.printrecord = self.build_mfdata("printrecord", printrecord)
        self._init_complete = True
