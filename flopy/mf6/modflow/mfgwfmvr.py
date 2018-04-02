# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfmvr(mfpackage.MFPackage):
    """
    ModflowGwfmvr defines a mvr package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of MVR
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of MVR flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    modelnames : boolean
        * modelnames (boolean) keyword to indicate that all package names will
          be preceded by the model name for the package. Model names are
          required when the Mover Package is used with a GWF-GWF Exchange. The
          MODELNAME keyword should not be used for a Mover Package that is for
          a single GWF Model.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    maxmvr : integer
        * maxmvr (integer) integer value specifying the maximum number of water
          mover entries that will specified for any stress period.
    maxpackages : integer
        * maxpackages (integer) integer value specifying the number of unique
          packages that are included in this water mover input file.
    packages : [mname, pname]
        * mname (string) name of model containing the package.
        * pname (string) is the name of a package that may be included in a
          subsequent stress period block.
    perioddata : [mname1, pname1, id1, mname2, pname2, id2, mvrtype, value]
        * mname1 (string) name of model containing the package, PNAME1.
        * pname1 (string) is the package name for the provider. The package
          PNAME1 must be designated to provide water through the MVR Package by
          specifying the keyword "MOVER" in its OPTIONS block.
        * id1 (integer) is the identifier for the provider. This is the well
          number, reach number, lake number, etc.
        * mname2 (string) name of model containing the package, PNAME2.
        * pname2 (string) is the package name for the receiver. The package
          PNAME2 must be designated to receive water from the MVR Package by
          specifying the keyword "MOVER" in its OPTIONS block.
        * id2 (integer) is the identifier for the receiver. This is the well
          number, reach number, lake number, etc.
        * mvrtype (string) is the character string signifying the method for
          determining how much water will be moved. Supported values are
          "FACTOR" "EXCESS" "THRESHOLD" and "UPTO". These four options
          determine how the receiver flow rate, :math:`Q_R`, is calculated.
          These options are based the options available in the SFR2 Package for
          diverting stream flow.
        * value (double) is the value to be used in the equation for
          calculating the amount of water to move. For the "FACTOR" option,
          VALUE is the :math:`\\alpha` factor. For the remaining options, VALUE
          is the specified flow rate, :math:`Q_S`.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    budget_filerecord = ListTemplateGenerator(('gwf6', 'mvr', 'options', 
                                               'budget_filerecord'))
    packages = ListTemplateGenerator(('gwf6', 'mvr', 'packages', 
                                      'packages'))
    perioddata = ListTemplateGenerator(('gwf6', 'mvr', 'period', 
                                        'perioddata'))
    package_abbr = "gwfmvr"
    package_type = "mvr"
    dfn_file_name = "gwf-mvr.dfn"

    dfn = [["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name modelnames", "type keyword", 
            "reader urword", "optional true"],
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
           ["block dimensions", "name maxmvr", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name maxpackages", "type integer", 
            "reader urword", "optional false"],
           ["block packages", "name packages", "type recarray mname pname", 
            "reader urword", "shape (npackages)", "optional false"],
           ["block packages", "name mname", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block packages", "name pname", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional false"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name perioddata", 
            "type recarray mname1 pname1 id1 mname2 pname2 id2 mvrtype value", 
            "shape (maxbound)", "reader urword"],
           ["block period", "name mname1", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block period", "name pname1", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name id1", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block period", "name mname2", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block period", "name pname2", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name id2", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block period", "name mvrtype", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name value", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"]]

    def __init__(self, model, loading_package=False, print_input=None,
                 print_flows=None, modelnames=None, budget_filerecord=None,
                 maxmvr=None, maxpackages=None, packages=None, perioddata=None,
                 fname=None, pname=None, parent_file=None):
        super(ModflowGwfmvr, self).__init__(model, "mvr", fname, pname,
                                            loading_package, parent_file)        

        # set up variables
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.modelnames = self.build_mfdata("modelnames",  modelnames)
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.maxmvr = self.build_mfdata("maxmvr",  maxmvr)
        self.maxpackages = self.build_mfdata("maxpackages",  maxpackages)
        self.packages = self.build_mfdata("packages",  packages)
        self.perioddata = self.build_mfdata("perioddata",  perioddata)
