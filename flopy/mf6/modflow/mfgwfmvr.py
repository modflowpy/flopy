# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfmvr(mfpackage.MFPackage):
    """
    ModflowGwfmvr defines a mvr package within a gwf6 model.

    Attributes
    ----------
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of MVR information will
          be written to the listing file immediately after it is read.
    print_flows : (print_flows : boolean)
        print_flows : keyword to indicate that the list of MVR flow rates will
          be printed to the listing file for every stress period time step in
          which ``BUDGET PRINT'' is specified in Output Control. If there is no
          Output Control option and PRINT\_FLOWS is specified, then flow rates
          are printed for the last time step of each stress period.
    modelnames : (modelnames : boolean)
        modelnames : keyword to indicate that all package names will be
          preceded by the model name for the package. Model names are required
          when the Mover Package is used with a GWF-GWF Exchange. The MODELNAME
          keyword should not be used for a Mover Package that is for a single
          GWF Model.
    budget_filerecord : [(budgetfile : string)]
        budgetfile : name of the output file to write budget information.
    maxmvr : (maxmvr : integer)
        maxmvr : integer value specifying the maximum number of water mover
          entries that will specified for any stress period.
    maxpackages : (maxpackages : integer)
        maxpackages : integer value specifying the number of unique packages
          that are included in this water mover input file.
    packagesrecarray : [(mname : string), (pname : string)]
        mname : name of model containing the package.
        pname : is the name of a package that may be included in a subsequent
          stress period block.
    periodrecarray : [(mname1 : string), (pname1 : string), (id1 : integer),
      (mname2 : string), (pname2 : string), (id2 : integer), (mvrtype :
      string), (value : double)]
        mname1 : name of model containing the package, pname1.
        pname1 : is the package name for the provider. The package
          pname1 must be designated to provide water through the MVR
          Package by specifying the keyword ``MOVER'' in its OPTIONS block.
        id1 : is the identifier for the provider. This is the well number,
          reach number, lake number, etc.
        mname2 : name of model containing the package, pname2.
        pname2 : is the package name for the receiver. The package
          pname2 must be designated to receive water from the MVR
          Package by specifying the keyword ``MOVER'' in its OPTIONS block.
        id2 : is the identifier for the receiver. This is the well number,
          reach number, lake number, etc.
        mvrtype : is the character string signifying the method for determining
          how much water will be moved. Supported values are ``FACTOR''
          ``EXCESS'' ``THRESHOLD'' and ``UPTO''. These four options determine
          how the receiver flow rate, $Q_R$, is calculated. These options are
          based the options available in the SFR2 Package for diverting stream
          flow.
        value : is the value to be used in the equation for calculating the
          amount of water to move. For the ``FACTOR'' option, value is
          the $\alpha$ factor. For the remaining options, value is the
          specified flow rate, $Q_S$.

    """
    budget_filerecord = ListTemplateGenerator(('gwf6', 'mvr', 'options', 
                                               'budget_filerecord'))
    packagesrecarray = ListTemplateGenerator(('gwf6', 'mvr', 'packages', 
                                              'packagesrecarray'))
    periodrecarray = ListTemplateGenerator(('gwf6', 'mvr', 'period', 
                                            'periodrecarray'))
    package_abbr = "gwfmvr"
    package_type = "mvr"
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
           ["block packages", "name packagesrecarray", 
            "type recarray mname pname", "reader urword", "shape (npackages)", 
            "optional false"],
           ["block packages", "name mname", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block packages", "name pname", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional false"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name periodrecarray", 
            "type recarray mname1 pname1 id1 mname2 pname2 id2 mvrtype value", 
            "shape (maxbound)", "reader urword"],
           ["block period", "name mname1", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block period", "name pname1", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name id1", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name mname2", "type string", "reader urword", 
            "shape", "tagged false", "in_record true", "optional true"],
           ["block period", "name pname2", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name id2", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name mvrtype", "type string", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name value", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"]]

    def __init__(self, model, add_to_package_list=True, print_input=None,
                 print_flows=None, modelnames=None, budget_filerecord=None,
                 maxmvr=None, maxpackages=None, packagesrecarray=None,
                 periodrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfmvr, self).__init__(model, "mvr", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.modelnames = self.build_mfdata("modelnames",  modelnames)
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.maxmvr = self.build_mfdata("maxmvr",  maxmvr)
        self.maxpackages = self.build_mfdata("maxpackages",  maxpackages)
        self.packagesrecarray = self.build_mfdata("packagesrecarray", 
                                                  packagesrecarray)
        self.periodrecarray = self.build_mfdata("periodrecarray", 
                                                periodrecarray)
