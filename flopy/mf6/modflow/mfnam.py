from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowNam(mfpackage.MFPackage):
    """
    ModflowNam defines a nam package.

    Attributes
    ----------
    continue_ : (continue : boolean)
        continue : keyword flag to indicate that the simulation should continue
          even if one or more solutions do not converge.
    nocheck : (nocheck : boolean)
        nocheck : keyword flag to indicate that the model input check routines
          should not be called prior to each time step. Checks are performed by
          default.
    memory_print_option : (memory_print_option : string)
        memory_print_option : is a flag that controls printing of detailed
          memory manager usage to the end of the simulation list file.
          NONE means do not print detailed information.
          SUMMARY means print only the total memory for each
          simulation component. ALL means print information for each
          variable stored in the memory manager. NONE is default if
          memory\_print\_option is not specified.
    tdis6 : (tdis6 : string)
        tdis6 : is the name of the Temporal Discretization (TDIS) Input File.
    modelrecarray : [(mtype : string), (mfname : string), (mname : string)]
        mtype : is the type of model to add to simulation.
        mfname : is the file name of the model name file.
        mname : is the user-assigned name of the model. The model name cannot
          exceed 16 characters and must not have blanks within the name. The
          model name is case insensitive; any lowercase letters are converted
          and stored as upper case letters.
    exchangerecarray : [(exgtype : string), (exgfile : string), (exgmnamea :
      string), (exgmnameb : string)]
        exgtype : is the exchange type.
        exgfile : is the input file for the exchange.
        exgmnamea : is the name of the first model that is part of this
          exchange.
        exgmnameb : is the name of the second model that is part of this
          exchange.
    mxiter : (mxiter : integer)
        mxiter : is the maximum number of outer iterations for this solution
          group. The default value is 1. If there is only one solution in the
          solution group, then MXITER must be 1.
    solutionrecarray : [(slntype : string), (slnfname : string), (slnmnames :
      string)]
        slntype : is the type of solution. The Integrated Model Solution (IMS6)
          is the only supported option in this version.
        slnfname : name of file containing solution input.
        slnmnames : is the array of model names to add to this solution.

    """
    modelrecarray = ListTemplateGenerator(('nam', 'models', 
                                           'modelrecarray'))
    exchangerecarray = ListTemplateGenerator(('nam', 'exchanges', 
                                              'exchangerecarray'))
    solutionrecarray = ListTemplateGenerator(('nam', 'solutiongroup', 
                                              'solutionrecarray'))
    package_abbr = "nam"
    package_type = "nam"
    dfn = [["block options", "name continue", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name nocheck", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name memory_print_option", "type string", 
            "reader urword", "optional true"],
           ["block timing", "name tdis6", "preserve_case true", 
            "type string", "reader urword", "optional"],
           ["block models", "name modelrecarray", 
            "type recarray mtype mfname mname", "reader urword", "optional"],
           ["block models", "name mtype", "in_record true", "type string", 
            "tagged false", "reader urword"],
           ["block models", "name mfname", "in_record true", "type string", 
            "preserve_case true", "tagged false", "reader urword"],
           ["block models", "name mname", "in_record true", "type string", 
            "tagged false", "reader urword"],
           ["block exchanges", "name exchangerecarray", 
            "type recarray exgtype exgfile exgmnamea exgmnameb", 
            "reader urword", "optional"],
           ["block exchanges", "name exgtype", "in_record true", 
            "type string", "tagged false", "reader urword"],
           ["block exchanges", "name exgfile", "in_record true", 
            "type string", "preserve_case true", "tagged false", 
            "reader urword"],
           ["block exchanges", "name exgmnamea", "in_record true", 
            "type string", "tagged false", "reader urword"],
           ["block exchanges", "name exgmnameb", "in_record true", 
            "type string", "tagged false", "reader urword"],
           ["block solutiongroup", "name group_num", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "reader urword"],
           ["block solutiongroup", "name mxiter", "type integer", 
            "reader urword", "optional true"],
           ["block solutiongroup", "name solutionrecarray", 
            "type recarray slntype slnfname slnmnames", "reader urword"],
           ["block solutiongroup", "name slntype", "type string", 
            "valid_values ims6", "in_record true", "tagged false", 
            "reader urword"],
           ["block solutiongroup", "name slnfname", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword"],
           ["block solutiongroup", "name slnmnames", "type string", 
            "in_record true", "shape (nslnmod)", "tagged false", 
            "reader urword"]]

    def __init__(self, simulation, add_to_package_list=True, continue_=None,
                 nocheck=None, memory_print_option=None, tdis6=None,
                 modelrecarray=None, exchangerecarray=None, mxiter=None,
                 solutionrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowNam, self).__init__(simulation, "nam", fname, pname,
                                         add_to_package_list, parent_file)        

        # set up variables
        self.continue_ = self.build_mfdata("continue",  continue_)
        self.nocheck = self.build_mfdata("nocheck",  nocheck)
        self.memory_print_option = self.build_mfdata("memory_print_option", 
                                                     memory_print_option)
        self.tdis6 = self.build_mfdata("tdis6",  tdis6)
        self.modelrecarray = self.build_mfdata("modelrecarray",  modelrecarray)
        self.exchangerecarray = self.build_mfdata("exchangerecarray", 
                                                  exchangerecarray)
        self.mxiter = self.build_mfdata("mxiter",  mxiter)
        self.solutionrecarray = self.build_mfdata("solutionrecarray", 
                                                  solutionrecarray)
