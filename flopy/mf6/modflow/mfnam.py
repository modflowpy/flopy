from .. import mfpackage
from ..data import mfdatautil


class ModflowNam(mfpackage.MFPackage):
    """
    ModflowNam defines a nam package.

    Attributes
    ----------
    continue_ : (continue : keyword)
        keyword flag to indicate that the simulation should continue even if one or more solutions do not converge.
    nocheck : (nocheck : keyword)
        keyword flag to indicate that the model input check routines should not be called prior to each time step. Checks are performed by default.
    memory_print_option : (memory_print_option : string)
        is a flag that controls printing of detailed memory manager usage to the end of the simulation list file. NONE means do not print detailed information. SUMMARY means print only the total memory for each simulation component. ALL means print information for each variable stored in the memory manager. NONE is default if memory\_print\_option is not specified.
    tdis6 : (tdis6 : string)
        is the name of the Temporal Discretization (TDIS) Input File.
    modelrecarray : [(mtype : string), (mfname : string), (mname : string)]
        is the list of model types, model name files, and model names.
        mtype : is the type of model to add to simulation.
        mfname : is the file name of the model name file.
        mname : is the user-assigned name of the model. The model name cannot exceed 16 characters and must not have blanks within the name. The model name is case insensitive; any lowercase letters are converted and stored as upper case letters.
    exchangerecarray : [(exgtype : string), (exgfile : string), (exgmnamea : string), (exgmnameb : string)]
        is the list of exchange types, exchange files, and model names.
        exgtype : is the exchange type.
        exgfile : is the input file for the exchange.
        exgmnamea : is the name of the first model that is part of this exchange.
        exgmnameb : is the name of the second model that is part of this exchange.
    mxiter : (mxiter : integer)
        is the maximum number of outer iterations for this solution group. The default value is 1. If there is only one solution in the solution group, then MXITER must be 1.
    solutionrecarray : [(slntype : string), (slnfname : string), (slnmnames : string)]
        is the list of solution types and models in the solution.
        slntype : is the type of solution. The Integrated Model Solution (IMS6) is the only supported option in this version.
        slnfname : name of file containing solution input.
        slnmnames : is the array of model names to add to this solution.

    """
    modelrecarray = mfdatautil.ListTemplateGenerator(('nam', 'models', 'modelrecarray'))
    exchangerecarray = mfdatautil.ListTemplateGenerator(('nam', 'exchanges', 'exchangerecarray'))
    solutionrecarray = mfdatautil.ListTemplateGenerator(('nam', 'solutiongroup', 'solutionrecarray'))
    package_abbr = "nam"

    def __init__(self, simulation, add_to_package_list=True, continue_=None, nocheck=None, memory_print_option=None,
                 tdis6=None, modelrecarray=None, exchangerecarray=None, mxiter=None,
                 solutionrecarray=None, fname=None, pname=None, parent_file=None):
        super(ModflowNam, self).__init__(simulation, "nam", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.continue_ = self.build_mfdata("continue", continue_)

        self.nocheck = self.build_mfdata("nocheck", nocheck)

        self.memory_print_option = self.build_mfdata("memory_print_option", memory_print_option)

        self.tdis6 = self.build_mfdata("tdis6", tdis6)

        self.modelrecarray = self.build_mfdata("modelrecarray", modelrecarray)

        self.exchangerecarray = self.build_mfdata("exchangerecarray", exchangerecarray)

        self.mxiter = self.build_mfdata("mxiter", mxiter)

        self.solutionrecarray = self.build_mfdata("solutionrecarray", solutionrecarray)


