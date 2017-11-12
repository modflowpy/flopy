from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfic(mfpackage.MFPackage):
    package_abbr = "gwfic"
    strt = mfdatautil.ArrayTemplateGenerator(('gwf6', 'ic', 'griddata', 'strt'))
    """
    ModflowGwfic defines a ic package within a gwf6 model.

    Attributes
    ----------
    strt : [(strt : double)]
        is the initial (starting) head---that is, head at the beginning of the GWF Model simulation. strt must be specified for all simulations, including steady-state simulations. One value is read for every model cell. For simulations in which the first stress period is steady state, the values used for STRT generally do not affect the simulation (exceptions may occur if cells go dry and (or) rewet). The execution time, however, will be less if STRT includes hydraulic heads that are close to the steady-state solution. A head value lower than the cell bottom can be provided if a cell should start as dry.

    """
    def __init__(self, model, add_to_package_list=True, strt=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfic, self).__init__(model, "ic", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.strt = self.build_mfdata("strt", strt)


