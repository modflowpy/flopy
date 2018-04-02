# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfic(mfpackage.MFPackage):
    """
    ModflowGwfic defines a ic package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    strt : [double]
        * strt (double) is the initial (starting) head---that is, head at the
          beginning of the GWF Model simulation. STRT must be specified for all
          simulations, including steady-state simulations. One value is read
          for every model cell. For simulations in which the first stress
          period is steady state, the values used for STRT generally do not
          affect the simulation (exceptions may occur if cells go dry and (or)
          rewet). The execution time, however, will be less if STRT includes
          hydraulic heads that are close to the steady-state solution. A head
          value lower than the cell bottom can be provided if a cell should
          start as dry.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    strt = ArrayTemplateGenerator(('gwf6', 'ic', 'griddata', 'strt'))
    package_abbr = "gwfic"
    package_type = "ic"
    dfn_file_name = "gwf-ic.dfn"

    dfn = [["block griddata", "name strt", "type double precision", 
            "shape (nodes)", "reader readarray", "default_value 1.0"]]

    def __init__(self, model, loading_package=False, strt=1.0, fname=None,
                 pname=None, parent_file=None):
        super(ModflowGwfic, self).__init__(model, "ic", fname, pname,
                                           loading_package, parent_file)        

        # set up variables
        self.strt = self.build_mfdata("strt",  strt)
