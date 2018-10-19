# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfsto(mfpackage.MFPackage):
    """
    ModflowGwfsto defines a sto package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that cell-by-cell flow terms
          will be written to the file specified with "BUDGET SAVE FILE" in
          Output Control.
    storagecoefficient : boolean
        * storagecoefficient (boolean) keyword to indicate that the SS array is
          read as storage coefficient rather than specific storage.
    iconvert : [integer]
        * iconvert (integer) is a flag for each cell that specifies whether or
          not a cell is convertible for the storage calculation. 0 indicates
          confined storage is used. :math:`>`0 indicates confined storage is
          used when head is above cell top and unconfined storage is used when
          head is below cell top. A mixed formulation is when when a cell
          converts from confined to unconfined (or vice versa) during a single
          time step.
    ss : [double]
        * ss (double) is specific storage (or the storage coefficient if
          STORAGECOEFFICIENT is specified as an option).
    sy : [double]
        * sy (double) is specific yield.
    steady_state : boolean
        * steady-state (boolean) keyword to indicate that stress-period IPER is
          steady-state. Steady-state conditions will apply until the TRANSIENT
          keyword is specified in a subsequent BEGIN PERIOD block.
    transient : boolean
        * transient (boolean) keyword to indicate that stress-period IPER is
          transient. Transient conditions will apply until the STEADY-STATE
          keyword is specified in a subsequent BEGIN PERIOD block.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    iconvert = ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 
                                       'iconvert'))
    ss = ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 'ss'))
    sy = ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 'sy'))
    package_abbr = "gwfsto"
    package_type = "sto"
    dfn_file_name = "gwf-sto.dfn"

    dfn = [["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name storagecoefficient", "type keyword", 
            "reader urword", "optional true"],
           ["block griddata", "name iconvert", "type integer", 
            "shape (nodes)", "valid", "reader readarray", "optional false", 
            "default_value 0"],
           ["block griddata", "name ss", "type double precision", 
            "shape (nodes)", "valid", "reader readarray", "optional false", 
            "default_value 1.e-5"],
           ["block griddata", "name sy", "type double precision", 
            "shape (nodes)", "valid", "reader readarray", "optional false", 
            "default_value 0.15"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name steady-state", "type keyword", "shape", 
            "valid", "reader urword", "optional true"],
           ["block period", "name transient", "type keyword", "shape", 
            "valid", "reader urword", "optional true"]]

    def __init__(self, model, loading_package=False, save_flows=None,
                 storagecoefficient=None, iconvert=0, ss=1.e-5, sy=0.15,
                 steady_state=None, transient=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfsto, self).__init__(model, "sto", fname, pname,
                                            loading_package, parent_file)        

        # set up variables
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.storagecoefficient = self.build_mfdata("storagecoefficient", 
                                                    storagecoefficient)
        self.iconvert = self.build_mfdata("iconvert",  iconvert)
        self.ss = self.build_mfdata("ss",  ss)
        self.sy = self.build_mfdata("sy",  sy)
        self.steady_state = self.build_mfdata("steady-state",  steady_state)
        self.transient = self.build_mfdata("transient",  transient)
