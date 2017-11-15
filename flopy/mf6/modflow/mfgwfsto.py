from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfsto(mfpackage.MFPackage):
    package_abbr = "gwfsto"
    iconvert = mfdatautil.ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 'iconvert'))
    ss = mfdatautil.ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 'ss'))
    sy = mfdatautil.ArrayTemplateGenerator(('gwf6', 'sto', 'griddata', 'sy'))
    """
    ModflowGwfsto defines a sto package within a gwf6 model.

    Attributes
    ----------
    save_flows : (save_flows : keyword)
        keyword to indicate that cell-by-cell flow terms will be written to the file specified with ``BUDGET SAVE FILE'' in Output Control.
    storagecoefficient : (storagecoefficient : keyword)
        keyword to indicate that the ss array is read as storage coefficient rather than specific storage.
    iconvert : [(iconvert : integer)]
        is a flag for each cell that specifies whether or not a cell is convertible for the storage calculation. 0 indicates confined storage is used. $>$0 indicates confined storage is used when head is above cell top and unconfined storage is used when head is below cell top. A mixed formulation is when when a cell converts from confined to unconfined (or vice versa) during a single time step.
    ss : [(ss : double)]
        is specific storage (or the storage coefficient if STORAGECOEFFICIENT is specified as an option).
    sy : [(sy : double)]
        is specific yield.
    steady_state : (steady-state : keyword)
        keyword to indicate that stress-period iper is steady-state. Steady-state conditions will apply until the TRANSIENT keyword is specified in a subsequent BEGIN PERIOD block.
    transient : (transient : keyword)
        keyword to indicate that stress-period iper is transient. Transient conditions will apply until the STEADY-STATE keyword is specified in a subsequent BEGIN PERIOD block.

    """
    def __init__(self, model, add_to_package_list=True, save_flows=None, storagecoefficient=None, iconvert=None,
                 ss=None, sy=None, steady_state=None, transient=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfsto, self).__init__(model, "sto", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.storagecoefficient = self.build_mfdata("storagecoefficient", storagecoefficient)

        self.iconvert = self.build_mfdata("iconvert", iconvert)

        self.ss = self.build_mfdata("ss", ss)

        self.sy = self.build_mfdata("sy", sy)

        self.steady_state = self.build_mfdata("steady-state", steady_state)

        self.transient = self.build_mfdata("transient", transient)


