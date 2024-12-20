# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlhpc(mfpackage.MFPackage):
    """
    ModflowUtlhpc defines a hpc package within a utl model.

    Parameters
    ----------
    parent_package : MFSimulation
        Parent_package that this package is a part of. Package is automatically
        added to parent_package when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_table : boolean
        * print_table (boolean) keyword to indicate that the partition table
          will be printed to the listing file.
    dev_log_mpi : boolean
        * dev_log_mpi (boolean) keyword to enable (extremely verbose) logging
          of mpi traffic to file.
    partitions : [mname, mrank]
        * mname (string) is the unique model name.
        * mrank (integer) is the zero-based partition number (also: MPI rank or
          processor id) to which the model will be assigned.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    partitions = ListTemplateGenerator(('hpc', 'partitions',
                                        'partitions'))
    package_abbr = "utlhpc"
    _package_type = "hpc"
    dfn_file_name = "utl-hpc.dfn"

    dfn = [
           ["header", ],
           ["block options", "name print_table", "type keyword",
            "reader urword", "optional true"],
           ["block options", "name dev_log_mpi", "type keyword",
            "reader urword", "optional true"],
           ["block partitions", "name partitions",
            "type recarray mname mrank", "reader urword", "optional true"],
           ["block partitions", "name mname", "in_record true",
            "type string", "tagged false", "reader urword"],
           ["block partitions", "name mrank", "in_record true",
            "type integer", "tagged false", "reader urword"]]

    def __init__(self, parent_package, loading_package=False, print_table=None,
                 dev_log_mpi=None, partitions=None, filename=None, pname=None,
                 **kwargs):
        super().__init__(parent_package, "hpc", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.print_table = self.build_mfdata("print_table", print_table)
        self.dev_log_mpi = self.build_mfdata("dev_log_mpi", dev_log_mpi)
        self.partitions = self.build_mfdata("partitions", partitions)
        self._init_complete = True
