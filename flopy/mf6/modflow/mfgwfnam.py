from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfnam(mfpackage.MFPackage):
    package_abbr = "gwfnam"
    packagerecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'nam', 'packages', 'packagerecarray'))
    """
    ModflowGwfnam defines a nam package within a gwf6 model.

    Attributes
    ----------
    list : (list : string)
        is name of the listing file to create for this GWF model. If not specified, then the name of the list file will be the basename of the GWF model name file and the '.lst' extension. For example, if the GWF name file is called ``my.model.nam'' then the list file will be called ``my.model.lst''.
    print_input : (print_input : keyword)
        keyword to indicate that the list of all model stress package information will be written to the listing file immediately after it is read.
    print_flows : (print_flows : keyword)
        keyword to indicate that the list of all model package flow rates will be printed to the listing file for every stress period time step in which ``BUDGET PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_FLOWS is specified, then flow rates are printed for the last time step of each stress period.
    save_flows : (save_flows : keyword)
        keyword to indicate that all model package flow terms will be written to the file specified with ``BUDGET FILEOUT'' in Output Control.
    newtonoptions : [(newton : keyword), (under_relaxation : keyword)]
        none
        newton : keyword that activates the Newton-Raphson formulation for groundwater flow between connected, convertible groundwater cells and stress packages that support calculation of Newton-Raphson terms for groundwater exchanges. Cells will not dry when this option is used. By default, the Newton-Raphson formulation is not applied.
        under_relaxation : keyword that indicates whether the groundwater head in a cell will be under-relaxed when water levels fall below the bottom of the model below any given cell. By default, Newton-Raphson UNDER\_RELAXATION is not applied.
    packagerecarray : [(ftype : string), (fname : string), (pname : string)]
        ftype : is the file type, which must be one of the following character values shown in tableftype. Ftype may be entered in any combination of uppercase and lowercase.
        fname : is the name of the file containing the package input. The path to the file should be included if the file is not located in the folder where the program was run.
        pname : is the user-defined name for the package. Pname is restricted to 16 characters. No spaces are allowed in Pname. Pname character values are read and stored by the program for stress packages only. These names may be useful for labeling purposes when multiple stress packages of the same type are located within a single GWF Model. If Pname is specified for a stress package, then Pname will be used in the flow budget table in the listing file; it will also be used for the text entry in the cell-by-cell budget file. Pname is case insensitive and is stored in all upper case letters.

    """
    def __init__(self, model, add_to_package_list=True, list=None, print_input=None, print_flows=None,
                 save_flows=None, newtonoptions=None, packagerecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfnam, self).__init__(model, "nam", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.list = self.build_mfdata("list", list)

        self.print_input = self.build_mfdata("print_input", print_input)

        self.print_flows = self.build_mfdata("print_flows", print_flows)

        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.newtonoptions = self.build_mfdata("newtonoptions", newtonoptions)

        self.packagerecarray = self.build_mfdata("packagerecarray", packagerecarray)


