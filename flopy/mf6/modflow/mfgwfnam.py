# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfnam(mfpackage.MFPackage):
    """
    ModflowGwfnam defines a nam package within a gwf6 model.

    Parameters
    ----------
    list : string
        * list (string) is name of the listing file to create for this GWF
        model. If not specified, then the name of the list file will be the
        basename of the GWF model name file and the '.lst' extension. For
        example, if the GWF name file is called ``my.model.nam'' then the list
        file will be called ``my.model.lst''.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of all model
        stress package information will be written to the listing file
        immediately after it is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of all model
        package flow rates will be printed to the listing file for every stress
        period time step in which ``BUDGET PRINT'' is specified in Output
        Control. If there is no Output Control option and PRINT\_FLOWS is
        specified, then flow rates are printed for the last time step of each
        stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that all model package flow
        terms will be written to the file specified with ``BUDGET FILEOUT'' in
        Output Control.
    newtonoptions : [under_relaxation]
        * under_relaxation (string) keyword that indicates whether the
        groundwater head in a cell will be under-relaxed when water levels fall
        below the bottom of the model below any given cell. By default, Newton-
        Raphson UNDER\_RELAXATION is not applied.
    packagerecarray : [ftype, fname, pname]
        * ftype (string) is the file type, which must be one of the following
        character values shown in tableftype. Ftype may be entered
        in any combination of uppercase and lowercase.
        * fname (string) is the name of the file containing the package input.
        The path to the file should be included if the file is not located in
        the folder where the program was run.
        * pname (string) is the user-defined name for the package.
        Pname is restricted to 16 characters. No spaces are allowed in
        Pname. Pname character values are read and stored by
        the program for stress packages only. These names may be useful for
        labeling purposes when multiple stress packages of the same type are
        located within a single GWF Model. If Pname is specified for a
        stress package, then Pname will be used in the flow budget
        table in the listing file; it will also be used for the text entry in
        the cell-by-cell budget file. Pname is case insensitive and is
        stored in all upper case letters.

    """
    packagerecarray = ListTemplateGenerator(('gwf6', 'nam', 'packages', 
                                             'packagerecarray'))
    package_abbr = "gwfnam"
    package_type = "nam"
    dfn = [["block options", "name list", "type string", "reader urword", 
            "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name newtonoptions", 
            "type record newton under_relaxation", "reader urword", 
            "optional true"],
           ["block options", "name newton", "in_record true", 
            "type keyword", "reader urword"],
           ["block options", "name under_relaxation", "in_record true", 
            "type keyword", "reader urword", "optional true"],
           ["block packages", "name packagerecarray", 
            "type recarray ftype fname pname", "reader urword", 
            "optional false"],
           ["block packages", "name ftype", "in_record true", "type string", 
            "ucase true", "tagged false", "reader urword"],
           ["block packages", "name fname", "in_record true", "type string", 
            "preserve_case true", "tagged false", "reader urword"],
           ["block packages", "name pname", "in_record true", "type string", 
            "tagged false", "reader urword", "optional true"]]

    def __init__(self, model, add_to_package_list=True, list=None,
                 print_input=None, print_flows=None, save_flows=None,
                 newtonoptions=None, packagerecarray=None, fname=None,
                 pname=None, parent_file=None):
        super(ModflowGwfnam, self).__init__(model, "nam", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.list = self.build_mfdata("list",  list)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.newtonoptions = self.build_mfdata("newtonoptions",  newtonoptions)
        self.packagerecarray = self.build_mfdata("packagerecarray", 
                                                 packagerecarray)
