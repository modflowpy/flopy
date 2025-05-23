# autogenerated file, do not modify

from os import PathLike, curdir
from typing import Union

from flopy.mf6.data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator
from flopy.mf6.mfpackage import MFChildPackages, MFPackage


class ModflowPrtnam(MFPackage):
    """
    ModflowPrtnam defines a NAM package.

    Parameters
    ----------
    list : string
        is name of the listing file to create for this prt model.  if not specified,
        then the name of the list file will be the basename of the prt model name file
        and the '.lst' extension.  for example, if the prt name file is called
        'my.model.nam' then the list file will be called 'my.model.lst'.
    print_input : keyword
        keyword to indicate that the list of all model stress package information will
        be written to the listing file immediately after it is read.
    print_flows : keyword
        keyword to indicate that the list of all model package flow rates will be
        printed to the listing file for every stress period time step in which 'budget
        print' is specified in output control.  if there is no output control option
        and 'print_flows' is specified, then flow rates are printed for the last time
        step of each stress period.
    save_flows : keyword
        keyword to indicate that all model package flow terms will be written to the
        file specified with 'budget fileout' in output control.
    packages : list

    """

    packages = ListTemplateGenerator(("prt6", "nam", "packages", "packages"))
    package_abbr = "prtnam"
    _package_type = "nam"
    dfn_file_name = "prt-nam.dfn"
    dfn = [
        ["header"],
        ["block options", "name list", "type string", "reader urword", "optional true"],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block packages",
            "name packages",
            "type recarray ftype fname pname",
            "reader urword",
            "optional false",
        ],
        [
            "block packages",
            "name ftype",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block packages",
            "name fname",
            "in_record true",
            "type string",
            "preserve_case true",
            "tagged false",
            "reader urword",
        ],
        [
            "block packages",
            "name pname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        list=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        packages=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        """
        ModflowPrtnam defines a NAM package.

        Parameters
        ----------
        model
            Model that this package is a part of. Package is automatically
            added to model when it is initialized.
        loading_package : bool
            Do not set this parameter. It is intended for debugging and internal
            processing purposes only.
        list : string
            is name of the listing file to create for this prt model.  if not specified,
            then the name of the list file will be the basename of the prt model name file
            and the '.lst' extension.  for example, if the prt name file is called
            'my.model.nam' then the list file will be called 'my.model.lst'.
        print_input : keyword
            keyword to indicate that the list of all model stress package information will
            be written to the listing file immediately after it is read.
        print_flows : keyword
            keyword to indicate that the list of all model package flow rates will be
            printed to the listing file for every stress period time step in which 'budget
            print' is specified in output control.  if there is no output control option
            and 'print_flows' is specified, then flow rates are printed for the last time
            step of each stress period.
        save_flows : keyword
            keyword to indicate that all model package flow terms will be written to the
            file specified with 'budget fileout' in output control.
        packages : list

        filename : str
            File name for this package.
        pname : str
            Package name for this package.
        parent_file : MFPackage
            Parent package file that references this package. Only needed for
            utility packages (mfutl*). For example, mfutllaktab package must have
            a mfgwflak package parent_file.
        """

        super().__init__(model, "nam", filename, pname, loading_package, **kwargs)

        self.list = self.build_mfdata("list", list)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.packages = self.build_mfdata("packages", packages)

        self._init_complete = True
