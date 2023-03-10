import shutil
from inspect import getsourcefile
from os import makedirs
from os.path import abspath, exists, join, split, splitext
from flopy.mf6.mfbase import PackageContainer
from flopy.mf6.utils import createpackages
from flopy.mf6.utils.flopy_plugins.plugin_interface import FPBMIPluginInterface


class TemplateUtils:
    """
    Class contains several static methods used internally by flopy.  This
    class is not intended for the end user.  Use the generate_plugin_template
    and create_python_package methods instead.

    """

    @staticmethod
    def update_init(package_name, package_path):
        """
        Updates the init file to include the new flopy plug-in.

        Parameters
        ----------
        package_name : str
            Name of flopy plug-in
        package_path : str
            Path to flopy plug-in
        """
        package_folder, package_file_name = split(package_path)
        init_file = join(package_folder, "__init__.py")
        init_lines = []
        package_file = package_file_name.split(".")[0]
        # read in existing file
        with open(init_file, "r") as fd_init:
            for line in fd_init:
                sline = line.strip()
                ssline = sline.split()
                if len(sline) > 0 and (
                    len(ssline) <= 1 or package_file != ssline[1][1:]
                ):
                    init_lines.append(line)
        # append line to list
        init_lines.append(
            f"from .{package_file} "
            f"import Flopy{package_name.capitalize()}\n"
        )
        # write list to file
        with open(init_file, "w") as fd_init:
            for line in init_lines:
                fd_init.write(line)

    @staticmethod
    def _write_spd_eval(
        fd_pkg, stress_period_data, api_package_support, indent
    ):
        """
        Writes the part of the flopy plug-in file that loops through stress
        period data.

        Parameters
        ----------
        fd_pkg : file
            File to write to
        stress_period_data : list
            Stress period data list
        api_package_support : bool
            Whether plug-in has support for an api package
        indent : str
            The "base" size of the indent to use when writing this file
        """
        # iteration_start - Get hcof, rhs, nodelist, and bound
        if api_package_support:
            fd_pkg.write(f"{indent}# get mf6 variables\n")
            fd_pkg.write(
                f"{indent}cur_nodelist = self.mf6_default_package."
                f'get_advanced_var("nodelist")\n'
            )
            fd_pkg.write(
                f"{indent}hcof_list = self.mf6_default_package." f"hcof\n"
            )
            fd_pkg.write(f"{indent}rhs_list = self.mf6_default_package.rhs\n")
            fd_pkg.write(
                f"{indent}bound = self.mf6_default_package."
                f'get_advanced_var("bound")\n\n'
            )

        # iteration_start - loop through period data, comment to
        # update nodelist, rhs, hcof arrays here
        fd_pkg.write(f"{indent}# loop through stress period data\n")
        fd_pkg.write(
            f"{indent}for row_num, data_row in "
            "enumerate(self.current_spd):\n"
        )
        fd_pkg.write(f"{indent}    aux_offset = 0\n")
        fd_pkg.write(f"{indent}    # get stress period variables\n")
        has_aux = False
        for idx, item in enumerate(stress_period_data):
            index = f"{idx}" " + aux_offset"
            if item == "aux":
                fd_pkg.write(
                    f"{indent}    if hasattr(self, "
                    '"auxiliary") and self.auxiliary is '
                    "not None:\n"
                )
                fd_pkg.write(f"{indent}        aux_vals = []\n")
                fd_pkg.write(
                    f"{indent}        _aux = self.auxiliary." "tolist()\n"
                )
                fd_pkg.write(
                    f"{indent}        for aux_offset, "
                    "aux_var in enumerate(_aux"
                    "[0][1:]):\n"
                )
                fd_pkg.write(
                    f"{indent}            aux_vals.append("
                    f"data_row[{index}])\n"
                )
                fd_pkg.write(f"{indent}    else:\n")
                fd_pkg.write(f"{indent}        aux_offset = -1\n")
            else:
                if item == "boundname":
                    fd_pkg.write(
                        f"{indent}    if hasattr(self, "
                        '"boundnames") and self.boundnames'
                        ":\n"
                    )
                    spaces = f"{indent}        "
                else:
                    spaces = f"{indent}    "
                fd_pkg.write(f"{spaces}{item} = " f"data_row[{index}]\n")
        if api_package_support:
            fd_pkg.write(
                f"\n{indent}    # ----------------------------"
                "-------------------------------\n"
            )
            fd_pkg.write(
                f"{indent}    # ADD YOUR CODE HERE TO UPDATE "
                "RHS, HCOF, BOUND, AND NODELIST\n"
            )
            fd_pkg.write(
                f"{indent}    # -----------------------------"
                "------------------------------\n"
            )
            fd_pkg.write(f"{indent}    # example code...\n")
            fd_pkg.write(
                f"{indent}    # rhs_list[row_num] = " "calculated_rhs_val\n"
            )
            fd_pkg.write(
                f"{indent}    # hcof_list[row_num] = " "calculated_hcof_val\n"
            )
            fd_pkg.write(
                f"{indent}    # cur_nodelist[row_num] = \\\n"
                f"{indent}    #     self.mf6_model.usertonode"
                f"[self.get_node(cellid)] + 1\n"
            )
            fd_pkg.write(
                f"{indent}    # bound[row_num, 0] = "
                "calculated_bound_val\n\n"
            )

            # iteration_start - at end of loop use BMI to write
            # nodelist, rhs, hcof arrays to API package
            fd_pkg.write(f"{indent}# update mf6 variables\n")
            fd_pkg.write(
                f"{indent}self.mf6_default_package.set_advanced_var"
                f'("nodelist", cur_nodelist)\n'
            )
            fd_pkg.write(f"{indent}self.mf6_default_package.rhs = rhs_list\n")
            fd_pkg.write(
                f"{indent}self.mf6_default_package.hcof = " f"hcof_list\n"
            )
            fd_pkg.write(
                f"{indent}self.mf6_default_package.set_advanced_var"
                f'("bound", bound)\n\n'
            )
        else:
            fd_pkg.write(
                f"\n{indent}    # ----------------------------"
                "-------------------------------\n"
            )
            fd_pkg.write(f"{indent}    # ADD YOUR CODE HERE\n")
            fd_pkg.write(
                f"{indent}    # -----------------------------"
                "------------------------------\n\n"
            )

    @staticmethod
    def write_class_template(
        package_name,
        package_path,
        options_dict,
        dimensions,
        package_data,
        stress_period_data,
        api_package_support=True,
        evaluation_code_at="iteration_start",
    ):
        """
        Writes templated code for a new flopy plug-in.

        Parameters
        ----------
        package_name : str
            Name of flopy plug-in
        package_path : str
            Path to flopy plug-in
        options_dict : dict
            Dictionary containing plug-in options
        dimensions: list
            List containing plug-in dimensions
        package_data: dict
            Dictionary containing plug-in package data
        stress_period_data: dict
            Dictionary containing plug-in stress period data
        api_package_support : bool
            Whether or not this plug-in will have API package support
        evaluation_code_at: str
            Location where user is instructed to add code
        """
        exclude_property_list = [
            "ts_filerecord",
            "ts6",
            "filein",
            "ts6_filename",
            "obs_filerecord",
            "obs6",
            "obs6_filename",
            "tas_filerecrod",
            "tas6",
            "tas6_filename",
        ]
        has_pkd = package_data is not None and len(package_data) > 0
        has_spd = (
            stress_period_data is not None and len(stress_period_data) > 0
        )
        # write out class template
        with open(package_path, "w") as fd_pkg:
            # class definition and init
            fd_pkg.write(
                "from flopy.mf6.utils.flopy_plugins import "
                "FPBMIPluginInterface\n\n\n"
            )
            fd_pkg.write(
                f"class Flopy{package_name.capitalize()}"
                f"(FPBMIPluginInterface):"
            )
            fd_pkg.write(f'\n    abbr = "{package_name}"\n\n')
            if api_package_support:
                fd_pkg.write("    def __init__(self, use_api_package=True):\n")
            else:
                fd_pkg.write(
                    "    def __init__(self, use_api_package=False):" "\n"
                )
            fd_pkg.write("        super().__init__(use_api_package)\n\n")
            fd_pkg.write("    def init_plugin(self):\n")
            fd_pkg.write("        super().init_plugin()\n")

            # read in options, dimensions, and stress period data
            if options_dict is not None:
                for option in options_dict:
                    if option not in exclude_property_list:
                        fd_pkg.write("        # load options data\n")
                        fd_pkg.write(
                            f"        self.{option} = "
                            f"self.package.{option}.get_data()\n"
                        )
            if dimensions is not None:
                for dimension in dimensions:
                    fd_pkg.write("        # load dimensions data\n")
                    fd_pkg.write(
                        f"        self.{dimension} = "
                        f"self.package.{dimension}.get_data()\n"
                    )

            if has_pkd:
                fd_pkg.write("        # load package data\n")
                fd_pkg.write(
                    f"        self.packagedata = "
                    f"self.package.packagedata.get_data()\n"
                )

            if "maxbound" not in dimensions and not has_spd:
                if has_pkd:
                    fd_pkg.write(
                        "\n        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        "        # PACKAGE DOES NOT HAVE MAXBOUND "
                        "DIMENSION\n"
                    )
                    fd_pkg.write(
                        "        # API PACKAGE'S MAXBOUND "
                        "ASSIGNED TO LENGTH OF PACKAGEDATA\n"
                    )
                    fd_pkg.write(
                        "        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        f"        self.maxbound = " f"len(self.packagedata)\n"
                    )
                elif len(dimensions) > 0:
                    fd_pkg.write(
                        "\n        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        "        # PACKAGE DOES NOT HAVE MAXBOUND "
                        "DIMENSION OR PACKAGEDATA\n"
                    )
                    fd_pkg.write(
                        "        # API PACKAGE'S MAXBOUND"
                        f" ASSIGNED TO {dimensions[0]}"
                        f"\n"
                    )
                    fd_pkg.write(
                        "        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        "        self.maxbound = " f"dimensions_list[0]\n"
                    )
                else:
                    fd_pkg.write(
                        "\n        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        "        # PACKAGE DOES NOT HAVE MAXBOUND "
                        "DIMENSION, PACKAGEDATA, OR DIMENSIONS\n"
                    )
                    fd_pkg.write(
                        "        # API PACKAGE'S MAXBOUND " f"ASSIGNED TO 1\n"
                    )
                    fd_pkg.write(
                        "        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write("        self.maxbound = 1\n")

            if has_spd:
                fd_pkg.write("        # load stress period data\n")
                fd_pkg.write(
                    "        self.spd = self.package." "stress_period_data\n"
                )
                fd_pkg.write("        self.current_spd = []\n")
                fd_pkg.write("\n")

                if "maxbound" not in dimensions:
                    fd_pkg.write(
                        "\n        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        "        # PACKAGE DOES NOT HAVE MAXBOUND "
                        "DIMENSION\n"
                    )
                    fd_pkg.write(
                        "        # API PACKAGE'S MAXBOUND "
                        "ASSIGNED TO MAX STRESSPERIODDATA LENGTH\n"
                    )
                    fd_pkg.write(
                        "        # -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        f"        self.maxbound = 1\n"
                        f"        spd_all = self.spd.get_data()\n"
                    )
                    fd_pkg.write(f"        for data in spd_all:\n")
                    fd_pkg.write(
                        f"            if len(data) > self.maxbound:\n"
                        f"                self.maxbound = len(data)"
                        f"\n\n"
                    )

                # stress_period_start
                fd_pkg.write(
                    "    def stress_period_start(self, kper, " "sln_group):\n"
                )
                fd_pkg.write(
                    "        super().stress_period_start(kper, "
                    "sln_group)\n\n"
                )
                fd_pkg.write("        # get data for this stress period\n")
                fd_pkg.write(
                    "        current_spd = self.spd.get_data" "(kper)\n"
                )
                fd_pkg.write("        if current_spd is not None:\n")
                fd_pkg.write(
                    "            current_spd = current_spd.tolist" "()\n"
                )
                fd_pkg.write("            if len(current_spd) > 0:\n")
                fd_pkg.write(
                    "                self.current_spd = " "current_spd\n\n"
                )
                if evaluation_code_at.lower() == "stress_period_start":
                    indent = "                "
                    TemplateUtils._write_spd_eval(
                        fd_pkg, stress_period_data, api_package_support, indent
                    )

                # time_step_start
                fd_pkg.write(
                    "    def time_step_start(self, kper, kstp, "
                    "sln_group):\n"
                )
                fd_pkg.write(
                    "        super().time_step_start(kper, kstp, "
                    "sln_group)\n\n"
                )
                if evaluation_code_at.lower() == "time_step_start":
                    indent = "        "
                    TemplateUtils._write_spd_eval(
                        fd_pkg, stress_period_data, api_package_support, indent
                    )

                # iteration_start
                fd_pkg.write(
                    "    def iteration_start(self, kper, kstp, "
                    "iter, sln_group):\n"
                )
                fd_pkg.write(
                    "        super().iteration_start(kper, kstp, "
                    "iter, sln_group)\n\n"
                )
                if evaluation_code_at.lower() == "iteration_start":
                    fd_pkg.write("        # record start of iteration\n")

                    indent = "        "
                    TemplateUtils._write_spd_eval(
                        fd_pkg, stress_period_data, api_package_support, indent
                    )
            else:
                # override receive_bmi
                fd_pkg.write("\n    def receive_bmi(self, mf6_sim):\n")
                fd_pkg.write("        super().receive_bmi(mf6_sim)\n\n")
                if api_package_support:
                    # get hcof, rhs, nodelist, and bound
                    fd_pkg.write("        # get mf6 variables\n")
                    fd_pkg.write(
                        "        cur_nodelist = "
                        "self.mf6_default_package.get_advanced_var"
                        '("nodelist")\n'
                    )
                    fd_pkg.write(
                        "        hcof_list = "
                        "self.mf6_default_package.hcof\n"
                    )
                    fd_pkg.write(
                        "        rhs_list = " "self.mf6_default_package.rhs\n"
                    )
                    fd_pkg.write(
                        "        bound = "
                        "self.mf6_default_package.get_advanced_var"
                        '("bound")\n'
                    )
                if has_pkd:
                    fd_pkg.write(
                        "        for row_num, data_row in "
                        "enumerate(self.packagedata):\n"
                    )
                    fd_pkg.write("            # get package variables\n")
                    for idx, item in enumerate(package_data):
                        fd_pkg.write(f"            {item} = data_row[{idx}]\n")
                    lspace = "            "
                else:
                    lspace = "        "

                if api_package_support:
                    # comments for user to set write code here
                    fd_pkg.write(
                        f"\n{lspace}# -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(
                        f"{lspace}# ADD YOUR CODE HERE TO UPDATE "
                        "RHS, HCOF, BOUND, AND NODELIST\n"
                    )
                    fd_pkg.write(
                        f"{lspace}# -----------------------------"
                        "------------------------------\n"
                    )
                    fd_pkg.write(f"{lspace}# example code...\n")
                    fd_pkg.write(
                        f"{lspace}# rhs_list[row_num] = "
                        "calculated_rhs_val\n"
                    )
                    fd_pkg.write(
                        f"{lspace}# hcof_list[row_num] = "
                        "calculated_hcof_val\n"
                    )
                    fd_pkg.write(
                        f"{lspace}# cur_nodelist[row_num] = \\\n"
                        f"{lspace}#    self.mf6_model.usertonode"
                        f"[self.get_node(cellid)] + 1\n"
                    )
                    fd_pkg.write(
                        f"{lspace}# bound[row_num, 0] = "
                        "calculated_bound_val\n\n"
                    )

                    # set hcof, rhs, nodelist, and bound
                    fd_pkg.write("        # update mf6 variables\n")
                    fd_pkg.write(
                        f"{lspace}self.mf6_default_package.set_advanced_var"
                        f'("nodelist", cur_nodelist)\n'
                    )
                    fd_pkg.write(
                        f"{lspace}self.mf6_default_package.rhs = rhs_list\n"
                    )
                    fd_pkg.write(
                        f"{lspace}self.mf6_default_package.hcof = "
                        f"hcof_list\n"
                    )
                    fd_pkg.write(
                        f"{lspace}self.mf6_default_package.set_advanced_var"
                        f'("bound", bound)\n'
                    )

                # stress_period_start
                fd_pkg.write(
                    "    def stress_period_start(self, kper, " "sln_group):\n"
                )
                fd_pkg.write(
                    "        super().stress_period_start(kper, "
                    "sln_group)\n\n"
                )

                # time_step_start
                fd_pkg.write(
                    "    def time_step_start(self, kper, kstp, "
                    "sln_group):\n"
                )
                fd_pkg.write(
                    "        super().time_step_start(kper, kstp, "
                    "sln_group)\n\n"
                )

                # iteration_start
                fd_pkg.write(
                    "    def iteration_start(self, kper, kstp, "
                    "iter, sln_group):\n"
                )
                fd_pkg.write(
                    "        super().iteration_start(kper, kstp, "
                    "iter, sln_group)\n\n"
                )

            # iteration_end
            fd_pkg.write(
                "    def iteration_end(self, kper, kstp, "
                "iter, sln_group):\n"
            )
            fd_pkg.write(
                "        return super().iteration_end(kper, kstp, "
                "iter, sln_group)\n\n"
            )

            # time_step_end
            fd_pkg.write(
                "    def time_step_end(self, kper, kstp, converged, "
                "sln_group):\n"
            )
            fd_pkg.write(
                "        super().time_step_end(kper, kstp, "
                "converged, sln_group)\n\n"
            )

            # stress_period_end
            fd_pkg.write(
                "    def stress_period_end(self, kper, " "sln_group):\n"
            )
            fd_pkg.write(
                "        super().stress_period_end(kper, " "sln_group)\n\n"
            )

    @staticmethod
    def get_packages_path():
        """
        Returns relative path to packages folder.
        """
        return split(abspath(getsourcefile(lambda: 0)))[0]

    @staticmethod
    def get_dfn_base_path():
        """
        Returns relative path to dfn folder.
        """
        fp_packages_path = TemplateUtils.get_packages_path()
        return join(fp_packages_path, "..", "..", "data", "dfn")

    @staticmethod
    def create_dfn(
        model_type,
        package_name,
        options,
        dimensions_list,
        package_vars=None,
        stress_period_vars=None,
    ):
        """
        Creates dfn for a new flopy plug-in.

        Parameters
        ----------
        model_type : str
            Type of model that the flopy plug-in supports
        package_name : str
            Name of flopy plug-in
        options : dict
            Dictionary containing plug-in options
        dimensions_list: list
            List containing plug-in dimensions
        package_vars: dict
            Dictionary containing plug-in package data
        stress_period_vars: dict
            Dictionary containing plug-in stress period data
        """
        dfn_path = TemplateUtils.get_dfn_base_path()
        new_dfn_file_path = join(
            dfn_path, f"{model_type}-fp_{package_name}" f".dfn"
        )
        with open(new_dfn_file_path, "w") as fd_dfn:
            # write headers
            if options is not None:
                fd_dfn.write(
                    f"# --------------------- {model_type} "
                    f"{package_name} options ---------------------\n"
                )
            fd_dfn.write(f"# flopy flopy-plugin {package_name}\n\n")
            if options is not None:
                # write options lines
                for name, values in options.items():
                    # write out as true/false option
                    fd_dfn.write("block options\n")
                    fd_dfn.write(f"name {name}\n")
                    fd_dfn.write(f'type {values["type"]}\n')
                    if "shape" in values:
                        fd_dfn.write(f'shape ({values["shape"]})\n')
                    fd_dfn.write("reader urword\n")
                    if "optional" in values:
                        fd_dfn.write(f'optional {values["optional"]}\n')
                    else:
                        fd_dfn.write("optional true\n")

                    fd_dfn.write("longname *** ADD LONGNAME HERE ***\n")
                    fd_dfn.write(
                        "description *** ADD DESCRIPTION HERE ***\n\n"
                    )

            fd_dfn.write(
                f"# --------------------- {model_type} {package_name}"
                f" dimensions ---------------------\n\n"
            )
            if dimensions_list is not None:
                for dimension in dimensions_list:
                    fd_dfn.write("block dimensions\n")
                    fd_dfn.write(f"name {dimension}\n")
                    fd_dfn.write("type integer\n")
                    fd_dfn.write("reader urword\n")
                    fd_dfn.write("optional false\n")
                    fd_dfn.write(f"longname maximum number... \n")
                    fd_dfn.write("description maximum number...\n\n")
            else:
                # write generic dimensions block with maxbound
                fd_dfn.write("block dimensions\n")
                fd_dfn.write("name maxbound\n")
                fd_dfn.write("type integer\n")
                fd_dfn.write("reader urword\n")
                fd_dfn.write("optional false\n")
                fd_dfn.write(
                    f"longname maximum number of {package_name} " "cells\n"
                )
                fd_dfn.write(
                    f"description maximum number of {package_name} "
                    "cells for any stress period\n\n"
                )

            if package_vars is not None:
                # write period block with package data
                fd_dfn.write(
                    f"# --------------------- {model_type} "
                    f"{package_name} packagedata "
                    f"---------------------\n\n"
                )
                # build package variables
                spv_names = []
                for spv in package_vars.keys():
                    spv_names.append(spv)
                spv_str = " ".join(spv_names)
                fd_dfn.write("block packagedata\n")
                fd_dfn.write("name packagedata\n")
                fd_dfn.write(f"type recarray {spv_str}\n")
                fd_dfn.write("shape (maxpackage)\n")
                fd_dfn.write("reader urword\n")
                fd_dfn.write("longname\n")
                fd_dfn.write("description\n\n")

                # write out package variables
                for name, value in package_vars.items():
                    fd_dfn.write("block packagedata\n")
                    fd_dfn.write(f"name {name}\n")
                    fd_dfn.write(f'type {value["type"]}\n')
                    if "shape" in value:
                        fd_dfn.write(f'shape ({value["shape"]})\n')

                    fd_dfn.write("tagged false\n")
                    fd_dfn.write("in_record true\n")
                    fd_dfn.write("reader urword\n")
                    if "optional" in value:
                        fd_dfn.write(f'optional {value["optional"]}\n')
                    fd_dfn.write("longname *** ADD LONGNAME HERE ***\n")
                    fd_dfn.write(
                        "description *** ADD DESCRIPTION HERE ***\n" "\n"
                    )

            if stress_period_vars is not None:
                # write period block with stress period data
                fd_dfn.write(
                    f"# --------------------- {model_type} "
                    f"{package_name} period ---------------------\n"
                    f"\n"
                )
                fd_dfn.write("block period\n")
                fd_dfn.write("name iper\n")
                fd_dfn.write("type integer\n")
                fd_dfn.write("block_variable True\n")
                fd_dfn.write("in_record true\n")
                fd_dfn.write("tagged false\n")
                fd_dfn.write("shape\nvalid\n")
                fd_dfn.write("reader urword\n")
                fd_dfn.write("optional false\n")
                fd_dfn.write("longname stress period number\n")
                fd_dfn.write("description REPLACE iper {}\n\n")

                # build stress period variables
                spv_names = []
                for spv in stress_period_vars.keys():
                    spv_names.append(spv)
                spv_str = " ".join(spv_names)
                fd_dfn.write("block period\n")
                fd_dfn.write("name stress_period_data\n")
                fd_dfn.write(f"type recarray {spv_str}\n")
                fd_dfn.write("shape (maxbound)\n")
                fd_dfn.write("reader urword\n")
                fd_dfn.write("longname\n")
                fd_dfn.write("description\n\n")

                # write out stress period variables
                for name, value in stress_period_vars.items():
                    fd_dfn.write("block period\n")
                    fd_dfn.write(f"name {name}\n")
                    fd_dfn.write(f'type {value["type"]}\n')
                    if "shape" in value:
                        fd_dfn.write(f'shape ({value["shape"]})\n')

                    fd_dfn.write("tagged false\n")
                    fd_dfn.write("in_record true\n")
                    fd_dfn.write("reader urword\n")
                    if "optional" in value:
                        fd_dfn.write(f'optional {value["optional"]}\n')
                    fd_dfn.write("longname *** ADD LONGNAME HERE ***\n")
                    fd_dfn.write(
                        "description *** ADD DESCRIPTION HERE ***\n" "\n"
                    )
        return new_dfn_file_path


def generate_plugin_template(
    model_type,
    new_package_abbr,
    options=None,
    stress_period_vars=None,
    package_vars=None,
    api_package_support=True,
    evaluation_code_at="iteration_start",
):
    """
    This method generates a template for a flopy package based on the
    user supplied parameters. The template generates:
        * A DFN file describing the package input file
        * A flopy package file so that flopy can load and save your package's
          input file.
        * A template to build your flopy package code from, which is created
          in the mf6/utils/flopy_plugins/plugins folder.

    Parameters
    ----------
    model_type : str
        Three letter abbreviation indicating model type (eg. "GWF")
    new_package_abbr : str
        Package abbreviation for your new package (eg. "DRN", "WEL")
    options : dict (optional)
        Dictionary of package options.  The option name is the dictionary key,
        and the dictionary value is another dictionary containing the option's
        attributes ("type", "shape", "optional").
    stress_period_vars : dict (optional)
        Dictionary of the package's stress period variables.  The variable
        name is the dictionary key, and the dictionary value is another
        dictionary containing the variable's attributes ("type", "shape",
        "optional").
    package_vars : dict (optional)
        Dictionary of the package's package data variables.  The variable
        name is the dictionary key, and the dictionary value is another
        dictionary containing the variable's attributes ("type", "shape",
        "optional").
    api_package_support : bool (optional)
        True if your plugin will be using the generic MF-6 API package.  Flopy
        plug-ins will automatically add a unique instance of the API package
        to any simulation that uses this plug-in and give the plug-in access
        to that API package.
    evaluation_code_at : str (optional)
        String that determines where your main code block that evaluates and
        changes MF-6 functionality will go.  The options are "iteration_start"
        (default), "time_step_start", or "stress_period_start".
        evaluation_code_at is only used when stress_period_vars exist.

    """
    dimensions_list = []
    if stress_period_vars:
        dimensions_list.append("maxbound")
    if package_vars:
        dimensions_list.append("maxpackage")
    # create dfn based on options and stress period vars supplied, using flopy
    # package name
    dfn_file = TemplateUtils.create_dfn(
        model_type,
        new_package_abbr,
        options,
        dimensions_list,
        package_vars,
        stress_period_vars,
    )
    print(f'Created new dfn file "{dfn_file}"')

    # run createpackages.py
    print(f"Running createpackages...")
    createpackages.create_packages()
    in_file_path = join(
        "..", "..", "modflow", f"mf{model_type}fp_{new_package_abbr}.py"
    )
    print(f"Package interface file {in_file_path} created.")

    # generate flopy package code template
    package_path = join(
        TemplateUtils.get_packages_path(),
        "plugins",
        f"flopy_{new_package_abbr}_plugin.py",
    )
    if stress_period_vars is None:
        stress_period_vars = []
    TemplateUtils.write_class_template(
        new_package_abbr,
        package_path,
        options,
        dimensions_list,
        package_vars,
        stress_period_vars,
        api_package_support,
        evaluation_code_at,
    )
    print(
        f'Flopy plugin template file "{package_path}" created.  Modify '
        f"this file to add functionality to your flopy plugin."
    )
    # update init file
    TemplateUtils.update_init(new_package_abbr, package_path)


def create_python_package(
    plugin_ext, package_type, model_type, python_package_name=None
):
    """
    This method generates the files necessary to install a python package
    containing a flopy plug-in.

    Parameters
    ----------
    plugin_ext : str
        Three letter abbreviation indicating the extension of the plug-in
        to be used.  This plug-in must exist in the "flopy_plugins/plugins"
        folder.
    package_type : str
        The abbreviation of the flopy package file associated with the plug-in.
        This can be found by opening the flopy plug-in's package file in the
        "mf6/modflow" folder and looking for the value of the "_package_type"
        variable.
    model_type : str
        Three letter abbreviation indicating model type (eg. "GWF")
    python_package_name : str
        The name of the python package this method will be generating
    """
    # use plugin_ext to find files, paths, class names, and abbreviations
    bmi_plugins = FPBMIPluginInterface.flopy_bmi_plugins()
    if plugin_ext not in bmi_plugins:
        raise Exception(
            f"Flopy plug-in with extension {plugin_ext} not "
            f"found.  Only flopy plug-ins installed to run "
            f"within flopy can be exported to an external "
            f"python package."
        )
    plugin_class = bmi_plugins[plugin_ext]
    plugin_file_path = abspath(getsourcefile(plugin_class))
    plugin_file_folder, plugin_file_name = split(plugin_file_path)

    package_class = PackageContainer.package_factory(package_type, model_type)
    if package_class is None:
        raise Exception(
            f"No package found with type {package_type} of "
            f"model {model_type}."
        )
    package_file_path = abspath(getsourcefile(package_class))
    dfn_file_name = package_class.dfn_file_name
    package_file_folder, package_file_name = split(package_file_path)
    dfn_file_path = join(
        package_file_folder, "..", "data", "dfn", dfn_file_name
    )
    if python_package_name is None:
        python_package_name = f"fp_{plugin_ext}_plugin"

    # create temp python package folder and move files there
    output_package_folder = f"{python_package_name}"
    output_source_folder = join(output_package_folder, python_package_name)
    if not exists(output_package_folder):
        makedirs(output_package_folder)
    if not exists(output_source_folder):
        makedirs(output_source_folder)
    shutil.copy(plugin_file_path, output_source_folder)
    shutil.copy(package_file_path, output_source_folder)
    shutil.copy(dfn_file_path, output_source_folder)

    # create dist-info entry_points
    plugin = (
        f"{plugin_ext} = {python_package_name}."
        f"{splitext(plugin_file_name)[0]}:"
        f"{plugin_class.__name__}"
    )
    package = (
        f"{package_class.package_abbr} = {python_package_name}."
        f"{splitext(package_file_name)[0]}:"
        f"{package_class.__name__}"
    )
    entry_points = (
        "{"
        f'"mf6api.plugin": ["{plugin}"], '
        f'"mf6api.package": ["{package}"]'
        "}"
    )

    # create setup.py file
    with open(join(output_package_folder, "setup.py"), "w") as fd_setup:
        fd_setup.write("from setuptools import setup\n")
        fd_setup.write("setup(\n")
        fd_setup.write(f"    name='{python_package_name}',\n")
        fd_setup.write("    version='1.0',\n")
        fd_setup.write(f"    entry_points={entry_points},\n")
        fd_setup.write(f"    packages=['{python_package_name}'],\n")
        fd_setup.write(")\n")

    # create __init__.py file
    with open(join(output_source_folder, "__init__.py"), "w") as fd_init:
        fd_init.write(f"from .{splitext(plugin_file_name)[0]} import ")
        fd_init.write(f"{plugin_class.__name__}\n")

    # command line to create wheel file
    out_full = abspath(output_package_folder)
    print(f"Your package files are set up in:\n {out_full}")
    print(
        f"To install your package run the following command from that "
        f"folder:  python setup.py install"
    )
