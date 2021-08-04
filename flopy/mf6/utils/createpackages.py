import os
import io
import datetime
import textwrap
from enum import Enum
from flopy.mf6.data import mfstructure, mfdatautil
from flopy.utils import datautil

"""
createpackages.py is a utility script that reads in the file definition
metadata in the .dfn files to create the package classes in the modflow folder.
Run this script any time changes are made to the .dfn files.
"""


class PackageLevel(Enum):
    sim_level = 0
    model_level = 1


def build_doc_string(param_name, param_type, param_desc, indent):
    return "{}{} : {}\n{}* {}".format(
        indent, param_name, param_type, indent * 2, param_desc
    )


def generator_type(data_type):
    if (
        data_type == mfstructure.DataType.scalar_keyword
        or data_type == mfstructure.DataType.scalar
    ):
        # regular scalar
        return "ScalarTemplateGenerator"
    elif (
        data_type == mfstructure.DataType.scalar_keyword_transient
        or data_type == mfstructure.DataType.scalar_transient
    ):
        # transient scalar
        return "ScalarTemplateGenerator"
    elif data_type == mfstructure.DataType.array:
        # array
        return "ArrayTemplateGenerator"
    elif data_type == mfstructure.DataType.array_transient:
        # transient array
        return "ArrayTemplateGenerator"
    elif data_type == mfstructure.DataType.list:
        # list
        return "ListTemplateGenerator"
    elif (
        data_type == mfstructure.DataType.list_transient
        or data_type == mfstructure.DataType.list_multiple
    ):
        # transient or multiple list
        return "ListTemplateGenerator"


def clean_class_string(name):
    if len(name) > 0:
        clean_string = name.replace(" ", "_")
        clean_string = clean_string.replace("-", "_")
        version = mfstructure.MFStructure().get_version_string()
        # FIX: remove all numbers
        if clean_string[-1] == version:
            clean_string = clean_string[:-1]
        return clean_string
    return name


def build_dfn_string(dfn_list):
    dfn_string = "    dfn = ["
    line_length = len(dfn_string)
    leading_spaces = " " * line_length
    first_di = True
    # process all data items
    for data_item in dfn_list:
        line_length += 1
        if not first_di:
            dfn_string = "{},\n{}".format(dfn_string, leading_spaces)
            line_length = len(leading_spaces)
        else:
            first_di = False
        dfn_string = "{}{}".format(dfn_string, "[")
        first_line = True
        # process each line in a data item
        for line in data_item:
            line = line.strip()
            # do not include the description of longname
            if not line.lower().startswith(
                "description"
            ) and not line.lower().startswith("longname"):
                line = line.replace('"', "'")
                line_length += len(line) + 4
                if not first_line:
                    dfn_string = "{},".format(dfn_string)
                if line_length < 77:
                    # added text fits on the current line
                    if first_line:
                        dfn_string = '{}"{}"'.format(dfn_string, line)
                    else:
                        dfn_string = '{} "{}"'.format(dfn_string, line)
                else:
                    # added text does not fit on the current line
                    line_length = len(line) + len(leading_spaces) + 2
                    if line_length > 79:
                        # added text too long to fit on a single line, wrap
                        # text as needed
                        line = '"{}"'.format(line)
                        lines = textwrap.wrap(
                            line,
                            75 - len(leading_spaces),
                            drop_whitespace=True,
                        )
                        lines[0] = "{} {}".format(leading_spaces, lines[0])
                        line_join = ' "\n{} "'.format(leading_spaces)
                        dfn_string = "{}\n{}".format(
                            dfn_string, line_join.join(lines)
                        )
                    else:
                        dfn_string = '{}\n{} "{}"'.format(
                            dfn_string, leading_spaces, line
                        )
            first_line = False

        dfn_string = "{}{}".format(dfn_string, "]")
    dfn_string = "{}{}".format(dfn_string, "]")
    return dfn_string


def create_init_var(clean_ds_name, data_structure_name, init_val=None):
    if init_val is None:
        init_val = clean_ds_name

    init_var = "        self.{} = self.build_mfdata(".format(clean_ds_name)
    leading_spaces = " " * len(init_var)
    if len(init_var) + len(data_structure_name) + 2 > 79:
        second_line = '\n            "{}",'.format(data_structure_name)
        if len(second_line) + len(clean_ds_name) + 2 > 79:
            init_var = "{}{}\n            {})".format(
                init_var, second_line, init_val
            )
        else:
            init_var = "{}{} {})".format(init_var, second_line, init_val)
    else:
        init_var = '{}"{}",'.format(init_var, data_structure_name)
        if len(init_var) + len(clean_ds_name) + 2 > 79:
            init_var = "{}\n{}{})".format(init_var, leading_spaces, init_val)
        else:
            init_var = "{} {})".format(init_var, init_val)
    return init_var


def create_basic_init(clean_ds_name):
    return "        self.{} = {}\n".format(clean_ds_name, clean_ds_name)


def create_property(clean_ds_name):
    return "    {} = property(get_{}, set_{})".format(
        clean_ds_name, clean_ds_name, clean_ds_name
    )


def format_var_list(base_string, var_list, is_tuple=False):
    if is_tuple:
        base_string = "{}(".format(base_string)
        extra_chars = 4
    else:
        extra_chars = 2
    line_length = len(base_string)
    leading_spaces = " " * line_length
    # determine if any variable name is too long to fit
    for item in var_list:
        if line_length + len(item) + extra_chars > 80:
            leading_spaces = "        "
            base_string = "{}\n{}".format(base_string, leading_spaces)
            line_length = len(leading_spaces)
            break

    for index, item in enumerate(var_list):
        if is_tuple:
            item = "'{}'".format(item)
        if index == len(var_list) - 1:
            next_var_str = item
        else:
            next_var_str = "{},".format(item)
        line_length += len(item) + extra_chars
        if line_length > 80:
            base_string = "{}\n{}{}".format(
                base_string, leading_spaces, next_var_str
            )
        else:
            if base_string[-1] == ",":
                base_string = "{} ".format(base_string)
            base_string = "{}{}".format(base_string, next_var_str)
    if is_tuple:
        return "{}))".format(base_string)
    else:
        return "{})".format(base_string)


def create_package_init_var(parameter_name, package_abbr, data_name):
    one_line = "        self._{}_package = self.build_child_package(".format(
        package_abbr
    )
    one_line_b = '"{}", {},'.format(package_abbr, parameter_name)
    leading_spaces = " " * len(one_line)
    two_line = '\n{}"{}",'.format(leading_spaces, data_name)
    three_line = "\n{}self._{}_filerecord)".format(
        leading_spaces, package_abbr
    )
    return "{}{}{}{}".format(one_line, one_line_b, two_line, three_line)


def add_var(
    init_vars,
    class_vars,
    init_param_list,
    package_properties,
    doc_string,
    data_structure_dict,
    default_value,
    name,
    python_name,
    description,
    path,
    data_type,
    basic_init=False,
    construct_package=None,
    construct_data=None,
    parameter_name=None,
    set_param_list=None,
):
    if set_param_list is None:
        set_param_list = []
    clean_ds_name = datautil.clean_name(python_name)
    if construct_package is None:
        # add variable initialization lines
        if basic_init:
            init_vars.append(create_basic_init(clean_ds_name))
        else:
            init_vars.append(create_init_var(clean_ds_name, name))
        # add to parameter list
        if default_value is None:
            default_value = "None"
        init_param_list.append("{}={}".format(clean_ds_name, default_value))
        # add to set parameter list
        set_param_list.append("{}={}".format(clean_ds_name, clean_ds_name))
    else:
        clean_parameter_name = datautil.clean_name(parameter_name)
        # init hidden variable
        init_vars.append(
            create_init_var("_{}".format(clean_ds_name), name, "None")
        )
        # init child package
        init_vars.append(
            create_package_init_var(
                clean_parameter_name, construct_package, construct_data
            )
        )
        # add to parameter list
        init_param_list.append("{}=None".format(clean_parameter_name))
        # add to set parameter list
        set_param_list.append(
            "{}={}".format(clean_parameter_name, clean_parameter_name)
        )

    package_properties.append(create_property(clean_ds_name))
    doc_string.add_parameter(description, model_parameter=True)
    data_structure_dict[python_name] = 0
    if class_vars is not None:
        gen_type = generator_type(data_type)
        if gen_type != "ScalarTemplateGenerator":
            new_class_var = "    {} = {}(".format(clean_ds_name, gen_type)
            class_vars.append(format_var_list(new_class_var, path, True))
            return gen_type
    return None


def build_init_string(
    init_string, init_param_list, whitespace="                 "
):
    line_chars = len(init_string)
    for index, param in enumerate(init_param_list):
        if index + 1 < len(init_param_list):
            line_chars += len(param) + 2
        else:
            line_chars += len(param) + 3
        if line_chars > 79:
            if len(param) + len(whitespace) + 1 > 79:
                # try to break apart at = sign
                param_list = param.split("=")
                if len(param_list) == 2:
                    init_string = "{},\n{}{}=\n{}{}".format(
                        init_string,
                        whitespace,
                        param_list[0],
                        whitespace,
                        param_list[1],
                    )
                    line_chars = len(param_list[1]) + len(whitespace) + 1
                    continue
            init_string = "{},\n{}{}".format(init_string, whitespace, param)
            line_chars = len(param) + len(whitespace) + 1
        else:
            init_string = "{}, {}".format(init_string, param)
    return "{}):\n".format(init_string)


def build_model_load(model_type):
    model_load_c = (
        "    Methods\n    -------\n"
        "    load : (simulation : MFSimulationData, model_name : "
        "string,\n        namfile : string, "
        "version : string, exe_name : string,\n        model_ws : "
        "string, strict : boolean) : MFSimulation\n"
        "        a class method that loads a model from files"
        '\n    """'
    )

    model_load = (
        "    @classmethod\n    def load(cls, simulation, structure, "
        "modelname='NewModel',\n             "
        "model_nam_file='modflowtest.nam', version='mf6',\n"
        "             exe_name='mf6.exe', strict=True, "
        "model_rel_path='.',\n"
        "             load_only=None):\n        "
        "return mfmodel.MFModel.load_base(simulation, structure, "
        "modelname,\n                                         "
        "model_nam_file, '{}', version,\n"
        "                                         exe_name, strict, "
        "model_rel_path,\n"
        "                                         load_only)"
        "\n".format(model_type)
    )
    return model_load, model_load_c


def build_model_init_vars(param_list):
    init_var_list = []
    for param in param_list:
        param_parts = param.split("=")
        init_var_list.append(
            "        self.name_file.{}.set_data({}"
            ")".format(param_parts[0], param_parts[0])
        )
    return "\n".join(init_var_list)


def create_packages():
    indent = "    "
    init_string_def = "    def __init__(self"

    # load JSON file
    file_structure = mfstructure.MFStructure(load_from_dfn_files=True)
    sim_struct = file_structure.sim_struct

    # assemble package list of buildable packages
    package_list = []
    package_list.append(
        (
            sim_struct.name_file_struct_obj,
            PackageLevel.sim_level,
            "",
            sim_struct.name_file_struct_obj.dfn_list,
            sim_struct.name_file_struct_obj.file_type,
        )
    )
    for package in sim_struct.package_struct_objs.values():
        # add simulation level package to list
        package_list.append(
            (
                package,
                PackageLevel.sim_level,
                "",
                package.dfn_list,
                package.file_type,
            )
        )
    for package in sim_struct.utl_struct_objs.values():
        # add utility packages to list
        package_list.append(
            (
                package,
                PackageLevel.model_level,
                "utl",
                package.dfn_list,
                package.file_type,
            )
        )
    for model_key, model in sim_struct.model_struct_objs.items():
        package_list.append(
            (
                model.name_file_struct_obj,
                PackageLevel.model_level,
                model_key,
                model.name_file_struct_obj.dfn_list,
                model.name_file_struct_obj.file_type,
            )
        )
        for package in model.package_struct_objs.values():
            package_list.append(
                (
                    package,
                    PackageLevel.model_level,
                    model_key,
                    package.dfn_list,
                    package.file_type,
                )
            )

    util_path, tail = os.path.split(os.path.realpath(__file__))
    init_file = io.open(
        os.path.join(util_path, "..", "modflow", "__init__.py"),
        "w",
        newline="\n",
    )
    init_file.write("# imports\n")
    init_file.write("from .mfsimulation import MFSimulation\n")

    nam_import_string = (
        "from .. import mfmodel\nfrom ..data.mfdatautil "
        "import ListTemplateGenerator, ArrayTemplateGenerator"
    )

    # loop through packages list
    for package in package_list:
        data_structure_dict = {}
        package_properties = []
        init_vars = []
        init_param_list = []
        set_param_list = []
        class_vars = []
        template_gens = []
        dfn_string = build_dfn_string(package[3])
        package_abbr = clean_class_string(
            "{}{}".format(clean_class_string(package[2]), package[0].file_type)
        ).lower()
        package_name = clean_class_string(
            "{}{}{}".format(
                clean_class_string(package[2]),
                package[0].file_prefix,
                package[0].file_type,
            )
        ).lower()
        if package[0].description:
            doc_string = mfdatautil.MFDocString(package[0].description)
        else:
            if package[2]:
                package_container_text = " within a {} model".format(
                    package[2]
                )
            else:
                package_container_text = ""
            ds = "Modflow{} defines a {} package{}.".format(
                package_name.title(),
                package[0].file_type,
                package_container_text,
            )
            if package[0].file_type == "mvr":
                # mvr package warning
                if package[2]:
                    ds = (
                        "{} This package\n    can only be used to move "
                        "water between packages within a single model."
                        "\n    To move water between models use ModflowMvr"
                        ".".format(ds)
                    )
                else:
                    ds = (
                        "{} This package can only be used to move\n    "
                        "water between two different models. To move "
                        "water between two packages\n    in the same "
                        'model use the "model level" mover package (ex. '
                        "ModflowGwfmvr).".format(ds)
                    )

            doc_string = mfdatautil.MFDocString(ds)

        if package[0].dfn_type == mfstructure.DfnType.exch_file:
            add_var(
                init_vars,
                None,
                init_param_list,
                package_properties,
                doc_string,
                data_structure_dict,
                None,
                "exgtype",
                "exgtype",
                build_doc_string(
                    "exgtype",
                    "<string>",
                    "is the exchange type (GWF-GWF or GWF-GWT).",
                    indent,
                ),
                None,
                None,
                True,
            )
            add_var(
                init_vars,
                None,
                init_param_list,
                package_properties,
                doc_string,
                data_structure_dict,
                None,
                "exgmnamea",
                "exgmnamea",
                build_doc_string(
                    "exgmnamea",
                    "<string>",
                    "is the name of the first model that is "
                    "part of this exchange.",
                    indent,
                ),
                None,
                None,
                True,
            )
            add_var(
                init_vars,
                None,
                init_param_list,
                package_properties,
                doc_string,
                data_structure_dict,
                None,
                "exgmnameb",
                "exgmnameb",
                build_doc_string(
                    "exgmnameb",
                    "<string>",
                    "is the name of the second model that is "
                    "part of this exchange.",
                    indent,
                ),
                None,
                None,
                True,
            )
            init_vars.append(
                "        simulation.register_exchange_file(self)\n"
            )

        # loop through all blocks
        for block in package[0].blocks.values():
            for data_structure in block.data_structures.values():
                # only create one property for each unique data structure name
                if data_structure.name not in data_structure_dict:
                    tg = add_var(
                        init_vars,
                        class_vars,
                        init_param_list,
                        package_properties,
                        doc_string,
                        data_structure_dict,
                        data_structure.default_value,
                        data_structure.name,
                        data_structure.python_name,
                        data_structure.get_doc_string(79, indent, indent),
                        data_structure.path,
                        data_structure.get_datatype(),
                        False,
                        data_structure.construct_package,
                        data_structure.construct_data,
                        data_structure.parameter_name,
                        set_param_list,
                    )
                    if tg is not None and tg not in template_gens:
                        template_gens.append(tg)

        import_string = "from .. import mfpackage"
        if template_gens:
            import_string = "{}\nfrom ..data.mfdatautil import ".format(
                import_string
            )
            first_string = True
            for template in template_gens:
                if first_string:
                    import_string = "{}{}".format(import_string, template)
                    first_string = False
                else:
                    import_string = "{}, {}".format(import_string, template)
        # add extra docstrings for additional variables
        doc_string.add_parameter(
            "    filename : String\n        File name for this package."
        )
        doc_string.add_parameter(
            "    pname : String\n        Package name for this package."
        )
        doc_string.add_parameter(
            "    parent_file : MFPackage\n        "
            "Parent package file that references this "
            "package. Only needed for\n        utility "
            "packages (mfutl*). For example, mfutllaktab "
            "package must have \n        a mfgwflak "
            "package parent_file."
        )

        # build package builder class string
        init_vars.append("        self._init_complete = True")
        init_vars = "\n".join(init_vars)
        package_short_name = clean_class_string(package[0].file_type).lower()
        class_def_string = "class Modflow{}(mfpackage.MFPackage):\n".format(
            package_name.title()
        )
        class_def_string = class_def_string.replace("-", "_")
        class_var_string = (
            '{}\n    package_abbr = "{}"\n    _package_type = '
            '"{}"\n    dfn_file_name = "{}"'
            "\n".format(
                "\n".join(class_vars),
                package_abbr,
                package[4],
                package[0].dfn_file_name,
            )
        )
        init_string_full = init_string_def
        init_string_model = "{}, simulation".format(init_string_def)
        # add variables to init string
        doc_string.add_parameter(
            "    loading_package : bool\n        "
            "Do not set this parameter. It is intended "
            "for debugging and internal\n        "
            "processing purposes only.",
            beginning_of_list=True,
        )
        if package[1] == PackageLevel.sim_level:
            doc_string.add_parameter(
                "    simulation : MFSimulation\n        "
                "Simulation that this package is a part "
                "of. Package is automatically\n        "
                "added to simulation when it is "
                "initialized.",
                beginning_of_list=True,
            )
            init_string_full = (
                "{}, simulation, loading_package="
                "False".format(init_string_full)
            )
        else:
            doc_string.add_parameter(
                "    model : MFModel\n        "
                "Model that this package is a part of.  "
                "Package is automatically\n        added "
                "to model when it is initialized.",
                beginning_of_list=True,
            )
            init_string_full = "{}, model, loading_package=False".format(
                init_string_full
            )
        init_param_list.append("filename=None")
        init_param_list.append("pname=None")
        init_param_list.append("parent_file=None")
        init_string_full = build_init_string(init_string_full, init_param_list)

        # build init code
        if package[1] == PackageLevel.sim_level:
            init_var = "simulation"
        else:
            init_var = "model"
        parent_init_string = "        super().__init__("
        spaces = " " * len(parent_init_string)
        parent_init_string = (
            '{}{}, "{}", filename, pname,\n{}'
            "loading_package, parent_file)\n\n"
            "        # set up variables".format(
                parent_init_string, init_var, package_short_name, spaces
            )
        )
        local_datetime = datetime.datetime.now(datetime.timezone.utc)
        comment_string = (
            "# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE "
            "MUST BE CREATED BY\n# mf6/utils/createpackages.py\n"
            "# FILE created on {} UTC".format(
                local_datetime.strftime("%B %d, %Y %H:%M:%S")
            )
        )
        # assemble full package string
        package_string = "{}\n{}\n\n\n{}{}\n{}\n{}\n\n{}{}\n{}\n".format(
            comment_string,
            import_string,
            class_def_string,
            doc_string.get_doc_string(),
            class_var_string,
            dfn_string,
            init_string_full,
            parent_init_string,
            init_vars,
        )

        # open new Packages file
        pb_file = io.open(
            os.path.join(
                util_path, "..", "modflow", "mf{}.py".format(package_name)
            ),
            "w",
            newline="\n",
        )
        pb_file.write(package_string)

        if package[2] == "utl" and package_abbr != "utltab":
            set_param_list.append("filename=filename")
            set_param_list.append("pname=pname")
            set_param_list.append("parent_file=self._cpparent")
            whsp_1 = "                   "
            whsp_2 = "                                    "

            chld_doc_string = (
                '    """\n    Utl{}Packages is a container '
                "class for the ModflowUtl{} class.\n\n    "
                "Methods\n    ----------"
                "\n".format(package_short_name, package_short_name)
            )

            # write out child packages class
            chld_cls = (
                "\n\nclass Utl{}Packages(mfpackage.MFChildPackage"
                "s):\n".format(package_short_name)
            )
            chld_var = '    package_abbr = "utl{}packages"\n\n'.format(
                package_short_name
            )
            chld_init = "    def initialize(self"
            chld_init = build_init_string(
                chld_init, init_param_list[:-1], whsp_1
            )
            init_pkg = "\n        self._init_package(new_package, filename)"
            params_init = (
                "        new_package = ModflowUtl{}("
                "self._model".format(package_short_name)
            )
            params_init = build_init_string(
                params_init, set_param_list, whsp_2
            )
            chld_doc_string = (
                "{}    initialize\n        Initializes a new "
                "ModflowUtl{} package removing any sibling "
                "child\n        packages attached to the same "
                "parent package. See ModflowUtl{} init\n "
                "       documentation for definition of "
                "parameters.\n".format(
                    chld_doc_string, package_short_name, package_short_name
                )
            )

            chld_appn = ""
            params_appn = ""
            append_pkg = ""
            if package_abbr != "utlobs":  # Hard coded obs no multi-pkg support
                chld_appn = "\n\n    def append_package(self"
                chld_appn = build_init_string(
                    chld_appn, init_param_list[:-1], whsp_1
                )
                append_pkg = (
                    "\n        self._append_package(new_package, filename)"
                )
                params_appn = (
                    "        new_package = ModflowUtl{}("
                    "self._model".format(package_short_name)
                )
                params_appn = build_init_string(
                    params_appn, set_param_list, whsp_2
                )
                chld_doc_string = (
                    "{}    append_package\n        Adds a "
                    "new ModflowUtl{} package to the container."
                    " See ModflowUtl{}\n        init "
                    "documentation for definition of "
                    "parameters.\n".format(
                        chld_doc_string, package_short_name, package_short_name
                    )
                )
            chld_doc_string = '{}    """\n'.format(chld_doc_string)
            packages_str = "{}{}{}{}{}{}{}{}{}\n".format(
                chld_cls,
                chld_doc_string,
                chld_var,
                chld_init,
                params_init[:-2],
                init_pkg,
                chld_appn,
                params_appn[:-2],
                append_pkg,
            )
            pb_file.write(packages_str)
        pb_file.close()

        init_file.write(
            "from .mf{} import "
            "Modflow{}\n".format(package_name, package_name.title())
        )

        if package[0].dfn_type == mfstructure.DfnType.model_name_file:
            # build model file
            model_param_list = init_param_list[:-3]
            init_vars = build_model_init_vars(model_param_list)

            model_param_list.insert(0, "model_rel_path='.'")
            model_param_list.insert(0, "exe_name='mf6.exe'")
            model_param_list.insert(0, "version='mf6'")
            model_param_list.insert(0, "model_nam_file=None")
            model_param_list.insert(0, "modelname='model'")
            model_param_list.append("**kwargs")
            init_string_model = build_init_string(
                init_string_model, model_param_list
            )
            model_name = clean_class_string(package[2])
            class_def_string = "class Modflow{}(mfmodel.MFModel):\n".format(
                model_name.capitalize()
            )
            class_def_string = class_def_string.replace("-", "_")
            doc_string.add_parameter(
                "    sim : MFSimulation\n        "
                "Simulation that this model is a part "
                "of.  Model is automatically\n        "
                "added to simulation when it is "
                "initialized.",
                beginning_of_list=True,
                model_parameter=True,
            )
            doc_string.description = "Modflow{} defines a {} model".format(
                model_name, model_name
            )
            class_var_string = "    model_type = '{}'\n".format(model_name)
            mparent_init_string = "        super().__init__("
            spaces = " " * len(mparent_init_string)
            mparent_init_string = (
                "{}simulation, model_type='{}6',\n{}"
                "modelname=modelname,\n{}"
                "model_nam_file=model_nam_file,\n{}"
                "version=version, exe_name=exe_name,\n{}"
                "model_rel_path=model_rel_path,\n{}"
                "**kwargs"
                ")\n".format(
                    mparent_init_string,
                    model_name,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                )
            )
            load_txt, doc_text = build_model_load(model_name)
            package_string = "{}\n{}\n\n\n{}{}\n{}\n{}\n{}{}\n{}\n\n{}".format(
                comment_string,
                nam_import_string,
                class_def_string,
                doc_string.get_doc_string(True),
                doc_text,
                class_var_string,
                init_string_model,
                mparent_init_string,
                init_vars,
                load_txt,
            )
            md_file = io.open(
                os.path.join(
                    util_path, "..", "modflow", "mf{}.py".format(model_name)
                ),
                "w",
                newline="\n",
            )
            md_file.write(package_string)
            md_file.close()
            init_file.write(
                "from .mf{} import "
                "Modflow{}\n".format(model_name, model_name.capitalize())
            )
    init_file.close()


if __name__ == "__main__":
    create_packages()
