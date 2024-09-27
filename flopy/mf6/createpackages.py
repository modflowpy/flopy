"""
createpackages.py is a utility script that reads in the file definition
metadata in the .dfn files and creates the package classes in the modflow
folder. Run this script any time changes are made to the .dfn files.

To create a new package that is part of an existing model, first create a new
dfn file for the package in the mf6/data/dfn folder.
1) Follow the file naming convention <model abbr>-<package abbr>.dfn.
2) Run this script (createpackages.py), and check in your new dfn file, and
   the package class and updated __init__.py that createpackages.py created.

A subpackage is a package referenced by another package (vs being referenced
in the name file).  The tas, ts, and obs packages are examples of subpackages.
There are a few additional steps required when creating a subpackage
definition file.  First, verify that the parent package's dfn file has a file
record for the subpackage to the option block.   For example, for the time
series package the file record definition starts with:

    block options
    name ts_filerecord
    type record ts6 filein ts6_filename

Verify that the same naming convention is followed as the example above,
specifically:

    name <subpackage-abbr>_filerecord
    record <subpackage-abbr>6 filein <subpackage-abbr>6_filename

Next, create the child package definition file in the mf6/data/dfn folder
following the naming convention above.

When your child package is ready for release follow the same procedure as
other packages along with these a few additional steps required for
subpackages.

At the top of the child dfn file add two lines describing how the parent and
child packages are related. The first line determines how the subpackage is
linked to the package:

# flopy subpackage <parent record> <abbreviation> <child data>
<data name>

* Parent record is the MF6 record name of the filerecord in parent package
  that references the child packages file name
* Abbreviation is the short abbreviation of the new subclass
* Child data is the name of the child class data that can be passed in as
  parameter to the parent class. Passing in this parameter to the parent class
  automatically creates the child class with the data provided.
* Data name is the parent class parameter name that automatically creates the
  child class with the data provided.

The example below is the first line from the ts subpackage dfn:

# flopy subpackage ts_filerecord ts timeseries timeseries

The second line determines the variable name of the subpackage's parent and
the type of parent (the parent package's object oriented parent):

# flopy parent_name_type <parent package variable name>
<parent package type>

An example below is the second line in the ts subpackage dfn:

# flopy parent_name_type parent_package MFPackage

There are three possible types (or combination of them) that can be used for
"parent package type", MFPackage, MFModel, and MFSimulation. If a package
supports multiple types of parents (for example, it can be either in the model
namefile or in a package, like the obs package), include all the types
supported, separating each type with a / (MFPackage/MFModel).

To create a new type of model choose a unique three letter model abbreviation
("gwf", "gwt", ...). Create a name file dfn with the naming convention
<model abbr>-nam.dfn. The name file must have only an options and packages
block (see gwf-nam.dfn as an example). Create a new dfn file for each of the
packages in your new model, following the naming convention described above.

When your model is ready for release make sure all the dfn files are in the
flopy/mf6/data/dfn folder, run createpackages.py, and check in your new dfn
files, the package classes, and updated init.py that createpackages.py created.

"""

import collections
import datetime
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from enum import Enum
from keyword import kwlist
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    ForwardRef,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
    get_args,
    get_origin,
)
from warnings import warn

import numpy as np
from jinja2 import Environment, PackageLoader
from modflow_devtools.misc import run_cmd
from numpy.typing import ArrayLike, NDArray

from flopy.mf6 import MFSimulation

# keep below as absolute imports
from flopy.mf6.data import mfdatautil, mfstructure
from flopy.mf6.mfmodel import MFModel
from flopy.mf6.mfpackage import MFPackage
from flopy.mf6.mfsimbase import MFSimulationBase
from flopy.utils import datautil


class PackageLevel(Enum):
    sim_level = 0
    model_level = 1


def build_doc_string(param_name, param_type, param_desc, indent):
    return f"{indent}{param_name} : {param_type}\n{indent * 2}* {param_desc}"


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


def build_dfn_string(dfn_list, header, package_abbr, flopy_dict):
    dfn_string = "    dfn = ["
    line_length = len(dfn_string)
    leading_spaces = " " * line_length
    first_di = True

    # process header
    dfn_string = f'{dfn_string}\n{leading_spaces}["header", '
    for key, value in header.items():
        if key == "multi-package":
            dfn_string = f'{dfn_string}\n{leading_spaces} "multi-package", '
        if key == "package-type":
            dfn_string = (
                f'{dfn_string}\n{leading_spaces} "package-type ' f'{value}"'
            )

    # process solution packages
    if package_abbr in flopy_dict["solution_packages"]:
        model_types = '", "'.join(
            flopy_dict["solution_packages"][package_abbr]
        )
        dfn_string = (
            f"{dfn_string}\n{leading_spaces} "
            f'["solution_package", "{model_types}"], '
        )
    dfn_string = f"{dfn_string}],\n{leading_spaces}"

    # process all data items
    for data_item in dfn_list:
        line_length += 1
        if not first_di:
            dfn_string = f"{dfn_string},\n{leading_spaces}"
            line_length = len(leading_spaces)
        else:
            first_di = False
        dfn_string = f"{dfn_string}["
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
                    dfn_string = f"{dfn_string},"
                if line_length < 77:
                    # added text fits on the current line
                    if first_line:
                        dfn_string = f'{dfn_string}"{line}"'
                    else:
                        dfn_string = f'{dfn_string} "{line}"'
                else:
                    # added text does not fit on the current line
                    line_length = len(line) + len(leading_spaces) + 2
                    if line_length > 79:
                        # added text too long to fit on a single line, wrap
                        # text as needed
                        line = f'"{line}"'
                        lines = textwrap.wrap(
                            line,
                            75 - len(leading_spaces),
                            drop_whitespace=True,
                        )
                        lines[0] = f"{leading_spaces} {lines[0]}"
                        line_join = f' "\n{leading_spaces} "'
                        dfn_string = f"{dfn_string}\n{line_join.join(lines)}"
                    else:
                        dfn_string = f'{dfn_string}\n{leading_spaces} "{line}"'
            first_line = False

        dfn_string = f"{dfn_string}]"
    dfn_string = f"{dfn_string}]"
    return dfn_string


def create_init_var(clean_ds_name, data_structure_name, init_val=None):
    if init_val is None:
        init_val = clean_ds_name

    init_var = f"        self.{clean_ds_name} = self.build_mfdata("
    leading_spaces = " " * len(init_var)
    if len(init_var) + len(data_structure_name) + 2 > 79:
        second_line = f'\n            "{data_structure_name}",'
        if len(second_line) + len(clean_ds_name) + 2 > 79:
            init_var = f"{init_var}{second_line}\n            {init_val})"
        else:
            init_var = f"{init_var}{second_line} {init_val})"
    else:
        init_var = f'{init_var}"{data_structure_name}",'
        if len(init_var) + len(clean_ds_name) + 2 > 79:
            init_var = f"{init_var}\n{leading_spaces}{init_val})"
        else:
            init_var = f"{init_var} {init_val})"
    return init_var


def create_basic_init(clean_ds_name):
    return f"        self.{clean_ds_name} = {clean_ds_name}\n"


def create_property(clean_ds_name):
    return f"    {clean_ds_name} = property(get_{clean_ds_name}, set_{clean_ds_name})"


def format_var_list(base_string, var_list, is_tuple=False):
    if is_tuple:
        base_string = f"{base_string}("
        extra_chars = 4
    else:
        extra_chars = 2
    line_length = len(base_string)
    leading_spaces = " " * line_length
    # determine if any variable name is too long to fit
    for item in var_list:
        if line_length + len(item) + extra_chars > 80:
            leading_spaces = "        "
            base_string = f"{base_string}\n{leading_spaces}"
            line_length = len(leading_spaces)
            break

    for index, item in enumerate(var_list):
        if is_tuple:
            item = f"'{item}'"
        if index == len(var_list) - 1:
            next_var_str = item
        else:
            next_var_str = f"{item},"
        line_length += len(item) + extra_chars
        if line_length > 80:
            base_string = f"{base_string}\n{leading_spaces}{next_var_str}"
        else:
            if base_string[-1] == ",":
                base_string = f"{base_string} "
            base_string = f"{base_string}{next_var_str}"
    if is_tuple:
        return f"{base_string}))"
    else:
        return f"{base_string})"


def create_package_init_var(
    parameter_name, package_abbr, data_name, clean_ds_name
):
    one_line = (
        f"        self._{package_abbr}_package = self.build_child_package("
    )
    one_line_b = f'"{package_abbr}", {parameter_name},'
    leading_spaces = " " * len(one_line)
    two_line = f'\n{leading_spaces}"{data_name}",'
    three_line = f"\n{leading_spaces}self._{clean_ds_name})"
    return f"{one_line}{one_line_b}{two_line}{three_line}"


def add_var(
    init_vars,
    class_vars,
    options_param_list,
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
    mf_nam=False,
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
        init_param_list.append(f"{clean_ds_name}={default_value}")
        if path is not None and "options" in path:
            options_param_list.append(f"{clean_ds_name}={default_value}")
        # add to set parameter list
        set_param_list.append(f"{clean_ds_name}={clean_ds_name}")
    else:
        clean_parameter_name = datautil.clean_name(parameter_name)
        # init hidden variable
        init_vars.append(create_init_var(f"_{clean_ds_name}", name, "None"))
        if mf_nam:
            options_param_list.append(
                [f"{parameter_name}_data=None", parameter_name]
            )
        else:
            # init child package
            init_vars.append(
                create_package_init_var(
                    clean_parameter_name,
                    construct_package,
                    construct_data,
                    clean_ds_name,
                )
            )
            # add to parameter list
            init_param_list.append(f"{clean_parameter_name}=None")
            # add to set parameter list
            set_param_list.append(
                f"{clean_parameter_name}={clean_parameter_name}"
            )

    package_properties.append(create_property(clean_ds_name))
    doc_string.add_parameter(description, model_parameter=True)
    data_structure_dict[python_name] = 0
    if class_vars is not None:
        gen_type = generator_type(data_type)
        if gen_type != "ScalarTemplateGenerator":
            new_class_var = f"    {clean_ds_name} = {gen_type}("
            class_vars.append(format_var_list(new_class_var, path, True))
            return gen_type
    return None


def build_init_string(
    init_string, init_param_list, whitespace="                 "
):
    line_chars = len(init_string)
    for index, param in enumerate(init_param_list):
        if isinstance(param, list):
            param = param[0]
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
            init_string = f"{init_string},\n{whitespace}{param}"
            line_chars = len(param) + len(whitespace) + 1
        else:
            init_string = f"{init_string}, {param}"
    return f"{init_string}):\n"


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
        "             exe_name='mf6', strict=True, "
        "model_rel_path='.',\n"
        "             load_only=None):\n        "
        "return mfmodel.MFModel.load_base(cls, simulation, structure, "
        "modelname,\n                                         "
        "model_nam_file, '{}6', version,\n"
        "                                         exe_name, strict, "
        "model_rel_path,\n"
        "                                         load_only)"
        "\n".format(model_type)
    )
    return model_load, model_load_c


def build_sim_load():
    sim_load_c = (
        "    Methods\n    -------\n"
        "    load : (sim_name : str, version : "
        "string,\n        exe_name : str or PathLike, "
        "sim_ws : str or PathLike, strict : bool,\n        verbosity_level : "
        "int, load_only : list, verify_data : bool,\n        "
        "write_headers : bool, lazy_io : bool, use_pandas : bool,\n        "
        ") : MFSimulation\n"
        "        a class method that loads a simulation from files"
        '\n    """'
    )

    sim_load = (
        "    @classmethod\n    def load(cls, sim_name='modflowsim', "
        "version='mf6',\n             "
        "exe_name: Union[str, os.PathLike] = 'mf6',\n             "
        "sim_ws: Union[str, os.PathLike] = os.curdir,\n             "
        "strict=True, verbosity_level=1, load_only=None,\n             "
        "verify_data=False, write_headers=True,\n             "
        "lazy_io=False, use_pandas=True):\n        "
        "return mfsimbase.MFSimulationBase.load(cls, sim_name, version, "
        "\n                                               "
        "exe_name, sim_ws, strict,\n"
        "                                               verbosity_level, "
        "load_only,\n                                               "
        "verify_data, write_headers, "
        "\n                                               lazy_io, use_pandas)"
        "\n"
    )
    return sim_load, sim_load_c


def build_model_init_vars(param_list):
    init_var_list = []
    # build set data calls
    for param in param_list:
        if not isinstance(param, list):
            param_parts = param.split("=")
            init_var_list.append(
                f"        self.name_file.{param_parts[0]}.set_data({param_parts[0]})"
            )
    init_var_list.append("")
    # build attributes
    for param in param_list:
        if isinstance(param, list):
            pkg_name = param[1]
            param_parts = param[0].split("=")
            init_var_list.append(
                f"        self.{param_parts[0]} = "
                f"self._create_package('{pkg_name}', {param_parts[0]})"
            )
        else:
            param_parts = param.split("=")
            init_var_list.append(
                f"        self.{param_parts[0]} = self.name_file.{param_parts[0]}"
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
    for package in sim_struct.utl_struct_objs.values():
        # add utility packages to list
        package_list.append(
            (
                package,
                PackageLevel.model_level,
                "utl",
                package.dfn_list,
                package.file_type,
                package.header,
            )
        )
    package_list.append(
        (
            sim_struct.name_file_struct_obj,
            PackageLevel.sim_level,
            "",
            sim_struct.name_file_struct_obj.dfn_list,
            sim_struct.name_file_struct_obj.file_type,
            sim_struct.name_file_struct_obj.header,
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
                package.header,
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
                model.name_file_struct_obj.header,
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
                    package.header,
                )
            )

    util_path, tail = os.path.split(os.path.realpath(__file__))
    init_file = open(
        os.path.join(util_path, "..", "modflow", "__init__.py"),
        "w",
        newline="\n",
    )
    init_file.write("from .mfsimulation import MFSimulation  # isort:skip\n")

    nam_import_string = (
        "from .. import mfmodel\nfrom ..data.mfdatautil "
        "import ArrayTemplateGenerator, ListTemplateGenerator"
    )

    # loop through packages list
    init_file_imports = []
    flopy_dict = file_structure.flopy_dict
    for package in package_list:
        data_structure_dict = {}
        package_properties = []
        init_vars = []
        init_param_list = []
        options_param_list = []
        set_param_list = []
        class_vars = []
        template_gens = []

        package_abbr = clean_class_string(
            f"{clean_class_string(package[2])}{package[0].file_type}"
        ).lower()
        dfn_string = build_dfn_string(
            package[3], package[5], package_abbr, flopy_dict
        )
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
                package_container_text = f" within a {package[2]} model"
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
            exgtype = (
                f'"{package_abbr[0:3].upper()}6-{package_abbr[3:].upper()}6"'
            )

            add_var(
                init_vars,
                None,
                options_param_list,
                init_param_list,
                package_properties,
                doc_string,
                data_structure_dict,
                exgtype,
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
                options_param_list,
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
                options_param_list,
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
                    mf_sim = (
                        "parent_name_type" in package[0].header
                        and package[0].header["parent_name_type"][1]
                        == "MFSimulation"
                    )
                    mf_nam = package[0].file_type == "nam"
                    if (
                        data_structure.construct_package is not None
                        and not mf_sim
                        and not mf_nam
                    ):
                        c_pkg = data_structure.construct_package
                    else:
                        c_pkg = None
                    tg = add_var(
                        init_vars,
                        class_vars,
                        options_param_list,
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
                        # c_pkg,
                        data_structure.construct_package,
                        data_structure.construct_data,
                        data_structure.parameter_name,
                        set_param_list,
                        mf_nam,
                    )
                    if tg is not None and tg not in template_gens:
                        template_gens.append(tg)

        import_string = "from .. import mfpackage"
        if template_gens:
            import_string += "\nfrom ..data.mfdatautil import "
            import_string += ", ".join(sorted(template_gens))
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
        init_string_sim = f"{init_string_def}, simulation"
        # add variables to init string
        doc_string.add_parameter(
            "    loading_package : bool\n        "
            "Do not set this parameter. It is intended "
            "for debugging and internal\n        "
            "processing purposes only.",
            beginning_of_list=True,
        )
        if "parent_name_type" in package[0].header:
            init_var = package[0].header["parent_name_type"][0]
            parent_type = package[0].header["parent_name_type"][1]
        elif package[1] == PackageLevel.sim_level:
            init_var = "simulation"
            parent_type = "MFSimulation"
        else:
            init_var = "model"
            parent_type = "MFModel"
        doc_string.add_parameter(
            f"    {init_var} : {parent_type}\n        "
            f"{init_var.capitalize()} that this package is a part "
            "of. Package is automatically\n        "
            f"added to {init_var} when it is "
            "initialized.",
            beginning_of_list=True,
        )
        init_string_full = (
            f"{init_string_full}, {init_var}, loading_package=False"
        )
        init_param_list.append("filename=None")
        init_param_list.append("pname=None")
        init_param_list.append("**kwargs")
        init_string_full = build_init_string(init_string_full, init_param_list)

        # build init code
        parent_init_string = "        super().__init__("
        spaces = " " * len(parent_init_string)
        parent_init_string = (
            '{}{}, "{}", filename, pname,\n{}'
            "loading_package, **kwargs)\n\n"
            "        # set up variables".format(
                parent_init_string, init_var, package_short_name, spaces
            )
        )
        local_datetime = datetime.datetime.now(datetime.timezone.utc)
        comment_string = (
            "# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE "
            "MUST BE CREATED BY\n# mf6/createpackages.py\n"
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
        pb_file = open(
            os.path.join(util_path, "..", "modflow", f"mf{package_name}.py"),
            "w",
            newline="\n",
        )
        pb_file.write(package_string)
        if (
            package[0].sub_package
            and package_abbr != "utltab"
            and (
                "parent_name_type" not in package[0].header
                or package[0].header["parent_name_type"][1] != "MFSimulation"
            )
        ):
            set_param_list.append("filename=filename")
            set_param_list.append("pname=pname")
            set_param_list.append("child_builder_call=True")
            whsp_1 = "                   "
            whsp_2 = "                                    "

            file_prefix = package[0].dfn_file_name[0:3]
            chld_doc_string = (
                '    """\n    {}Packages is a container '
                "class for the Modflow{} class.\n\n    "
                "Methods\n    ----------"
                "\n".format(package_name.title(), package_name.title())
            )

            # write out child packages class
            chld_cls = (
                "\n\nclass {}Packages(mfpackage.MFChildPackage" "s):\n".format(
                    package_name.title()
                )
            )
            chld_var = (
                f"    package_abbr = "
                f'"{package_name.title().lower()}packages"\n\n'
            )
            chld_init = "    def initialize(self"
            chld_init = build_init_string(
                chld_init, init_param_list[:-1], whsp_1
            )
            init_pkg = "\n        self.init_package(new_package, filename)"
            params_init = (
                "        new_package = Modflow"
                f"{package_name.title()}(self._cpparent"
            )
            params_init = build_init_string(
                params_init, set_param_list, whsp_2
            )
            chld_doc_string = (
                "{}    initialize\n        Initializes a new "
                "Modflow{} package removing any sibling "
                "child\n        packages attached to the same "
                "parent package. See Modflow{} init\n "
                "       documentation for definition of "
                "parameters.\n".format(
                    chld_doc_string, package_name.title(), package_name.title()
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
                    "        new_package = Modflow"
                    f"{file_prefix.capitalize()}"
                    f"{package_short_name}(self._cpparent"
                )
                params_appn = build_init_string(
                    params_appn, set_param_list, whsp_2
                )
                chld_doc_string = (
                    "{}    append_package\n        Adds a "
                    "new Modflow{}{} package to the container."
                    " See Modflow{}{}\n        init "
                    "documentation for definition of "
                    "parameters.\n".format(
                        chld_doc_string,
                        file_prefix.capitalize(),
                        package_short_name,
                        file_prefix.capitalize(),
                        package_short_name,
                    )
                )
            chld_doc_string = f'{chld_doc_string}    """\n'
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

        init_file_imports.append(
            f"from .mf{package_name} import Modflow{package_name.title()}\n"
        )

        if package[0].dfn_type == mfstructure.DfnType.model_name_file:
            # build model file
            init_vars = build_model_init_vars(options_param_list)

            options_param_list.insert(0, "model_rel_path='.'")
            options_param_list.insert(0, "exe_name='mf6'")
            options_param_list.insert(0, "version='mf6'")
            options_param_list.insert(0, "model_nam_file=None")
            options_param_list.insert(0, "modelname='model'")
            options_param_list.append("**kwargs,")
            init_string_sim = build_init_string(
                init_string_sim, options_param_list
            )
            sim_name = clean_class_string(package[2])
            class_def_string = "class Modflow{}(mfmodel.MFModel):\n".format(
                sim_name.capitalize()
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
            doc_string.description = (
                f"Modflow{sim_name} defines a {sim_name} model"
            )
            class_var_string = f"    model_type = '{sim_name}'\n"
            mparent_init_string = "        super().__init__("
            spaces = " " * len(mparent_init_string)
            mparent_init_string = (
                "{}simulation, model_type='{}6',\n{}"
                "modelname=modelname,\n{}"
                "model_nam_file=model_nam_file,\n{}"
                "version=version, exe_name=exe_name,\n{}"
                "model_rel_path=model_rel_path,\n{}"
                "**kwargs,"
                ")\n".format(
                    mparent_init_string,
                    sim_name,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                )
            )
            load_txt, doc_text = build_model_load(sim_name)
            package_string = "{}\n{}\n\n\n{}{}\n{}\n{}\n{}{}\n{}\n\n{}".format(
                comment_string,
                nam_import_string,
                class_def_string,
                doc_string.get_doc_string(True),
                doc_text,
                class_var_string,
                init_string_sim,
                mparent_init_string,
                init_vars,
                load_txt,
            )
            md_file = open(
                os.path.join(util_path, "..", "modflow", f"mf{sim_name}.py"),
                "w",
                newline="\n",
            )
            md_file.write(package_string)
            md_file.close()
            init_file_imports.append(
                f"from .mf{sim_name} import Modflow{sim_name.capitalize()}\n"
            )
        elif package[0].dfn_type == mfstructure.DfnType.sim_name_file:
            # build simulation file
            init_vars = build_model_init_vars(options_param_list)

            options_param_list.insert(0, "lazy_io=False")
            options_param_list.insert(0, "use_pandas=True")
            options_param_list.insert(0, "write_headers=True")
            options_param_list.insert(0, "verbosity_level=1")
            options_param_list.insert(
                0, "sim_ws: Union[str, os.PathLike] = " "os.curdir"
            )
            options_param_list.insert(
                0, "exe_name: Union[str, os.PathLike] " '= "mf6"'
            )
            options_param_list.insert(0, "version='mf6'")
            options_param_list.insert(0, "sim_name='sim'")
            init_string_sim = "    def __init__(self"
            init_string_sim = build_init_string(
                init_string_sim, options_param_list
            )
            class_def_string = (
                "class MFSimulation(mfsimbase." "MFSimulationBase):\n"
            )
            doc_string.add_parameter(
                "    sim_name : str\n" "       Name of the simulation",
                beginning_of_list=True,
                model_parameter=True,
            )
            doc_string.description = (
                "MFSimulation is used to load, build, and/or save a MODFLOW "
                "6 simulation. \n    A MFSimulation object must be created "
                "before creating any of the MODFLOW 6 \n    model objects."
            )
            sparent_init_string = "        super().__init__("
            spaces = " " * len(sparent_init_string)
            sparent_init_string = (
                "{}sim_name=sim_name,\n{}"
                "version=version,\n{}"
                "exe_name=exe_name,\n{}"
                "sim_ws=sim_ws,\n{}"
                "verbosity_level=verbosity_level,\n{}"
                "write_headers=write_headers,\n{}"
                "lazy_io=lazy_io,\n{}"
                "use_pandas=use_pandas,\n{}"
                ")\n".format(
                    sparent_init_string,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                    spaces,
                )
            )
            sim_import_string = (
                "import os\n"
                "from typing import Union\n"
                "from .. import mfsimbase"
            )

            load_txt, doc_text = build_sim_load()
            package_string = "{}\n{}\n\n\n{}{}\n{}\n{}{}\n{}\n\n{}".format(
                comment_string,
                sim_import_string,
                class_def_string,
                doc_string.get_doc_string(False, True),
                doc_text,
                init_string_sim,
                sparent_init_string,
                init_vars,
                load_txt,
            )
            sim_file = open(
                os.path.join(util_path, "..", "modflow", "mfsimulation.py"),
                "w",
                newline="\n",
            )
            sim_file.write(package_string)
            sim_file.close()
            init_file_imports.append(
                "from .mfsimulation import MFSimulation\n"
            )

    # Sort the imports
    for line in sorted(init_file_imports, key=lambda x: x.split()[3]):
        init_file.write(line)
    init_file.close()


# ------------------------------------------


DFN_PATH = Path(__file__).parent / "data" / "dfn"
TGT_PATH = Path(__file__).parent / "modflow"
SCALAR_TYPES = {
    "keyword": bool,
    "integer": int,
    "double precision": float,
    "string": str,
}
NP_SCALAR_TYPES = {
    "keyword": np.bool_,
    "integer": np.int_,
    "double precision": np.float64,
    "string": np.str_,
}
TEMPLATE_ENV = Environment(loader=PackageLoader("flopy", "mf6/templates/"))


Scalar = Union[bool, int, float, str]
Variable = Dict[str, Scalar]
Variables = Dict[str, Variable]
Metadata = List[str]
Definition = Tuple[Variables, Metadata]
Definitions = Dict[str, Definition]


def load_dfn(f) -> Definition:
    """
    Load an input definition file. Returns a tuple containing
    a dictionary variables and a list of metadata attributes.
    """
    meta = list()
    vars = dict()
    var = dict()

    for line in f:
        # remove whitespace/etc from the line
        line = line.strip()

        # record flopy metadata attributes but
        # skip all other comment lines
        if line.startswith("#"):
            _, sep, tail = line.partition("flopy")
            if sep == "flopy":
                meta.append(tail.strip())
            continue

        # if we hit a newline and the parameter dict
        # is nonempty, we've reached the end of its
        # block of attributes
        if not any(line):
            if any(var):
                vars[var["name"]] = var
                var = dict()
            continue

        # split the attribute's key and value and
        # store it in the parameter dictionary
        key, _, value = line.partition(" ")
        var[key] = value

    # add the final parameter
    if any(var):
        vars[var["name"]] = var

    return vars, meta


class DefinitionName(NamedTuple):
    l: str
    r: str

    @property
    def title(self) -> str:
        l, r = self
        if l == "sln":
            return r
        elif l == "exg":
            return r
        return f"{l}{r}"

    @property
    def parent(self) -> type:
        l, r = self
        if l in ["exg", "sln"]:
            return MFSimulation
        if l in ["utl"]:
            return MFPackage
        return MFModel

    @property
    def parent_kind(self) -> str:
        return self.parent.__name__.lower().replace("mf", "")

    @property
    def description(self) -> str:
        l, r = self
        title = self.title.title()
        parent = self.parent
        parent_kind = self.parent_kind
        context = ContextType.from_dfn_name(self)
        if context == ContextType.Package:
            return (
                f"Modflow{title} defines a {r} package "
                f"within a {l} {parent_kind}."
            )
        elif context == ContextType.Model:
            return f"Modflow{title} defines a {l.upper()} model."
        elif context == ContextType.Simulation:
            return """
    MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation.
    A MFSimulation object must be created before creating any of the MODFLOW 6
    model objects."""

    @property
    def target(self) -> str:
        l, r = self

        def _name():
            if l == "sim":
                if r == "nam":
                    return "simulation"
                else:
                    return r
            elif l == "sln":
                return r
            elif l == "exg":
                return r
            return f"{l}{r}"

        return f"mf{_name()}.py"


class ContextType(Enum):
    Model = "model"
    Package = "package"
    Simulation = "simulation"

    @property
    def base(self):
        if self == ContextType.Model:
            return MFModel
        elif self == ContextType.Package:
            return MFPackage
        elif self == ContextType.Simulation:
            return MFSimulationBase

    @classmethod
    def from_dfn_name(cls, dfn_name: DefinitionName) -> "ContextType":
        l, r = dfn_name
        if l == "sim" and r == "nam":
            return ContextType.Simulation
        elif r == "nam":
            return ContextType.Model
        else:
            return ContextType.Package


def _try_qualify(t) -> str:
    """Convert a type to a name suitable for templating."""
    origin = get_origin(t)
    args = get_args(t)
    if origin is Literal:
        args = ['"' + a + '"' for a in args]
        return f"{Literal.__name__}[{', '.join(args)}]"
    elif origin is Union:
        if len(args) == 2 and args[1] is type(None):
            return f"{Optional.__name__}[{_try_qualify(args[0])}]"
        return (
            f"{Union.__name__}[{', '.join([_try_qualify(a) for a in args])}]"
        )
    elif origin is tuple:
        return (
            f"{Tuple.__name__}[{', '.join([_try_qualify(a) for a in args])}]"
        )
    elif origin is collections.abc.Iterable:
        return f"{Iterable.__name__}[{', '.join([_try_qualify(a) for a in args])}]"
    elif origin is list:
        return f"{List.__name__}[{', '.join([_try_qualify(a) for a in args])}]"
    elif origin is np.ndarray:
        return f"NDArray[np.{_try_qualify(args[1].__args__[0])}]"
    elif origin is np.dtype:
        return str(t)
    elif isinstance(t, ForwardRef):
        return t.__forward_arg__
    elif t is Ellipsis:
        return "..."
    elif isinstance(t, type):
        return t.__qualname__
    return t


def _try_enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def _try_de_underscore(k: str) -> str:
    return k[1:] if k.startswith("_") else k


def _render(d: dict) -> dict:
    """
    Recursively transform a dictionary for rendering in a template.
    """

    def _render_key(k):
        return _try_de_underscore(k)

    def _render_val(v):
        return (
            _render(v)
            if isinstance(v, dict)
            else _try_qualify(_try_enum_value(v))
        )

    # drop nones, except for default
    _d = {
        _render_key(k): _render_val(v)
        for k, v in d.items()
        if (k == "default" or v is not None)
    }

    # wrap default in quotation marks if it's a string
    default = _d.get("default", None)
    if default is not None and isinstance(default, str):
        _d["default"] = f'"{default}"'

    return _d


class Renderable:
    def render(self) -> dict:
        """
        Recursively transform the dataclass instance for rendering.
        """
        return _render(asdict(self))


@dataclass(frozen=True)
class Subpackage(Renderable):
    parent: type
    abbreviation: str
    data_name: str
    param_name: str
    record_name: str


class ParameterKind(Enum):
    Array = "array"
    Scalar = "scalar"
    Record = "record"
    Union = "union"
    List = "list"

    @classmethod
    def from_type(cls, t: type) -> Optional["ParameterKind"]:
        origin = get_origin(t)
        try:
            if t is np.ndarray or origin is NDArray or origin is ArrayLike:
                return ParameterKind.Array
            elif origin is collections.abc.Iterable or origin is list:
                return ParameterKind.List
            elif origin is tuple:
                return ParameterKind.Record
            elif origin is Union:
                return ParameterKind.Union
            elif issubclass(t, Scalar):
                return ParameterKind.Scalar
            return None
        except:
            pass


@dataclass
class Parameter(Renderable):
    name: str
    _type: type
    description: Optional[str]
    default: Optional[Any]
    children: Optional[Dict[str, "Parameter"]]
    subpackage: Optional[Subpackage]
    kind: Optional[ParameterKind]
    is_choice: bool = False
    set_self: bool = False

    def __init__(
        self,
        name: str,
        _type: type,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        parent: Optional["Parameter"] = None,
        children: Optional[Dict[str, "Parameter"]] = None,
        subpackage: Optional[Subpackage] = None,
        kind: Optional[ParameterKind] = None,
        is_choice: bool = False,
        set_self: bool = False,
    ):
        self.name = name
        self._type = _type
        self.description = description
        self.default = default
        self.parent = parent
        self.children = children
        self.subpackage = subpackage
        # note the general kind of the parameter,
        # i.e. the general type theoretic concept.
        # this is ofc derivable on demand, but Jinja
        # doesn't allow arbitrary expressions, and it
        # doesn't seem to have `subclass`-ish filters.
        self.kind = kind or ParameterKind.from_type(_type)
        self.is_choice = is_choice
        self.set_self = set_self


Parameters = Dict[str, Parameter]


def _try_get_subpkg(metadata: Metadata) -> Optional[Subpackage]:
    lines = {
        "subpkg": next(
            iter(m for m in metadata if m.startswith("subpac")), None
        ),
        "parent": next(
            iter(m for m in metadata if m.startswith("parent")), None
        ),
    }

    def _subpkg():
        line = lines["subpkg"]
        _, record_name, abbreviation, param_name, data_name = line.split()
        return {
            "abbreviation": abbreviation,
            "data_name": data_name,
            "param_name": param_name,
            "record_name": record_name,
        }

    def _parent():
        line = lines["parent"]
        _, _, param_type = line.split()
        param_type = param_type.lower()
        if "simulation" in param_type:
            return MFSimulation
        elif "model" in param_type:
            return MFModel
        return MFPackage

    return (
        Subpackage(parent=_parent(), **_subpkg())
        if all(v for v in lines.values())
        else None
    )


def get_template_context(
    name: DefinitionName,
    base: Union[MFSimulationBase, MFModel, MFPackage],
    common: Variables,
    definition: Definition,
    references: Optional[Definitions] = None,
) -> dict:
    """
    Convert an input definition to a template rendering context.

    Notes
    -----
    Each input definition corresponds to a generated file/class.

    A map of other definitions may be provided, in which case a
    parameter in this context may act as kind of "foreign key",
    identifying another context (e.g. a subpackage) which this
    context "owns". This allows conditionally initializing any
    subpackages in `__init__` for input contexts that use them.
    """

    variables, metadata = definition

    def _convert(variable: Variable, wrap: bool = False) -> Parameter:
        """
        Transform a variable from its original representation in
        an input definition to a parameter for inclusion in type
        hints, docstrings, and the `__init__` method's signature.

        This involves expanding nested input hierarchies, mapping
        types to roughly equivalent Python primitives/composites,
        and other shaping.

        Notes
        -----
        If a `default_value` is not provided, keywords are `False`
        by default. Everything else is `None` by default.

        If `wrap` is true, scalars will be wrapped as records with
        keywords represented as string literals. This is useful for
        unions, to distinguish between choices having the same type.
        """

        _name = variable["name"]
        _type = variable.get("type", "unknown")
        shape = variable.get("shape", None)
        shape = None if shape == "" else shape
        optional = variable.get("optional", True)
        in_record = variable.get("in_record", False)
        tagged = variable.get("tagged, False")
        description = variable.get("description", "")
        children = None

        def _description(description: str) -> str:
            """
            Make substitutions from common variables,
            remove backslashes from the description.
            TODO: insert citations.
            """
            description = description.replace("\\", "")
            _, replace, tail = description.strip().partition("REPLACE")
            if replace:
                key, _, replacements = tail.strip().partition(" ")
                replacements = eval(replacements)
                val = common.get(key, None).get("description", "")
                if val is None:
                    raise ValueError(f"Common variable not found: {key}")
                if any(replacements):
                    return val.replace("{#1}", replacements["{#1}"])
                return val
            return description

        def _fields(record_name: str) -> Parameters:
            """
            Load a record's fields and recursively convert them.

            Notes
            -----
            This function is provided because records
            need extra processing; we remove keywords
            and 'filein'/'fileout', which are details
            of the mf6io format, not of python/flopy.
            """
            record = variables[record_name]
            field_names = record["type"].split()[1:]
            fields: Dict[str, Parameter] = {
                n: _convert(field)
                for n, field in variables.items()
                if n in field_names
            }
            field_names = list(fields.keys())

            # if the record represents a file...
            if "file" in record_name:
                # remove filein/fileout
                for term in ["filein", "fileout"]:
                    if term in field_names:
                        fields.pop(term)

                # remove leading keyword
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

                # set the type
                n = list(fields.keys())[0]
                path_field = fields[n]
                path_field._type = os.PathLike
                fields[n] = path_field

            # if tagged, remove the leading keyword
            elif record.get("tagged", False):
                keyword = next(iter(fields), None)
                if keyword:
                    fields.pop(keyword)

            return fields

        # now, proceed through the possible input types
        # from top (composites) to bottom (primitives):
        #   - list
        #   - union
        #   - record
        #   - array
        #   - scalar

        # list input can have records or unions as rows.
        # lists which have a consistent record type are
        # regular, inconsistent record types irregular.
        if _type.startswith("recarray"):
            # make sure columns are defined
            names = _type.split()[1:]
            n_names = len(names)
            if n_names < 1:
                raise ValueError(f"Missing recarray definition: {_type}")

            # regular tabular/columnar data (1 record type) can be
            # defined with a nested record (i.e. explicit) or with
            # fields directly inside the recarray (implicit). list
            # data for unions/keystrings necessarily comes nested.

            def _is_explicit_record():
                return len(names) == 1 and variables[names[0]][
                    "type"
                ].startswith("record")

            def _is_implicit_record():
                types = [
                    _try_qualify(v["type"])
                    for n, v in variables.items()
                    if n in names
                ]
                scalar_types = list(SCALAR_TYPES.keys())
                return all(t in scalar_types for t in types)

            if _is_explicit_record():
                record_name = names[0]
                record_type = _convert(variables[record_name])
                children = {record_name: record_type}
                type_ = Iterable[record_type._type]
            elif _is_implicit_record():
                # record implicitly defined, make it on the fly
                record_name = _name
                record_fields = _fields(record_name)
                record_type = Tuple[
                    tuple([f._type for f in record_fields.values()])
                ]
                record = Parameter(
                    name=record_name,
                    _type=record_type,
                    children=record_fields,
                )
                children = {record_name: record}
                type_ = Iterable[record_type]
            else:
                # irregular recarray, rows can be any of several types
                children = {n: _convert(variables[n]) for n in names}
                type_ = Iterable[
                    Union[tuple([c._type for c in children.values()])]
                ]

        # basic composite types...
        # union (product), children are record choices
        elif _type.startswith("keystring"):
            names = _type.split()[1:]
            children = {n: _convert(variables[n], wrap=True) for n in names}
            type_ = Union[tuple([c._type for c in children.values()])]

        # record (sum) type, children are fields
        elif _type.startswith("record"):
            children = _fields(_name)
            if len(children) > 1:
                record_type = Tuple[
                    tuple([c._type for c in children.values()])
                ]
            elif len(children) == 1:
                t = list(children.values())[0]._type
                # make sure we don't double-wrap tuples
                record_type = t if get_origin(t) is tuple else Tuple[(t,)]
            # TODO: if record has 1 field, accept value directly?
            type_ = record_type

        # are we wrapping a record which is a
        # choice within a union? if so, use a
        # literal for the keyword as tag e.g.
        # `Tuple[Literal[...], T]`
        elif wrap:
            field_name = _name
            field = _convert(variable)
            field_type = (
                Literal[field_name] if field._type is bool else field._type
            )
            record_type = (
                Tuple[Literal[field_name]]
                if field._type is bool
                else Tuple[Literal[field_name], field._type]
            )
            children = {
                field_name: replace(field, _type=field_type, is_choice=True)
            }
            type_ = record_type

        # at this point, if it has a shape, it's an array..
        # but if it's in a record make it a variadic tuple,
        # and if its item type is a string use an iterable.
        elif shape is not None:
            scalars = list(SCALAR_TYPES.keys())
            if in_record:
                if _type not in scalars:
                    raise TypeError(f"Unsupported repeating type: {_type}")
                type_ = Tuple[SCALAR_TYPES[_type], ...]
            elif _type in scalars and SCALAR_TYPES[_type] is str:
                type_ = Iterable[SCALAR_TYPES[_type]]
            else:
                if _type not in NP_SCALAR_TYPES.keys():
                    raise TypeError(f"Unsupported array type: {_type}")
                type_ = NDArray[NP_SCALAR_TYPES[_type]]

        # finally a bog standard scalar
        else:
            # if it's a keyword, there are two cases we want to convert it to
            # a string literal: if it's 1) tagging another variable, or 2) it
            # is being wrapped into a record to represent a choice in a union
            tag = _type == "keyword" and (tagged or wrap)
            type_ = Literal[_name] if tag else SCALAR_TYPES.get(_type, _type)

        # make optional if needed
        if optional:
            type_ = (
                Optional[type_]
                if (type_ is not bool and not in_record and not wrap)
                else type_
            )

        # keywords default to False, everything else to None
        default = variable.get("default", False if type_ is bool else None)

        return Parameter(
            # if name is a reserved keyword, add a trailing underscore to it
            name=f"{_name}_" if _name in kwlist else _name,
            # MF6IO type is mapped to a (lenient) Python equivalent
            _type=type_,
            # default value
            default=default,
            # do substitutions, expand citations, apply formatting, etc
            description=_description(description),
            # child parameters
            children=children,
            # does variable reference another context (e.g. subpkg)?
            subpackage=references.get(_name, None) if references else None,
        )

    def _add_exg_params(params: Parameters) -> Parameters:
        a = name.r[:3]
        b = name.r[:3]
        default = f"{a.upper()}6-{b.upper()}6"
        return {
            "exgtype": Parameter(
                **{
                    "name": "exgtype",
                    "_type": str,
                    "default": default,
                    "description": "The exchange type.",
                    "set_self": True,
                }
            ),
            "exgmnamea": Parameter(
                **{
                    "name": "exgmnamea",
                    "_type": str,
                    "description": "The name of the first model in the exchange.",
                    "set_self": True,
                }
            ),
            "exgmnameb": Parameter(
                **{
                    "name": "exgmnameb",
                    "_type": str,
                    "description": "The name of the second model in the exchange.",
                    "set_self": True,
                }
            ),
            **params,
        }

    def _add_pkg_params(params: Parameters) -> Parameters:
        parent_type = name.parent
        parent_name = parent_type.__name__.lower().replace("mf", "")
        is_subpkg = name[0] in ["utl"]
        out = {
            parent_name: Parameter(
                **{
                    "name": parent_name,
                    "_type": parent_type,
                    "description": (
                        f"{parent_name.title()} that this package is part of. "
                        f"Package is automatically added to the {parent_name} "
                        "when it is initialized."
                    ),
                }
            ),
            "loading_package": Parameter(
                **{
                    "name": "loading_package",
                    "_type": bool,
                    "description": (
                        "Do not set this variable. It is intended for debugging "
                        "and internal processing purposes only."
                    ),
                }
            ),
            **params,
            "filename": Parameter(
                **{
                    "name": "filename",
                    "_type": str,
                    "description": "File name for this package.",
                }
            ),
            "pname": Parameter(
                **{
                    "name": "pname",
                    "_type": str,
                    "description": "Package name for this package.",
                }
            ),
        }
        if is_subpkg:
            out["parent_file"] = Parameter(
                **{
                    "name": "parent_file",
                    "_type": PathLike,
                    "description": (
                        "Parent package file that references this package. Only needed "
                        "for utility packages (mfutl*). For example, mfutllaktab package "
                        "must have a mfgwflak package parent_file."
                    ),
                }
            )
        return out

    def _add_mdl_params(params: Parameters) -> Parameters:
        return {
            "simulation": Parameter(
                **{
                    "name": "simulation",
                    "_type": MFSimulation,
                    "description": (
                        "Simulation that this model is part of. "
                        "Model is automatically added to the simulation "
                        "when it is initialized."
                    ),
                }
            ),
            "modelname": Parameter(
                **{
                    "name": "modelname",
                    "_type": str,
                    "description": "The name of the model.",
                }
            ),
            "model_nam_file": Parameter(
                **{
                    "name": "model_nam_file",
                    "_type": PathLike,
                    "description": (
                        "The relative path to the model name file from model working folder."
                    ),
                }
            ),
            "version": Parameter(
                **{
                    "name": "version",
                    "_type": str,
                    "description": "The version of modflow",
                }
            ),
            "exe_name": Parameter(
                **{
                    "name": "exe_name",
                    "_type": str,
                    "description": "The executable name.",
                }
            ),
            "model_ws": Parameter(
                **{
                    "name": "model_ws",
                    "_type": PathLike,
                    "description": "The model working folder path.",
                }
            ),
            **params,
        }

    def _add_sim_params(params: Parameters) -> Parameters:
        sim_params = {
            "sim_name": Parameter(
                name="sim_name",
                _type=str,
                default="sim",
                description="Name of the simulation.",
            ),
            "version": Parameter(
                name="version",
                _type=str,
                default="mf6",
            ),
            "exe_name": Parameter(
                name="exe_name",
                _type=PathLike,
                default="mf6",
            ),
            "sim_ws": Parameter(
                name="sim_ws",
                _type=PathLike,
                default=os.curdir,
            ),
            "verbosity_level": Parameter(
                name="verbosity_level",
                _type=int,
                default=1,
            ),
            "write_headers": Parameter(
                name="write_headers",
                _type=bool,
                default=True,
            ),
            "use_pandas": Parameter(
                name="use_pandas",
                _type=bool,
                default=True,
            ),
            "lazy_io": Parameter(
                name="lazy_io",
                _type=bool,
                default=False,
            ),
        }
        return {
            **sim_params,
            **params,
        }

    def _parameters(definition: Definition) -> Parameters:
        """
        Convert the input variables to parameters for an input
        context class. Context-specific parameters may also be
        added depending.

        Notes
        -----
        Not all variables become parameters; nested variables
        will become components of composite parameters, e.g.,
        record fields, keystring (union) choices, list items.
        """

        vars, _ = definition
        _vars = vars.copy()
        params = {
            name: _convert(var)
            for name, var in _vars.items()
            # filter composite components
            # since we've already inflated
            # parent types in the hierarchy
            if not var.get("in_record", False)
        }
        l, r = name
        if l == "sim" and r == "nam":
            params = _add_sim_params(params)
        elif l == "exg":
            params = _add_exg_params(params)
        elif r == "nam":
            params = _add_mdl_params(params)
        else:
            params = _add_pkg_params(params)
        return {n: p.render() for n, p in params.items()}

    def _dfn(definition: Definition) -> List[Metadata]:
        """
        Get a list of the class' original definition attributes,
        as an internal/partial reproduction of the DFN contents.

        Notes
        -----
        Currently, generated classes have a `.dfn` property that
        reproduces the corresponding DFN sans a few attributes.
        This represents the DFN in raw form, before adapting to
        Python, consolidating nested types, etc.
        """

        vars, meta = definition

        def _to_dfn_fmt(var: Variable) -> List[str]:
            exclude = ["longname", "description"]
            return [
                " ".join([k, v]) for k, v in var.items() if k not in exclude
            ]

        return [["header"] + [attr for attr in meta]] + [
            _to_dfn_fmt(var) for var in vars.values()
        ]

    return {
        # the input definition's name.
        "name": name,
        # the corresponding context class name.
        "class": name.title,
        # the base class the context class inherits from.
        "base": _try_qualify(base),
        # contexts may be attached to a parent context type.
        "parent": {
            "name": name.title,
            "type": _try_qualify(name.parent),
            "kind": name.parent_kind,
        },
        # the context's description. header of the docstring.
        "description": name.description,
        # the context's parameters. inserted into docstring
        # and __init__ method, optionally set in __init__.
        "parameters": _parameters(definition),
        # reproduction of (most of the) original DFN.
        "dfn": _dfn(definition),
    }


def generate_target(
    dfn_name: DefinitionName,
    out_path: Path,
    common: Variables,
    definition: Definition,
    subpackages: Definitions,
    verbose: bool = False,
):
    """
    Generate Python source file(s) from the given input definition.

    Notes
    -----

    Model definitions will produce two files / classes, one for the
    model itself and one for its corresponding control file package.
    """

    template = TEMPLATE_ENV.get_template("context.jinja")
    context_type = ContextType.from_dfn_name(dfn_name)
    context_base = context_type.base
    target_path = out_path / dfn_name.target
    with open(target_path, "w") as f:
        context = {
            "context": context_type.value,
            **get_template_context(
                name=dfn_name,
                base=context_base,
                common=common,
                definition=definition,
                references=subpackages,
            ),
        }
        source = template.render(**context)
        if verbose:
            print(f"Generating {target_path}")
        f.write(source)

    # bc models are defined in `<mdl>-nam.dfn` files,
    # we wrote the model class file above, also need
    # to write the package class file.
    if context_type == ContextType.Model:
        _dfn_name = DefinitionName(l=dfn_name.l, r="")
        target_path = out_path / _dfn_name.target
        with open(target_path, "w") as f:
            context = {
                "context": "package",
                **get_template_context(
                    name=_dfn_name,
                    base=MFPackage,
                    common=common,
                    definition=definition,
                    references=subpackages,
                ),
            }
            source = template.render(**context)
            if verbose:
                print(f"Generating {target_path}")
            f.write(source)


def generate_targets(dfn_path: Path, out_path: Path, verbose: bool = False):
    """Generate Python source files from the DFN files in the given location."""

    # find definition files
    dfn_paths = [
        dfn
        for dfn in dfn_path.glob("*.dfn")
        if dfn.stem not in ["common", "flopy"]
    ]

    # try to load common variables
    if not any(p.stem == "common" for p in dfn_paths):
        warn("No common input definition file...")
    with open(dfn_path / "common.dfn") as f:
        common, _ = load_dfn(f)

    # load and generate subpackages first so we can
    # identify references to them in other contexts
    subpkg_paths = [dfn for dfn in dfn_paths if dfn.stem.startswith("utl")]
    subpkgs = dict()
    for p in subpkg_paths:
        l, r = p.stem.split("-")
        name = DefinitionName(l, r)
        with open(p) as f:
            variables, metadata = load_dfn(f)
            definition = (variables, metadata)
            subpackage = _try_get_subpkg(metadata)
            if subpackage:
                subpkgs[r] = subpackage
            else:
                warn(f"{name} lacks subpackage metadata, only: {metadata}")
            generate_target(
                name, out_path, common, definition, subpkgs, verbose
            )

    # load and generate the rest of the contexts
    context_paths = [p for p in dfn_paths if p not in subpkg_paths]
    for p in context_paths:
        name = DefinitionName(*p.stem.split("-"))
        with open(p) as f:
            definition = load_dfn(f)
            generate_target(
                name, out_path, common, definition, subpkgs, verbose
            )

    # format the generated file
    run_cmd("ruff", "format", out_path, verbose=verbose)


if __name__ == "__main__":
    generate_targets(DFN_PATH, TGT_PATH)
