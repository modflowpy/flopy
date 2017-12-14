import os
import keyword
from enum import Enum
from flopy.mf6.data import mfstructure, mfdatautil


class PackageLevel(Enum):
    sim_level = 0
    model_level = 1


def generator_type(data_type):
    if data_type == mfstructure.DataType.scalar_keyword or \
                    data_type == mfstructure.DataType.scalar:
        # regular scalar
        return 'ScalarTemplateGenerator'
    elif data_type == mfstructure.DataType.scalar_keyword_transient or \
                    data_type == mfstructure.DataType.scalar_transient:
        # transient scalar
        return 'ScalarTemplateGenerator'
    elif data_type == mfstructure.DataType.array:
        # array
        return 'ArrayTemplateGenerator'
    elif data_type == mfstructure.DataType.array_transient:
        # transient array
        return 'ArrayTemplateGenerator'
    elif data_type == mfstructure.DataType.list:
        # list
        return 'ListTemplateGenerator'
    elif data_type == mfstructure.DataType.list_transient or \
                    data_type == mfstructure.DataType.list_multiple:
        # transient or multiple list
        return 'ListTemplateGenerator'


def clean_name(name):
    # remove bad characters
    clean_string = name.replace(' ', '_')
    clean_string = clean_string.replace('-', '_')
    # remove anything after a parenthesis
    index = clean_string.find('(')
    if index != -1:
        clean_string = clean_string[0:index]

    return clean_string


def clean_class_string(name):
    if len(name) > 0:
        clean_string = name.replace(' ', '_')
        clean_string = clean_string.replace('-', '_')
        version = mfstructure.MFStructure().get_version_string()
        # FIX: remove all numbers
        if clean_string[-1] == version:
            clean_string = clean_string[:-1]
        return clean_string
    return name


def create_init_var(clean_ds_name, data_structure_name):
    init_var = '        self.{} = self.build_mfdata('.format(clean_ds_name)
    leading_spaces = ' ' * len(init_var)
    if len(init_var) + len(data_structure_name) + 2 > 79:
        second_line = '\n            "{}", '.format(data_structure_name)
        if len(second_line) + len(clean_ds_name) + 2 > 79:
            init_var = '{}{}\n            {})'.format(init_var, second_line,
                                                      clean_ds_name)
        else:
            init_var = '{}{} {})'.format(init_var, second_line, clean_ds_name)
    else:
        init_var = '{}"{}", '.format(init_var, data_structure_name)
        if len(init_var) + len(clean_ds_name) + 2 > 79:
            init_var = '{}\n{}{})'.format(init_var, leading_spaces,
                                          clean_ds_name)
        else:
            init_var = '{} {})'.format(init_var, clean_ds_name)
    return init_var


def create_basic_init(clean_ds_name):
    return '        self.{} = {}\n'.format(clean_ds_name, clean_ds_name)


def create_property(clean_ds_name):
    return "    {} = property(get_{}, set_{}" \
           ")".format(clean_ds_name,
                      clean_ds_name,
                      clean_ds_name)


def format_var_list(base_string, var_list, is_tuple=False):
    if is_tuple:
        base_string = '{}('.format(base_string)
        extra_chars = 4
    else:
        extra_chars = 2
    line_length = len(base_string)
    leading_spaces = ' ' * line_length
    # determine if any variable name is too long to fit
    for item in var_list:
        if line_length + len(item) + extra_chars  > 80:
            leading_spaces = '        '
            base_string = '{}\n{}'.format(base_string, leading_spaces)
            line_length = len(leading_spaces)
            break

    for index, item in enumerate(var_list):
        if is_tuple:
            item = "'{}'".format(item)
        if index == len(var_list) - 1:
            next_var_str = item
        else:
            next_var_str = '{}, '.format(item)
        line_length += len(item) + extra_chars
        if line_length > 80:
            base_string = '{}\n{}{}'.format(base_string, leading_spaces,
                                            next_var_str)
        else:
            base_string = '{}{}'.format(base_string, next_var_str)
    if is_tuple:
        return '{}))'.format(base_string)
    else:
        return '{})'.format(base_string)

def add_var(init_vars, class_vars, init_param_list, package_properties,
            doc_string, data_structure_dict, name,
            python_name, type_string, description, path, data_type,
            basic_init=False):
    clean_ds_name = clean_name(python_name)
    if basic_init:
        init_vars.append(create_basic_init(clean_ds_name))
    else:
        init_vars.append(create_init_var(clean_ds_name, name))
    init_param_list.append('{}=None'.format(clean_ds_name))
    package_properties.append(create_property(clean_ds_name))
    doc_string.add_parameter(python_name,
                             type_string,
                             description)
    data_structure_dict[python_name] = 0
    if class_vars is not None:
        gen_type = generator_type(data_type)
        if gen_type != 'ScalarTemplateGenerator':
            new_class_var = '    {} = {}('.format(clean_ds_name,
                                                  gen_type)
            class_vars.append(format_var_list(new_class_var, path, True))


def create_packages():
    init_string_def = '    def __init__(self'

    # load JSON file
    file_structure = mfstructure.MFStructure()
    sim_struct = file_structure.sim_struct

    # assemble package list of buildable packages
    package_list = []
    package_list.append(
        (sim_struct.name_file_struct_obj, PackageLevel.sim_level, ''))
    for key, package in sim_struct.package_struct_objs.items():
        # add simulation level package to list
        package_list.append((package, PackageLevel.sim_level, ''))
    for key, package in sim_struct.utl_struct_objs.items():
        # add utility packages to list
        package_list.append((package, PackageLevel.model_level, 'utl'))
    for model_key, model in sim_struct.model_struct_objs.items():
        package_list.append(
            (model.name_file_struct_obj, PackageLevel.model_level, model_key))
        for key, package in model.package_struct_objs.items():
            package_list.append((package, PackageLevel.model_level, model_key))

    util_path, tail = os.path.split(os.path.realpath(__file__))
    init_file = open(os.path.join(util_path, '..', 'modflow', '__init__.py'),
                     'w')
    init_file.write('# imports\n')

    # loop through packages list
    for package in package_list:
        data_structure_dict = {}
        package_properties = []
        init_vars = []
        init_param_list = []
        class_vars = []
        package_name = clean_class_string(
            '{}{}'.format(clean_class_string(package[2]),
                          package[0].file_type)).lower()
        if package[0].description:
            doc_string = mfdatautil.MFDocString(package[0].description)
        else:
            if package[2]:
                package_container_text = ' within a {} model'.format(
                    package[2])
            else:
                package_container_text = ''
            doc_string = mfdatautil.MFDocString(
                'Modflow{} defines a {} package'
                '{}.'.format(package_name.title(),
                             package[0].file_type,
                             package_container_text))
        import_string = 'from .. import mfpackage\nfrom ..data.mfdatautil ' \
                        'import ListTemplateGenerator, ArrayTemplateGenerator'

        if package[0].dfn_type == mfstructure.DfnType.exch_file:
            add_var(init_vars, None, init_param_list, package_properties,
                    doc_string, data_structure_dict,
                    'exgtype', 'exgtype', '<string>',
                    'is the exchange type (GWF-GWF or GWF-GWT).', None, None,
                    True)
            add_var(init_vars, None, init_param_list, package_properties,
                    doc_string, data_structure_dict,
                    'exgmnamea', 'exgmnamea', '<string>',
                    'is the name of the first model that is part of this '
                    'exchange.', None, None, True)
            add_var(init_vars, None, init_param_list, package_properties,
                    doc_string, data_structure_dict,
                    'exgmnameb', 'exgmnameb', '<string>',
                    'is the name of the second model that is part of this '
                    'exchange.', None, None, True)
            init_vars.append(
                '        simulation.register_exchange_file(self)\n')

        # loop through all blocks
        for bl_key, block in package[0].blocks.items():
            for ds_key, data_structure in block.data_structures.items():
                # only create one property for each unique data structure name
                if data_structure.name not in data_structure_dict:
                    add_var(init_vars, class_vars, init_param_list,
                            package_properties, doc_string,
                            data_structure_dict,
                            data_structure.name, data_structure.python_name,
                            data_structure.get_type_string(),
                            data_structure.get_description(),
                            data_structure.path,
                            data_structure.get_datatype())

        # build package builder class string
        init_vars = '\n'.join(init_vars)
        package_short_name = clean_class_string(package[0].file_type).lower()
        class_def_string = 'class Modflow{}(mfpackage.MFPackage):\n'.format(
            package_name.title())
        class_def_string = class_def_string.replace('-', '_')
        class_var_string = '{}\n    package_abbr = "{}"'.format(
            '\n'.join(class_vars), package_name)
        init_string_full = init_string_def
        # add variables to init string
        if package[1] == PackageLevel.sim_level:
            init_string_full = '{}, simulation, add_to_package_list=' \
                               'True'.format(init_string_full)
        else:
            init_string_full = '{}, model, add_to_package_list=True'.format(
                init_string_full)
        line_chars = len(init_string_full)
        init_param_list.append('fname=None')
        init_param_list.append('pname=None')
        init_param_list.append('parent_file=None')
        for index, param in enumerate(init_param_list):
            if index + 1 < len(init_param_list):
                line_chars += len(param) + 2
            else:
                line_chars += len(param) + 3
            if line_chars > 79:
                init_string_full = '{},\n                 {}'.format(
                    init_string_full, param)
                line_chars = len(param) + len('                 ') + 1
            else:
                init_string_full = '{}, {}'.format(init_string_full, param)
        init_string_full = '{}):\n'.format(init_string_full)

        # build init code
        if package[1] == PackageLevel.sim_level:
            init_var = 'simulation'
        else:
            init_var = 'model'
        parent_init_string = '        super(Modflow{}, self)' \
                             '.__init__('.format(package_name.title())
        spaces = ' ' * len(parent_init_string)
        parent_init_string = '{}{}, "{}", fname, pname,\n{}' \
                             'add_to_package_list, parent_file)        \n\n' \
                             '        # set up variables'.format(
            parent_init_string, init_var, package_short_name, spaces)

        # assemble full package string
        package_string = '{}\n\n\n{}{}\n{}\n\n{}{}\n{}\n'.format(
            import_string, class_def_string,
            doc_string.get_doc_string(), class_var_string,
            init_string_full, parent_init_string, init_vars)

        # open new Packages file
        pb_file = open(os.path.join(util_path, '..', 'modflow',
                                    'mf{}.py'.format(package_name)), 'w')
        pb_file.write(package_string)
        pb_file.close()

        init_file.write('from .mf{} import '
                        'Modflow{}\n'.format(package_name,
                                             package_name.title()))
    init_file.close()


if __name__ == '__main__':
    create_packages()
