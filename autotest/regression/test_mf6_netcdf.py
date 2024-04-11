import math
import os
import shutil

import pytest
from modflow_devtools.markers import requires_exe

import flopy
from flopy.utils.datautil import DatumUtil


def check_datum(test_entry, orig_entry):
    if DatumUtil.is_float(test_entry):
        return round(float(test_entry), 5) == round(float(orig_entry), 5)
    else:
        return test_entry == orig_entry


def check_data(test_vals, orig_vals):
    if isinstance(test_vals, list):
        assert len(test_vals) == len(orig_vals), (
            f"Lists not same size " f"{test_vals} and {orig_vals}"
        )
        for test_entry, orig_entry in zip(test_vals, orig_vals):
            if isinstance(test_entry, list):
                check_data(test_entry, orig_entry)
            else:
                if not check_datum(test_entry, orig_entry):
                    return False, test_entry, orig_entry
    else:
        if not check_datum(test_vals, orig_vals):
            return False, test_vals, orig_vals
    return True, None, None


class TestInfo:
    def __init__(
        self,
        original_simulation_folder,
        netcdf_simulation_folder,
        netcdf_output_file,
    ):
        self.original_simulation_folder = original_simulation_folder
        self.netcdf_simulation_folder = netcdf_simulation_folder
        self.netcdf_output_file = netcdf_output_file


class Variable:
    def __init__(self, var_name, var_type, var_dimensions):
        self.var_name = var_name
        self.var_type = var_type
        self.var_dimensions = var_dimensions
        self.var_attribs = {}


class NcOut:
    def __init__(self, nc_out_file):
        self.name = None
        self.dimensions = {}
        self.variables = {}
        self.global_attr = {}
        self.data = {}
        self._load(nc_out_file)

    def _load(self, nc_out_file):
        with open(nc_out_file, "r") as fd_nc:
            first_line = fd_nc.readline()
            first_line_lst = first_line.split()
            self.name = first_line_lst[1]
            loading = None
            var_name = None
            end_data = True
            for line in fd_nc:
                line_s = line.strip()
                if len(line_s) == 0:
                    continue
                if line_s == "dimensions:":
                    loading = "dimensions"
                elif line_s == "variables:":
                    loading = "variables"
                elif line_s == "// global attributes:":
                    loading = "global attributes"
                elif line_s == "data:":
                    loading = "data"
                elif loading == "dimensions":
                    line_lst = line_s.split()
                    if len(line_lst) > 2:
                        self.dimensions[line_lst[0].strip()] = line_lst[2]
                elif loading == "variables":
                    if line.startswith("\t\t"):
                        if var_name is not None:
                            line_lst = line_s.split("=")
                            attr_name = line_lst[0].split(":")[1].strip()
                            self.variables[var_name].var_attribs[attr_name] = (
                                line_lst[1]
                            )
                    else:
                        line_lst = line_s.split()
                        if "(" in line_lst[1]:
                            line_lst_end = line_lst[1].split("(")
                            var_name = line_lst_end[0].strip()
                            dimensions = line_lst_end[1].strip()
                        else:
                            var_name = line_lst[1]
                            dimensions = ""
                        var = Variable(var_name, line_lst[0], dimensions)
                        self.variables[var_name] = var
                elif loading == "global attributes":
                    line_lst = line_s.split()
                    if len(line_lst) > 2:
                        self.global_attr[line_lst[0].strip()[1:]] = line_lst[
                            2
                        ].strip()
                elif loading == "data":
                    if "=" in line_s:
                        line_lst = line_s.split("=")
                        var_name = line_lst[0].strip()
                        line_lst_end = line_lst[1].split(",")
                        line_lst_end[-1] = (
                            line_lst_end[-1].strip(";").strip(",")
                        )
                        if len(line_lst_end) > 0:
                            self.data[var_name] = line_lst_end
                        end_data = line_s.endswith(";")
                    elif not end_data and len(line_s) > 0:
                        line_lst = line_s.split(",")
                        line_lst[-1].strip(";")
                        if var_name in self.data:
                            self.data[var_name] = (
                                self.data[var_name] + line_lst
                            )
                        else:
                            self.data[var_name] = line_lst
                        end_data = line_s.endswith(";")


@requires_exe("mf6")
@pytest.mark.regression
def test_save_netcdf(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        # "temp": TestInfo(
        # "test001h_evt_list2",
        #    "test028_sfr_mvr_dev",
        # "test006_gwf3_disv_ext",
        #    "test",
        #    "test.nc.out",
        # ),
        "test_gwf_rch03": TestInfo(
            "test_mf6model[0-rch03]0",
            "test_mf6model_nc[0-rch03]0",
            "rch.nc.out",
        ),
        "test_gwf_disv_uzf": TestInfo(
            "test_mf6model[disv_with_uzf]0",
            "test_mf6model_nc[disv_with_uzf]0",
            "disv_with_uzf.nc.out",
        ),
        "test_gwf_boundname01": TestInfo(
            "test_mf6model[0-bndname01]0",
            "test_mf6model_nc[0-bndname01]0",
            "gwf_bndname01.nc.out",
        ),
        "test_gwf_npf02_rewet": TestInfo(
            "test_mf6model[0-npf02_hreweta]0",
            "test_mf6model_nc[0-npf02_hreweta]0",
            "gwf0.nc.out",
        ),
        "test_gwf_csub_sub03_spd": TestInfo(
            "test_mf6model[0-csub_sub03a]0",
            "test_mf6model_nc[0-csub_sub03a]0",
            "csub_sub03a.nc.out",
        ),
        "test_gwf_disu01a_ascii": TestInfo(
            "disu01a_ascii", "disu01a", "disu01a.nc.out"
        ),
        "test_gwt_mst03": TestInfo(
            "test_mf6model[mst03]0",
            "test_mf6model_nc[mst03]0",
            "gwf_mst03.nc.out",
        ),
        "test_gwf_evt02": TestInfo(
            "test_mf6model[0-evt02]0",
            "test_mf6model_nc[0-evt02]0",
            "evt02.nc.out",
        ),
        "test_gwf_csub_sub03": TestInfo(
            "test_mf6model[0-csub_sub03a]0",
            "test_mf6model_nc[0-csub_sub03a]0",
            "csub_sub03a.nc.out",
        ),
    }
    ws = function_tmpdir / "ws"
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(
            data_path_base, base_folder, test_info.original_simulation_folder
        )
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_model_folder)
        # change simulation path
        sim.set_sim_path(test_model_folder)
        # write example as netcdf
        sim.write_simulation(write_netcdf=True, to_cdl=True)
        # compare .nc.out files
        np_path = os.path.join(
            test_model_folder,
            test_info.netcdf_output_file,
        )
        comp_nc_path = os.path.join(
            data_path_base,
            base_folder,
            test_info.netcdf_simulation_folder,
            test_info.netcdf_output_file,
        )

        ignore_list = [
            "_Storage",
            "_ChunkSizes",
            "_DeflateLevel",
            "_Shuffle",
            "_Endianness",
            "_NCProperties",
            "_SuperblockVersion",
            "_IsNetcdf4",
            "_FillValue",
            "_Format",
            "source",
            "Conventions",
        ]
        nc_test = NcOut(np_path)
        nc_orig = NcOut(comp_nc_path)

        # compare name
        assert nc_test.name == nc_orig.name

        # compare dimensions
        for orig_name, orig_val in nc_orig.dimensions.items():
            assert orig_name in nc_test.dimensions
            assert nc_test.dimensions[orig_name] == orig_val, (
                f"dimension size for {orig_name} do not match ({orig_val}, "
                f"{nc_test.dimensions[orig_name]})"
            )
        for test_name, test_val in nc_test.dimensions.items():
            assert test_name in nc_orig.dimensions
            assert nc_orig.dimensions[test_name] == test_val

        # compare variables
        for orig_name, orig_var in nc_orig.variables.items():
            assert orig_name in nc_test.variables
            assert orig_var.var_type == nc_test.variables[orig_name].var_type
            assert (
                orig_var.var_dimensions
                == nc_test.variables[orig_name].var_dimensions
            )
            for attrib_name, attrib_val in orig_var.var_attribs.items():
                if attrib_name in ignore_list:
                    continue
                attrib_dict = nc_test.variables[orig_name].var_attribs
                assert attrib_name in attrib_dict
                attrib_match = (
                    attrib_val
                    == nc_test.variables[orig_name].var_attribs[attrib_name]
                )
                assert attrib_match, (
                    f"Variable {orig_name} attribute {attrib_name} does not "
                    f"match ({attrib_val}, {attrib_dict[attrib_name]})"
                )
        for test_name, test_var in nc_test.variables.items():
            assert test_name in nc_orig.variables
            assert test_var.var_type == nc_orig.variables[test_name].var_type
            assert (
                test_var.var_dimensions
                == nc_orig.variables[test_name].var_dimensions
            )
            for attrib_name, attrib_val in test_var.var_attribs.items():
                if attrib_name in ignore_list:
                    continue
                assert attrib_name in nc_orig.variables[test_name].var_attribs
                assert (
                    attrib_val
                    == nc_orig.variables[test_name].var_attribs[attrib_name]
                )
        # compare global
        for orig_name, orig_val in nc_orig.global_attr.items():
            if orig_name in ignore_list:
                continue
            assert orig_name in nc_test.global_attr
            assert nc_test.global_attr[orig_name] == orig_val
        for test_name, test_val in nc_test.global_attr.items():
            if test_name in ignore_list:
                continue
            assert test_name in nc_orig.global_attr
            assert nc_orig.global_attr[test_name] == test_val
        # compare data
        for orig_name, orig_val in nc_orig.data.items():
            print(f"   Comparing {orig_name}...")
            assert orig_name in nc_test.data
            success, check_1, check_2 = check_data(
                nc_test.data[orig_name], orig_val
            )
            assert success, (
                f"Data check on {orig_name} failed comparing "
                f" {check_1} to {check_2}"
            )
        for test_name, test_val in nc_test.data.items():
            print(f"   Comparing {test_name}...")
            assert test_name in nc_orig.data
            success, check_1, check_2 = check_data(
                nc_orig.data[test_name], test_val
            )
            assert success, (
                f"Data check on {test_name} failed comparing "
                f" {check_1} to {check_2}"
            )
