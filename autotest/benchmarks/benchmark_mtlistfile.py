import pytest

from flopy.utils.mtlistfile import MtListBudget


@pytest.mark.benchmark
def test_mtlistfile_load(benchmark, example_data_path):
    list_file = example_data_path / "mt3d_test" / "mcomp.list"

    def load_and_parse():
        mt = MtListBudget(str(list_file))
        mt.parse()
        return mt.gw_data

    benchmark(load_and_parse)
