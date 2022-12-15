from flopy.utils.datautil import PyListUtil


def test_split_data_line():
    # fixes: https://github.com/modflowpy/flopy/runs/7581629193?check_suite_focus=true#step:11:1753

    line = "13,14,15, 16,  17,"
    spl = PyListUtil.split_data_line(line)
    exp = ["13", "14", "15", "16", "17"]
    assert len(spl) == len(exp)
    # whitespace is not removed, todo: can it be?
    # or is it needed to support Modflow input file format?
    assert all(any([e in s for s in spl]) for e in exp)
