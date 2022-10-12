from flopy.utils.flopy_io import line_parse


def test_line_parse():
    """t027 test line_parse method in MNW2 Package class"""
    # ensure that line_parse is working correctly
    # comment handling
    line = line_parse("Well-A  -1                   ; 2a. WELLID,NNODES")
    assert line == ["Well-A", "-1"]
