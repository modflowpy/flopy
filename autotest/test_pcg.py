"""
This simple script tests that the pcg load function works for both free-
and fixed-format pcg files for mf2005 models.

Refer to pull request #311: "Except block to forgive old fixed format pcg
for post-mf2k model versions" for more details.
"""

from flopy.modflow import Modflow, ModflowPcg


def test_pcg_fmt(example_data_path):
    pcg_fname = example_data_path / "pcg_fmt_test" / "fixfmt.pcg"
    # mf2k container - this will pass
    m2k = Modflow(version="mf2k")
    m2k.pcg = ModflowPcg.load(model=m2k, f=pcg_fname)

    # mf2005 container
    m05 = Modflow(version="mf2005")
    m05.pcg = ModflowPcg.load(model=m05, f=pcg_fname)
    # this will exit with ValueError without the except block added in pull req

    assert m2k.pcg.rclose == m05.pcg.rclose
    assert m2k.pcg.damp == m05.pcg.damp
