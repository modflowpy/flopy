# -*- coding: utf-8 -*-
"""
This simple script tests that the pcg load function works for both free-
and fixed-format pcg files for mf2005 models.

Refer to pull request #311: "Except block to forgive old fixed format pcg
for post-mf2k model versions" for more details.
"""

import os
import flopy as fp

pcg_fname = os.path.join("..","examples", "data", "pcg_fmt_test", "fixfmt.pcg")

def pcg_fmt_test():
    # mf2k container - this will pass
    m2k = fp.modflow.Modflow(version='mf2k')
    m2k.pcg = fp.modflow.ModflowPcg.load(model=m2k, 
                                     f=pcg_fname)
    
    # mf2005 container
    m05 = fp.modflow.Modflow(version='mf2005')
    m05.pcg = fp.modflow.ModflowPcg.load(model=m05, 
                                 f=pcg_fname)
    # this will exit with ValueError without the except block added in pull req
    
    assert m2k.pcg.rclose == m05.pcg.rclose
    assert m2k.pcg.damp == m05.pcg.damp
    
    return
    
if __name__ == '__main__':
    pcg_fmt_test()
