"""
mfwel module.  Contains the ModflowWel class. Note that the user can access
the ModflowWel class as `flopy.modflow.ModflowWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?wel.htm>`_.

"""

import numpy as np
from flopy.modflow.mfzon import ModflowZon
from flopy.modflow.mfpval import ModflowPval
from flopy.modflow.mfmlt import ModflowMlt

class ModflowPar(object):
    """

    """
    def __init__(self):
        """
        Package constructor.

        """
        self.pval = None
        self.mult = None
        self.zone = None
        return

    def set_zone(self, model, ext_unit_dict):
        zone = None
        zone_key = None
        for key, item in ext_unit_dict.iteritems():
            if item.filetype.lower() == "zone":
                zone = item
                zone_key = key
        if zone_key is not None:
            self.zone = ModflowZon.load(zone.filename, model,
                                        ext_unit_dict=ext_unit_dict)
            ext_unit_dict.pop(zone_key)
            model.remove_package("ZONE")
        return

    def set_mult(self, model, ext_unit_dict):
        mult = None
        mult_key = None
        for key, item in ext_unit_dict.iteritems():
            if item.filetype.lower() == "mult":
                mult = item
                mult_key = key
        if mult_key is not None:
            self.mult = ModflowMlt.load(mult.filename, model,
                                        ext_unit_dict=ext_unit_dict)
            ext_unit_dict.pop(mult_key)
            model.remove_package("MULT")
        return

    def set_pval(self, model, ext_unit_dict):
        pval = None
        pval_key = None
        for key, item in ext_unit_dict.iteritems():
            if item.filetype.lower() == "pval":
                pval = item
                pval_key = key
        if pval_key is not None:
            self.pval = ModflowPval.load(pval.filename, model,
                                         ext_unit_dict=ext_unit_dict)
            ext_unit_dict.pop(pval_key)
            model.remove_package("PVAL")
        return


