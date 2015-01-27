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


    @staticmethod
    def load(f, npar):
        """
        Load property parameters from an existing package.

        Parameters
        ----------
        f : file handle

        npar : int
            The number of parameters.

        Returns
        -------
        list : list object of unique par_types in file f
        dictionary : dictionary object with parameters in file f

        Examples
        --------


        """
        #--read parameter data
        if npar > 0:
            parm_dict = {}
            par_types = []
            for nprm in xrange(npar):
                line = f.readline()
                t = line.strip().split()
                parnam = t[0].lower()
                print 'loading parameter "{}"...'.format(parnam)
                partyp = t[1].lower()
                if partyp not in par_types:
                    par_types.append(partyp)
                parval = np.float(t[2])
                nclu = np.int(t[3])
                clusters = []
                for nc in xrange(nclu):
                    line = f.readline()
                    t = line.strip().split()
                    lay = np.int(t[0])
                    mltarr = t[1]
                    zonarr = t[2]
                    iarr = []
                    for iv in t[3:]:
                        iarr.append(np.int(iv))
                    clusters.append([lay, mltarr, zonarr, iarr])
                #--add parnam to parm_dict
                parm_dict[parnam] = {'partyp':partyp, 'parval':parval, 'nclu':nclu, 'clusters':clusters}

        return par_types, parm_dict


    @staticmethod
    def parameter_fill(model, shape, findkey, parm_dict, findlayer=None):
        dtype = np.float32
        data = np.zeros(shape, dtype=dtype)
        for key, tdict in parm_dict.iteritems():
            partyp, parval = tdict['partyp'], tdict['parval']
            nclu, clusters = tdict['nclu'], tdict['clusters']
            #print partyp, parval, nclu, clusters
            if partyp == findkey:
                for [layer, mltarr, zonarr, izones] in clusters:
                    #print layer, mltarr, zonarr, izones
                    foundlayer = False
                    if findlayer == None:
                        foundlayer = True
                    else:
                        if layer == (findlayer + 1):
                            foundlayer = True
                    if foundlayer:
                        cluster_data = np.zeros(shape, dtype=dtype)
                        if mltarr.lower() == 'none':
                            mult = np.ones(shape, dtype=dtype)
                        else:
                            mult = model.mfpar.mult.mult_dict[mltarr.lower()][:, :]
                        if zonarr.lower() == 'all':
                            cluster_data = parval * mult
                        else:
                            mult_save = np.copy(mult)
                            za = model.mfpar.zone.zone_dict[zonarr.lower()][:, :]
                            #--build a multiplier for all of the izones
                            for iz in izones:
                                mult = np.zeros(shape, dtype=dtype)
                                filtarr = za == iz
                                mult[filtarr] += np.copy(mult_save[filtarr])
                            #--calculate parameter value for this cluster
                            cluster_data = parval * mult
                        #--add data
                        data += cluster_data

        return data
