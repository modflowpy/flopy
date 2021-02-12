"""
mfpar module.  Contains the ModflowPar class. Note that the user can access
the ModflowPar class as `flopy.modflow.ModflowPar`.


"""

import sys
import numpy as np
from .mfzon import ModflowZon
from .mfpval import ModflowPval
from .mfmlt import ModflowMlt


class ModflowPar(object):
    """
    Class for loading mult, zone, pval, and parameter data for MODFLOW packages
    that use array data (LPF, UPW, RCH, EVT). Class also includes methods to
    create data arrays using mult, zone, pval, and parameter data (not used
    for boundary conditions).

    Notes
    -----
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.


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
        """
        Load an existing zone package and set zone data for a model.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------


        Examples
        --------

        >>> ml.mfpar.set_zone(ml, ext_unit_dict)

        """
        zone = None
        zone_key = None
        for key, item in ext_unit_dict.items():
            if item.filetype.lower() == "zone":
                zone = item
                zone_key = key
        if zone_key is not None:
            try:
                self.zone = ModflowZon.load(
                    zone.filename, model, ext_unit_dict=ext_unit_dict
                )
                if model.verbose:
                    sys.stdout.write(
                        "   {} package load...success\n".format(
                            self.zone.name[0]
                        )
                    )
                ext_unit_dict.pop(zone_key)
                model.remove_package("ZONE")
            except BaseException as o:
                sys.stdout.write(
                    "   {} package load...failed\n      {!s}".format("ZONE", o)
                )
        return

    def set_mult(self, model, ext_unit_dict):
        """
        Load an existing mult package and set mult data for a model.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------


        Examples
        --------

        >>> ml.mfpar.set_mult(ml, ext_unit_dict)

        """
        mult = None
        mult_key = None
        for key, item in ext_unit_dict.items():
            if item.filetype.lower() == "mult":
                mult = item
                mult_key = key
        if mult_key is not None:
            try:
                self.mult = ModflowMlt.load(
                    mult.filename, model, ext_unit_dict=ext_unit_dict
                )
                if model.verbose:
                    sys.stdout.write(
                        "   {} package load...success\n".format(
                            self.mult.name[0]
                        )
                    )
                ext_unit_dict.pop(mult_key)
                model.remove_package("MULT")
            except BaseException as o:
                sys.stdout.write(
                    "   {} package load...failed\n      {!s}".format("MULT", o)
                )

        return

    def set_pval(self, model, ext_unit_dict):
        """
        Load an existing pval package and set pval data for a model.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------


        Examples
        --------

        >>> ml.mfpar.set_pval(ml, ext_unit_dict)

        """
        pval = None
        pval_key = None
        for key, item in ext_unit_dict.items():
            if item.filetype.lower() == "pval":
                pval = item
                pval_key = key
        if pval_key is not None:
            try:
                self.pval = ModflowPval.load(
                    pval.filename, model, ext_unit_dict=ext_unit_dict
                )
                if model.verbose:
                    sys.stdout.write(
                        "   {} package load...success\n".format(
                            self.pval.name[0]
                        )
                    )
                ext_unit_dict.pop(pval_key)
                model.remove_package("PVAL")
            except BaseException as o:
                sys.stdout.write(
                    "   {} package load...failed\n      {!s}".format("PVAL", o)
                )

        return

    @staticmethod
    def load(f, npar, verbose=False):
        """
        Load property parameters from an existing package.

        Parameters
        ----------
        f : file handle

        npar : int
            The number of parameters.

        verbose : bool
            Boolean flag to control output. (default is False)

        Returns
        -------
        list : list object of unique par_types in file f
        dictionary : dictionary object with parameters in file f

        Examples
        --------

        >>>par_types, parm_dict = flopy.modflow.mfpar.ModflowPar.load(f, np)


        """
        # read parameter data
        if npar > 0:
            parm_dict = {}
            par_types = []
            for nprm in range(npar):
                line = f.readline()
                t = line.strip().split()
                parnam = t[0].lower()
                if verbose:
                    print('   loading parameter "{}"...'.format(parnam))
                partyp = t[1].lower()
                if partyp not in par_types:
                    par_types.append(partyp)
                parval = float(t[2])
                nclu = int(t[3])
                clusters = []
                for nc in range(nclu):
                    line = f.readline()
                    t = line.strip().split()
                    lay = int(t[0])
                    s = t[1]
                    if len(s) > 10:
                        s = s[0:10]
                    mltarr = s
                    s = t[2]
                    if len(s) > 10:
                        s = s[0:10]
                    zonarr = s
                    iarr = []
                    for iv in t[3:]:
                        try:
                            iz = int(iv)
                            if iz != 0:
                                iarr.append(iz)
                        except:
                            break

                    clusters.append([lay, mltarr, zonarr, iarr])
                # add parnam to parm_dict
                parm_dict[parnam] = {
                    "partyp": partyp,
                    "parval": parval,
                    "nclu": nclu,
                    "clusters": clusters,
                }

        return par_types, parm_dict

    @staticmethod
    def parameter_fill(model, shape, findkey, parm_dict, findlayer=None):
        """
        Fill an array with parameters using zone, mult, and pval data.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.

        shape : tuple
            The shape of the returned data array. Typically shape is (nrow, ncol)

        findkey : string
            the parameter array to be constructed,

        parm_dict : dict
            dictionary that includes all of the parameter data for a package

        findlayer : int
            Layer that will be filled. Not required for array boundary condition data.

        Returns
        -------
        data : numpy array
            Filled array resulting from applications of zone, mult, pval, and
            parameter data.

        Examples
        --------

        for lpf and upw:

        >>> data = flopy.modflow.mfpar.ModflowPar.parameter_fill(m, (nrow, ncol), 'vkcb',
        >>> .....................................................parm_dict, findlayer=1)


        """
        dtype = np.float32
        data = np.zeros(shape, dtype=dtype)
        for key, tdict in parm_dict.items():
            partyp, parval = tdict["partyp"], tdict["parval"]
            nclu, clusters = tdict["nclu"], tdict["clusters"]
            if model.mfpar.pval is None:
                pv = float(parval)
            else:
                try:
                    pv = float(model.mfpar.pval.pval_dict[key.lower()])
                except:
                    pv = float(parval)
            # print partyp, parval, nclu, clusters
            if partyp == findkey:
                for [layer, mltarr, zonarr, izones] in clusters:
                    # print layer, mltarr, zonarr, izones
                    foundlayer = False
                    if findlayer == None:
                        foundlayer = True
                    else:
                        if layer == (findlayer + 1):
                            foundlayer = True
                    if foundlayer:
                        model.parameter_load = True
                        cluster_data = np.zeros(shape, dtype=dtype)
                        if mltarr.lower() == "none":
                            mult = np.ones(shape, dtype=dtype)
                        else:
                            mult = model.mfpar.mult.mult_dict[mltarr.lower()][
                                :, :
                            ]
                        if zonarr.lower() == "all":
                            cluster_data = pv * mult
                        else:
                            mult_save = np.copy(mult)
                            za = model.mfpar.zone.zone_dict[zonarr.lower()][
                                :, :
                            ]
                            # build a multiplier for all of the izones
                            mult = np.zeros(shape, dtype=dtype)
                            for iz in izones:
                                filtarr = za == iz
                                mult[filtarr] += np.copy(mult_save[filtarr])
                            # calculate parameter value for this cluster
                            cluster_data = pv * mult
                        # add data
                        data += cluster_data

        return data
