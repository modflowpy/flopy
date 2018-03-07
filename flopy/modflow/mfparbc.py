"""
mfparbc module.  Contains the ModflowParBc class. Note that the user can access
the ModflowParBc class as `flopy.modflow.ModflowParBc`.

"""

import numpy as np


class ModflowParBc(object):
    """
    Class for loading boundary condition parameter data for MODFLOW packages
    that use list data (WEL, GHB, DRN, etc.). This Class is also used to
    create hfb6 data from hfb parameters. Class also includes methods to
    create data arrays using pval and boundary condition parameter data.

    Notes
    -----
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.


    """

    def __init__(self, bc_parms):
        """
        Package constructor.

        """
        self.bc_parms = bc_parms

    def get(self, fkey):
        """
        overload get to return a value from the bc_parms dictionary

        """
        for key, value in self.bc_parms.items():
            if fkey == key:
                return self.bc_parms[key]
        return None

    @staticmethod
    def load(f, npar, dt, verbose=False):
        """
        Load bc property parameters from an existing bc package
        that uses list data (e.g. WEL, RIV, etc.).

        Parameters
        ----------
        f : file handle

        npar : int
            The number of parameters.

        dt : numpy.dtype
            numpy.dtype for the particular list boundary condition.

        verbose : bool
            Boolean flag to control output. (default is False)

        Returns
        -------
        dictionary : dictionary object with parameters in file f

        Examples
        --------


        """
        nitems = len(dt.names)
        # read parameter data
        if npar > 0:
            bc_parms = {}
            for idx in range(npar):
                line = f.readline()
                t = line.strip().split()
                parnam = t[0].lower()
                if parnam.startswith("'"):
                    parnam = parnam[1:]
                if parnam.endswith("'"):
                    parnam = parnam[:-1]
                if verbose:
                    print('   loading parameter "{}"...'.format(parnam))
                partyp = t[1].lower()
                parval = t[2]
                nlst = np.int(t[3])
                numinst = 1
                timeVarying = False
                if len(t) > 4:
                    if 'instances' in t[4].lower():
                        numinst = np.int(t[5])
                        timeVarying = True
                pinst = {}
                for inst in range(numinst):
                    # read instance name
                    if timeVarying:
                        line = f.readline()
                        t = line.strip().split()
                        instnam = t[0].lower()
                    else:
                        instnam = 'static'
                    bcinst = []
                    for nw in range(nlst):
                        line = f.readline()
                        t = line.strip().split()
                        bnd = []
                        for jdx in range(nitems):
                            # if jdx < 3:
                            if issubclass(dt[jdx].type, np.integer):
                                # conversion to zero-based occurs in package load method in mbase.
                                bnd.append(int(t[jdx]))
                            else:
                                bnd.append(float(t[jdx]))
                        bcinst.append(bnd)
                    pinst[instnam] = bcinst
                bc_parms[parnam] = [{'partyp': partyp, 'parval': parval,
                                     'nlst': nlst, 'timevarying': timeVarying},
                                    pinst]

        # print bc_parms
        bcpar = ModflowParBc(bc_parms)
        return bcpar

    @staticmethod
    def loadarray(f, npar, verbose=False):
        """
        Load bc property parameters from an existing bc package
        that uses array data (e.g. RCH, EVT).

        Parameters
        ----------
        f : file handle

        npar : int
            The number of parameters.

        verbose : bool
            Boolean flag to control output. (default is False)

        Returns
        -------
        dictionary : dictionary object with parameters in file f

        Examples
        --------


        """
        # read parameter data
        if npar > 0:
            bc_parms = {}
            for idx in range(npar):
                line = f.readline()
                t = line.strip().split()
                parnam = t[0].lower()
                if verbose:
                    print('   loading parameter "{}"...'.format(parnam))
                partyp = t[1].lower()
                parval = t[2]
                nclu = np.int(t[3])
                numinst = 1
                timeVarying = False
                if len(t) > 4:
                    if 'instances' in t[4].lower():
                        numinst = np.int(t[5])
                        timeVarying = True
                pinst = {}
                for inst in range(numinst):
                    # read instance name
                    if timeVarying:
                        line = f.readline()
                        t = line.strip().split()
                        instnam = t[0].lower()
                    else:
                        instnam = 'static'
                    bcinst = []

                    for nc in range(nclu):
                        line = f.readline()
                        t = line.strip().split()
                        bnd = [t[0], t[1]]
                        if t[1].lower() == 'all':
                            bnd.append([])
                        else:
                            iz = []
                            for jdx in range(2, len(t)):
                                try:
                                    ival = int(t[jdx])
                                    if ival > 0:
                                        iz.append(ival)
                                except:
                                    break
                            bnd.append(iz)
                        bcinst.append(bnd)
                    pinst[instnam] = bcinst
                bc_parms[parnam] = [{'partyp': partyp, 'parval': parval, 'nclu': nclu, 'timevarying': timeVarying},
                                    pinst]

        # print bc_parms
        bcpar = ModflowParBc(bc_parms)
        return bcpar

    @staticmethod
    def parameter_bcfill(model, shape, parm_dict, pak_parms):
        """
        Fill an array with parameters using zone, mult, and pval data.

        Parameters
        ----------
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.

        shape : tuple
            The shape of the returned data array. Typically shape is (nrow, ncol)

        parm_dict : list
            dictionary of parameter instances

        pak_parms : dict
            dictionary that includes all of the parameter data for a package

        Returns
        -------
        data : numpy array
            Filled array resulting from applications of zone, mult, pval, and
            parameter data.

        Examples
        --------

        for rch and evt
        >>> data = flopy.modflow.mfparbc.ModflowParBc.parameter_bcfill(m, (nrow, ncol),
        >>> .......'rech', parm_dict, pak_parms)


        """
        dtype = np.float32
        data = np.zeros(shape, dtype=dtype)
        for key, value in parm_dict.items():
            # print key, value
            pdict, idict = pak_parms.bc_parms[key]
            inst_data = idict[value]
            if model.mfpar.pval is None:
                pv = np.float(pdict['parval'])
            else:
                try:
                    pv = np.float(model.mfpar.pval.pval_dict[key.lower()])
                except:
                    pv = np.float(pdict['parval'])
            for [mltarr, zonarr, izones] in inst_data:
                model.parameter_load = True
                # print mltarr, zonarr, izones
                if mltarr.lower() == 'none':
                    mult = np.ones(shape, dtype=dtype)
                else:
                    mult = model.mfpar.mult.mult_dict[mltarr.lower()][:, :]
                if zonarr.lower() == 'all':
                    t = pv * mult
                else:
                    mult_save = np.copy(mult)
                    za = model.mfpar.zone.zone_dict[zonarr.lower()][:, :]
                    # build a multiplier for all of the izones
                    mult = np.zeros(shape, dtype=dtype)
                    for iz in izones:
                        filtarr = za == iz
                        mult[filtarr] += np.copy(mult_save[filtarr])
                    # calculate parameter value for this instance
                    t = pv * mult
                data += t

        return data
