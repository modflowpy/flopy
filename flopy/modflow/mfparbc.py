"""
mfwel module.  Contains the ModflowWel class. Note that the user can access
the ModflowWel class as `flopy.modflow.ModflowWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?wel.htm>`_.

"""

import numpy as np

class ModflowParBc(object):
    """
    MODFLOW Boundary Condition Package Parameter Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    iwelcb : int
        is a flag and a unit number. (the default is 0).
    layer_row_column_data : list of records
        In its most general form, this is a triple list of well records  The
        innermost list is the layer, row, column, and flux rate for a single
        well.  Then for a stress period, there can be a list of wells.  Then
        for a simulation, there can be a separate list for each stress period.
        This gives the form of
            lrcq = [
                     [  #stress period 1
                       [l1, r1, c1, q1],
                       [l2, r2, c2, q2],
                       [l3, r3, c3, q3],
                       ],
                     [  #stress period 2
                       [l1, r1, c1, q1],
                       [l2, r2, c2, q2],
                       [l3, r3, c3, q3],
                       ], ...
                     [  #stress period kper
                       [l1, r1, c1, q1],
                       [l2, r2, c2, q2],
                       [l3, r3, c3, q3],
                       ],
                    ]
        Note that if there are not records in layer_row_column_data, then the
        last group of wells will apply until the end of the simulation.
    layer_row_column_Q : list of records
        Deprecated - use layer_row_column_data instead.
    options : list of strings
        Package options. (default is None).
    naux : int
        number of auxiliary variables
    extension : string
        Filename extension (default is 'wel')
    unitnumber : int
        File unit number (default is 11).

    Attributes
    ----------
    mxactw : int
        Maximum number of wells for a stress period.  This is calculated
        automatically by FloPy based on the information in
        layer_row_column_data.

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> lrcq = [[[2, 3, 4, -100.]]]  #this well will be applied to all stress
    >>>                              #periods
    >>> wel = flopy.modflow.ModflowWel(m, layer_row_column_data=lrcq)

    """
    def __init__(self, bc_parms):
        """
        Package constructor.

        """
        self.bc_parms = bc_parms


    def __repr__( self ):
        return 'Boundary Condition Package Parameter Class'
        
        
    def get(self, fkey):
        for key, value in self.bc_parms.iteritems():
            if fkey == key:
                return self.bc_parms[key]
        return None


    @staticmethod
    def load(f, npar, nitems):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        wel : ModflowWel object
            ModflowWel object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflow.mfwel.load('test.wel', m)

        """
        #--read parameter data
        if npar > 0:
            bc_parms = {}
            for idx in xrange(npar):
                line = f.readline()
                t = line.strip().split()
                parnam = t[0].lower()
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
                for inst in xrange(numinst):
                    #--read instance name
                    if timeVarying:
                        line = f.readline()
                        t = line.strip().split()
                        instnam = t[0].lower()
                    else:
                        instnam = 'static'
                    wellinst = []
                    for nw in xrange(nlst):
                        line = f.readline()
                        t = line.strip().split()
                        bnd = []
                        for jdx in xrange(nitems):
                            if jdx < 3:
                                #--conversion to zero-based occurs in package load method in mbase.
                                bnd.append(int(t[jdx]))
                            else:
                                bnd.append(float(t[jdx]))
                        wellinst.append(bnd)
                    pinst[instnam] = wellinst
                bc_parms[parnam] = [{'partyp':partyp, 'parval':parval, 'nlst':nlst, 'timevarying':timeVarying}, pinst]
        
        print bc_parms
        bcpar = ModflowParBc(bc_parms)
        return bcpar

