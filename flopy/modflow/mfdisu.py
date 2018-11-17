"""
mfdisu module.  Contains the ModflowDisU class. Note that the user can access
the ModflowDisU class as `flopy.modflow.ModflowDisU`.

"""

import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Util3d, read1d

ITMUNI = {"u": 0, "s": 1, "m": 2, "h": 3, "d": 4, "y": 5}
LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}


class ModflowDisU(Package):
    """
    MODFLOW Unstructured Discretization Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    nodes : int
        Number of nodes in the model grid (default is 2).
    nlay : int
        Number of layers in the model grid (default is 1).
    njag : int
        Total number of connections of an unstructured grid. njag is used to
        dimension the sparse matrix in a compressed row storage format. For
        symmetric arrays, only the upper triangle of the matrix may be
        entered. For that case, the symmetric portion (minus the diagonal
        terms) is dimensioned as njags = (njag - nodes) / 2.
        (default is None).
    ivsd : int
        is the vertical sub-discretization index. For purposes of this flag,
        vertical sub-discretization is defined to occur when all layers are
        not a stacked representation of each other.
        If IVSD = 0 there is no sub-discretization of layers within the model
        domain. That is, grids are not nested in the vertical direction.
        However, one layer may have a different grid structure from the next
        due to different sub-gridding structures within each layer.
        If IVSD = 1 there could be sub-discretization of layers with
        vertically nested grids (as shown in Figure 5c in the MODFLOW-USG
        document) within the domain. For this case, the vertical connection
        index IVC is required to determine the vertical connections of every
        node. Otherwise, the vertical connections are internally computed and
        IVC is not read.
        If IVSD = -1 there is no vertical sub-discretization of layers, and
        further, the horizontal discretization of all layers is the same. For
        this case, the cell areas (AREA) are read only for one layer and are
        computed to be the same for all the stacked layers. A structured
        finite-difference grid is an example of this condition.
        (default is 0).
    nper : int
        Number of model stress periods (the default is 1).
    itmuni : int
        Time units, default is days (4)
    lenuni : int
        Length units, default is meters (2)
    idsymrd : int
        A flag indicating if the finite-volume connectivity information of an
        unstructured grid is input as a full matrix or as a symmetric matrix
        in the input file.
        If idsymrd is 0 the finite-volume connectivity information is provided
        for the full matrix of the porous matrix grid-block connections of an
        unstructured grid. The code internally stores only the symmetric
        portion of this information. This input structure (IDSYMRD=0) is easy
        to organize but contains unwanted information which is parsed out
        when the information is stored.
        If idsymrd is 1 then finite-volume connectivity information is
        provided only for the upper triangular portion of the porous matrix
        grid-block connections within the unstructured grid. This input
        structure (IDSYMRD=1) is compact but is slightly more complicated to
        organize. Only the non-zero upper triangular items of each row are
        read in sequence for all symmetric matrices.
        (default is 0).
    laycbd : int or array of ints (nlay), optional
        An array of flags indicating whether or not a layer has a Quasi-3D
        confining bed below it. 0 indicates no confining bed, and not zero
        indicates a confining bed. LAYCBD for the bottom layer must be 0. (the
        default is 0)
    nodelay : int or array of ints (nlay)
        The number of cells in each layer. (the default is None, which means
        the number of cells in a layer is equal to nodes / nlay).
    top : float or array of floats (nodes), optional
        An array of the top elevation for every cell. For the situation in
        which the top layer represents a water-table aquifer, it may be
        reasonable to set Top equal to land-surface elevation (the default is
        1.0)
    bot : float or array of floats (nodes), optional
        An array of the bottom elevation for each model cell (the default is
        0.)
    area : float or array of floats
        Surface area for model cells.  Area is for only one layer if IVSD = -1
        to indicate that the grid is vertically stacked. Otherwise, area is
        required for each layer in the model grid. Note that there may be
        different number of nodes per layer (ndslay) for an unstructured grid.
        (default is 1.0)
    iac : array of integers
        is a vector indicating the number of connections plus 1 for each
        node. Note that the IAC array is only supplied for the GWF cells;
        the IAC array is internally expanded to include CLN or GNC nodes if
        they are present in a simulation.
        (default is None. iac must be provided).
    ja : array of integers
        is a list of cell number (n) followed by its connecting cell numbers
        (m) for each of the m cells connected to cell n. This list is
        sequentially provided for the first to the last GWF cell. Note that
        the cell and its connections are only supplied for the GWF cells and
        their connections to the other GWF cells. This connectivity is
        internally expanded if CLN or GNC nodes are present in a simulation.
        Also note that the JA list input may be chopped up to have every node
        number and its connectivity list on a separate line for ease in
        readability of the file. To further ease readability of the file, the
        node number of the cell whose connectivity is subsequently listed,
        may be expressed as a negative number the sign of which is
        subsequently corrected by the code.
        (default is None.  ja must be provided).
    ivc : int or array of integers
        is an index array indicating the direction between a node n and all
        its m connections. IVC = 0 if the connection between n and m is
        horizontal.  IVC = 1 if the connecting node m is vertically oriented
        to node n.  Note that if the CLN Process is active, the connection
        between two CLN cells has IVC = 2 and the connection between a CLN
        cell and a GWF cell has IVC = 3.
        (default is None.  ivc must be provided if ivsd = 1)
    cl1 : float or array of floats
        is the perpendicular length between the center of a node (node 1) and
        the interface between the node and its adjoining node (node 2).
        (default is None.  cl1 and cl2 must be specified, or cl12 must be
        specified)
    cl2 : float or array of floats
        is the perpendicular length between node 2 and the interface between
        nodes 1 and 2, and is at the symmetric location of CL1.
        (default is None.  cl1 and cl2 must be specified, or cl12 must be
        specified)
    cl12 : float or array of floats
        is the array containing CL1 and CL2 lengths, where CL1 is the
        perpendicular length between the center of a node (node 1) and the
        interface between the node and its adjoining node (node 2). CL2,
        which is the perpendicular length between node 2 and the interface
        between nodes 1 and 2 is at the symmetric location of CL1. The array
        CL12 reads both CL1 and CL2 in the upper and lower triangular
        portions of the matrix respectively. Note that the CL1 and CL2 arrays
        are only supplied for the GWF cell connections and are internally
        expanded if CLN or GNC nodes exist in a simulation.
        (default is None.  cl1 and cl2 must be specified, or cl12 must be
        specified)
    fahl : float or arry of floats
        Area of the interface Anm between nodes n and m.
        (default is None.  fahl must be specified.)
    perlen : float or array of floats (nper)
        An array of the stress period lengths.
    nstp : int or array of ints (nper)
        Number of time steps in each stress period (default is 1).
    tsmult : float or array of floats (nper)
        Time step multiplier (default is 1.0).
    steady : boolean or array of boolean (nper)
        true or False indicating whether or not stress period is steady state
        (default is True).
    extension : string
        Filename extension (default is 'dis')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.


    Attributes
    ----------
    heading : str
        Text string written to top of package input file.

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Does not work yet for multi-layer USG models because top and bot cannot
    be u3d instances until u3d is modified to handle multiple u2d instances
    of different size.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> disu = flopy.modflow.ModflowDisU(m)

    """

    def __init__(self, model, nodes=2, nlay=1, njag=None, ivsd=0, nper=1,
                 itmuni=4, lenuni=2, idsymrd=0, laycbd=0, nodelay=None,
                 top=1, bot=0, area=1.0, iac=None, ja=None, ivc=None,
                 cl1=None, cl2=None, cl12=None, fahl=None, perlen=1, nstp=1,
                 tsmult=1, steady=True, extension='disu',
                 unitnumber=None, filenames=None, start_datetime="1/1/1970"):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDisU.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowDisU.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        # Set values of all parameters
        self.url = 'dis.htm'
        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'

        self.nodes = nodes
        self.nlay = nlay
        self.njag = njag
        self.ivsd = ivsd
        self.nper = nper
        try:
            self.itmuni = int(itmuni)
        except:
            self.itmuni = ITMUNI[itmuni.lower()[0]]
        try:
            self.lenuni = int(lenuni)
        except:
            self.lenuni = LENUNI[lenuni.lower()[0]]
        self.idsymrd = idsymrd

        # LAYCBD
        self.laycbd = Util2d(model, (self.nlay,), np.int32, laycbd,
                              name='laycbd')
        self.laycbd[-1] = 0  # bottom layer must be zero

        # NODELAY
        if nodelay is None:
            npl = int(nodes / nlay)
            nodelay = []
            for k in range(self.nlay):
                nodelay.append(npl)
        self.nodelay = Util2d(model, (self.nlay,), np.int32, nodelay,
                               name='nodelay', locat=self.unit_number[0])

        # set ncol and nrow for array readers
        nrow = None
        ncol = self.nodelay.array[:]

        # Top and bot are both 1d arrays of size nodes
        self.top = Util3d(model, (nlay, nrow, ncol), np.float32, top, name='top',
                           locat=self.unit_number[0])
        self.bot = Util3d(model, (nlay, nrow, ncol), np.float32, bot, name='bot',
                           locat=self.unit_number[0])


        # Area is Util2d if ivsd == -1, otherwise it is Util3d
        if ivsd == -1:
            self.area = Util2d(model, (self.nodelay[0],), np.float32, area,
                                'area', locat=self.unit_number[0])
        else:
            self.area = Util3d(model, (nlay, nrow, ncol), np.float32, area,
                                name='area', locat=self.unit_number[0])

        # Connectivity and ivc
        if iac is None:
            raise Exception('iac must be provided')
        self.iac = Util2d(model, (self.nodes,), np.int32,
                           iac, name='iac', locat=self.unit_number[0])
        assert self.iac.array.sum() == njag, 'The sum of iac must equal njag.'
        if ja is None:
            raise Exception('ja must be provided')
        self.ja = Util2d(model, (self.njag,), np.int32,
                          ja, name='ja', locat=self.unit_number[0])
        self.ivc = None
        if self.ivsd == 1:
            if ivc is None:
                raise Exception('ivc must be provided if ivsd is 1.')
            self.ivc = Util2d(model, (self.njag,), np.int32,
                               ivc, name='ivc', locat=self.unit_number[0])

        # Connection lengths
        if idsymrd == 1:
            njags = int((njag - nodes) / 2)
            if cl1 is None:
                raise Exception('idsymrd is 1 but cl1 was not specified.')
            if cl2 is None:
                raise Exception('idsymrd is 1 but cl2 was not specified.')
            self.cl1 = Util2d(model, (njags,), np.float32,
                               cl1, name='cl1', locat=self.unit_number[0])
            self.cl2 = Util2d(model, (njags,), np.float32,
                               cl2, name='cl2', locat=self.unit_number[0])

        if idsymrd == 0:
            if cl12 is None:
                raise Exception('idsymrd is 0 but cl12 was not specified')
            self.cl12 = Util2d(model, (self.njag,), np.float32,
                                cl12, name='cl12', locat=self.unit_number[0])

        # Flow area (set size of array to njag or njags depending on idsymrd)
        if fahl is None:
            raise Exception('fahl must be provided')
        if idsymrd == 1:
            n = njags
        elif idsymrd == 0:
            n = self.njag
        self.fahl = Util2d(model, (n,), np.float32,
                            fahl, name='fahl', locat=self.unit_number[0])

        # Stress period information
        self.perlen = Util2d(model, (self.nper,), np.float32, perlen,
                              name='perlen')
        self.nstp = Util2d(model, (self.nper,), np.int32, nstp, name='nstp')
        self.tsmult = Util2d(model, (self.nper,), np.float32, tsmult,
                              name='tsmult')
        self.steady = Util2d(model, (self.nper,), np.bool,
                              steady, name='steady')

        self.itmuni_dict = {0: "undefined", 1: "seconds", 2: "minutes",
                            3: "hours", 4: "days", 5: "years"}

#        self.sr = reference.SpatialReference(self.delr.array, self.delc.array,
#                                             self.lenuni, xul=xul,
#                                             yul=yul, rotation=rotation)
        self.start_datetime = start_datetime

        # calculate layer thicknesses
        self.__calculate_thickness()

        # Add package and return
        self.parent.add_package(self)
        return

    def __calculate_thickness(self):
        # set ncol and nrow for array readers
        nrow = None
        ncol = self.nodelay.array
        nlay = self.nlay
        thk = []
        for k in range(self.nlay):
            thk.append(self.top[k] - self.bot[k])
        self.__thickness = Util3d(self.parent, (nlay, nrow, ncol),
                                   np.float32, thk, name='thickness')
        return

    @property
    def thickness(self):
        """
        Get a Util2d array of cell thicknesses.

        Returns
        -------
        thickness : util2d array of floats (nodes,)

        """
        return self.__thickness

    def checklayerthickness(self):
        """
        Check layer thickness.

        """
        return (self.thickness > 0).all()

    def get_cell_volumes(self):
        """
        Get an array of cell volumes.

        Returns
        -------
        vol : array of floats (nodes)

        """
        vol = np.empty((self.nodes))
        for n in range(self.nodes):
            nn = n
            if self.ivsd == -1:
                nn = n % self.nodelay[0]
            area = self.area[nn]
            vol[n] = area * (self.top[n] - self.bot[n])
        return vol

    @property
    def zcentroids(self):
        """
        Return an array of size nodes that contains the vertical cell center
        elevation.

        """
        z = np.empty((self.nodes))
        z[:] = (self.top[:] - self.bot[:]) / 2.
        return z

    @property
    def ncpl(self):
        return self.nodes/self.nlay

    @staticmethod
    def load(f, model, ext_unit_dict=None, check=False):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.
        check : boolean
            Check package data for common errors. (default False; not setup yet)

        Returns
        -------
        dis : ModflowDisU object
            ModflowDisU object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> disu = flopy.modflow.ModflowDisU.load('test.disu', m)

        """

        if model.verbose:
            sys.stdout.write('loading disu package file...\n')

        if model.version != 'mfusg':
            msg = "Warning: model version was reset from " + \
                  "'{}' to 'mfusg' in order to load a DISU file".format(
                          model.version)
            print(msg)
            model.version = 'mfusg'

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # dataset 1
        if model.verbose:
            print('   loading NODES, NLAY, NJAG, IVSD, NPER, ITMUNI, LENUNI,'
                  ' IDSYMRD...')
        ll = line.strip().split()
        nodes = int(ll.pop(0))
        nlay = int(ll.pop(0))
        njag = int(ll.pop(0))
        ivsd = int(ll.pop(0))
        nper = int(ll.pop(0))
        # mimic urword behavior in case these values aren't present on line
        if len(ll) > 0:
            itmuni = int(ll.pop(0))
        else:
            itmuni = 0
        if len(ll) > 0:
            lenuni = int(ll.pop(0))
        else:
            lenuni = 0
        if len(ll) > 0:
            idsymrd = int(ll.pop(0))
        else:
            idsymrd = 0
        if model.verbose:
            print('   NODES {}'.format(nodes))
            print('   NLAY {}'.format(nlay))
            print('   NJAG {}'.format(njag))
            print('   IVSD {}'.format(ivsd))
            print('   NPER {}'.format(nper))
            print('   ITMUNI {}'.format(itmuni))
            print('   LENUNI {}'.format(lenuni))
            print('   IDSYMRD {}'.format(idsymrd))

        # Calculate njags
        njags = int((njag - nodes) / 2)
        if model.verbose:
            print('   NJAGS calculated as {}'.format(njags))

        # dataset 2 -- laycbd
        if model.verbose:
            print('   loading LAYCBD...')
        laycbd = np.empty((nlay,), np.int32)
        laycbd = read1d(f, laycbd)
        if model.verbose:
            print('   LAYCBD {}'.format(laycbd))

        # dataset 3 -- nodelay
        if model.verbose:
            print('   loading NODELAY...')
        nodelay = Util2d.load(f, model, (nlay,), np.int32, 'nodelay',
                              ext_unit_dict)
        if model.verbose:
            print('   NODELAY {}'.format(nodelay))

        # dataset 4 -- top
        if model.verbose:
            print('   loading TOP...')
        top = [0] * nlay
        for k in range(nlay):
            tpk = Util2d.load(f, model, (nodelay[k],), np.float32, 'top',
                              ext_unit_dict)
            top[k] = tpk
        if model.verbose:
            for k, tpk in enumerate(top):
                print('   TOP layer {}: {}'.format(k, tpk.array))

        # dataset 5 -- bot
        if model.verbose:
            print('   loading BOT...')
        bot = [0] * nlay
        for k in range(nlay):
            btk = Util2d.load(f, model, (nodelay[k],), np.float32, 'btk',
                              ext_unit_dict)
            bot[k] = btk
        if model.verbose:
            for k, btk in enumerate(bot):
                print('   BOT layer {}: {}'.format(k, btk.array))

        # dataset 6 -- area
        if model.verbose:
            print('   loading AREA...')
        if ivsd == -1:
            area = Util2d.load(f, model, (nodelay[0],), np.float32, 'area',
                               ext_unit_dict)
        else:
            area = [0] * nlay
            for k in range(nlay):
                ak = Util2d.load(f, model, (nodelay[k],), np.float32, 'ak',
                                 ext_unit_dict)
                area[k] = ak
        if model.verbose:
            for k, ak in enumerate(area):
                print('   AREA layer {}: {}'.format(k, ak))

        # dataset 7 -- iac
        if model.verbose:
            print('   loading IAC...')
        iac = Util2d.load(f, model, (nodes,), np.int32, 'iac', ext_unit_dict)
        if model.verbose:
            print('   IAC {}'.format(iac))

        # dataset 8 -- ja
        if model.verbose:
            print('   loading JA...')
        ja = Util2d.load(f, model, (njag,), np.int32, 'ja', ext_unit_dict)
        if model.verbose:
            print('   JA {}'.format(ja))

        # dataset 9 -- ivc
        ivc = None
        if ivsd == 1:
            if model.verbose:
                print('   loading IVC...')
            ivc = Util2d.load(f, model, (njag,), np.int32, 'ivc', ext_unit_dict)
            if model.verbose:
                print('   IVC {}'.format(ivc))

        # dataset 10a -- cl1
        cl1 = None
        if idsymrd == 1:
            if model.verbose:
                print('   loading CL1...')
            cl1 = Util2d.load(f, model, (njags,), np.float32, 'cl1',
                              ext_unit_dict)
            if model.verbose:
                print('   CL1 {}'.format(cl1))

        # dataset 10b -- cl2
        cl2 = None
        if idsymrd == 1:
            if model.verbose:
                print('   loading CL2...')
            cl2 = Util2d.load(f, model, (njags,), np.float32, 'cl2',
                              ext_unit_dict)
            if model.verbose:
                print('   CL2 {}'.format(cl2))

        # dataset 11 -- cl12
        cl12 = None
        if idsymrd == 0:
            if model.verbose:
                print('   loading CL12...')
            cl12 = Util2d.load(f, model, (njag,), np.float32, 'cl12',
                               ext_unit_dict)
            if model.verbose:
                print('   CL12 {}'.format(cl12))

        # dataset 12 -- fahl
        fahl = None
        if idsymrd == 0:
            n = njag
        elif idsymrd == 1:
            n = njags
        if model.verbose:
            print('   loading FAHL...')
        fahl = Util2d.load(f, model, (n,), np.float32, 'fahl', ext_unit_dict)
        if model.verbose:
            print('   FAHL {}'.format(fahl))

        # dataset 7 -- stress period info
        if model.verbose:
            print('   loading stress period data...')
        perlen = []
        nstp = []
        tsmult = []
        steady = []
        for k in range(nper):
            line = f.readline()
            a1, a2, a3, a4 = line.strip().split()[0:4]
            a1 = float(a1)
            a2 = int(a2)
            a3 = float(a3)
            if a4.upper() == 'TR':
                a4 = False
            else:
                a4 = True
            perlen.append(a1)
            nstp.append(a2)
            tsmult.append(a3)
            steady.append(a4)
        if model.verbose:
            print('   PERLEN {}'.format(perlen))
            print('   NSTP {}'.format(nstp))
            print('   TSMULT {}'.format(tsmult))
            print('   STEADY {}'.format(steady))

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowDisU.ftype())

        # create dis object instance
        disu = ModflowDisU(model, nodes=nodes, nlay=nlay, njag=njag, ivsd=ivsd,
                           nper=nper, itmuni=itmuni, lenuni=lenuni,
                           idsymrd=idsymrd, laycbd=laycbd, nodelay=nodelay,
                           top=top, bot=bot, area=area, iac=iac, ja=ja,
                           ivc=ivc, cl1=cl1, cl2=cl2, cl12=cl12, fahl=fahl,
                           perlen=perlen, nstp=nstp, tsmult=tsmult,
                           steady=steady, unitnumber=unitnumber,
                           filenames=filenames)

        # return dis object instance
        return disu

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing
        f_dis = open(self.fn_path, 'w')

        # Item 0: heading
        f_dis.write('{0:s}\n'.format(self.heading))

        # Item 1: NODES NLAY NJAG IVSD NPER ITMUNI LENUNI IDSYMRD
        s = ''
        for var in [self.nodes, self.nlay, self.njag, self.ivsd, self.nper,
                    self.itmuni, self.lenuni, self.idsymrd]:
            s += '{} '.format(var)
        f_dis.write(s + '\n')

        # Item 2: LAYCBD
        for k in range(self.nlay):
            f_dis.write('{0:3d}'.format(self.laycbd[k]))
        f_dis.write('\n')

        # Item 3: NODELAY
        f_dis.write(self.nodelay.get_file_entry())

        # Item 4: TOP
        f_dis.write(self.top.get_file_entry())

        # Item 5: BOT
        f_dis.write(self.bot.get_file_entry())

        # Item 6: AREA
        f_dis.write(self.area.get_file_entry())

        # Item 7: IAC
        f_dis.write(self.iac.get_file_entry())

        # Item 8: JA
        f_dis.write(self.ja.get_file_entry())

        # Item 9: IVC
        if self.ivsd == 1:
            f_dis.write(self.ivc.get_file_entry())

        # Item 10a: CL1
        if self.idsymrd == 1:
            f_dis.write(self.cl1.get_file_entry())

        # Item 10b: CL2
        if self.idsymrd == 1:
            f_dis.write(self.cl2.get_file_entry())

        # Item 11: CL12
        if self.idsymrd == 0:
            f_dis.write(self.cl12.get_file_entry())

        # Item 12: FAHL
        f_dis.write(self.fahl.get_file_entry())

        # Item 13: NPER, NSTP, TSMULT, Ss/tr
        for t in range(self.nper):
            f_dis.write('{0:14f}{1:14d}{2:10f} '.format(self.perlen[t],
                                                        self.nstp[t],
                                                        self.tsmult[t]))
            if self.steady[t]:
                f_dis.write(' {0:3s}\n'.format('SS'))
            else:
                f_dis.write(' {0:3s}\n'.format('TR'))

        # Close and return
        f_dis.close()
        return

    @staticmethod
    def ftype():
        return 'DISU'

    @staticmethod
    def defaultunit():
        return 11

            # def get_node_coordinates(self):
    #     """
    #     Get y, x, and z cell centroids.
    #
    #     Returns
    #     -------
    #     y : list of cell y-centroids
    #
    #     x : list of cell x-centroids
    #
    #     z : array of floats (nlay, nrow, ncol)
    #     """
    #     # In row direction
    #     y = np.empty((self.nrow))
    #     for r in range(self.nrow):
    #         if (r == 0):
    #             y[r] = self.delc[r] / 2.
    #         else:
    #             y[r] = y[r - 1] + (self.delc[r] + self.delc[r - 1]) / 2.
    #     # Invert y to convert to a cartesian coordinate system
    #     y = y[::-1]
    #     # In column direction
    #     x = np.empty((self.ncol))
    #     for c in range(self.ncol):
    #         if (c == 0):
    #             x[c] = self.delr[c] / 2.
    #         else:
    #             x[c] = x[c - 1] + (self.delr[c] + self.delr[c - 1]) / 2.
    #     # In layer direction
    #     z = np.empty((self.nlay, self.nrow, self.ncol))
    #     for l in range(self.nlay):
    #         if (l == 0):
    #             z[l, :, :] = (self.top[:, :] + self.botm[l, :, :]) / 2.
    #         else:
    #             z[l, :, :] = (self.botm[l - 1, :, :] + self.botm[l, :, :]) / 2.
    #     return y, x, z
    #
    # def get_lrc(self, nodes):
    #     """
    #     Get layer, row, column from a list of MODFLOW node numbers.
    #
    #     Returns
    #     -------
    #     v : list of tuples containing the layer (k), row (i),
    #         and column (j) for each node in the input list
    #     """
    #     if not isinstance(nodes, list):
    #         nodes = [nodes]
    #     nrc = self.nrow * self.ncol
    #     v = []
    #     for node in nodes:
    #         k = int(node / nrc)
    #         if (k * nrc) < node:
    #             k += 1
    #         ij = int(node - (k - 1) * nrc)
    #         i = int(ij / self.ncol)
    #         if (i * self.ncol) < ij:
    #             i += 1
    #         j = ij - (i - 1) * self.ncol
    #         v.append((k, i, j))
    #     return v
    #
    # def get_node(self, lrc_list):
    #     """
    #     Get node number from a list of MODFLOW layer, row, column tuples.
    #
    #     Returns
    #     -------
    #     v : list of MODFLOW nodes for each layer (k), row (i),
    #         and column (j) tuple in the input list
    #     """
    #     if not isinstance(lrc_list, list):
    #         lrc_list = [lrc_list]
    #     nrc = self.nrow * self.ncol
    #     v = []
    #     for [k, i, j] in lrc_list:
    #         node = int(((k - 1) * nrc) + ((i - 1) * self.ncol) + j)
    #         v.append(node)
    #     return v
    #
    # def read_from_cnf(self, cnf_file_name, n_per_line=0):
    #     """
    #     Read discretization information from an MT3D configuration file.
    #
    #     """
    #
    #     def getn(ii, jj):
    #         if (jj == 0):
    #             n = 1
    #         else:
    #             n = int(ii / jj)
    #             if (ii % jj != 0):
    #                 n = n + 1
    #
    #         return n
    #
    #     try:
    #         f_cnf = open(cnf_file_name, 'r')
    #
    #         # nlay, nrow, ncol
    #         line = f_cnf.readline()
    #         s = line.split()
    #         cnf_nlay = int(s[0])
    #         cnf_nrow = int(s[1])
    #         cnf_ncol = int(s[2])
    #
    #         # ncol column widths delr[c]
    #         line = ''
    #         for dummy in range(getn(cnf_ncol, n_per_line)):
    #             line = line + f_cnf.readline()
    #         cnf_delr = [float(s) for s in line.split()]
    #
    #         # nrow row widths delc[r]
    #         line = ''
    #         for dummy in range(getn(cnf_nrow, n_per_line)):
    #             line = line + f_cnf.readline()
    #         cnf_delc = [float(s) for s in line.split()]
    #
    #         # nrow * ncol htop[r, c]
    #         line = ''
    #         for dummy in range(getn(cnf_nrow * cnf_ncol, n_per_line)):
    #             line = line + f_cnf.readline()
    #         cnf_top = [float(s) for s in line.split()]
    #         cnf_top = np.reshape(cnf_top, (cnf_nrow, cnf_ncol))
    #
    #         # nlay * nrow * ncol layer thickness dz[l, r, c]
    #         line = ''
    #         for dummy in range(
    #                 getn(cnf_nlay * cnf_nrow * cnf_ncol, n_per_line)):
    #             line = line + f_cnf.readline()
    #         cnf_dz = [float(s) for s in line.split()]
    #         cnf_dz = np.reshape(cnf_dz, (cnf_nlay, cnf_nrow, cnf_ncol))
    #
    #         # cinact, cdry, not used here so commented
    #         '''line = f_cnf.readline()
    #         s = line.split()
    #         cinact = float(s[0])
    #         cdry = float(s[1])'''
    #
    #         f_cnf.close()
    #     finally:
    #         self.nlay = cnf_nlay
    #         self.nrow = cnf_nrow
    #         self.ncol = cnf_ncol
    #
    #         self.delr = Util2d(model, (self.ncol,), np.float32, cnf_delr,
    #                             name='delr', locat=self.unit_number[0])
    #         self.delc = Util2d(model, (self.nrow,), np.float32, cnf_delc,
    #                             name='delc', locat=self.unit_number[0])
    #         self.top = Util2d(model, (self.nrow, self.ncol), np.float32,
    #                            cnf_top, name='model_top',
    #                            locat=self.unit_number[0])
    #
    #         cnf_botm = np.empty((self.nlay + sum(self.laycbd), self.nrow,
    #                              self.ncol))
    #
    #         # First model layer
    #         cnf_botm[0:, :, :] = cnf_top - cnf_dz[0, :, :]
    #         # All other layers
    #         for l in range(1, self.nlay):
    #             cnf_botm[l, :, :] = cnf_botm[l - 1, :, :] - cnf_dz[l, :, :]
    #
    #         self.botm = Util3d(model, (self.nlay + sum(self.laycbd),
    #                                     self.nrow, self.ncol), np.float32,
    #                             cnf_botm, 'botm',
    #                             locat=self.unit_number[0])
    #
    # def gettop(self):
    #     """
    #     Get the top array.
    #
    #     Returns
    #     -------
    #     top : array of floats (nrow, ncol)
    #     """
    #     return self.top.array
    #
    # def getbotm(self, k=None):
    #     """
    #     Get the bottom array.
    #
    #     Returns
    #     -------
    #     botm : array of floats (nlay, nrow, ncol), or
    #
    #     botm : array of floats (nrow, ncol) if k is not none
    #     """
    #     if k is None:
    #         return self.botm.array
    #     else:
    #         return self.botm.array[k, :, :]
    #
    # def check(self, f=None, verbose=True, level=1):
    #     """
    #     Check dis package data for zero and negative thicknesses.
    #
    #     Parameters
    #     ----------
    #     f : str or file handle
    #         String defining file name or file handle for summary file
    #         of check method output. If a sting is passed a file handle
    #         is created. If f is None, check method does not write
    #         results to a summary file. (default is None)
    #     verbose : bool
    #         Boolean flag used to determine if check method results are
    #         written to the screen
    #     level : int
    #         Check method analysis level. If level=0, summary checks are
    #         performed. If level=1, full checks are performed.
    #
    #     Returns
    #     -------
    #     None
    #
    #     Examples
    #     --------
    #
    #     >>> import flopy
    #     >>> m = flopy.modflow.Modflow.load('model.nam')
    #     >>> m.dis.check()
    #     """
    #     if f is not None:
    #         if isinstance(f, str):
    #             pth = os.path.join(self.parent.model_ws, f)
    #             f = open(pth, 'w', 0)
    #
    #     errors = False
    #     txt = '\n{} PACKAGE DATA VALIDATION:\n'.format(self.name[0])
    #     t = ''
    #     t1 = ''
    #     inactive = self.parent.bas6.ibound.array == 0
    #     # thickness errors
    #     d = self.thickness.array
    #     d[inactive] = 1.
    #     if d.min() <= 0:
    #         errors = True
    #         t = '{}  ERROR: Negative or zero cell thickness specified.\n'.format(
    #             t)
    #         if level > 0:
    #             idx = np.column_stack(np.where(d <= 0.))
    #             t1 = self.level1_arraylist(idx, d, self.thickness.name, t1)
    #     else:
    #         t = '{}  Specified cell thickness is OK.\n'.format(t)
    #
    #     # add header to level 0 text
    #     txt += t
    #
    #     if level > 0:
    #         if errors:
    #             txt += '\n  DETAILED SUMMARY OF {} ERRORS:\n'.format(
    #                 self.name[0])
    #             # add level 1 header to level 1 text
    #             txt += t1
    #
    #     # write errors to summary file
    #     if f is not None:
    #         f.write('{}\n'.format(txt))
    #
    #     # write errors to stdout
    #     if verbose:
    #         print(txt)

