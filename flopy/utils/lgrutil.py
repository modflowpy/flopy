import numpy as np

from ..discretization import StructuredGrid
from ..modflow import Modflow
from .cvfdutil import get_disv_gridprops
from .util_array import Util2d, Util3d


class SimpleRegularGrid:
    """
    Simple object for representing regular MODFLOW grid information.

    Parameters
    ----------
    nlay : int
        number of layers
    nrow : int
        number of rows
    ncol : int
        number of columns
    delr : ndarray
        delr array
    delc : ndarray
        delc array
    top : ndarray
        top array (nrow, ncol)
    botm : ndarray
        botm array (nlay, nrow, ncol)
    idomain : ndarray
        idomain array (nlay, nrow, ncol)
    xorigin : float
        x location of grid lower left corner
    yorigin : float
        y location of grid lower left corner
    """

    def __init__(
        self,
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        top,
        botm,
        idomain,
        xorigin,
        yorigin,
    ):
        # enforce compliance
        assert delr.shape == (ncol,)
        assert delc.shape == (nrow,)
        assert top.shape == (nrow, ncol)
        assert botm.shape == (nlay, nrow, ncol)
        assert idomain.shape == (nlay, nrow, ncol)

        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.delr = delr
        self.delc = delc
        self.top = top
        self.botm = botm
        self.idomain = idomain
        self.xorigin = xorigin
        self.yorigin = yorigin
        return

    @property
    def modelgrid(self):
        mg = StructuredGrid(
            delc=self.delc,
            delr=self.delr,
            top=self.top,
            botm=self.botm,
            idomain=self.idomain,
            xoff=self.xorigin,
            yoff=self.yorigin,
        )
        return mg

    def get_gridprops_dis6(self):
        gridprops = {
            "xorigin": self.xorigin,
            "yorigin": self.yorigin,
            "nlay": self.nlay,
            "nrow": self.nrow,
            "ncol": self.ncol,
            "delr": self.delr,
            "delc": self.delc,
            "top": self.top,
            "botm": self.botm,
            "idomain": self.idomain,
        }
        return gridprops


class Lgr:
    def __init__(
        self,
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=3,
        ncppl=1,
        xllp=0.0,
        yllp=0.0,
    ):
        """

        Parameters
        ----------
        nlayp : int
            parent layers
        nrowp : int
            parent number of rows
        ncolp : int
            parent number of columns
        delrp : ndarray
            parent delr array
        delcp : ndarray
            parent delc array
        topp : ndarray
            parent top array (nrowp, ncolp)
        botmp : ndarray
            parent botm array (nlayp, nrowp, ncolp)
        idomainp : ndarray
            parent idomain array used to create the child grid.  Ones indicate
            a parent cell and zeros indicate a child cell.  The domain of the
            child grid will span a rectangular region that spans all idomain
            cells with a value of zero. idomain must be of shape
            (nlayp, nrowp, ncolp)
        ncpp : int
            number of child cells along the face of a parent cell
        ncppl : list of ints
            number of child layers per parent layer
        xllp : float
            x location of parent grid lower left corner
        yllp : float
            y location of parent grid lower left corner

        """

        # parent grid properties
        self.nlayp = nlayp
        self.nrowp = nrowp
        self.ncolp = ncolp

        m = Modflow()
        self.delrp = Util2d(m, (ncolp,), np.float32, delrp, "delrp").array
        self.delcp = Util2d(m, (nrowp,), np.float32, delcp, "delcp").array
        self.topp = Util2d(m, (nrowp, ncolp), np.float32, topp, "topp").array
        self.botmp = Util3d(m, (nlayp, nrowp, ncolp), np.float32, botmp, "botmp").array

        # idomain
        assert idomainp.shape == (nlayp, nrowp, ncolp)
        self.idomain = idomainp
        idxl, idxr, idxc = np.asarray(idomainp == 0).nonzero()
        assert idxl.shape[0] > 0, "no zero values found in idomain"

        # child cells per parent and child cells per parent layer
        self.ncpp = ncpp
        self.ncppl = Util2d(m, (nlayp,), np.int32, ncppl, "ncppl").array

        # calculate ibcl which is the bottom child layer (one based) in each
        # parent layer
        self.ibcl = np.zeros(self.nlayp, dtype=int)
        self.ibcl[0] = self.ncppl[0]
        for k in range(1, self.nlayp):
            if self.ncppl[k] > 0:
                self.ibcl[k] = self.ibcl[k - 1] + self.ncppl[k]

        # parent lower left
        self.xllp = xllp
        self.yllp = yllp

        # child grid properties
        self.nplbeg = int(idxl.min())
        self.nplend = int(idxl.max())
        self.npcbeg = int(idxc.min())
        self.npcend = int(idxc.max())
        self.nprbeg = int(idxr.min())
        self.nprend = int(idxr.max())

        # child grid dimensions
        self.nlay = int(self.ncppl.sum())
        self.nrow = (self.nprend - self.nprbeg + 1) * ncpp
        self.ncol = (self.npcend - self.npcbeg + 1) * ncpp

        # assign child properties
        self.delr, self.delc = self.get_delr_delc()
        self.top, self.botm = self.get_top_botm()
        self.xll = xllp + float(self.delrp[0 : self.npcbeg].sum())
        self.yll = yllp + float(self.delcp[self.nprend + 1 :].sum())

    def get_shape(self):
        """
        Return the shape of the child grid

        Returns
        -------
        (nlay, nrow, ncol) : tuple
            shape of the child grid

        """
        return self.nlay, self.nrow, self.ncol

    def get_lower_left(self):
        """
        Return the lower left corner of the child grid

        Returns
        -------
        (xll, yll) : tuple
            location of lower left corner of the child grid

        """
        return self.xll, self.yll

    def get_delr_delc(self):
        # create the delr and delc arrays for this child grid
        delr = np.zeros((self.ncol), dtype=float)
        delc = np.zeros((self.nrow), dtype=float)
        jstart = 0
        jend = self.ncpp
        for j in range(self.npcbeg, self.npcend + 1):
            delr[jstart:jend] = float(self.delrp[j]) / self.ncpp
            jstart = jend
            jend = jstart + self.ncpp
        istart = 0
        iend = self.ncpp
        for i in range(self.nprbeg, self.nprend + 1):
            delc[istart:iend] = float(self.delcp[i]) / self.ncpp
            istart = iend
            iend = istart + self.ncpp
        return delr, delc

    def get_top_botm(self):
        bt = self.botmp
        tp = self.topp
        shp = tp.shape
        tp = tp.reshape(1, shp[0], shp[1])
        pbotm = np.vstack((tp, bt))
        botm = np.zeros((self.nlay + 1, self.nrow, self.ncol), dtype=float)
        for ip in range(self.nprbeg, self.nprend + 1):
            for jp in range(self.npcbeg, self.npcend + 1):
                top = float(pbotm[0, ip, jp])
                icrowstart = (ip - self.nprbeg) * self.ncpp
                icrowend = icrowstart + self.ncpp
                iccolstart = (jp - self.npcbeg) * self.ncpp
                iccolend = iccolstart + self.ncpp
                botm[0, icrowstart:icrowend, iccolstart:iccolend] = top
                kc = 1
                for kp in range(self.nplbeg, self.nplend + 1):
                    top = float(pbotm[kp, ip, jp])
                    bot = float(pbotm[kp + 1, ip, jp])
                    dz = (top - bot) / self.ncppl[kp]
                    for _ in range(self.ncppl[kp]):
                        botm[kc, icrowstart:icrowend, iccolstart:iccolend] = (
                            botm[kc - 1, icrowstart:icrowend, iccolstart:iccolend] - dz
                        )
                        kc += 1
        return botm[0], botm[1:]

    def get_replicated_parent_array(self, parent_array):
        """
        Get a two-dimensional array the size of the child grid that has values
        replicated from the provided parent array.

        Parameters
        ----------
        parent_array : ndarray
            A two-dimensional array that is the size of the parent model rows
            and columns.

        Returns
        -------
        child_array : ndarray
            A two-dimensional array that is the size of the child model rows
            and columns

        """
        assert parent_array.shape == (self.nrowp, self.ncolp)
        child_array = np.empty((self.nrow, self.ncol), dtype=parent_array.dtype)
        for ip in range(self.nprbeg, self.nprend + 1):
            for jp in range(self.npcbeg, self.npcend + 1):
                icrowstart = (ip - self.nprbeg) * self.ncpp
                icrowend = icrowstart + self.ncpp
                iccolstart = (jp - self.npcbeg) * self.ncpp
                iccolend = iccolstart + self.ncpp
                value = int(parent_array[ip, jp])
                child_array[icrowstart:icrowend, iccolstart:iccolend] = value
        return child_array

    def get_idomain(self):
        """
        Return the idomain array for the child model.  This will normally
        be all ones unless the idomain array for the parent model is
        non-rectangular and irregularly shaped.  Then, parts of the child
        model will have idomain zero cells.

        Returns
        -------
        idomain : ndarray
            idomain array for the child model

        """
        idomain = np.ones((self.nlay, self.nrow, self.ncol), dtype=int)
        for kc in range(self.nlay):
            for ic in range(self.nrow):
                for jc in range(self.ncol):
                    kp, ip, jp = self.get_parent_indices(kc, ic, jc)
                    if self.idomain[kp, ip, jp] == 1:
                        idomain[kc, ic, jc] = 0
        return idomain

    def get_parent_indices(self, kc, ic, jc):
        """
        Method returns the parent cell indices for this child.
        The returned indices are in zero-based indexing.

        """
        ip = self.nprbeg + int(ic / self.ncpp)
        jp = self.npcbeg + int(jc / self.ncpp)
        kp = 0
        kcstart = 0
        for k in range(self.nplbeg, self.nplend + 1):
            kcend = kcstart + self.ncppl[k] - 1
            if kcstart <= kc <= kcend:
                kp = k
                break
            kcstart = kcend + 1
        return kp, ip, jp

    def get_parent_connections(self, kc, ic, jc):
        """
        Return a list of parent cell indices that are connected to child
        cell kc, ic, jc.

        """

        assert 0 <= kc < self.nlay, "layer must be >= 0 and < child nlay"
        assert 0 <= ic < self.nrow, "layer must be >= 0 and < child nrow"
        assert 0 <= jc < self.ncol, "layer must be >= 0 and < child ncol"

        parentlist = []
        (kp, ip, jp) = self.get_parent_indices(kc, ic, jc)

        # parent cell to left
        if jc % self.ncpp == 0:
            if jp - 1 >= 0:
                if self.idomain[kp, ip, jp - 1] != 0:
                    parentlist.append(((kp, ip, jp - 1), -1))

        # parent cell to right
        if (jc + 1) % self.ncpp == 0:
            if jp + 1 < self.ncolp:
                if self.idomain[kp, ip, jp + 1] != 0:
                    parentlist.append(((kp, ip, jp + 1), 1))

        # parent cell to back
        if ic % self.ncpp == 0:
            if ip - 1 >= 0:
                if self.idomain[kp, ip - 1, jp] != 0:
                    parentlist.append(((kp, ip - 1, jp), 2))

        # parent cell to front
        if (ic + 1) % self.ncpp == 0:
            if ip + 1 < self.nrowp:
                if self.idomain[kp, ip + 1, jp] != 0:
                    parentlist.append(((kp, ip + 1, jp), -2))

        # parent cell to top is not possible

        # parent cell to bottom
        if kc + 1 == self.ibcl[kp]:
            if kp + 1 < self.nlayp:
                if self.idomain[kp + 1, ip, jp] != 0:
                    parentlist.append(((kp + 1, ip, jp), -3))

        return parentlist

    def get_exchange_data(self, angldegx=False, cdist=False):
        """
        Get the list of parent/child connections

        <cellidm1> <cellidm2> <ihc> <cl1> <cl2> <hwva> <angledegx>

        Returns
        -------
            exglist : list
                list of connections between parent and child

        """

        exglist = []
        nlayc = self.nlay
        nrowc = self.nrow
        ncolc = self.ncol
        delrc = self.delr
        delcc = self.delc
        delrp = self.delrp
        delcp = self.delcp
        topp = self.topp
        botp = self.botmp
        topc = self.top
        botc = self.botm

        if cdist:
            # child xy meshgrid
            xc = np.add.accumulate(delrc) - 0.5 * delrc
            Ly = np.add.reduce(delcc)
            yc = Ly - (np.add.accumulate(delcc) - 0.5 * delcc)
            xc += self.xll
            yc += self.yll
            xc, yc = np.meshgrid(xc, yc)

            # parent xy meshgrid
            xp = np.add.accumulate(delrp) - 0.5 * delrp
            Ly = np.add.reduce(delcp)
            yp = Ly - (np.add.accumulate(delcp) - 0.5 * delcp)
            xc += self.xllp
            yc += self.yllp
            xp, yp = np.meshgrid(xp, yp)

        cidomain = self.get_idomain()

        for kc in range(nlayc):
            for ic in range(nrowc):
                for jc in range(ncolc):
                    plist = self.get_parent_connections(kc, ic, jc)
                    for (kp, ip, jp), idir in plist:
                        if cidomain[kc, ic, jc] == 0:
                            continue

                        # horizontal or vertical connection
                        # 1 if a child cell horizontally connected to a parent
                        #   cell
                        # 2 if more than one child cells horizontally connected
                        #   to parent cell
                        # 0 if a vertical connection

                        ihc = 1
                        if self.ncppl[kp] > 1:
                            ihc = 2
                        if abs(idir) == 3:
                            ihc = 0

                        # angldegx
                        angle = None
                        if angldegx:
                            angle = 180.0  # -x, west
                            if idir == 2:
                                angle = 270.0  # -y, south
                            elif idir == -1:
                                angle = 0.0  # +x, east
                            elif idir == -2:
                                angle = 90.0  # +y, north

                        # vertical connection
                        cl1 = None
                        cl2 = None
                        hwva = None

                        tpp = float(topp[ip, jp])
                        btp = float(botp[kp, ip, jp])
                        if kp > 0:
                            tpp = float(botp[kp - 1, ip, jp])

                        tpc = float(topc[ic, jc])
                        btc = float(botc[kc, ic, jc])
                        if kc > 0:
                            tpc = float(botc[kc - 1, ic, jc])

                        if ihc == 0:
                            cl1 = 0.5 * (tpp - btp)
                            cl2 = 0.5 * (tpc - btc)
                            hwva = float(delrc[jc]) * float(delcc[ic])
                        else:
                            if abs(idir) == 1:
                                cl1 = 0.5 * float(delrp[jp])
                                cl2 = 0.5 * float(delrc[jc])
                                hwva = float(delcc[ic])
                            elif abs(idir) == 2:
                                cl1 = 0.5 * float(delcp[ip])
                                cl2 = 0.5 * float(delcc[ic])
                                hwva = delrc[jc]

                        # connection distance
                        cd = None
                        if cdist:
                            if abs(idir) == 3:
                                cd = cl1 + cl2
                            else:
                                x1 = float(xc[ic, jc])
                                y1 = float(yc[ic, jc])
                                x2 = float(xp[ip, jp])
                                y2 = float(yp[ip, jp])
                                cd = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                        exg = [(kp, ip, jp), (kc, ic, jc), ihc, cl1, cl2, hwva]
                        if angldegx:
                            exg.append(float(angle))
                        if cdist:
                            exg.append(float(cd))
                        exglist.append(exg)
        return exglist

    @property
    def parent(self):
        """
        Return a SimpleRegularGrid object for the parent model

        Returns
        -------
            simple_regular_grid : SimpleRegularGrid
                simple grid object containing grid information for the parent

        """
        simple_regular_grid = SimpleRegularGrid(
            self.nlayp,
            self.nrowp,
            self.ncolp,
            self.delrp,
            self.delcp,
            self.topp,
            self.botmp,
            self.idomain,
            self.xllp,
            self.yllp,
        )
        return simple_regular_grid

    @property
    def child(self):
        """
        Return a SimpleRegularGrid object for the child model

        Returns
        -------
            simple_regular_grid : SimpleRegularGrid
                simple grid object containing grid information for the child

        """
        delrc, delcc = self.get_delr_delc()
        idomainc = self.get_idomain()  # child idomain
        topc = self.top
        botmc = self.botm
        child_dis_shp = self.get_shape()
        nlayc = child_dis_shp[0]
        nrowc = child_dis_shp[1]
        ncolc = child_dis_shp[2]
        xorigin = self.xll
        yorigin = self.yll
        simple_regular_grid = SimpleRegularGrid(
            nlayc, nrowc, ncolc, delrc, delcc, topc, botmc, idomainc, xorigin, yorigin
        )
        return simple_regular_grid

    def to_disv_gridprops(self):
        """
        Create and return a gridprops dictionary that can be
        used to create a disv grid (instead of a separate parent
        and child representation).  The gridprops dictionary can
        be unpacked into the flopy.mf6.Modflowdisv() constructor
        and flopy.discretization.VertexGrid() constructor.

        Note that export capability will only work if the parent
        and child models have corresponding layers.

        Returns
        -------
        gridprops : dict
            Dictionary containing ncpl, nvert, vertices, cell2d,
            nlay, top, and botm

        """
        return LgrToDisv(self).get_disv_gridprops()


class LgrToDisv:
    def __init__(self, lgr):
        """
        Helper class used to convert and Lgr() object into
        the grid properties needed to create a disv vertex
        nested grid.  After instantiation, self.verts and
        self.iverts are available.

        The primary work of this class is identify hanging
        vertices along the shared parent-child boundary and
        include these hanging vertices in the vertex indicence
        list for parent cells.

        Parameters
        ----------
        lgr : Lgr instance
            Lgr() object describing a parent-child relation

        """

        # store information
        self.lgr = lgr
        self.pgrid = lgr.parent.modelgrid
        self.cgrid = lgr.child.modelgrid

        # count active parent and child cells
        self.ncpl_parent = np.count_nonzero(self.pgrid.idomain[0] > 0)
        self.ncpl_child = np.count_nonzero(self.cgrid.idomain[0] > 0)
        self.ncpl = self.ncpl_child + self.ncpl_parent

        # find child vertices that act as hanging vertices on parent
        # model cells
        self.right_face_hanging = None
        self.left_face_hanging = None
        self.front_face_hanging = None
        self.back_face_hanging = None
        self.parent_ij_to_global = None
        self.child_ij_to_global = None
        self.find_hanging_vertices()

        # build global verts and iverts keeping only idomain > 0
        self.verts = None
        self.iverts = None
        self.build_verts_iverts()

        # todo: remove unused vertices?

    def find_hanging_vertices(self):
        """
        Hanging vertices are vertices that must be included
        along the edge of parent cells.  These hanging vertices
        mark the locations of corners of adjacent child cells.
        Hanging vertices are not strictly
        necessary to define the shape of a parent cell, but they are
        required by modflow to describe connections between
        parent and child cells.

        This routine finds hanging vertices parent cells along
        a parent-child boundary.  These hanging vertices are
        stored in 4 member dictionaries, called right_face_hanging,
        left_face_hanging, front_face_hanging, and back_face_hanging.
        These dictionaries are used subsequently to insert
        hanging vertices into the iverts array.

        """

        # create dictionaries for parent left, right, back, and front
        # faces that have a key that is parent (row, col)
        # and a value that is a list of child vertex numbers

        # this list of child vertex numbers will be ordered from
        # left to right (back/front) and from back to front (left/right)
        # so when they are used later, two of them will need to be
        # reversed so that clockwise ordering is maintained

        nrowc = self.lgr.nrow
        ncolc = self.lgr.ncol
        iverts = self.cgrid.iverts
        cidomain = self.lgr.get_idomain()

        self.right_face_hanging = {}
        self.left_face_hanging = {}
        self.front_face_hanging = {}
        self.back_face_hanging = {}

        # map (i, j) to global cell number
        self.parent_ij_to_global = {}
        self.child_ij_to_global = {}

        kc = 0
        nodec = 0
        for ic in range(nrowc):
            for jc in range(ncolc):
                plist = self.lgr.get_parent_connections(kc, ic, jc)
                for (kp, ip, jp), idir in plist:
                    if cidomain[kc, ic, jc] == 0:
                        continue

                    if idir == -1:  # left child face connected to right parent face
                        # child vertices 0 and 3 added as hanging nodes
                        if (ip, jp) in self.right_face_hanging:
                            hlist = self.right_face_hanging.pop((ip, jp))
                        else:
                            hlist = []
                        ivlist = iverts[nodec]
                        for iv in (ivlist[0], ivlist[3]):
                            if iv not in hlist:
                                hlist.append(iv)
                        self.right_face_hanging[(ip, jp)] = hlist

                    elif idir == 1:
                        # child vertices 1 and 2 added as hanging nodes
                        if (ip, jp) in self.left_face_hanging:
                            hlist = self.left_face_hanging.pop((ip, jp))
                        else:
                            hlist = []
                        ivlist = iverts[nodec]
                        for iv in (ivlist[1], ivlist[2]):
                            if iv not in hlist:
                                hlist.append(iv)
                        self.left_face_hanging[(ip, jp)] = hlist

                    elif idir == 2:
                        # child vertices 0 and 1 added as hanging nodes
                        if (ip, jp) in self.front_face_hanging:
                            hlist = self.front_face_hanging.pop((ip, jp))
                        else:
                            hlist = []
                        ivlist = iverts[nodec]
                        for iv in (ivlist[0], ivlist[1]):
                            if iv not in hlist:
                                hlist.append(iv)
                        self.front_face_hanging[(ip, jp)] = hlist

                    elif idir == -2:
                        # child vertices 3 and 2 added as hanging nodes
                        if (ip, jp) in self.back_face_hanging:
                            hlist = self.back_face_hanging.pop((ip, jp))
                        else:
                            hlist = []
                        ivlist = iverts[nodec]
                        for iv in (ivlist[3], ivlist[2]):
                            if iv not in hlist:
                                hlist.append(iv)
                        self.back_face_hanging[(ip, jp)] = hlist

                nodec += 1

    def build_verts_iverts(self):
        """
        Build the verts and iverts members.  self.verts is a 2d
        numpy array of size (nvert, 2).  Column 1 is x and column 2
        is y.  self.iverts is a list of size ncpl (number of cells
        per layer) with each entry being the list of vertex indices
        that define the cell.

        """

        # stack vertex arrays; these will have more points than necessary,
        # because parent and child vertices will overlap at corners, but
        # duplicate vertices will be filtered later
        pverts = self.pgrid.verts
        cverts = self.cgrid.verts
        nverts_parent = pverts.shape[0]
        nverts_child = cverts.shape[0]
        verts = np.vstack((pverts, cverts))

        # build iverts list first with active parent cells
        iverts = []
        iglo = 0
        for i in range(self.pgrid.nrow):
            for j in range(self.pgrid.ncol):
                if self.pgrid.idomain[0, i, j] > 0:
                    ivlist = self.pgrid._build_structured_iverts(i, j)

                    # merge hanging vertices if they exist
                    ivlist = self.merge_hanging_vertices(i, j, ivlist)

                    iverts.append(ivlist)
                    self.parent_ij_to_global[(i, j)] = iglo
                    iglo += 1

        # now add active child cells
        for i in range(self.cgrid.nrow):
            for j in range(self.cgrid.ncol):
                if self.cgrid.idomain[0, i, j] > 0:
                    ivlist = [
                        iv + nverts_parent
                        for iv in self.cgrid._build_structured_iverts(i, j)
                    ]
                    iverts.append(ivlist)
                    self.child_ij_to_global[(i, j)] = iglo
                    iglo += 1
        self.verts = verts
        self.iverts = iverts

    def merge_hanging_vertices(self, ip, jp, ivlist):
        """
        Given a list of vertices (ivlist) for parent row and column
        (ip, jp) merge hanging vertices from adjacent child cells
        into ivlist.

        Parameters
        ----------
        ip : int
            parent cell row number

        jp : int
            parent cell column number

        ivlist : list of ints
            list of vertex indices that define the parent
            cell (ip, jp)

        Returns
        -------
        ivlist : list of ints
            modified list of vertices that now also contains
            any hanging vertices needed to properly define
            a parent cell adjacent to child cells

        """
        assert len(ivlist) == 4
        child_ivlist_offset = self.pgrid.verts.shape[0]

        # construct back edge
        idx = 0
        reverse = False
        face_hanging = self.back_face_hanging
        back_edge = [ivlist[idx]]
        if (ip, jp) in face_hanging:
            hlist = face_hanging[(ip, jp)]
            if len(hlist) > 2:
                hlist = hlist[1:-1]  # do not include two ends
                hlist = [h + child_ivlist_offset for h in hlist]
                if reverse:
                    hlist = hlist[::-1]
            else:
                hlist = []
            back_edge = [ivlist[idx]] + hlist

        # construct right edge
        idx = 1
        reverse = False
        face_hanging = self.right_face_hanging
        right_edge = [ivlist[idx]]
        if (ip, jp) in face_hanging:
            hlist = face_hanging[(ip, jp)]
            if len(hlist) > 2:
                hlist = hlist[1:-1]  # do not include two ends
                hlist = [h + child_ivlist_offset for h in hlist]
                if reverse:
                    hlist = hlist[::-1]
            else:
                hlist = []
            right_edge = [ivlist[idx]] + hlist

        # construct front edge
        idx = 2
        reverse = True
        face_hanging = self.front_face_hanging
        front_edge = [ivlist[idx]]
        if (ip, jp) in face_hanging:
            hlist = face_hanging[(ip, jp)]
            if len(hlist) > 2:
                hlist = hlist[1:-1]  # do not include two ends
                hlist = [h + child_ivlist_offset for h in hlist]
                if reverse:
                    hlist = hlist[::-1]
            else:
                hlist = []
            front_edge = [ivlist[idx]] + hlist

        # construct left edge
        idx = 3
        reverse = True
        face_hanging = self.left_face_hanging
        left_edge = [ivlist[idx]]
        if (ip, jp) in face_hanging:
            hlist = face_hanging[(ip, jp)]
            if len(hlist) > 2:
                hlist = hlist[1:-1]  # do not include two ends
                hlist = [h + child_ivlist_offset for h in hlist]
                if reverse:
                    hlist = hlist[::-1]
            else:
                hlist = []
            left_edge = [ivlist[idx]] + hlist

        ivlist = back_edge + right_edge + front_edge + left_edge

        return ivlist

    def get_xcyc(self):
        """
        Construct a 2d array of size (nvert, 2) that
        contains the cell centers.

        Returns
        -------
        xcyc : ndarray
            2d array of x, y positions for cell centers

        """
        xcyc = np.empty((self.ncpl, 2))
        pidx = self.pgrid.idomain[0] > 0
        cidx = self.cgrid.idomain[0] > 0
        px = self.pgrid.xcellcenters[pidx].flatten()
        cx = self.cgrid.xcellcenters[cidx].flatten()
        xcyc[:, 0] = np.vstack((np.atleast_2d(px).T, np.atleast_2d(cx).T)).flatten()
        py = self.pgrid.ycellcenters[pidx].flatten()
        cy = self.cgrid.ycellcenters[cidx].flatten()
        xcyc[:, 1] = np.vstack((np.atleast_2d(py).T, np.atleast_2d(cy).T)).flatten()
        return xcyc

    def get_top(self):
        """
        Construct a 1d array of size (ncpl) that
        contains the cell tops.

        Returns
        -------
        top : ndarray
            1d array of top elevations

        """
        top = np.empty((self.ncpl,))
        pidx = self.pgrid.idomain[0] > 0
        cidx = self.cgrid.idomain[0] > 0
        pa = self.pgrid.top[pidx].flatten()
        ca = self.cgrid.top[cidx].flatten()
        top[:] = np.hstack((pa, ca))
        return top

    def get_botm(self):
        """
        Construct a 2d array of size (nlay, ncpl) that
        contains the cell bottoms.

        Returns
        -------
        botm : ndarray
            2d array of bottom elevations

        """
        botm = np.empty((self.lgr.nlay, self.ncpl))
        pidx = self.pgrid.idomain[0] > 0
        cidx = self.cgrid.idomain[0] > 0
        for k in range(self.lgr.nlay):
            pa = self.pgrid.botm[k, pidx].flatten()
            ca = self.cgrid.botm[k, cidx].flatten()
            botm[k, :] = np.hstack((pa, ca))
        return botm

    def get_disv_gridprops(self):
        """
        Create and return a gridprops dictionary that can be
        used to create a disv grid (instead of a separate parent
        and child representation).  The gridprops dictionary can
        be unpacked into the flopy.mf6.Modflowdisv() constructor
        and flopy.discretization.VertexGrid() constructor.

        Note that export capability will only work if the parent
        and child models have corresponding layers.

        Returns
        -------
        gridprops : dict
            Dictionary containing ncpl, nvert, vertices, cell2d,
            nlay, top, and botm

        """

        # check
        assert self.lgr.ncppl.min() == self.lgr.ncppl.max(), (
            "Exporting disv grid properties requires ncppl to be 1."
        )
        assert self.lgr.nlayp == self.lgr.nlay, (
            "Exporting disv grid properties requires parent and child models "
            "to have the same number of layers."
        )
        for k in range(self.lgr.nlayp - 1):
            assert np.allclose(self.lgr.idomain[k], self.lgr.idomain[k + 1]), (
                "Exporting disv grid properties requires parent idomain "
                "is same for all layers."
            )

        # get information and build gridprops
        xcyc = self.get_xcyc()
        top = self.get_top()
        botm = self.get_botm()
        gridprops = get_disv_gridprops(self.verts, self.iverts, xcyc=xcyc)
        gridprops["nlay"] = self.lgr.nlay
        gridprops["top"] = top
        gridprops["botm"] = botm
        return gridprops
