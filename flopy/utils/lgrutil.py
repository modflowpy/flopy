import numpy as np
from ..modflow import Modflow
from .util_array import Util2d, Util3d


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
        parent : flopy.modflow.Modflow
            parent model
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
        self.botmp = Util3d(
            m, (nlayp, nrowp, ncolp), np.float32, botmp, "botmp"
        ).array

        # idomain
        assert idomainp.shape == (nlayp, nrowp, ncolp)
        self.idomain = idomainp
        idxl, idxr, idxc = np.where(idomainp == 0)
        assert idxl.shape[0] > 1, "no zero values found in idomain"

        # # child cells per parent and child cells per parent layer
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
        self.nplbeg = idxl.min()
        self.nplend = idxl.max()
        self.npcbeg = idxc.min()
        self.npcend = idxc.max()
        self.nprbeg = idxr.min()
        self.nprend = idxr.max()

        # child grid dimensions
        self.nlay = self.ncppl.sum()
        self.nrow = (self.nprend - self.nprbeg + 1) * ncpp
        self.ncol = (self.npcend - self.npcbeg + 1) * ncpp

        # assign child properties
        self.delr, self.delc = self.get_delr_delc()
        self.top, self.botm = self.get_top_botm()
        self.xll = xllp + self.delrp[0 : self.npcbeg].sum()
        self.yll = yllp + self.delcp[self.nprend + 1 :].sum()

        return

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
            delr[jstart:jend] = self.delrp[j - 1] / self.ncpp
            jstart = jend
            jend = jstart + self.ncpp
        istart = 0
        iend = self.ncpp
        for i in range(self.nprbeg, self.nprend + 1):
            delc[istart:iend] = self.delcp[i - 1] / self.ncpp
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
                top = pbotm[0, ip, jp]
                icrowstart = (ip - self.nprbeg) * self.ncpp
                icrowend = icrowstart + self.ncpp
                iccolstart = (jp - self.npcbeg) * self.ncpp
                iccolend = iccolstart + self.ncpp
                botm[0, icrowstart:icrowend, iccolstart:iccolend] = top
                kc = 1
                for kp in range(self.nplbeg, self.nplend + 1):
                    top = pbotm[kp, ip, jp]
                    bot = pbotm[kp + 1, ip, jp]
                    dz = (top - bot) / self.ncppl[kp]
                    for _ in range(self.ncppl[kp]):
                        botm[kc, icrowstart:icrowend, iccolstart:iccolend] = (
                            botm[
                                kc - 1,
                                icrowstart:icrowend,
                                iccolstart:iccolend,
                            ]
                            - dz
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
        child_array = np.empty(
            (self.nrow, self.ncol), dtype=parent_array.dtype
        )
        for ip in range(self.nprbeg, self.nprend + 1):
            for jp in range(self.npcbeg, self.npcend + 1):
                icrowstart = (ip - self.nprbeg) * self.ncpp
                icrowend = icrowstart + self.ncpp
                iccolstart = (jp - self.npcbeg) * self.ncpp
                iccolend = iccolstart + self.ncpp
                value = parent_array[ip, jp]
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

                        tpp = topp[ip, jp]
                        btp = botp[kp, ip, jp]
                        if kp > 0:
                            tpp = botp[kp - 1, ip, jp]

                        tpc = topc[ic, jc]
                        btc = botc[kc, ic, jc]
                        if kc > 0:
                            tpc = botc[kc - 1, ic, jc]

                        if ihc == 0:
                            cl1 = 0.5 * (tpp - btp)
                            cl2 = 0.5 * (tpc - btc)
                            hwva = delrc[jc] * delcc[ic]
                        else:
                            if abs(idir) == 1:
                                cl1 = 0.5 * delrp[jp]
                                cl2 = 0.5 * delrc[jc]
                                hwva = delcc[ic]
                            elif abs(idir) == 2:
                                cl1 = 0.5 * delcp[ip]
                                cl2 = 0.5 * delcc[ic]
                                hwva = delrc[jc]

                        # connection distance
                        cd = None
                        if cdist:
                            if abs(idir) == 3:
                                cd = cl1 + cl2
                            else:
                                x1 = xc[ic, jc]
                                y1 = yc[ic, jc]
                                x2 = xp[ip, jp]
                                y2 = yp[ip, jp]
                                cd = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                        exg = [(kp, ip, jp), (kc, ic, jc), ihc, cl1, cl2, hwva]
                        if angldegx:
                            exg.append(angle)
                        if cdist:
                            exg.append(cd)
                        exglist.append(exg)
        return exglist
