import numpy as np
from ..modflow import Modflow, ModflowDis


class Lgr(object):

    def __init__(self, parent, nplbeg, nplend, npcbeg, npcend, nprbeg, nprend,
                 ncpp, ncppl, ibndp=None):
        """

        Parameters
        ----------
        parent : flopy.modflow.Modflow
            parent model
        nplbeg : int
            parent beginning layer
        nplend : int
            parent ending layer
        npcbeg : int
            parent beginning column
        npcend : int
            parent ending column
        nprbeg : int
            parent beginning row
        nprend : int
            parent ending row
        ncpp : int
            number of child cells along the face of a parent cell
        ncppl : list of ints
            number of child layers per parent layer
        ibndp : ndarray
            optional ibound-like array to allow irregular parent/child
            boundary shape.  If not specified, then array will be created
            so that child grid cuts out the parent grid along its rectangular
            boundary.  If specified, ibndp should have zeros where the child
            grid exists.

        """
        self.parent = parent
        self.nplbeg = nplbeg
        self.nplend = nplend
        self.npcbeg = npcbeg
        self.npcend = npcend
        self.nprbeg = nprbeg
        self.nprend = nprend
        self.ncpp = ncpp
        ncppl = np.array(ncppl)
        self.ncppl = ncppl
        self.nlay = ncppl.sum()
        self.nrow = (nprend - nprbeg + 1) * ncpp
        self.ncol = (npcend - npcbeg + 1) * ncpp

        if ibndp is None:
            self.ibndp = np.ones((parent.dis.nlay, parent.dis.nrow,
                                   parent.dis.ncol), np.int)
            self.ibndp[nplbeg:nplend+1, nprbeg:nprend+1, npcbeg:npcend+1] = 0
        else:
            self.ibndp = ibndp
            assert ibndp.shape == (parent.dis.nlay, parent.dis.nrow,
                                   parent.dis.ncol)

        self.delr, self.delc = self.get_delr_delc()
        self.top, self.botm = self.get_top_botm()
        xll = parent.sr.xll + parent.dis.delr[0: npcbeg].sum()
        yll = parent.sr.yll + parent.dis.delc[nprend + 1:].sum()

        child = Modflow()
        dis = ModflowDis(child, self.nlay, self.nrow, self.ncol,
                         delr=self.delr, delc=self.delc, top=self.top,
                         botm=self.botm)
        dis.sr.xll = xll
        dis.sr.yll = yll
        self.dis = dis
        self.child = child

        return

    def get_delr_delc(self):
        # create the delr and delc arrays for this child grid
        delr = np.zeros((self.ncol), dtype=float)
        delc = np.zeros((self.nrow), dtype=float)
        jstart = 0
        jend = self.ncpp
        for j in range(self.npcbeg, self.npcend + 1):
            delr[jstart: jend] = self.parent.dis.delr[j - 1] / self.ncpp
            jstart = jend
            jend = jstart + self.ncpp
        istart = 0
        iend = self.ncpp
        for i in range(self.nprbeg, self.nprend + 1):
            delc[istart: iend] = self.parent.dis.delc[i - 1] / self.ncpp
            istart = iend
            iend = istart + self.ncpp
        return delr, delc

    def get_top_botm(self):
        bt = self.parent.dis.botm.array
        tp = self.parent.dis.top.array
        shp = tp.shape
        tp = tp.reshape(1, shp[0], shp[1])
        pbotm = np.vstack((tp, bt))
        botm = np.zeros( (self.nlay + 1, self.nrow, self.ncol), dtype=float)
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
                    dz = (top - bot) / self.ncppl[kp - 1]
                    for n in range(self.ncppl[kp - 1]):
                        botm[kc, icrowstart:icrowend,
                                 iccolstart: iccolend] = botm[kc - 1,
                                 icrowstart:icrowend,
                                 iccolstart: iccolend] - dz
                        kc += 1
        return botm[0], botm[1:]

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
        return (kp, ip, jp)

    def get_parent_connections(self, kc, ic, jc):
        """
        Return a list of parent cell indices that are connected to child
        cell kc, ic, jc.

        """

        if self.parent is None:
            return []

        parentlist = []
        (kp, ip, jp) = self.get_parent_indices(kc, ic, jc)

        # parent cell to left
        if jc % self.ncpp == 0:
            if jp - 1 >= 0:
                if self.ibndp[kp, ip, jp - 1] != 0:
                    parentlist.append(((kp, ip, jp - 1), -1))

        # parent cell to right
        if (jc + 1) % self.ncpp == 0:
            if jp + 1 < self.parent.dis.ncol:
                if self.ibndp[kp, ip, jp + 1] != 0:
                    parentlist.append(((kp, ip, jp + 1), 1))

        # parent cell to back
        if ic % self.ncpp == 0:
            if ip - 1 >= 0:
                if self.ibndp[kp, ip - 1, jp] != 0:
                    parentlist.append(((kp, ip - 1, jp), 2))

        # parent cell to front
        if (ic + 1) % self.ncpp == 0:
            if ip + 1 < self.parent.dis.nrow:
                if self.ibndp[kp, ip + 1, jp] != 0:
                    parentlist.append(((kp, ip + 1, jp), -2))

        # parent cell to top is not possible

        # parent cell to bottom
        if kc + 1 == self.ncppl[kp]:
            if kp + 1 < self.parent.dis.nlay:
                if self.ibndp[kp + 1, ip, jp] != 0:
                    parentlist.append(((kp + 1, ip, jp), -3))

        return parentlist

    def get_exchange_data(self):
        """
        Get the list of parent/child connections

        <cellidm1> <cellidm2> <ihc> <cl1> <cl2> <hwva> <angledegx>

        Returns
        -------
            exglist : list
                list of connections between parent and child

        """

        exglist = []
        nlayc, nrowc, ncolc = self.dis.nlay, self.dis.nrow, self.dis.ncol
        delrc = self.dis.delr.array
        delcc = self.dis.delc.array
        delrp = self.parent.dis.delr.array
        delcp = self.parent.dis.delc.array
        topp = self.parent.dis.top.array
        botp = self.parent.dis.botm.array
        topc = self.dis.top.array
        botc = self.dis.botm.array
        for kc in range(nlayc):
            for ic in range(nrowc):
                for jc in range(ncolc):
                    plist = self.get_parent_connections(kc, ic, jc)
                    for (kp, ip, jp), idir in plist:

                        # horizontal or vertical connection
                        ihc = 1
                        if self.ncppl[kp] > 1:
                            ihc = 2
                        if abs(idir) == 3:
                            ihc = 0

                        # angldegx
                        angldegx = 0
                        if idir == 2:
                            angldegx = 90.
                        elif idir == -1:
                            angldegx = 180.
                        elif idir == -2:
                            angldegx = 270

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

                        exg = (kp, ip, jp, kc, ic, jc, ihc, cl1, cl2, hwva,
                               angldegx)
                        exglist.append(exg)
        return exglist
