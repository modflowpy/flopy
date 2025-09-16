import inspect

import numpy as np

from ...mf6 import modflow
from ...plot import plotutil
from ...utils import import_optional_dependency
from ..data import mfdataarray, mfdatalist, mfdataplist, mfdatascalar
from ..mfbase import PackageContainer

OBS_ID1_LUT = {
    "gwf": "cellid",
    "csub": {
        "csub": "icsubno",
        "ineslastic-csub": "icsubno",
        "elastic-csub": "icsubno",
        "coarse-csub": "cellid",
        "csub-cell": "cellid",
        "wcomp-csub-cell": "cellid",
        "sk": "icsubno",
        "ske": "icsubno",
        "sk-cell": "cellid",
        "ske-cell": "cellid",
        "estress-cell": "cellid",
        "gstress-cell": "cellid",
        "interbed-compaction": "icsubno",
        "inelastic-compaction": "icsubno",
        "elastic-compaction": "icsubno",
        "coarse-compaction": "cellid",
        "inelastic-compaction-cell": "cellid",
        "elastic-compaction-cell": "cellid",
        "compaction-cell": "cellid",
        "thickness": "icsubno",
        "coarse-thickness": "cellid",
        "thickness-cell": "cellid",
        "theta": "icsubno",
        "coarse-theta": "cellid",
        "theta-cell": "cellid",
        "delay-flowtop": "icsubno",
        "delay-flowbot": "icsubno",
        "delay-head": "icsubno",
        "delay-gstress": "icsubno",
        "delay-estress": "icsubno",
        "delay-preconstress": "icsubno",
        "delay-compaction": "icsubno",
        "delay-thickness": "icsubno",
        "delay-theta": "icsubno",
        "preconstress-cell": "cellid",
    },
    "chd": "cellid",
    "drn": "cellid",
    "evt": "cellid",
    "ghb": "cellid",
    "rch": "cellid",
    "riv": "cellid",
    "wel": "cellid",
    "lak": {
        "stage": "ifno",
        "ext-inflow": "ifno",
        "outlet-inflow": "ifno",
        "inflow": "ifno",
        "from-mvr": "ifno",
        "rainfall": "ifno",
        "runoff": "ifno",
        "lak": "ifno",
        "withdrawal": "ifno",
        "evaporation": "ifno",
        "ext-outflow": "ifno",
        "to-mvr": "outletno",
        "storage": "ifno",
        "constant": "ifno",
        "outlet": "outletno",
        "volume": "ifno",
        "surface-area": "ifno",
        "wetted-area": "ifno",
        "conductance": "ifno",
    },
    "maw": "ifno",
    "sfr": "ifno",
    "uzf": "ifno",
    # transport
    "gwt": "cellid",
    "cnc": "cellid",
    "src": "cellid",
    "sft": "ifno",
    "lkt": "ifno",
    "mwt": "mawno",
    "uzt": "ifno",
    # energy
    "gwe": "cellid",
    "ctp": "cellid",
    "sfe": "ifno",
    "uze": "ifno",
}

OBS_ID2_LUT = {
    "gwf": "cellid",
    "csub": "icsubno",
    "chd": None,
    "drn": None,
    "evt": None,
    "ghb": None,
    "rch": None,
    "riv": None,
    "wel": None,
    "sfr": None,
    "lak": None,
    "maw": None,
    "uzf": None,
    "gwt": "cellid",
    "cnc": None,
    "src": None,
    "sft": "ifno",
    "lkt": {
        "flow-ja-face": "ifno",
        "lkt": None,
    },
    "mwt": None,
    "uzt": "ifno",
    "gwe": "cellid",
    "sfe": "ifno",
    "uze": "ifno"
}


class Mf6Splitter:
    """
    A class for splitting a single model into a multi-model simulation

    Parameters
    ----------
    sim : flopy.mf6.MFSimulation
    modelname : str, None
        name of model to split
    """

    def __init__(self, sim, modelname=None):
        self._sim = sim
        self._model = self._sim.get_model(modelname)
        if modelname is None:
            self._modelname = self._model.name
        self._model_type = self._model.model_type
        if self._model_type.endswith("6"):
            self._model_type = self._model_type[:-1]
        self._modelgrid = self._model.modelgrid
        self._fast_neighbors = False
        if self._modelgrid.grid_type in ("structured", "vertex"):
            self._ncpl = self._modelgrid.ncpl
            if self._modelgrid.grid_type == "structured":
                if self._modelgrid.nrow > 2 and self._modelgrid.ncol > 2:
                    self._fast_neighbors = True
        else:
            self._ncpl = self._modelgrid.nnodes
        self._shape = self._modelgrid.shape
        self._grid_type = self._modelgrid.grid_type
        self._node_map = {}
        self._node_map_r = {}
        self._new_connections = None
        self._new_ncpl = None
        self._grid_info = None
        self._exchange_metadata = None
        self._connection = None
        self._uconnection = None
        self._usg_metadata = None
        self._has_angldegx = False
        self._model_dict = None
        self._ivert_vert_remap = None
        self._sfr_mover_connections = []
        self._mover = False
        self._pkg_mover = False
        self._pkg_mover_name = None
        self._mover_remaps = {}
        self._sim_mover_data = {}
        self._new_sim = None
        self._offsets = {}
        # dictionaries of remaps necessary for piping GWF info to GWT
        self._uzf_remaps = {}
        self._lak_remaps = {}
        self._sfr_remaps = {}
        self._maw_remaps = {}
        self._allow_splitting = True

        self._fdigits = 1

        # multi-model splitting attr
        self._multimodel_exchange_gwf_names = {}

    @property
    def new_simulation(self):
        """
        Returns the new simulation object after model splitting
        """
        return self._new_sim

    def switch_models(self, modelname, remap_nodes=False):
        """
        Method to switch which model within a simulation that is being split.
        Models must be congruent. Ex. GWF model followed by a GWT model.
        If the models are not congruent remap_nodes must be set to True.

        Parameters
        ----------
        modelname : str
            model name to switch to
        remap_nodes : bool
            boolean flag to force the class to remap the node look up table.
            This is used when models do not overlap (ex. two separate
            GWF models). Exchanges between original models are not preserved
            currently.

        Returns
        -------
        None
        """
        self._model = self._sim.get_model(modelname)
        self._modelname = self._model.name
        self._model_type = self._model.model_type
        if self._model_type.endswith("6"):
            self._model_type = self._model_type[:-1]

        self._model_dict = None

        if remap_nodes:
            self._modelgrid = self._model.modelgrid
            self._node_map = {}
            self._node_map_r = {}
            self._new_connections = None
            self._new_ncpl = None
            self._grid_info = None
            self._exchange_metadata = None
            self._connection = None
            self._uconnection = None
            self._usg_metadata = None
            self._has_angldegx = False
            self._ivert_vert_remap = None
            self._sfr_mover_connections = []
            self._mover = False
            self._pkg_mover = False
            self._pkg_mover_name = None
            self._mover_remaps = {}
            self._sim_mover_data = {}
            self._offsets = {}

    @property
    def reversed_node_map(self):
        """
        Returns a lookup table of {model number : {model node: original node}}

        """
        if not self._node_map_r:
            self._node_map_r = {mkey: {} for mkey in self._model_dict.keys()}
            for onode, (mkey, nnode) in self._node_map.items():
                self._node_map_r[mkey][nnode] = onode
        return self._node_map_r

    @property
    def original_modelgrid(self):
        """
        Method to return the original model's modelgrid object. This is
        used for re-assembling the split models when analyzing output

        """
        return self._modelgrid

    def _map_grids_to_hdf5(self, f):
        """
        Method to map modelgrid information to hdf5 for easier save/reconstruction
        process

        Parameters
        ----------
        f : h5py.File object

        Returns
        -------
            None
        """
        import h5py

        kwargs = {
            "compression": "gzip",
        }

        mgg = f.create_group("modelgrids")
        grid_type = f["grid_type"][0].decode("utf8")
        mkeys = f["mkey"][:]
        mnames = [self._modelname] + list(f["new_modelnames"][:])
        grids = [self._modelgrid, ] + [self._model_dict[mk].modelgrid for mk in mkeys]

        for ix, name in enumerate(mnames):
            if hasattr(name, "decode"):
                name = name.decode("utf8")

            # name = name.decode("utf8")
            grd_grp = mgg.create_group(name)
            grid = grids[ix]
            if grid_type == "structured":
                delc = grid.delc
                dc = grd_grp.create_dataset("delc", (len(delc),), dtype=float, **kwargs)
                dc[:] = delc

                delr = grid.delr
                dr = grd_grp.create_dataset("delr", (len(delr),), dtype=float, **kwargs)
                dr[:] = delr

            else:
                # need cell2d information, idomain, top, botm, xoff, yoff, angrot.
                if grid.is_valid:
                    vert_dt = np.dtype([("ivert", int), ("xvert", float), ("yvert", float)])

                    # construct c2d dtype
                    cell2d = grid.cell2d
                    nverts = [row[3] for row in cell2d]
                    max_vert = np.max(nverts)
                    c2d_dt = np.dtype(
                        [("icell2d", int), ('xc', float), ("yc", float), ("ncvert", int)]
                        + [(f"iv_{i}", int) for i in range(max_vert)]
                    )

                    # now create datasets and set data
                    vert_ds = grd_grp.create_dataset(
                        "vertices", (len(grid._vertices),), dtype=vert_dt, **kwargs
                    )
                    for ix, verts in enumerate(grid._vertices):
                        vert_ds[ix] = tuple(verts)

                    c2d_ds = grd_grp.create_dataset(
                        "cell2d", (len(grid.cell2d),), dtype=c2d_dt, **kwargs
                    )
                    for ix, rec in enumerate(cell2d):
                        nvert = rec[3]
                        nfill = max_vert - nvert
                        fill = [-1,] * nfill
                        frec = tuple(list(rec) + fill)
                        c2d_ds[ix] = frec

                if grid_type == "unstructured":
                    # ncpl is also needed
                    iac = grid.iac
                    ja = grid.ja
                    ncpl = grid.ncpl

                    iac_ds = grd_grp.create_dataset(
                        "iac", (len(iac),), dtype=int, **kwargs
                    )
                    iac_ds[:] = iac

                    ja_ds = grd_grp.create_dataset(
                        "ja", (len(ja),), dtype=int, **kwargs
                    )
                    ja_ds[:] = ja

                    ncpl_ds = grd_grp.create_dataset(
                        "ncpl", (len(ncpl),), dtype=int, **kwargs
                    )
                    ncpl_ds[:] = ncpl

            top = grid.top
            botm = grid.botm
            idomain = grid.idomain
            if idomain is None:
                idomain = np.ones(botm.shape, dtype=int)

            top_ds = grd_grp.create_dataset("top", top.shape, dtype=float, **kwargs)
            top_ds[:] = top[:]

            botm_ds = grd_grp.create_dataset("botm", botm.shape, dtype=float, **kwargs)
            botm_ds[:] = botm[:]

            id_ds = grd_grp.create_dataset(
                "idomain", idomain.shape, dtype=int, **kwargs
            )
            id_ds[:] = idomain[:]

            xoff_ds = grd_grp.create_dataset("xoff", (1,), dtype=float)
            xoff_ds[0] = grid.xoffset

            yoff_ds = grd_grp.create_dataset("yoff", (1,), dtype=float)
            yoff_ds[0] = grid.yoffset

            rot_ds = grd_grp.create_dataset("angrot", (1,), dtype=float)
            rot_ds[0] = grid.angrot

    def save_node_mapping(self, filename):
        """
        Method to save the Mf6Splitter's node mapping to file for
        use in reconstructing arrays

        Parameters
        ----------
        filename : str, Path
            HDF5 file name
        hdf5 : bool
            flag to save as a hdf5 binary file

        Returns
        -------
            None
        """
        node_map = {
            int(k): (int(v[0]), int(v[1])) for k, v in self._node_map.items()
        }

        h5py = import_optional_dependency("h5py")
        # import h5py
        f = h5py.File(filename, "w")

        string_dt = h5py.string_dtype(encoding="ascii")

        gt_ds = f.create_dataset("grid_type", (1,), dtype=string_dt)
        gt_ds[:] = [self._grid_type,]

        mname_ds = f.create_dataset("modelname", (1,), dtype=string_dt)
        mname_ds[:] = [self._modelname,]

        mnames = [m.name for m in self._model_dict.values()]
        nname_ds = f.create_dataset("new_modelnames", (len(mnames),), string_dt)
        nname_ds[:] = mnames

        mkeys = list(self._model_dict.keys())
        mkey_ds = f.create_dataset("mkey", (len(mkeys),), dtype=int)
        mkey_ds[:] = mkeys

        # if int doesn't work try "i8"
        ncpl_ds = f.create_dataset("original_ncpl", (1,), dtype=int)
        ncpl_ds[:] = [self._ncpl]

        shape_ds = f.create_dataset("shape", (len(self._shape),), dtype=int)
        shape_ds[:] = list(self._shape)

        # finally, think about the node_map arrangement.... onode: (model, nnode)
        # maybe put it in a group "node_map/original_node", etc...
        nm_grp = f.create_group("node_map")
        onode, nmodel, nnode = [], [], []
        for k, (m, n) in node_map.items():
            onode.append(k)
            nmodel.append(m)
            nnode.append(n)

        onode_ds = nm_grp.create_dataset(
            "original_node", (len(onode),), dtype=int, compression="gzip"
        )
        nmodel_ds = nm_grp.create_dataset(
            "new_model", (len(nmodel),), dtype=int, compression="gzip"
        )
        nnode_ds = nm_grp.create_dataset(
            "new_node", (len(nnode),), dtype=int, compression="gzip"
        )

        onode_ds[:] = onode
        nmodel_ds[:] = nmodel
        nnode_ds[:] = nnode

        self._map_grids_to_hdf5(f)

        f.close()


    @staticmethod
    def load_node_mapping(filename):
        """
        Method that loads and instantiate a Mf6Splitter object from hdf5 file. Once
        Mf6Splitter has been instantiated, it can be used to reconstruct arrays and
        recarrays of output data

        Parameters
        ----------
        filename : str
            hdf5 file name

        Returns
        -------
            Mf6Splitter object
        """
        h5py = import_optional_dependency("h5py")

        class FakeSim:
            def __init__(self, modeldict):
                self._modeldict = modeldict
                self._model_names = list(modeldict.keys())

            def get_model(self, name=None):
                if name is None:
                    name = self._model_names[0]
                return self._modeldict[name]

        class FakeModel:
            def __init__(self, name, modelgrid):
                self.name = name
                self.modelgrid = modelgrid
                self.model_type = "FAK6"

        def construct_modelgrid(f, name, grid_type):
            """
            Method to construct a modelgrid instance from HDF5 stored grid information

            Parameters
            ----------
            f : h5py.File object
            name : str
                model name string
            grid_type : str
                grid type string

            Returns
            -------
                flopy.discretization.Grid object
            """
            from ...discretization import StructuredGrid, UnstructuredGrid, VertexGrid
            grid = f[f"modelgrids/{name}"]

            gridprops = dict()
            gridprops["top"] = grid["top"][:]
            gridprops["botm"] = grid["botm"][:]
            gridprops["idomain"] = grid["idomain"][:]
            gridprops["xoff"] = grid["xoff"][0]
            gridprops["yoff"] = grid["yoff"][0]
            gridprops["angrot"] = grid["angrot"][0]
            if grid_type == "structured":
                delc = grid["delc"][:]
                delr = grid["delr"][:]
                modelgrid = StructuredGrid(delc=delc, delr=delr, **gridprops)
            else:
                gridkeys = list(grid.keys())
                if "vertices" in gridkeys:
                    vertices = grid["vertices"][:]
                    gridprops["vertices"] = [list(v) for v in vertices]

                if "cell2d" in gridkeys:
                    c2d = grid["cell2d"][:]
                    gridprops["cell2d"] = [[i for i in row if i != -1] for row in c2d]

                if grid_type == "vertex":
                    modelgrid = VertexGrid(**gridprops)

                else:
                    iac = grid["iac"][:]
                    ja = grid["ja"][:]
                    ncpl = grid["ncpl"][:]
                    modelgrid = UnstructuredGrid(iac=iac, ja=ja, ncpl=ncpl, **gridprops)

            return modelgrid

        try:
            f = h5py.File(filename)
        except OSError:
            raise TypeError(
                "Text based files no longer supported, re-save node "
                "mapping in hdf5 format using Mf6Splitter.save_node_mapping()"
            )

        # construct the node map
        node_map = {}
        onode = f["node_map/original_node"][:]
        mnum = f["node_map/new_model"][:]
        nnode = f["node_map/new_node"][:]
        for ix, node in enumerate(onode):
            node_map[node] = (mnum[ix], nnode[ix])

        # construct representation of original model geometry
        modelname = f["modelname"][0].decode("utf8")
        grid_type = f["grid_type"][0].decode("utf8")
        grid = construct_modelgrid(f, modelname, grid_type)
        ml = FakeModel(modelname, grid)
        sim = FakeSim({modelname: ml})
        mfs = Mf6Splitter(sim)
        mfs._model_dict = {}
        mfs._new_ncpl = {}

        # construct representation of new model geometries
        mkeys = [int(i) for i in f["mkey"][:]]
        modelnames = [i.decode("utf8") for i in f["new_modelnames"][:]]
        for ix, mname in enumerate(modelnames):
            mkey = mkeys[ix]
            sgrid = construct_modelgrid(f, mname, grid_type)
            sml = FakeModel(mname, sgrid)
            mfs._model_dict[mkey] = sml
            if grid_type in ("structured", "vertex"):
                mfs._new_ncpl[mkey] = sgrid.ncpl
            else:
                mfs._new_ncpl[mkey] = sgrid.nnodes

        f.close()

        # create additional splitting data efficient reconstruction
        split_array = np.zeros((mfs._ncpl,), dtype=int)
        model_array = np.zeros((mfs._ncpl,), dtype=int)
        for k, v in node_map.items():
            k = int(k)
            model_array[k] = v[0]
            split_array[k] = v[1]

        grid_info = {}
        for mkey in mkeys:
            ncpl = mfs._new_ncpl[mkey]
            array = np.full((ncpl,), -1, dtype=int)
            onode = np.asarray(model_array == mkey).nonzero()[0]
            nnode = split_array[onode]
            array[nnode] = onode
            grid_info[mkey] = (array,)

        mfs._grid_info = grid_info
        mfs._node_map = node_map
        mfs._allow_splitting = False
        return mfs

    def optimize_splitting_mask(self, nparts, active_only=False, options=None, verbose=False):
        """
        Method to create a splitting array with a balanced number of active
        cells per model. This method uses the program METIS and pymetis to
        create the subsetting array

        Parameters
        ----------
        nparts: int
            number of parts to split the model in to
        active_only : bool
            only consider active cells when building adjacency graph. Default is False
        options : None or pymetis.Options
            optional pymetis.Options class that gets passed through to the
            pymetis.part_graph() function. Example
            `options=pymetis.Options(seed=42, contig=1)`
        verbose : bool
            Default False. Prints progress statements if True
        Returns
        -------
            np.ndarray
        """
        if active_only:
            import_optional_dependency("sklearn")
            from sklearn.neighbors import NearestNeighbors

        pymetis = import_optional_dependency(
            "pymetis",
            "please install pymetis using: "
            "conda install -c conda-forge pymetis",
        )
        # create graph of nodes
        nlay = self._modelgrid.nlay
        if self._modelgrid.grid_type in ("structured", "vertex"):
            ncpl = self._modelgrid.ncpl
            shape = self._modelgrid.shape[1:]
        else:
            nlay = 1
            ncpl = self._modelgrid.nnodes
            shape = self._modelgrid.shape
        idomain = self._modelgrid.idomain
        if idomain is None:
            idomain = np.full((nlay, ncpl), 1.0, dtype=float)
        else:
            idomain = idomain.reshape(nlay, ncpl)
        adv_pkg_weights = np.zeros((ncpl,), dtype=int)
        lak_array = np.zeros((ncpl,), dtype=int)
        laks = []
        hfbs = []
        for package in self._model.get_package():
            if isinstance(
                package,
                (
                    modflow.ModflowGwfsfr,
                    modflow.ModflowGwfuzf,
                    modflow.ModflowGwflak,
                    modflow.ModflowGwfhfb,
                    modflow.ModflowGwtsft,
                    modflow.ModflowGwtuzt,
                    modflow.ModflowGwtlkt,
                    modflow.ModflowGwesfe,
                    modflow.ModflowGweuze,
                    modflow.ModflowGwelke
                ),
            ):
                if isinstance(package, modflow.ModflowGwfhfb):
                    hfbs.append(package)
                    continue

                if isinstance(package, (modflow.ModflowGwflak, modflow.ModflowGwtlkt, modflow.ModflowGwelke)):
                    cellids = package.connectiondata.array.cellid
                else:
                    cellids = package.packagedata.array.cellid
                if self._modelgrid.grid_type == "structured":
                    cellids = [(0, i[1], i[2]) for i in cellids]
                    nodes = self._modelgrid.get_node(cellids)
                elif self._modelgrid.grid_type == "vertex":
                    nodes = [i[1] for i in cellids]
                else:
                    nodes = [i[0] for i in cellids]

                if isinstance(package, (modflow.ModflowGwflak, modflow.ModflowGwtlkt, modflow.ModflowGwelke)):
                    lakenos = package.connectiondata.array.ifno + 1
                    lak_array[nodes] = lakenos
                    laks += [i for i in np.unique(lakenos)]
                else:
                    adv_pkg_weights[nodes] += 1
        if verbose:
            print("Mapping neighbors")
        neighbors = self._modelgrid.neighbors(reset=True, fast=self._fast_neighbors)

        if active_only:
            if verbose:
                print("Filtering inactive cells")
            # filter out inactive cells here to avoid messing with neighbor algo.
            iact = np.where(np.sum(idomain, axis=0), 1, 0)
            # set this to a dict because key lookup is O(1) vs O(n) for lists
            inactive = dict.fromkeys([int(i) for i in np.where(iact == 0)[0]])
            # for k in inactive:
            [neighbors.pop(k) for k in inactive]
            node_map = {i: ix for ix, i in enumerate(np.where(iact > 0)[0])}
            neighbors = {
                k : [node_map[i] for i in v if i not in inactive] for k, v in neighbors.items()
            }

        if verbose:
            print("Creating graph and weights")
        neighbors = dict(sorted(neighbors.items()))
        weights = [np.count_nonzero(idomain[:, nn]) + adv_pkg_weights[nn] for nn in neighbors.keys()]
        graph = [np.array(neigh, dtype=int) for neigh in neighbors.values()]

        if verbose:
            print("Running Metis")
        n_cuts, membership = pymetis.part_graph(
            nparts, adjacency=graph, vweights=weights, options=options
        )
        membership = np.array(membership, dtype=int)

        if active_only:
            # reamp everything to original domain
            if verbose:
                print("Remapping inactive to model domains")
            if len(inactive) > 0:
                xc = self._modelgrid.xcellcenters.ravel()
                yc = self._modelgrid.ycellcenters.ravel()
                axc = xc[list(node_map.keys())]
                ayc = yc[list(node_map.keys())]
                fit_points = np.array(list(zip(axc, ayc)))
                ixc = xc[list(inactive.keys())]
                iyc = yc[list(inactive.keys())]
                pred_points = np.array(list(zip(ixc, iyc)))

                nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
                nn.fit(fit_points)
                ind = nn.kneighbors(pred_points, return_distance=False).ravel()
                data = membership[ind]
                member_array = np.full((ncpl,), -1)
                member_array[list(node_map.keys())] = membership[list(node_map.values())]
                member_array[list(inactive.keys())] = data
                membership = member_array

        if laks:
            for lak in laks:
                idx = np.asarray(lak_array == lak).nonzero()[0]
                mnum = np.unique(membership[idx])[0]
                membership[idx] = mnum

        if hfbs:
            for hfb in hfbs:
                for recarray in hfb.stress_period_data.data.values():
                    cellids1 = recarray.cellid1
                    cellids2 = recarray.cellid2
                    _, nodes1 = self._cellid_to_layer_node(cellids1)
                    _, nodes2 = self._cellid_to_layer_node(cellids2)
                    mnums1 = membership[nodes1]
                    mnums2 = membership[nodes2]
                    ev = np.equal(mnums1, mnums2)
                    if np.all(ev):
                        continue
                    idx = np.asarray(~ev).nonzero()[0]
                    mnum_to = mnums1[idx]
                    adj_nodes = nodes2[idx]
                    membership[adj_nodes] = mnum_to

        return membership.reshape(shape)

    def reconstruct_array(self, arrays):
        """
        Method to reconstruct model output arrays into a single array
        with the dimensions of the original model

        arrays : dict
            dictionary of model number and the associated array

        Returns
        -------
            np.ndarray of original model shape
        """
        for ix, mkey in enumerate(arrays.keys()):
            model = self._model_dict[mkey]
            array = arrays[mkey]
            if ix == 0:
                nlay, shape = self._get_nlay_shape_models(model, array)
            else:
                nlay1, shape1 = self._get_nlay_shape_models(model, array)

                if shape != shape1 and nlay != nlay1:
                    raise AssertionError(
                        f"Supplied array for model {mkey} is not "
                        f"consistent with output shape: {shape}"
                    )

        dtype = arrays[list(arrays.keys())[0]].dtype
        new_array = np.zeros(shape, dtype=dtype)
        new_array = new_array.ravel()
        oncpl = self._ncpl

        for mkey, array in arrays.items():
            array = array.ravel()
            ncpl = self._new_ncpl[mkey]
            mapping = self._grid_info[mkey][-1]
            old_nodes = np.asarray(mapping != -1).nonzero()
            new_nodes = mapping[old_nodes]

            old_nodes = np.tile(old_nodes, (nlay, 1))
            old_adj_array = np.arange(nlay, dtype=int) * ncpl
            old_adj_array = np.expand_dims(old_adj_array, axis=1)
            old_nodes += old_adj_array
            old_nodes = old_nodes.ravel()

            new_nodes = np.tile(new_nodes, (nlay, 1))
            new_adj_array = np.arange(nlay, dtype=int) * oncpl
            new_adj_array = np.expand_dims(new_adj_array, axis=1)
            new_nodes += new_adj_array
            new_nodes = new_nodes.ravel()

            new_array[new_nodes] = array[old_nodes]

        new_array.shape = shape
        return new_array

    def reconstruct_recarray(self, recarrays):
        """
        Method to reconstruct model recarrays into a single recarray
        that represents the original model data

        Parameters
        ----------
        recarrays : dict
            dictionary of model number and recarray

        Returns
        -------
            np.recarray
        """
        rlen = 0
        dtype = None
        for recarray in recarrays.values():
            rlen += len(recarray)
            dtype = recarray.dtype
            if "cellid" not in recarray.dtype.names:
                raise AssertionError("cellid must be present in recarray")

        if rlen == 0:
            return

        new_recarray = np.recarray((rlen,), dtype=dtype)
        idx = 0
        for mkey, recarray in recarrays.items():
            remapper = self.reversed_node_map[mkey]
            orec = recarray.copy()
            modelgrid = self._model_dict[mkey].modelgrid
            if self._grid_type in ("structured", "vertex"):
                layer = [i[0] for i in orec.cellid]
                if self._grid_type == "structured":
                    cellid = [(0, i[1], i[2]) for i in orec.cellid]
                    node = modelgrid.get_node(cellid)
                else:
                    node = [i[-1] for i in orec.cellid]
            else:
                node = [i[0] for i in orec.cellid]

            new_node = [remapper[i] for i in node if i in remapper]

            if modelgrid.grid_type == "structured":
                if self._modelgrid is None:
                    new_cellid = list(
                        zip(*np.unravel_index(new_node, self._shape))
                    )
                else:
                    new_cellid = self._modelgrid.get_lrc(new_node)
                new_cellid = [
                    (layer[ix], i[1], i[2]) for ix, i in enumerate(new_cellid)
                ]
            elif modelgrid.grid_type == "vertex":
                new_cellid = [(layer[ix], i) for ix, i in enumerate(new_node)]
            else:
                new_cellid = [(i,) for i in new_node]

            orec["cellid"] = new_cellid
            new_recarray[idx : idx + len(orec)] = orec[:]
            idx += len(orec)

        return new_recarray

    def recarray_bc_array(self, recarray, pkgtype=None, color="c"):
        """
        Method to take a reconstructed recarray and create a plottable
        boundary condition location array from it.

        Parameters
        ----------
        recarray : np.recarray
        pkgtype : str
            optional package type. used to apply flopy's default color to
            package

        Returns
        -------
        tuple: numpy array and a dict of matplotlib kwargs

        """
        import matplotlib

        bc_array = np.zeros(self._modelgrid.shape, dtype=int)
        idx = tuple(zip(*recarray.cellid))
        if len(idx) == 1:
            bc_array[list(idx)] = 1
        else:
            bc_array[idx] = 1
        bc_array = np.ma.masked_equal(bc_array, 0)
        if pkgtype is not None:
            key = pkgtype[:3].upper()
            if key in plotutil.bc_color_dict:
                color = plotutil.bc_color_dict[key]
            else:
                color = plotutil.bc_color_dict["default"]
        elif color is not None:
            pass
        else:
            color = plotutil.bc_color_dict["default"]

        cmap = matplotlib.colors.ListedColormap(["0", color])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        return bc_array, {"cmap": cmap, "norm": norm}

    def _get_nlay_shape_models(self, model, array):
        """
        Method to assert user provided arrays are either ncpl or nnodes

        Parameters
        ----------
        model : flopy.mf6.MFModel
            Modflow model instance
        array : np.ndarray
            Array of model data

        Returns
        -------
            tuple : (nlay, grid_shape)
        """
        if array.size == model.modelgrid.size:
            if self._modelgrid.grid_type in ("structured", "vertex"):
                nlay = model.modelgrid.nlay
            else:
                nlay = 1
            shape = self._shape

        elif array.size == model.modelgrid.ncpl:
            nlay = 1
            if self._modelgrid.grid_type in ("structured", "vertex"):
                shape = self._shape[1:]
            else:
                shape = self._shape
        else:
            raise AssertionError("Array is not of size ncpl or nnodes")

        return nlay, shape

    def _remap_nodes(self, array):
        """
        Method to remap existing nodes to new models

        Parameters
        ----------
        array : numpy array
            numpy array of dimension ncpl (nnodes for DISU models)

        """
        array = np.ravel(array)

        idomain = self._modelgrid.idomain.reshape((-1, self._ncpl))
        mkeys = [int(i) for i in np.unique(array)]
        bad_keys = []
        for mkey in mkeys:
            count = 0
            mask = np.asarray(array == mkey).nonzero()
            for arr in idomain:
                check = arr[mask]
                count += np.count_nonzero(check)
            if count == 0:
                bad_keys.append(mkey)

        if bad_keys:
            raise KeyError(
                f"{bad_keys} are not in the active model extent; "
                f"please adjust the model splitting array"
            )
        if self._modelgrid.grid_type == "unstructured":
            self._map_iac_ja_connections()
        else:
            self._connection = self._modelgrid.neighbors(
                reset=True, method="rook", fast=self._fast_neighbors
            )

        grid_info = {}
        if self._modelgrid.grid_type == "structured":
            a = array.reshape(self._modelgrid.nrow, self._modelgrid.ncol)
            for m in np.unique(a):
                cells = np.asarray(a == m).nonzero()
                rmin, rmax = np.min(cells[0]), np.max(cells[0])
                cmin, cmax = np.min(cells[1]), np.max(cells[1])
                cellids = list(zip([0] * len(cells[0]), cells[0], cells[1]))
                self._offsets[m] = {
                    "xorigin": self._modelgrid.xvertices[rmax + 1, cmin],
                    "yorigin": self._modelgrid.yvertices[rmax + 1, cmin],
                }
                # get new nrow and ncol information
                nrow = (rmax - rmin) + 1
                ncol = (cmax - cmin) + 1
                mapping = np.ones((nrow, ncol), dtype=int) * -1
                nodes = self._modelgrid.get_node(cellids)
                mapping[cells[0] - rmin, cells[1] - cmin] = nodes
                grid_info[m] = [
                    (nrow, ncol),
                    (rmin, rmax),
                    (cmin, cmax),
                    np.ravel(mapping),
                ]
        else:
            for m in mkeys:
                cells = np.asarray(array == m).nonzero()[0]
                mapping = np.zeros((len(cells),), dtype=int)
                mapping[:] = cells
                grid_info[m] = [(len(cells),), None, None, mapping]

                self._offsets[m] = {
                    "xorigin": self._modelgrid.xoffset,
                    "yorigin": self._modelgrid.yoffset,
                }

        new_ncpl = {}
        for m in mkeys:
            new_ncpl[m] = 1
            for i in grid_info[m][0]:
                new_ncpl[m] *= i

        for mdl in mkeys:
            mnodes = np.asarray(array == mdl).nonzero()[0]
            mg_info = grid_info[mdl]
            if mg_info is not None:
                mapping = mg_info[-1]
                new_nodes = np.asarray(mapping != -1).nonzero()[0]
                old_nodes = mapping[new_nodes]
                for ix, nnode in enumerate(new_nodes):
                    self._node_map[old_nodes[ix]] = (mdl, nnode)
            else:
                for nnode, onode in enumerate(mnodes):
                    self._node_map[onode] = (mdl, nnode)

        new_connections = {
            i: {"internal": {}, "external": {}} for i in mkeys
        }
        exchange_meta = {i: {} for i in mkeys}
        usg_meta = {i: {} for i in mkeys}
        for node, conn in self._connection.items():
            mdl, nnode = self._node_map[node]
            for ix, cnode in enumerate(conn):
                cmdl, cnnode = self._node_map[cnode]
                if cmdl == mdl:
                    if nnode in new_connections[mdl]["internal"]:
                        new_connections[mdl]["internal"][nnode].append(cnnode)
                        if self._uconnection is not None:
                            usg_meta[mdl][nnode]["ihc"].append(
                                int(self._uconnection[node]["ihc"][ix + 1])
                            )
                            usg_meta[mdl][nnode]["cl12"].append(
                                self._uconnection[node]["cl12"][ix + 1]
                            )
                            usg_meta[mdl][nnode]["hwva"].append(
                                self._uconnection[node]["hwva"][ix + 1]
                            )
                            if self._has_angldegx:
                                usg_meta[mdl][nnode]["angldegx"].append(
                                    self._uconnection[node]["angldegx"][ix + 1]
                                )

                    else:
                        new_connections[mdl]["internal"][nnode] = [cnnode]
                        if self._uconnection is not None:
                            usg_meta[mdl][nnode] = {
                                "ihc": [
                                    self._uconnection[node]["ihc"][0],
                                    self._uconnection[node]["ihc"][ix + 1],
                                ],
                                "cl12": [
                                    self._uconnection[node]["cl12"][0],
                                    self._uconnection[node]["cl12"][ix + 1],
                                ],
                                "hwva": [
                                    self._uconnection[node]["hwva"][0],
                                    self._uconnection[node]["hwva"][ix + 1],
                                ],
                            }
                            if self._has_angldegx:
                                usg_meta[mdl][nnode]["angldegx"] = [
                                    self._uconnection[node]["angldegx"][0],
                                    self._uconnection[node]["angldegx"][
                                        ix + 1
                                    ],
                                ]

                else:
                    if nnode in new_connections[mdl]["external"]:
                        new_connections[mdl]["external"][nnode].append(
                            (cmdl, cnnode)
                        )
                        if self._uconnection is None:
                            exchange_meta[mdl][nnode][cnnode] = [
                                node,
                                cnode,
                                self._modelgrid.get_shared_edge(node, cnode)
                            ]
                        else:
                            exchange_meta[mdl][nnode][cnnode] = [
                                node,
                                cnode,
                                self._uconnection[node]["ihc"][ix + 1],
                                self._uconnection[node]["cl12"][ix + 1],
                                self._uconnection[node]["hwva"][ix + 1],
                            ]
                    else:
                        new_connections[mdl]["external"][nnode] = [
                            (cmdl, cnnode)
                        ]
                        if self._uconnection is None:
                            exchange_meta[mdl][nnode] = {
                                cnnode: [
                                    node,
                                    cnode,
                                    self._modelgrid.get_shared_edge(node, cnode)
                                ]
                            }
                        else:
                            exchange_meta[mdl][nnode] = {
                                cnnode: [
                                    node,
                                    cnode,
                                    self._uconnection[node]["ihc"][ix + 1],
                                    self._uconnection[node]["cl12"][ix + 1],
                                    self._uconnection[node]["hwva"][ix + 1],
                                ]
                            }

        if self._modelgrid.grid_type in ("vertex", "unstructured"):
            self._map_verts_iverts(array)

        self._new_connections = new_connections
        self._new_ncpl = new_ncpl
        self._grid_info = grid_info
        self._exchange_metadata = exchange_meta
        self._usg_metadata = usg_meta

    def _map_iac_ja_connections(self):
        """
        Method to map connections in unstructured grids when no
        vertex information has been supplied

        """
        conn = {}
        uconn = {}
        iac = self._modelgrid.iac
        ja = self._modelgrid.ja
        cl12 = self._model.disu.cl12.array
        ihc = self._model.disu.ihc.array
        hwva = self._model.disu.hwva.array
        angldegx = self._model.disu.angldegx.array
        if angldegx is not None:
            self._has_angldegx = True
        idx0 = 0
        for ia in iac:
            idx1 = idx0 + ia
            cn = ja[idx0 + 1 : idx1]
            conn[ja[idx0]] = list(cn)
            uconn[ja[idx0]] = {
                "cl12": list(cl12[idx0:idx1]),
                "ihc": list(ihc[idx0:idx1]),
                "hwva": list(hwva[idx0:idx1]),
            }

            if self._has_angldegx:
                uconn[ja[idx0]]["angldegx"] = list(angldegx[idx0:idx1])

            idx0 = idx1

        self._connection = conn
        self._uconnection = uconn

    def _map_verts_iverts(self, array):
        """
        Method to create vertex and ivert look up tables

        Parameters
        ----------
        array : np.array
            integer array of model numbers

        """
        iverts = self._modelgrid.iverts
        verts = self._modelgrid.verts
        if iverts is None:
            return

        mkeys = [int(i) for i in np.unique(array)]
        ivlut = {mkey: {} for mkey in mkeys}
        for mkey in mkeys:
            new_iv = 0
            new_iverts = []
            new_verts = []
            tmp_vert_dict = {}
            for node, ivert in enumerate(iverts):
                tiverts = []
                mk, nnode = self._node_map[node]
                if mk == mkey:
                    for iv in ivert:
                        vert = tuple(verts[iv].tolist())
                        if vert in tmp_vert_dict:
                            tiverts.append(tmp_vert_dict[vert])
                        else:
                            tiverts.append(new_iv)
                            new_verts.append([new_iv] + list(vert))
                            tmp_vert_dict[vert] = new_iv
                            new_iv += 1

                    new_iverts.append(tiverts)

            ivlut[mkey]["iverts"] = new_iverts
            ivlut[mkey]["vertices"] = new_verts

        self._ivert_vert_remap = ivlut

    def _create_sln_tdis(self):
        """
        Method to create and add new TDIS and Solution Group objects
        from an existing Simulation

        Parameters
        ----------
        sim : MFSimulation object
            Simulation object that has a model that's being split
        new_sim : MFSimulation object
            New simulation object that will hold the split models

        Returns
        -------
            new_sim : MFSimulation object
        """
        for pak in self._sim.sim_package_list:
            if pak.package_abbr in ("gwfgwt", "gwfgwf", "gwfgwe", "utlats"):
                continue
            pak_cls = PackageContainer.package_factory(pak.package_abbr, "")
            signature = inspect.signature(pak_cls)
            d = {"simulation": self._new_sim, "loading_package": False}
            for key, value in signature.parameters.items():
                if key in ("simulation", "loading_package", "pname", "kwargs"):
                    continue
                elif key == "ats_perioddata":
                    data = getattr(pak, "ats")
                    if len(data._packages) > 0:
                        data = data._packages[0].perioddata.array
                        d[key] = data

                else:
                    data = getattr(pak, key)
                    if hasattr(data, "array"):
                        d[key] = data.array
                    else:
                        d[key] = data

            new_pak = pak_cls(**d)

    def _remap_cell2d(self, item, cell2d, mapped_data):
        """
        Method to remap vertex grid cell2d

        Parameters
        ----------
        item : str
            parameter name string
        cell2d : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        cell2d = cell2d.array

        for mkey in self._model_dict.keys():
            idx = []
            for node, (nmkey, nnode) in self._node_map.items():
                if nmkey == mkey:
                    idx.append(node)

            recarray = cell2d[idx]
            recarray["icell2d"] = range(len(recarray))
            iverts = plotutil.UnstructuredPlotUtilities.irregular_shape_patch(
                self._ivert_vert_remap[mkey]["iverts"]
            ).T
            for ix, ivert_col in enumerate(iverts[:-1]):
                recarray[f"icvert_{ix}"] = ivert_col

            mapped_data[mkey][item] = recarray

        return mapped_data

    def _remap_filerecords(self, item, value, mapped_data, namfile=False):
        """
        Method to create new file record names and map them to their
        associated models

        Parameters
        ----------
        item : str
            parameter name string
        value : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if item in (
            "budget_filerecord",
            "head_filerecord",
            "budgetcsv_filerecord",
            "stage_filerecord",
            "obs_filerecord",
            "concentration_filerecord",
            "ts_filerecord",
            "temperature_filerecord",
            "nc_mesh2d_filerecord"
        ):
            value = value.array
            if value is None:
                pass
            else:
                value = value[0][0]
                for mdl in mapped_data.keys():
                    if mapped_data[mdl] or namfile:
                        new_val = value.split(".")
                        new_val = f"{'.'.join(new_val[0:-1])}_{mdl :0{self._fdigits}d}.{new_val[-1]}"
                        mapped_data[mdl][item] = new_val
        return mapped_data

    def _remap_disu(self, mapped_data):
        """
        Method to remap DISU inputs to new grids

        Parameters
        ----------
        mapped_data : dict
            dictionary of model number and new package data

        Returns
        -------
        mapped_data : dict

        """
        for mkey, metadata in self._usg_metadata.items():
            iac, ja, ihc, cl12, hwva, angldegx = [], [], [], [], [], []
            for node, params in metadata.items():
                conns = [node] + self._new_connections[mkey]["internal"][node]
                iac.append(len(conns))
                ja.extend(conns)
                ihc.extend(params["ihc"])
                cl12.extend(params["cl12"])
                hwva.extend(params["hwva"])
                if self._has_angldegx:
                    angldegx.extend(params["angldegx"])

            assert np.sum(iac) == len(ja)

            mapped_data[mkey]["nja"] = len(ja)
            mapped_data[mkey]["iac"] = iac
            mapped_data[mkey]["ja"] = ja
            mapped_data[mkey]["ihc"] = ihc
            mapped_data[mkey]["cl12"] = cl12
            mapped_data[mkey]["hwva"] = hwva
            if self._has_angldegx:
                mapped_data[mkey]["angldegx"] = angldegx

        return mapped_data

    def _remap_transient_array(self, item, mftransient, mapped_data):
        """
        Method to split and remap transient arrays to new models

        Parameters
        ----------
        item : str
            variable name
        mftransient : mfdataarray.MFTransientArray
            transient array object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if mftransient.array is None:
            return mapped_data

        d0 = {mkey: {} for mkey in self._model_dict.keys()}
        for per, array in enumerate(mftransient.array):
            if per in mftransient._data_storage.keys():
                storage = mftransient._data_storage[per]
                how = [
                    i.data_storage_type.value
                    for i in storage.layer_storage.multi_dim_list
                ]
                binary = [
                    i.binary for i in storage.layer_storage.multi_dim_list
                ]
                fnames = [
                    i.fname for i in storage.layer_storage.multi_dim_list
                ]

                d = self._remap_array(
                    item,
                    array,
                    mapped_data,
                    how=how,
                    binary=binary,
                    fnames=fnames,
                )

                for mkey in d.keys():
                    d0[mkey][per] = d[mkey][item]

        for mkey, values in d0.items():
            mapped_data[mkey][item] = values

        return mapped_data

    def _remap_array(self, item, mfarray, mapped_data, **kwargs):
        """
        Method to remap array nodes to each model

        Parameters
        ----------
        item : str
            variable name
        value : MFArray
            MFArray object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        how = kwargs.pop("how", [])
        binary = kwargs.pop("binary", [])
        fnames = kwargs.pop("fnames", None)
        if not hasattr(mfarray, "size"):
            if mfarray.array is None:
                if item == "idomain":
                    mfarray.set_data(1)
                else:
                    return mapped_data

            how = [
                i.data_storage_type.value
                for i in mfarray._data_storage.layer_storage.multi_dim_list
            ]
            binary = [
                i.binary
                for i in mfarray._data_storage.layer_storage.multi_dim_list
            ]
            fnames = [
                i.fname
                for i in mfarray._data_storage.layer_storage.multi_dim_list
            ]
            mfarray = mfarray.array

        nlay = 1
        if isinstance(self._modelgrid.ncpl, (list, np.ndarray)):
            ncpl = self._modelgrid.nnodes
        else:
            ncpl = self._modelgrid.ncpl

        if mfarray.size == self._modelgrid.size:
            nlay = int(mfarray.size / ncpl)

        original_arr = mfarray.ravel()
        dtype = original_arr.dtype

        for mkey in self._model_dict.keys():
            new_ncpl = self._new_ncpl[mkey]
            new_array = np.zeros(new_ncpl * nlay, dtype=dtype)
            mapping = self._grid_info[mkey][-1]
            new_nodes = np.asarray(mapping != -1).nonzero()
            old_nodes = mapping[new_nodes]

            old_nodes = np.tile(old_nodes, (nlay, 1))
            old_adj_array = np.arange(nlay, dtype=int) * ncpl
            old_adj_array = np.expand_dims(old_adj_array, axis=1)
            old_nodes += old_adj_array
            old_nodes = old_nodes.ravel()

            new_nodes = np.tile(new_nodes, (nlay, 1))
            new_adj_array = np.arange(nlay, dtype=int) * new_ncpl
            new_adj_array = np.expand_dims(new_adj_array, axis=1)
            new_nodes += new_adj_array
            new_nodes = new_nodes.ravel()

            new_array[new_nodes] = original_arr[old_nodes]

            if how and item != "idomain":
                new_input = []
                i0 = 0
                i1 = new_ncpl
                lay = 0
                for h in how:
                    if h == 1:
                        # internal array
                        new_input.append(new_array[i0:i1])
                    elif h == 2:
                        # constant, parse the original array data
                        new_input.append(original_arr[ncpl * lay])
                    else:
                        # external array
                        tmp = fnames[lay].split(".")
                        filename = f"{'.'.join(tmp[:-1])}.{mkey :0{self._fdigits}d}.{tmp[-1]}"

                        cr = {
                            "filename": filename,
                            "factor": 1,
                            "iprn": 1,
                            "data": new_array[i0:i1],
                            "binary": binary[lay],
                        }

                        new_input.append(cr)

                    i0 += new_ncpl
                    i1 += new_ncpl
                    lay += 1

                new_array = new_input

            mapped_data[mkey][item] = new_array

        return mapped_data

    def _remap_mflist(
        self, item, mflist, mapped_data, transient=False, **kwargs
    ):
        """
        Method to remap mflist data to each model

        Parameters
        ----------
        item : str
            variable name
        value : MFList
            MFList object
        mapped_data : dict
            dictionary of remapped package data
        transient : bool
            flag to indicate this is transient stress period data
            flag is needed to trap for remapping mover data.

        Returns
        -------
            dict
        """
        mvr_remap = {}
        how = kwargs.pop("how", 1)
        binary = kwargs.pop("binary", False)
        fname = kwargs.pop("fname", None)
        if hasattr(mflist, "array"):
            if mflist.array is None:
                return mapped_data
            recarray = mflist.array
            how = mflist._data_storage._data_storage_type.value
            binary = mflist._data_storage.layer_storage.multi_dim_list[
                0
            ].binary
        else:
            recarray = mflist

        if "cellid" not in recarray.dtype.names:
            for model in self._model_dict.keys():
                mapped_data[model][item] = recarray.copy()
        else:
            cellids = recarray.cellid
            lay_num, nodes = self._cellid_to_layer_node(cellids)
            new_model, new_node = self._get_new_model_new_node(nodes)

            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model == mkey).nonzero()[0]
                if self._pkg_mover and transient:
                    mvr_remap = {
                        idx[i]: (model.name, i) for i in range(len(idx))
                    }

                if len(idx) == 0:
                    new_recarray = None
                else:
                    new_cellids = self._new_node_to_cellid(
                        model, new_node, lay_num, idx
                    )
                    new_recarray = recarray[idx]
                    new_recarray["cellid"] = new_cellids

                if how == 3 and new_recarray is not None:
                    tmp = fname.split(".")
                    filename = f"{'.'.join(tmp[:-1])}.{mkey :0{self._fdigits}d}.{tmp[-1]}"

                    new_recarray = {
                        "data": new_recarray,
                        "binary": binary,
                        "filename": filename,
                    }

                mapped_data[mkey][item] = new_recarray

        if not transient:
            return mapped_data
        else:
            return mapped_data, mvr_remap

    def _remap_adv_transport(self, package, item, pkg_remap, mapped_data):
        """
        General method to remap advanced transport packages

        Parameters
        ----------
        package : flopy.mf6.Package
        item : str
        pkg_remap : dict
        mapped_data : dict

        Returns
        -------
            mapped_data : dict
        """
        flow_package_name = package.flow_package_name.array
        packagedata = package.packagedata.array
        perioddata = package.perioddata.data

        for mkey in self._model_dict.keys():
            flow_package_const = flow_package_name.split(".")
            new_packagedata = self._remap_adv_tag(
                mkey, packagedata, item, pkg_remap
            )
            if new_packagedata is None:
                continue

            spd = {}
            for per, recarray in perioddata.items():
                new_recarray = self._remap_adv_tag(
                    mkey, recarray, item, pkg_remap
                )
                spd[per] = new_recarray

            flow_package_const[-2] += f"_{mkey :0{self._fdigits}d}"
            new_flow_package_name = ".".join(flow_package_const)
            mapped_data[mkey]["packagedata"] = new_packagedata
            mapped_data[mkey]["perioddata"] = spd
            mapped_data[mkey]["flow_package_name"] = new_flow_package_name
        return mapped_data

    def _remap_uzf(self, package, mapped_data):
        """
        Method to remap a UZF package, probably will work for UZT also
        need to check the input structure of UZT

        Parameters
        ----------
        package : ModflowGwfuzf
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        obs_map = {"ifno": {}}
        if isinstance(package, modflow.ModflowGwfuzf):
            packagedata = package.packagedata.array
            perioddata = package.perioddata.data
            cellids = packagedata.cellid
            layers, nodes = self._cellid_to_layer_node(cellids)
            new_model, new_node = self._get_new_model_new_node(nodes)

            mvr_remap = {}
            name = package.filename
            self._uzf_remaps[name] = {}
            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model == mkey).nonzero()[0]
                if len(idx) == 0:
                    new_recarray = None
                else:
                    new_recarray = packagedata[idx]

                if new_recarray is not None:
                    uzf_remap = {
                        i: ix for ix, i in enumerate(new_recarray.ifno)
                    }
                    if "boundname" in new_recarray.dtype.names:
                        for bname in new_recarray.boundname:
                            uzf_remap[bname] = bname

                    uzf_nodes = [i for i in uzf_remap.keys()]
                    uzf_remap[-1] = -1
                    for oid, nid in uzf_remap.items():
                        mvr_remap[oid] = (model.name, nid)
                        self._uzf_remaps[name][oid] = (mkey, nid)
                        obs_map["ifno"][oid] = (mkey, nid)

                    new_cellids = self._new_node_to_cellid(
                        model, new_node, layers, idx
                    )
                    new_recarray["cellid"] = new_cellids
                    new_recarray["ifno"] = [
                        uzf_remap[i] for i in new_recarray["ifno"]
                    ]
                    new_recarray["ivertcon"] = [
                        uzf_remap[i] for i in new_recarray["ivertcon"]
                    ]

                    obs_map = self._set_boundname_remaps(
                        new_recarray, obs_map, list(obs_map.keys()), mkey
                    )

                    spd = {}
                    for per, recarray in perioddata.items():
                        idx = np.asarray(
                            np.isin(recarray.ifno, uzf_nodes)
                        ).nonzero()
                        new_period = recarray[idx]
                        new_period["ifno"] = [
                            uzf_remap[i] for i in new_period["ifno"]
                        ]
                        spd[per] = new_period

                    mapped_data[mkey]["packagedata"] = new_recarray
                    mapped_data[mkey]["nuzfcells"] = len(new_recarray)
                    mapped_data[mkey]["ntrailwaves"] = (
                        package.ntrailwaves.array
                    )
                    mapped_data[mkey]["nwavesets"] = package.nwavesets.array
                    mapped_data[mkey]["perioddata"] = spd

            if self._pkg_mover:
                for per in range(self._model.nper):
                    if per in self._mover_remaps:
                        self._mover_remaps[per][package.name[0]] = mvr_remap
                    else:
                        self._mover_remaps[per] = {package.name[0]: mvr_remap}
        else:
            name = package.flow_package_name.array
            uzf_remap = self._uzf_remaps[name]
            mapped_data = self._remap_adv_transport(
                package, "ifno", uzf_remap, mapped_data
            )

        for obspak in package.obs._packages:
            mapped_data = self._remap_obs(
                obspak,
                mapped_data,
                obs_map["ifno"],
                pkg_type=package.package_type,
            )

        return mapped_data

    def _remap_mvr(self, package, mapped_data):
        """
        Method to remap internal and external movers from an existing
        MVR package

        Parameters
        ----------
        package : ModflowGwfmvr
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        # self._mvr_remaps = {}
        if isinstance(package, (modflow.ModflowGwtmvt, modflow.ModflowGwemve)):
            return mapped_data

        perioddata = package.perioddata.data
        for mkey, model in self._model_dict.items():
            spd = {}
            maxmvr = 0
            for per, recarray in perioddata.items():
                mover_remaps = self._mover_remaps[per]
                new_records = []
                externals = []
                for rec in recarray:
                    mname1, nid1 = mover_remaps[rec.pname1][rec.id1]
                    if mname1 != model.name:
                        continue
                    mname2, nid2 = mover_remaps[rec.pname2][rec.id2]
                    if mname1 != mname2:
                        new_rec = (
                            mname1,
                            rec.pname1,
                            nid1,
                            mname2,
                            rec.pname2,
                            nid2,
                            rec.mvrtype,
                            rec.value,
                        )
                        externals.append(new_rec)
                    else:
                        new_rec = (
                            rec.pname1,
                            nid1,
                            rec.pname2,
                            nid2,
                            rec.mvrtype,
                            rec.value,
                        )
                        new_records.append(new_rec)

                if new_records:
                    if len(new_records) > maxmvr:
                        maxmvr = len(new_records)

                    spd[per] = new_records

                if externals:
                    if per in self._sim_mover_data:
                        for rec in externals:
                            self._sim_mover_data[per].append(rec)
                    else:
                        self._sim_mover_data[per] = externals

            if spd:
                mapped_data[mkey]["perioddata"] = spd
                mapped_data[mkey]["maxmvr"] = maxmvr
                mapped_data[mkey]["maxpackages"] = len(package.packages.array)
                mapped_data[mkey]["packages"] = package.packages.array

        return mapped_data

    def _remap_lak(self, package, mapped_data):
        """
        Method to remap an existing LAK package

        Parameters
        ----------
        package : ModflowGwflak
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        packagedata = package.packagedata.array
        perioddata = package.perioddata.data

        obs_map = {"ifno": {}, "outletno": {}}
        if isinstance(package, modflow.ModflowGwflak):
            connectiondata = package.connectiondata.array
            tables = package.tables.array
            outlets = package.outlets.array

            lak_remaps = {}
            name = package.filename
            self._lak_remaps[name] = {}
            cellids = connectiondata.cellid
            layers, nodes = self._cellid_to_layer_node(cellids)

            new_model, new_node = self._get_new_model_new_node(nodes)

            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model == mkey).nonzero()[0]
                if len(idx) == 0:
                    new_recarray = None
                else:
                    new_recarray = connectiondata[idx]

                if new_recarray is not None:
                    new_cellids = self._new_node_to_cellid(
                        model, new_node, layers, idx
                    )
                    new_recarray["cellid"] = new_cellids

                    for nlak, lak in enumerate(
                        sorted(np.unique(new_recarray.ifno))
                    ):
                        lak_remaps[lak] = (mkey, nlak)
                        self._lak_remaps[name][lak] = (mkey, nlak)
                        obs_map["ifno"][lak] = (mkey, nlak)

                    new_lak = [lak_remaps[i][-1] for i in new_recarray.ifno]
                    new_recarray["ifno"] = new_lak

                    new_packagedata = self._remap_adv_tag(
                        mkey, packagedata, "ifno", lak_remaps
                    )

                    new_tables = None
                    if tables is not None:
                        new_tables = self._remap_adv_tag(
                            mkey, tables, "ifno", lak_remaps
                        )

                    new_outlets = None
                    if outlets is not None:
                        mapnos = []
                        for lak, meta in lak_remaps.items():
                            if meta[0] == mkey:
                                mapnos.append(lak)

                        idxs = np.asarray(
                            np.isin(outlets.lakein, mapnos)
                        ).nonzero()[0]
                        if len(idxs) == 0:
                            new_outlets = None
                        else:
                            new_outlets = outlets[idxs]
                            lakein = [
                                lak_remaps[i][-1] for i in new_outlets.lakein
                            ]
                            lakeout = [
                                lak_remaps[i][-1] if i in lak_remaps else -1
                                for i in new_outlets.lakeout
                            ]
                            for nout, out in enumerate(
                                sorted(np.unique(new_outlets.outletno))
                            ):
                                obs_map["outletno"][out] = (mkey, nout)

                            outletno = list(range(len(new_outlets)))
                            new_outlets["outletno"] = outletno
                            new_outlets["lakein"] = lakein
                            new_outlets["lakeout"] = lakeout

                    # set boundnames to the observation remapper
                    obs_map = self._set_boundname_remaps(
                        new_packagedata, obs_map, list(obs_map.keys()), mkey
                    )

                    spd = {}
                    for k, recarray in perioddata.items():
                        new_ra = self._remap_adv_tag(
                            mkey, recarray, "number", lak_remaps
                        )
                        spd[k] = new_ra

                    if new_recarray is not None:
                        mapped_data[mkey]["connectiondata"] = new_recarray
                        mapped_data[mkey]["packagedata"] = new_packagedata
                        mapped_data[mkey]["tables"] = new_tables
                        mapped_data[mkey]["outlets"] = new_outlets
                        mapped_data[mkey]["perioddata"] = spd
                        mapped_data[mkey]["nlakes"] = len(new_packagedata.ifno)
                        if new_outlets is not None:
                            mapped_data[mkey]["noutlets"] = len(new_outlets)
                        if new_tables is not None:
                            mapped_data[mkey]["ntables"] = len(new_tables)

            if self._pkg_mover:
                self._set_mover_remaps(package, lak_remaps)
        else:
            name = package.flow_package_name.array
            lak_remap = self._lak_remaps[name]
            mapped_data = self._remap_adv_transport(
                package, "lakno", lak_remap, mapped_data
            )

        for obspak in package.obs._packages:
            mapped_data = self._remap_obs(
                obspak, mapped_data, obs_map, pkg_type=package.package_type
            )

        return mapped_data

    def _remap_sfr(self, package, mapped_data):
        """
        Method to remap an existing SFR package

        Parameters
        ----------
        package : ModflowGwfsfr
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        obs_map = {"ifno": {}}
        if isinstance(package, modflow.ModflowGwfsfr):
            packagedata = package.packagedata.array
            crosssections = package.crosssections.array
            connectiondata = package.connectiondata.array
            diversions = package.diversions.array
            perioddata = package.perioddata.data
            name = package.filename
            self._sfr_remaps[name] = {}
            sfr_remaps = {}
            div_mvr_conn = {}
            sfr_mvr_conn = []
            cellids = packagedata.cellid

            # trap for cells that aren't connected to the GWF model and try to put them
            #  in the model that makes the most sense
            messy_val = (-1,)
            if self._modelgrid.grid_type == "vertex":
                messy_val += (-1,)
            elif self._modelgrid.grid_type == "structured":
                messy_val += (-1, -1)
            messy_idx = [ix for ix, i in enumerate(cellids) if i in (messy_val, "None", None)]
            if messy_idx:
                messy_rch = packagedata.ifno[messy_idx] # store these for later
                rcids = []
                for rch in messy_rch:
                    con = np.abs(connectiondata[connectiondata.ifno == rch].ic_0)[0]
                    cid = packagedata[packagedata.ifno == con].cellid[0]
                    # use absolute value in case this is connected to cell(s) that are
                    # also not connected to the model
                    rcids.append(tuple(np.abs(cid)))
                cellids[messy_idx] = rcids

            layers, nodes = self._cellid_to_layer_node(cellids)
            new_model, new_node = self._get_new_model_new_node(nodes)

            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model == mkey).nonzero()[0]
                if len(idx) == 0:
                    new_recarray = None
                    continue
                else:
                    new_recarray = packagedata[idx]

                if new_recarray is not None:
                    new_cellids = self._new_node_to_cellid(
                        model, new_node, layers, idx
                    )
                    new_recarray["cellid"] = new_cellids
                    # here we reset the "messy val", cells not connected to model
                    if messy_idx:
                        rp_idx = np.where(np.isin(new_recarray.ifno, messy_rch))[0]
                        for ix in rp_idx:
                            new_recarray["cellid"][ix] = messy_val

                    new_rno = []
                    old_rno = []
                    for ix, ifno in enumerate(new_recarray.ifno):
                        new_rno.append(ix)
                        old_rno.append(ifno)
                        sfr_remaps[ifno] = (mkey, ix)
                        sfr_remaps[-1 * ifno] = (mkey, -1 * ix)
                        self._sfr_remaps[name][ifno] = (mkey, ix)
                        obs_map["ifno"][ifno] = (mkey, ix)

                    new_recarray["ifno"] = new_rno
                    obs_map = self._set_boundname_remaps(
                        new_recarray, obs_map, ["ifno"], mkey
                    )

                    # now let's remap connection data and tag external exchanges
                    idx = np.asarray(
                        np.isin(connectiondata.ifno, old_rno)
                    ).nonzero()[0]
                    new_connectiondata = connectiondata[idx]
                    ncons = []
                    for ix, rec in enumerate(new_connectiondata):
                        new_rec = []
                        nan_count = 0
                        for item in new_connectiondata.dtype.names:
                            if rec[item] in sfr_remaps:
                                mn, nrno = sfr_remaps[rec[item]]
                                if mn != mkey:
                                    nan_count += 1
                                else:
                                    new_rec.append(sfr_remaps[rec[item]][-1])
                            elif np.isnan(rec[item]):
                                nan_count += 1
                            else:
                                # this is an instance where we need to map
                                # external connections!
                                nan_count += 1
                                if rec[item] < 0:
                                    # downstream connection
                                    sfr_mvr_conn.append(
                                        (rec["ifno"], int(abs(rec[item])))
                                    )
                                else:
                                    sfr_mvr_conn.append(
                                        (int(rec[item]), rec["ifno"])
                                    )
                        # sort the new_rec so nan is last
                        ncons.append(len(new_rec) - 1)
                        if nan_count > 0:
                            new_rec += [
                                np.nan,
                            ] * nan_count
                        new_connectiondata[ix] = tuple(new_rec)

                    # now we need to go back and change ncon....
                    new_recarray["ncon"] = ncons

                    new_crosssections = None
                    if crosssections is not None:
                        new_crosssections = self._remap_adv_tag(
                            mkey, crosssections, "ifno", sfr_remaps
                        )

                    new_diversions = None
                    div_mover_ix = []
                    if diversions is not None:
                        # first check if diversion outlet is outside the model
                        for ix, rec in enumerate(diversions):
                            ifno = rec.ifno
                            iconr = rec.iconr
                            if (
                                ifno not in sfr_remaps
                                and iconr not in sfr_remaps
                            ):
                                continue
                            elif (
                                ifno in sfr_remaps and iconr not in sfr_remaps
                            ):
                                div_mover_ix.append(ix)
                            else:
                                m0 = sfr_remaps[ifno][0]
                                m1 = sfr_remaps[iconr][0]
                                if m0 != m1:
                                    div_mover_ix.append(ix)

                        idx = np.asarray(
                            np.isin(diversions.ifno, old_rno)
                        ).nonzero()[0]
                        idx = np.asarray(
                            ~np.isin(idx, div_mover_ix)
                        ).nonzero()[0]

                        new_diversions = diversions[idx]
                        new_rno = [
                            sfr_remaps[i][-1] for i in new_diversions.ifno
                        ]
                        new_iconr = [
                            sfr_remaps[i][-1] for i in new_diversions.iconr
                        ]
                        new_idv = list(range(len(new_diversions)))
                        new_diversions["ifno"] = new_rno
                        new_diversions["iconr"] = new_iconr
                        new_diversions["idv"] = new_idv

                        externals = diversions[div_mover_ix]
                        for rec in externals:
                            div_mvr_conn[rec["idv"]] = [
                                rec["ifno"],
                                rec["iconr"],
                                rec["cprior"],
                            ]

                    # now we can do the stress period data
                    spd = {}
                    for kper, recarray in perioddata.items():
                        idx = np.asarray(
                            np.isin(recarray.ifno, old_rno)
                        ).nonzero()[0]
                        new_spd = recarray[idx]
                        if diversions is not None:
                            external_divs = np.asarray(
                                np.isin(new_spd.idv, list(div_mvr_conn.keys()))
                            ).nonzero()[0]
                            if len(external_divs) > 0:
                                for ix in external_divs:
                                    rec = recarray[ix]
                                    idv = recarray["idv"]
                                    div_mvr_conn[idv].append(rec["divflow"])

                            idx = np.asarray(
                                ~np.isin(
                                    new_spd.idv, list(div_mvr_conn.keys())
                                )
                            ).nonzero()[0]

                            new_spd = new_spd[idx]

                        # now to renamp the rnos...
                        new_rno = [sfr_remaps[i][-1] for i in new_spd.ifno]
                        new_spd["ifno"] = new_rno
                        spd[kper] = new_spd

                    mapped_data[mkey]["packagedata"] = new_recarray
                    mapped_data[mkey]["connectiondata"] = new_connectiondata
                    mapped_data[mkey]["crosssections"] = new_crosssections
                    mapped_data[mkey]["diversions"] = new_diversions
                    mapped_data[mkey]["perioddata"] = spd
                    mapped_data[mkey]["nreaches"] = len(new_recarray)

            # connect model network through movers between models
            mvr_recs = []
            mvr_mdl_set = set()
            for rec in sfr_mvr_conn:
                m0, n0 = sfr_remaps[rec[0]]
                m1, n1 = sfr_remaps[rec[1]]
                mvr_mdl_set = mvr_mdl_set | {m0, m1}
                mvr_recs.append(
                    (
                        self._model_dict[m0].name,
                        package.name[0],
                        n0,
                        self._model_dict[m1].name,
                        package.name[0],
                        n1,
                        "FACTOR",
                        1,
                    )
                )

            for idv, rec in div_mvr_conn.items():
                m0, n0 = sfr_remaps[rec[0]]
                m1, n1 = sfr_remaps[rec[1]]
                mvr_mdl_set = mvr_mdl_set | {m0, m1}
                mvr_recs.append(
                    (
                        self._model_dict[m0].name,
                        package.name[0],
                        n0,
                        self._model_dict[m1].name,
                        package.name[0],
                        n1,
                        rec[2],
                        rec[3],
                    )
                )

            if mvr_recs:
                for mkey in mvr_mdl_set:
                    if not mapped_data[mkey]:
                        continue
                    mapped_data[mkey]["mover"] = True
                for per in range(self._model.nper):
                    if per in self._sim_mover_data:
                        for rec in mvr_recs:
                            self._sim_mover_data[per].append(rec)
                    else:
                        self._sim_mover_data[per] = mvr_recs

            # create a remap table for movers between models
            if self._pkg_mover:
                self._set_mover_remaps(package, sfr_remaps)
        else:
            flow_package_name = package.flow_package_name.array
            sfr_remap = self._sfr_remaps[flow_package_name]
            mapped_data = self._remap_adv_transport(
                package, "ifno", sfr_remap, mapped_data
            )

        for obspak in package.obs._packages:
            mapped_data = self._remap_obs(
                obspak,
                mapped_data,
                obs_map["ifno"],
                pkg_type=package.package_type,
            )

        return mapped_data

    def _remap_maw(self, package, mapped_data):
        """
        Method to remap a Multiaquifer well package

        Parameters
        ----------
        package : ModflowGwfmaw
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        obs_map = {"ifno": {}}
        if isinstance(package, modflow.ModflowGwfmaw):
            connectiondata = package.connectiondata.array
            packagedata = package.packagedata.array
            perioddata = package.perioddata.data
            name = package.filename
            self._maw_remaps[name] = {}

            cellids = connectiondata.cellid
            layers, nodes = self._cellid_to_layer_node(cellids)
            new_model, new_node = self._get_new_model_new_node(nodes)
            maw_remaps = {}

            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model == mkey).nonzero()[0]
                new_connectiondata = connectiondata[idx]
                if len(new_connectiondata) == 0:
                    continue
                else:
                    new_cellids = self._new_node_to_cellid(
                        model, new_node, layers, idx
                    )

                    maw_wellnos = []
                    for nmaw, maw in enumerate(
                        sorted(np.unique(new_connectiondata.ifno))
                    ):
                        maw_wellnos.append(maw)
                        maw_remaps[maw] = (mkey, nmaw)
                        self._maw_remaps[maw] = (mkey, nmaw)
                        obs_map["ifno"][maw] = (mkey, nmaw)

                    new_wellno = [
                        maw_remaps[wl][-1] for wl in new_connectiondata.ifno
                    ]
                    new_connectiondata["cellid"] = new_cellids
                    new_connectiondata["ifno"] = new_wellno

                    obs_map = self._set_boundname_remaps(
                        new_connectiondata, obs_map, ["ifno"], mkey
                    )

                    new_packagedata = self._remap_adv_tag(
                        mkey, packagedata, "ifno", maw_remaps
                    )

                    spd = {}
                    for per, recarray in perioddata.items():
                        idx = np.asarray(
                            np.isin(recarray.ifno, maw_wellnos)
                        ).nonzero()[0]
                        if len(idx) > 0:
                            new_recarray = recarray[idx]
                            new_wellno = [
                                maw_remaps[wl][-1] for wl in new_recarray.ifno
                            ]
                            new_recarray["ifno"] = new_wellno
                            spd[per] = new_recarray

                    mapped_data[mkey]["nmawwells"] = len(new_packagedata)
                    mapped_data[mkey]["packagedata"] = new_packagedata
                    mapped_data[mkey]["connectiondata"] = new_connectiondata
                    mapped_data[mkey]["perioddata"] = spd

            if self._pkg_mover:
                self._set_mover_remaps(package, maw_remaps)
        else:
            flow_package_name = package.flow_package_name.array
            maw_remap = self._maw_remaps[flow_package_name]
            mapped_data = self._remap_adv_transport(
                package, "mawno", maw_remap, mapped_data
            )

        for obspak in package.obs._packages:
            mapped_data = self._remap_obs(
                obspak,
                mapped_data,
                obs_map["ifno"],
                pkg_type=package.package_type,
            )

        return mapped_data

    def _remap_csub(self, package, mapped_data):
        """
        Method to remap the CSUB package

        Parameters
        ----------
        package : ModflowGwfcsub
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        mapped_data = self._remap_array(
            "cg_ske_cr", package.cg_ske_cr, mapped_data
        )
        mapped_data = self._remap_array(
            "cg_theta", package.cg_theta, mapped_data
        )
        mapped_data = self._remap_array("sgm", package.sgm, mapped_data)
        mapped_data = self._remap_array("sgs", package.sgs, mapped_data)

        packagedata = package.packagedata.array
        stress_period_data = package.stress_period_data.array

        cellids = packagedata.cellid
        layers, nodes = self._cellid_to_layer_node(cellids)
        new_model, new_node = self._get_new_model_new_node(nodes)

        ninterbeds = None
        for mkey, model in self._model_dict.items():
            idx = np.asarray(new_model == mkey).nonzero()[0]
            if len(idx) == 0:
                new_packagedata = None
            else:
                new_packagedata = packagedata[idx]

            if new_packagedata is not None:
                new_cellids = self._new_node_to_cellid(
                    model, new_node, layers, idx
                )
                new_packagedata["cellid"] = new_cellids
                new_packagedata["ninterbeds"] = list(
                    range(len(new_packagedata))
                )
                ninterbeds = len(new_packagedata)

            spd = {}
            maxsigo = 0
            for per, recarray in stress_period_data.items():
                layers, nodes = self._cellid_to_layer_node(recarray.cellid)
                new_model, new_node = self._get_new_model_new_node(nodes)

                idx = np.asarray(new_model == mkey).nonzero()[0]
                if len(idx) == 0:
                    continue

                new_recarray = recarray[idx]
                new_cellids = self._new_node_to_cellid(
                    model, new_node, layers, idx
                )
                new_recarray["cellid"] = new_cellids

                if len(new_recarray) > maxsigo:
                    maxsigo = len(new_recarray)

                spd[per] = new_recarray

            mapped_data["packagedata"] = new_packagedata
            mapped_data["stress_period_data"] = spd
            mapped_data["ninterbeds"] = ninterbeds
            mapped_data["maxsigo"] = maxsigo

        return mapped_data

    def _remap_buy(self, package, mapped_data):
        """
        Method to remap a Buoyancy package

        Parameters
        ----------
        package : ModflowGwfbuy
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        recarray = package.packagedata.array
        mnames = [i for i in recarray["modelname"]]

        for mkey in mapped_data.keys():
            new_recarray = recarray.copy()
            new_mnames = [
                f"{mn}_{mkey :0{self._fdigits}d}" for mn in mnames
            ]
            new_recarray["modelname"] = new_mnames
            mapped_data[mkey]["packagedata"] = new_recarray

        return mapped_data

    def _set_boundname_remaps(self, recarray, obs_map, variables, mkey):
        """
        Method for creating observation remaps based on boundnames

        Parameters
        ----------
        recarray:
        obs_map:
        variables:
        mkey:

        Returns
        -------
            dict : obs_map
        """
        if "boundname" in recarray.dtype.names:
            for bname in recarray.boundname:
                for variable in variables:
                    if bname in obs_map[variable]:
                        if not isinstance(obs_map[variable][bname], list):
                            obs_map[variable][bname] = [
                                obs_map[variable][bname]
                            ]
                        obs_map[variable][bname].append((mkey, bname))
                    else:
                        obs_map[variable][bname] = (mkey, bname)

        return obs_map

    def _set_mover_remaps(self, package, pkg_remap):
        """
        Method to set remaps that can be used later to remap the mover
        package

        Parameters
        ----------
        package : flopy.mf6.Package
            any flopy package that can be a receiver or provider in the
            mover package
        pkg_remap : dict
            dictionary of remapped package data, such as
            {old well number : (new model name, new well number)}

        """
        mvr_remap = {}
        for oid, (mkey, nid) in pkg_remap.items():
            if oid < 0:
                continue
            name = self._model_dict[mkey].name
            mvr_remap[oid] = (name, nid)

        for per in range(self._model.nper):
            if per in self._mover_remaps:
                self._mover_remaps[per][package.name[0]] = mvr_remap

            else:
                self._mover_remaps[per] = {package.name[0]: mvr_remap}

    def _remap_hfb(self, package, mapped_data):
        """
        Method to remap a horizontal flow barrier package

        Parameters
        ----------
        package : ModflowGwfhfb
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        spd = {}
        for per, recarray in package.stress_period_data.data.items():
            per_dict = {}
            cellids1 = recarray.cellid1
            cellids2 = recarray.cellid2
            layers1, nodes1 = self._cellid_to_layer_node(cellids1)
            layers2, nodes2 = self._cellid_to_layer_node(cellids2)
            new_model1, new_node1 = self._get_new_model_new_node(nodes1)
            new_model2, new_node2 = self._get_new_model_new_node(nodes2)
            if not (new_model1 == new_model2).all():
                raise AssertionError("Models cannot be split along faults")

            for mkey, model in self._model_dict.items():
                idx = np.asarray(new_model1 == mkey).nonzero()[0]
                if len(idx) == 0:
                    new_recarray = None
                else:
                    new_recarray = recarray[idx]

                if new_recarray is not None:
                    new_cellids1 = self._new_node_to_cellid(
                        model, new_node1, layers1, idx
                    )
                    new_cellids2 = self._new_node_to_cellid(
                        model, new_node2, layers2, idx
                    )
                    new_recarray["cellid1"] = new_cellids1
                    new_recarray["cellid2"] = new_cellids2
                    per_dict[mkey] = new_recarray

            for mkey, rec in per_dict.items():
                if "stress_period_data" not in mapped_data[mkey]:
                    mapped_data[mkey]["stress_period_data"] = {per: rec}
                else:
                    mapped_data[mkey]["stress_period_data"][per] = rec

        return mapped_data

    def _remap_fmi(self, package, mapped_data):
        """
        Method to remap a flow model interface package

        Parameters
        ----------
        package : ModflowGwtfmi
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        packagedata = package.packagedata.array
        if packagedata is not None:
            fnames = packagedata.fname

            for mkey in self._model_dict.keys():
                new_fnames = []
                for fname in fnames:
                    new_val = fname.split(".")
                    new_val = f"{'.'.join(new_val[0:-1])}_{mkey :0{self._fdigits}d}.{new_val[-1]}"
                    new_fnames.append(new_val)

                new_packagedata = packagedata.copy()
                new_packagedata["fname"] = new_fnames
                mapped_data[mkey]["packagedata"] = new_packagedata

        return mapped_data

    def _remap_ssm(self, package, mapped_data):
        """
        Method to remap the source and sink mixing package

        Parameters
        ----------
        package : ModflowGwtssm or ModflowGwessm package
        mapped_data : dict
            dictionary of remapped data for package construction

        Returns
        -------
            mapped_data : dict
        """
        sources = package.sources.array
        if sources is not None:
            if self._multimodel_exchange_gwf_names:
                for mkey, mname in self._multimodel_exchange_gwf_names.items():
                    records = []
                    gwf = self._new_sim.get_model(mname)
                    for rec in sources:
                        tmp = gwf.get_package(rec.pname)
                        if tmp is None:
                            continue
                        records.append(tuple(rec))

                    if records:
                        mapped_data[mkey]["sources"] = records
                    else:
                        mapped_data[mkey]["sources"] = None

        return mapped_data

    def _remap_obs(self, package, mapped_data, remapper, pkg_type=None):
        """
        Method to remap an observation package

        Parameters
        ----------
        package : ModflowUtlobs
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        if isinstance(package, modflow.ModflowUtlobs):
            obs_packages = [
                package,
            ]
        else:
            if hasattr(package, "obs"):
                obs_packages = package.obs._packages
            else:
                obs_packages = []

        obs_data = {mkey: {} for mkey in self._model_dict.keys()}
        mm_keys = {}
        mm_idx = []
        for obs_package in obs_packages:
            continuous_data = obs_package.continuous.data
            for ofile, recarray in continuous_data.items():
                if pkg_type is None:
                    layers1, node1 = self._cellid_to_layer_node(recarray.id)
                    new_node1 = np.array(
                        [remapper[i][-1] for i in node1], dtype=int
                    )
                    new_model1 = np.array(
                        [remapper[i][0] for i in node1], dtype=int
                    )

                    new_cellid1 = np.full(
                        (
                            len(
                                recarray,
                            )
                        ),
                        None,
                        dtype=object,
                    )
                    for mkey, model in self._model_dict.items():
                        idx = np.asarray(new_model1 == mkey).nonzero()
                        tmp_cellid = self._new_node_to_cellid(
                            model, new_node1, layers1, idx
                        )
                        new_cellid1[idx] = tmp_cellid
                else:
                    obstype = OBS_ID1_LUT[pkg_type]
                    if obstype != "cellid":
                        obsid = []
                        for i in recarray.id:
                            try:
                                obsid.append(int(i) - 1)
                            except ValueError:
                                obsid.append(i)

                        obsid = np.array(obsid, dtype=object)
                        if isinstance(obstype, dict):
                            new_cellid1 = np.full(
                                len(recarray), None, dtype=object
                            )
                            new_model1 = np.full(
                                len(recarray), None, dtype=object
                            )
                            obstypes = [
                                obstype for obstype in recarray.obstype
                            ]
                            idtype = np.array(
                                [
                                    OBS_ID1_LUT[pkg_type][otype]
                                    for otype in obstypes
                                ],
                                dtype=object,
                            )
                            for idt in set(idtype):
                                remaps = remapper[idt]
                                idx = np.asarray(idtype == idt).nonzero()
                                new_cellid1[idx] = [
                                    (
                                        remaps[i][-1] + 1
                                        if isinstance(i, int)
                                        else i
                                    )
                                    for i in obsid[idx]
                                ]
                                new_model1[idx] = [
                                    remaps[i][0] for i in obsid[idx]
                                ]

                        else:
                            new_cellid1 = np.array(
                                [
                                    (
                                        remapper[i][-1] + 1
                                        if isinstance(i, int)
                                        else i
                                    )
                                    for i in obsid
                                ],
                                dtype=object,
                            )
                            new_model1 = np.array(
                                [remapper[i][0] for i in obsid], dtype=object
                            )

                    else:
                        new_node1 = np.full(
                            (len(recarray),), None, dtype=object
                        )
                        new_model1 = np.full(
                            (len(recarray),), None, dtype=object
                        )

                        bidx = [
                            ix
                            for ix, i in enumerate(recarray.id)
                            if isinstance(i, str)
                        ]
                        idx = [
                            ix
                            for ix, i in enumerate(recarray.id)
                            if not isinstance(i, str)
                        ]
                        if idx:
                            layers1, node1 = self._cellid_to_layer_node(
                                recarray.id[idx]
                            )
                            new_node1[idx] = [remapper[i][-1] for i in node1]
                            new_model1[idx] = [remapper[i][0] for i in node1]

                        new_node1[bidx] = [i for i in recarray.id[bidx]]
                        new_model1[bidx] = [
                            remapper[i][0] for i in recarray.id[bidx]
                        ]

                        new_cellid1 = np.full(
                            (
                                len(
                                    recarray,
                                )
                            ),
                            None,
                            dtype=object,
                        )
                        for mkey, model in self._model_dict.items():
                            idx = np.asarray(new_model1 == mkey).nonzero()[0]
                            idx = [
                                ix
                                for ix in idx
                                if not isinstance(recarray.id[ix], str)
                            ]
                            if idx:
                                tmp_cellid = self._new_node_to_cellid(
                                    model, new_node1, layers1, idx
                                )
                                new_cellid1[idx] = tmp_cellid

                        new_cellid1[bidx] = new_node1[bidx]

                    # check if any boundnames cross model boundaries
                    if isinstance(obstype, dict):
                        remap = remapper[list(remapper.keys())[0]]
                    else:
                        remap = remapper
                    mm_idx = [
                        idx
                        for idx, v in enumerate(new_model1)
                        if isinstance(v, tuple)
                    ]
                    for idx in mm_idx:
                        key = new_model1[idx][-1]
                        for mdl, bname in remap[key]:
                            if key in mm_keys:
                                mm_keys[key].append(mdl)
                            else:
                                mm_keys[key] = [mdl]

                    tmp_models = [new_model1[idx][0] for idx in mm_idx]
                    new_model1[mm_idx] = tmp_models

                cellid2 = recarray.id2
                conv_idx = np.asarray(cellid2 != None).nonzero()[0]  # noqa: E711
                if len(conv_idx) > 0:
                    # need to trap layers...
                    if pkg_type is None:
                        if self._modelgrid.grid_type in (
                            "structured",
                            "vertex",
                        ):
                            layers2 = [
                                cid[0] if cid is not None else None
                                for cid in cellid2
                            ]
                            if self._modelgrid.grid_type == "structured":
                                cellid2 = [
                                    (
                                        (0, cid[1], cid[2])
                                        if cid is not None
                                        else None
                                    )
                                    for cid in cellid2
                                ]
                            else:
                                cellid2 = [
                                    (0, cid[1]) if cid is not None else None
                                    for cid in cellid2
                                ]

                        node2 = self._modelgrid.get_node(
                            list(cellid2[conv_idx])
                        )
                        new_node2 = np.full(
                            (len(recarray),), None, dtype=object
                        )
                        new_model2 = np.full(
                            (len(recarray),), None, dtype=object
                        )

                        new_node2[conv_idx] = [remapper[i][-1] for i in node2]
                        new_model2[conv_idx] = [remapper[i][0] for i in node2]
                        for ix in range(len(recarray)):
                            if ix in conv_idx:
                                continue
                            else:
                                new_node2.append(None)
                                new_model2.append(new_model1[ix])

                        if not np.allclose(new_model1, new_model2):
                            raise AssertionError(
                                "One or more observation records cross model boundaries"
                            )

                        new_cellid2 = np.full(
                            (len(new_node2),), None, dtype=object
                        )
                        for mkey, model in self._model_dict.items():
                            idx = np.asarray(new_model2 == mkey).nonzero()
                            tmp_node = new_node2[idx]
                            cidx = np.asarray(tmp_node != None).nonzero()  # noqa: E711
                            tmp_cellid = model.modelgrid.get_lrc(
                                tmp_node[cidx].to_list()
                            )
                            if self._modelgrid.grid_type in (
                                "structured",
                                "vertex",
                            ):
                                tmp_layers = layers2[cidx]
                                tmp_cellid = [
                                    (tmp_layers[ix],) + cid[1:]
                                    for ix, cid in enumerate(tmp_cellid)
                                ]

                            tmp_node[cidx] = tmp_cellid
                            new_cellid2[idx] = tmp_node
                    else:
                        obstype = OBS_ID2_LUT[pkg_type]
                        if obstype is None:
                            new_cellid2 = cellid2
                        else:
                            obsid = []
                            for i in recarray.id2:
                                try:
                                    obsid.append(int(i) - 1)
                                except ValueError:
                                    obsid.append(i)
                            if isinstance(obstype, dict):
                                new_cellid2 = np.full(
                                    len(recarray), None, dtype=object
                                )
                                obstypes = [
                                    obstype for obstype in recarray.obstype
                                ]
                                idtype = np.array(
                                    [
                                        OBS_ID2_LUT[pkg_type][otype]
                                        for otype in obstypes
                                    ],
                                    dtype=object,
                                )
                                for idt in set(idtype):
                                    if idt is None:
                                        continue
                                    remaps = remapper[idt]
                                    idx = np.asarray(idtype == idt).nonzero()
                                    new_cellid2[idx] = [
                                        (
                                            remaps[i][-1] + 1
                                            if isinstance(i, int)
                                            else i
                                        )
                                        for i in obsid[idx]
                                    ]
                            else:
                                new_cellid2 = np.array(
                                    [
                                        (
                                            remapper[i][-1] + 1
                                            if isinstance(i, int)
                                            else i
                                        )
                                        for i in obsid
                                    ],
                                    dtype=object,
                                )

                else:
                    new_cellid2 = cellid2

                for mkey in self._model_dict.keys():
                    # adjust model numbers if boundname crosses models
                    if mm_keys:
                        for idx in mm_idx:
                            bname = new_cellid1[idx]
                            model_nums = mm_keys[bname]
                            if mkey in model_nums:
                                new_model1[idx] = mkey

                    # now we remap the continuous data!!!!
                    idx = np.asarray(new_model1 == mkey).nonzero()[0]
                    if len(idx) == 0:
                        continue

                    new_recarray = recarray[idx]
                    new_recarray.id = new_cellid1[idx]
                    new_recarray.id2 = new_cellid2[idx]

                    # remap file names
                    if isinstance(ofile, (list, tuple)):
                        rm_ofile = [i for i in ofile]
                        fname = ofile[0]
                        tmp = fname.split(".")
                        tmp[-2] += f"_{mkey :0{self._fdigits}d}"
                        rm_ofile[0] = ".".join(tmp)
                    else:
                        tmp = ofile.split(".")
                        tmp[-2] += f"_{mkey :0{self._fdigits}d}"
                        rm_ofile = ".".join(tmp)

                    if pkg_type is None:
                        if "continuous" not in mapped_data[mkey]:
                            mapped_data[mkey]["continuous"] = {
                                rm_ofile: new_recarray
                            }
                        else:
                            mapped_data[mkey]["continuous"][rm_ofile] = (
                                new_recarray
                            )
                    else:
                        if "observations" not in mapped_data:
                            mapped_data["observations"] = {
                                mkey: {} for mkey in self._model_dict.keys()
                            }

                        if "continuous" not in mapped_data[mkey]:
                            mapped_data["observations"][mkey]["continuous"] = {
                                rm_ofile: new_recarray
                            }
                        else:
                            mapped_data["observations"][mkey]["continuous"][
                                rm_ofile
                            ] = new_recarray

        return mapped_data

    def _cellid_to_layer_node(self, cellids):
        """
        Method to convert cellids to node numbers

        Parameters
        ----------
        cellids : np.array
            array of cellids

        Returns
        -------
        tuple (list of layers, list of nodes)
        """
        if self._modelgrid.grid_type == "structured":
            layers = np.array([i[0] for i in cellids])
            cellids = [(0, i[1], i[2]) for i in cellids]
            nodes = self._modelgrid.get_node(cellids)
        elif self._modelgrid.grid_type == "vertex":
            layers = np.array([i[0] for i in cellids])
            nodes = [i[1] for i in cellids]
        else:
            nodes = [i[0] for i in cellids]
            layers = None

        return layers, nodes

    def _get_new_model_new_node(self, nodes):
        """
        Method to get new model number and node number from the node map

        Parameters
        ----------
        nodes : list, np.ndarray
            iterable of model node numbers

        Returns
        -------
        tuple (list, list) list of new model numbers and new node numbers

        """
        new_model = np.zeros((len(nodes),), dtype=int)
        new_node = np.zeros((len(nodes),), dtype=int)
        for ix, node in enumerate(nodes):
            nm, nn = self._node_map[node]
            new_model[ix] = nm
            new_node[ix] = nn

        return new_model, new_node

    def _new_node_to_cellid(self, model, new_node, layers, idx):
        """
        Method to convert nodes to cellids

        Parameters
        ----------
        model : flopy.mf6.MFModel object
        new_node : list
            list of node numbers
        layers : list
            list of layer numbers
        idx: : list
            index of node numbers to convert to cellids

        Returns
        -------
            list : new cellid numbers
        """

        new_node = new_node[idx].astype(int)
        if self._modelgrid.grid_type == "structured":
            new_node += layers[idx] * model.modelgrid.ncpl
            new_cellids = model.modelgrid.get_lrc(new_node.astype(int))
        elif self._modelgrid.grid_type == "vertex":
            new_cellids = [tuple(cid) for cid in zip(layers[idx], new_node)]

        else:
            new_cellids = [(i,) for i in new_node]

        return new_cellids

    def _remap_adv_tag(self, mkey, recarray, item, mapper):
        """
        Method to remap advanced package ids such as SFR's ifno variable

        Parameters
        ----------
        recarray : np.recarray
        item : str
            variable name to remap
        mapper : dict
            dictionary of {old id: (new model number, new id)}

        Returns
        -------
            np.recarray
        """
        mapnos = []
        for lak, meta in mapper.items():
            if meta[0] == mkey:
                mapnos.append(lak)

        idxs = np.asarray(np.isin(recarray[item], mapnos)).nonzero()[0]
        if len(idxs) == 0:
            new_recarray = None
        else:
            new_recarray = recarray[idxs]
            newnos = [mapper[i][-1] for i in new_recarray[item]]
            new_recarray[item] = newnos
        return new_recarray

    def _remap_transient_list(self, item, mftransientlist, mapped_data):
        """
        Method to remap transient list data to each model

        Parameters
        ----------
        item : str
            parameter name
        mftransientlist : MFTransientList
            MFTransientList object
        mapped_data : dict
            dictionary of remapped package data

        Returns
        -------
            dict
        """
        d0 = {mkey: {} for mkey in self._model_dict.keys()}
        if mftransientlist.data is None:
            for mkey in self._model_dict.keys():
                mapped_data[mkey][item] = None
            return mapped_data

        how = {
            p: i.layer_storage.multi_dim_list[0].data_storage_type.value
            for p, i in mftransientlist._data_storage.items()
        }
        binary = {
            p: i.layer_storage.multi_dim_list[0].binary
            for p, i in mftransientlist._data_storage.items()
        }
        fnames = {
            p: i.layer_storage.multi_dim_list[0].fname
            for p, i in mftransientlist._data_storage.items()
        }

        for per, recarray in mftransientlist.data.items():
            d, mvr_remaps = self._remap_mflist(
                item,
                recarray,
                mapped_data,
                transient=True,
                how=how[per],
                binary=binary[per],
                fname=fnames[per],
            )
            for mkey in self._model_dict.keys():
                if mapped_data[mkey][item] is None:
                    continue
                d0[mkey][per] = mapped_data[mkey][item]

            if mvr_remaps:
                if per in self._mover_remaps:
                    self._mover_remaps[per][self._pkg_mover_name] = mvr_remaps
                else:
                    self._mover_remaps[per] = {
                        self._pkg_mover_name: mvr_remaps
                    }

        for mkey in self._model_dict.keys():
            mapped_data[mkey][item] = d0[mkey]

        return mapped_data

    def _remap_package(self, package, ismvr=False):
        """
        Method to remap package data to new packages in each model

        Parameters
        ----------
        package : flopy.mf6.Package
            Package object
        ismvr : bool
            boolean flag to indicate that this is a mover package
            to remap

        Returns
        -------
            dict
        """
        mapped_data = {mkey: {} for mkey in self._model_dict.keys()}

        # check to see if the package has active movers
        self._pkg_mover = False
        if hasattr(package, "mover"):
            if package.mover.array:
                self._pkg_mover = True
                self._pkg_mover_name = package.name[0]

        if isinstance(
            package,
            (
                modflow.ModflowGwfdis,
                modflow.ModflowGwfdisu,
                modflow.ModflowGwtdis,
                modflow.ModflowGwtdisu,
                modflow.ModflowGwedis,
                modflow.ModflowGwedisu
            ),
        ):
            for item, value in package.__dict__.items():
                if item in ("nja", "ja", "cl12", "ihc", "hwva", "angldegx"):
                    continue

                if item in ("delr", "delc"):
                    for mkey, d in self._grid_info.items():
                        if item == "delr":
                            i0, i1 = d[2]
                        else:
                            i0, i1 = d[1]

                        mapped_data[mkey][item] = value.array[i0 : i1 + 1]

                elif item in ("nrow", "ncol"):
                    for mkey, d in self._grid_info.items():
                        if item == "nrow":
                            i0, i1 = d[1]
                        else:
                            i0, i1 = d[2]

                        mapped_data[mkey][item] = (i1 - i0) + 1

                elif item == "nlay":
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = value.array

                elif item == "nodes":
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = self._grid_info[mkey][0][0]

                elif item == "iac":
                    mapped_data = self._remap_disu(mapped_data)

                elif item == "xorigin":
                    for mkey in self._model_dict.keys():
                        for k, v in self._offsets[mkey].items():
                            mapped_data[mkey][k] = v

                elif item in ("vertices", "cell2d"):
                    if value.array is not None:
                        if item == "cell2d":
                            mapped_data = self._remap_cell2d(
                                item, value, mapped_data
                            )
                        else:
                            for mkey in self._model_dict.keys():
                                mapped_data[mkey][item] = (
                                    self._ivert_vert_remap[mkey][item]
                                )
                                mapped_data[mkey]["nvert"] = len(
                                    self._ivert_vert_remap[mkey][item]
                                )

                elif isinstance(value, mfdataarray.MFArray):
                    mapped_data = self._remap_array(item, value, mapped_data)

        elif isinstance(package, modflow.ModflowGwfhfb):
            mapped_data = self._remap_hfb(package, mapped_data)

        elif isinstance(package, modflow.ModflowGwfcsub):
            mapped_data = self._remap_csub(package, mapped_data)

        elif isinstance(package, modflow.ModflowGwfbuy):
            mapped_data = self._remap_buy(package, mapped_data)

        elif isinstance(package, (modflow.ModflowGwtssm, modflow.ModflowGwessm)):
            mapped_data = self._remap_ssm(package, mapped_data)

        elif isinstance(
            package, (modflow.ModflowGwfuzf, modflow.ModflowGwtuzt, modflow.ModflowGweuze)
        ):
            mapped_data = self._remap_uzf(package, mapped_data)

        elif isinstance(
            package, (modflow.ModflowGwfmaw, modflow.ModflowGwtmwt, modflow.ModflowGwemwe)
        ):
            mapped_data = self._remap_maw(package, mapped_data)

        elif ismvr:
            self._remap_mvr(package, mapped_data)

        elif isinstance(
            package, (modflow.ModflowGwfmvr, modflow.ModflowGwtmvt, modflow.ModflowGwemve)
        ):
            self._mover = True
            return {}

        elif isinstance(
            package, (modflow.ModflowGwflak, modflow.ModflowGwtlkt, modflow.ModflowGwelke)
        ):
            mapped_data = self._remap_lak(package, mapped_data)

        elif isinstance(
            package, (modflow.ModflowGwfsfr, modflow.ModflowGwtsft, modflow.ModflowGwesfe)
        ):
            mapped_data = self._remap_sfr(package, mapped_data)

        elif isinstance(package, modflow.ModflowUtlobs):
            if package.parent_file is not None:
                return {}
            mapped_data = self._remap_obs(package, mapped_data, self._node_map)

        elif isinstance(package, modflow.ModflowGwtfmi):
            mapped_data = self._remap_fmi(package, mapped_data)

        else:
            for item, value in package.__dict__.items():
                if item.startswith("_"):
                    continue

                elif item == "nvert":
                    continue

                elif item in ("ncpl", "nodes"):
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = self._grid_info[mkey][0][0]

                elif item.endswith("_filerecord"):
                    mapped_data = self._remap_filerecords(
                        item, value, mapped_data
                    )

                elif item in ("vertices", "cell2d"):
                    if value.array is not None:
                        if item == "cell2d":
                            mapped_data = self._remap_cell2d(
                                item, value, mapped_data
                            )
                        else:
                            for mkey in self._model_dict.keys():
                                mapped_data[mkey][item] = (
                                    self._ivert_vert_remap[mkey][item]
                                )
                                mapped_data[mkey]["nvert"] = len(
                                    self._ivert_vert_remap[mkey][item]
                                )

                elif item == "xorigin":
                    for mkey in self._model_dict.keys():
                        for k, v in self._offsets[mkey].items():
                            mapped_data[mkey][k] = v

                elif isinstance(value, mfdataarray.MFTransientArray):
                    mapped_data = self._remap_transient_array(
                        item, value, mapped_data
                    )

                elif isinstance(value, mfdataarray.MFArray):
                    mapped_data = self._remap_array(item, value, mapped_data)

                elif isinstance(
                    value,
                    (
                        mfdatalist.MFTransientList,
                        mfdataplist.MFPandasTransientList,
                    ),
                ):
                    if isinstance(value, mfdataplist.MFPandasTransientList):
                        list_data = mfdatalist.MFTransientList(
                            value._simulation_data,
                            value._model_or_sim,
                            value.structure,
                            True,
                            value.path,
                            value.data_dimensions.package_dim,
                            value._package,
                            value._block,
                        )
                        list_data.set_record(value.get_record())
                        value = list_data
                    mapped_data = self._remap_transient_list(
                        item, value, mapped_data
                    )

                elif isinstance(
                    value, (mfdatalist.MFList, mfdataplist.MFPandasList)
                ):
                    if isinstance(value, mfdataplist.MFPandasList):
                        list_data = mfdatalist.MFList(
                            value._simulation_data,
                            value._model_or_sim,
                            value.structure,
                            None,
                            True,
                            value.path,
                            value.data_dimensions.package_dim,
                            value._package,
                            value._block,
                        )
                        list_data.set_record(value.get_record())
                        value = list_data
                    mapped_data = self._remap_mflist(item, value, mapped_data)

                elif isinstance(value, mfdatascalar.MFScalarTransient):
                    for mkey in self._model_dict.keys():
                        val_dict = {}
                        for (
                            perkey,
                            data_storage,
                        ) in value._data_storage.items():
                            val_dict[perkey] = data_storage.get_data()
                        mapped_data[mkey][item] = val_dict
                elif isinstance(value, mfdatascalar.MFScalar):
                    for mkey in self._model_dict.keys():
                        mapped_data[mkey][item] = value.data

                else:
                    pass

            if hasattr(package, "obs"):
                obs_map = {"cellid": self._node_map}
                for mkey, mdict in mapped_data.items():
                    if "stress_period_data" in mdict:
                        for _, ra in mdict["stress_period_data"].items():
                            if isinstance(ra, dict):
                                if "data" not in ra:
                                    continue
                                ra = ra["data"]
                            obs_map = self._set_boundname_remaps(
                                ra, obs_map, list(obs_map.keys()), mkey
                            )
                for obspak in package.obs._packages:
                    mapped_data = self._remap_obs(
                        obspak,
                        mapped_data,
                        obs_map["cellid"],
                        pkg_type=package.package_type,
                    )

        observations = mapped_data.pop("observations", None)

        # trap for timeseries files, in both basic and advanced packages
        if hasattr(package, "tas"):
            value = package.tas
        elif hasattr(package, "ts"):
            value = package.ts
        else:
            value = None
        if value is not None:
            if value._filerecord.array is not None:
                tspkg = value._packages[0]
                for mkey in self._model_dict.keys():
                    new_fname = tspkg.filename.split(".")
                    new_fname = f"{'.'.join(new_fname[0:-1])}_{mkey :0{self._fdigits}d}.{new_fname[-1]}"
                    tsdict = {
                        "filename": new_fname,
                        "timeseries": tspkg.timeseries.array,
                        "time_series_namerecord": [i for i in
                                                   tspkg.time_series_namerecord.array[
                                                       0]],
                        "interpolation_methodrecord": [i for i in
                                                       tspkg.interpolation_methodrecord.array[
                                                           0]]
                    }
                    mapped_data[mkey]["timeseries"] = tsdict

        if "options" in package.blocks:
            for item, value in package.blocks["options"].datasets.items():
                if item.endswith("_filerecord"):
                    mapped_data = self._remap_filerecords(
                        item, value, mapped_data
                    )
                    continue

                elif item in ("flow_package_name", "xorigin", "yorigin"):
                    continue

                for mkey in mapped_data.keys():
                    if mapped_data[mkey]:
                        if item in mapped_data[mkey]:
                            continue
                        elif isinstance(value, mfdatascalar.MFScalar):
                            mapped_data[mkey][item] = value.data
                        elif isinstance(value, mfdatalist.MFList):
                            mapped_data[mkey][item] = value.array

        pak_cls = PackageContainer.package_factory(
            package.package_type, self._model_type
        )
        paks = {}
        for mdl, data in mapped_data.items():
            _ = mapped_data.pop("maxbound", None)
            if data:
                if "stress_period_data" in data:
                    if not data["stress_period_data"]:
                        continue
                paks[mdl] = pak_cls(
                    self._model_dict[mdl], pname=package.name[0], **data
                )

        if observations is not None:
            for mdl, data in observations.items():
                if data:
                    parent = paks[mdl]
                    filename = f"{parent.quoted_filename}.obs"
                    obs = modflow.ModflowUtlobs(
                        parent, pname="obs", filename=filename, **data
                    )

        return paks

    def _create_exchanges(self):
        """
        Method to create exchange packages for fluxes between models

        Returns
        -------
            dict
        """
        d = {}
        exchange_kwargs = {}
        built = []
        nmodels = list(self._model_dict.keys())
        if hasattr(self._model.name_file, "newtonoptions"):
            if self._model.name_file.newtonoptions is not None:
                newton = self._model.name_file.newtonoptions.array
                if isinstance(newton, list):
                    exchange_kwargs["newton"] = True

        if hasattr(self._model, "npf"):
            if self._model.npf.xt3doptions is not None:
                xt3d = self._model.npf.xt3doptions.array
                if isinstance(xt3d, list):
                    exchange_kwargs["xt3d"] = True

        if self._model_type.lower() == "gwf":
            extension = "gwfgwf"
            exchgcls = modflow.ModflowGwfgwf
            check_multi_model = False
        elif self._model_type.lower() == "gwt":
            extension = "gwtgwt"
            exchgcls = modflow.ModflowGwtgwt
            check_multi_model = True
        else:
            raise NotImplementedError()

        if self._modelgrid.grid_type == "unstructured":
            # use existing connection information
            aux = False
            grid_dict = {i: m.modelgrid for i, m in self._model_dict.items()}
            for m0, model in self._model_dict.items():
                grid0 = grid_dict[m0]
                exg_nodes = self._new_connections[m0]["external"]
                for m1 in nmodels:
                    grid1 = grid_dict[m1]
                    if m1 in built:
                        continue
                    if m1 == m0:
                        continue
                    exchange_data = []
                    if check_multi_model:
                        if self._multimodel_exchange_gwf_names:
                            exchange_kwargs["gwfmodelname1"] = (
                                self._multimodel_exchange_gwf_names[m0]
                            )
                            exchange_kwargs["gwfmodelname2"] = (
                                self._multimodel_exchange_gwf_names[m1]
                            )
                    for node0, exg_list in exg_nodes.items():
                        if grid0.idomain[node0] < 1:
                            continue
                        for exg in exg_list:
                            if exg[0] != m1:
                                continue
                            node1 = exg[-1]
                            if grid1.idomain[node1] < 1:
                                continue
                            exg_meta0 = self._exchange_metadata[m0][node0][
                                node1
                            ]
                            exg_meta1 = self._exchange_metadata[m1][node1][
                                node0
                            ]
                            rec = (
                                (node0,),
                                (node1,),
                                1,
                                exg_meta0[3],
                                exg_meta1[3],
                                exg_meta0[-1],
                            )
                            exchange_data.append(rec)

                    if exchange_data:
                        mname0 = self._model_dict[m0].name
                        mname1 = self._model_dict[m1].name
                        exchg = exchgcls(
                            self._new_sim,
                            exgmnamea=mname0,
                            exgmnameb=mname1,
                            nexg=len(exchange_data),
                            exchangedata=exchange_data,
                            filename=f"sim_{mname0}_{mname1}.{extension}",
                            pname=f"{mname0}_{mname1}",
                            **exchange_kwargs,
                        )
                        d[f"{mname0}_{mname1}"] = exchg

                built.append(m0)

            for _, model in self._model_dict.items():
                # turn off save_specific_discharge if it's on
                if hasattr(model, "npf"):
                    model.npf.save_specific_discharge = None

        else:
            xc = self._modelgrid.xcellcenters.ravel()
            yc = self._modelgrid.ycellcenters.ravel()
            verts = self._modelgrid.verts
            for m0, model in self._model_dict.items():
                exg_nodes = self._new_connections[m0]["external"]
                for m1 in nmodels:
                    exchange_data = []
                    if m1 in built:
                        continue
                    if m1 == m0:
                        continue

                    if check_multi_model:
                        if self._multimodel_exchange_gwf_names:
                            exchange_kwargs["gwfmodelname1"] = (
                                self._multimodel_exchange_gwf_names[m0]
                            )
                            exchange_kwargs["gwfmodelname2"] = (
                                self._multimodel_exchange_gwf_names[m1]
                            )

                    modelgrid0 = model.modelgrid
                    modelgrid1 = self._model_dict[m1].modelgrid
                    ncpl0 = modelgrid0.ncpl
                    ncpl1 = modelgrid1.ncpl
                    idomain0 = modelgrid0.idomain
                    idomain1 = modelgrid1.idomain
                    for node0, exg_list in exg_nodes.items():
                        for exg in exg_list:
                            if exg[0] != m1:
                                continue

                            node1 = exg[1]
                            for layer in range(self._modelgrid.nlay):
                                if self._modelgrid.grid_type == "structured":
                                    tmpnode0 = node0 + (ncpl0 * layer)
                                    tmpnode1 = node1 + (ncpl1 * layer)
                                    cellidm0 = modelgrid0.get_lrc([tmpnode0])[
                                        0
                                    ]
                                    cellidm1 = modelgrid1.get_lrc([tmpnode1])[
                                        0
                                    ]
                                elif self._modelgrid.grid_type == "vertex":
                                    cellidm0 = (layer, node0)
                                    cellidm1 = (layer, node1)
                                else:
                                    cellidm0 = node0
                                    cellidm1 = node1

                                if idomain0 is not None:
                                    if idomain0[cellidm0] <= 0:
                                        continue
                                if idomain1 is not None:
                                    if idomain1[cellidm1] <= 0:
                                        continue
                                # calculate CL1, CL2 from exchange metadata
                                meta = self._exchange_metadata[m0][node0][
                                    node1
                                ]
                                ivrt = meta[2]
                                x1 = xc[meta[0]]
                                y1 = yc[meta[0]]
                                x2 = xc[meta[1]]
                                y2 = yc[meta[1]]
                                x3, y3 = verts[ivrt[0]]
                                x4, y4 = verts[ivrt[1]]

                                numa = (x4 - x3) * (y1 - y3) - (y4 - y3) * (
                                    x1 - x3
                                )
                                denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (
                                    y2 - y1
                                )
                                ua = numa / denom
                                x = x1 + ua * (x2 - x1)
                                y = y1 + ua * (y2 - y1)

                                cl0 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                                cl1 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                                hwva = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)

                                # calculate angledegx and cdist
                                angledegx = np.arctan2([y2 - y1], [x2 - x1])[
                                    0
                                ] * (180 / np.pi)
                                if angledegx < 0:
                                    angledegx = 360 + angledegx

                                cdist = np.sqrt(
                                    (x1 - x2) ** 2 + (y1 - y2) ** 2
                                )

                                rec = [
                                    cellidm0,
                                    cellidm1,
                                    1,
                                    cl0,
                                    cl1,
                                    hwva,
                                    angledegx,
                                    cdist,
                                ]
                                exchange_data.append(rec)

                    mvr_data = {}
                    packages = []
                    maxmvr = 0
                    for per, mvrs in self._sim_mover_data.items():
                        mname0 = self._model_dict[m0].name
                        mname1 = self._model_dict[m1].name
                        records = []
                        for rec in mvrs:
                            if rec[0] == mname0 or rec[3] == mname0:
                                if rec[0] == mname1 or rec[3] == mname1:
                                    records.append(rec)
                                    if rec[1] not in packages:
                                        packages.append((rec[0], rec[1]))
                                    if rec[4] not in packages:
                                        packages.append((rec[3], rec[4]))
                        if records:
                            if len(records) > maxmvr:
                                maxmvr = len(records)
                            mvr_data[per] = records

                    if exchange_data or mvr_data:
                        mname0 = self._model_dict[m0].name
                        mname1 = self._model_dict[m1].name
                        exchg = exchgcls(
                            self._new_sim,
                            exgmnamea=mname0,
                            exgmnameb=mname1,
                            auxiliary=["ANGLDEGX", "CDIST"],
                            nexg=len(exchange_data),
                            exchangedata=exchange_data,
                            filename=f"sim_{mname0}_{mname1}.{extension}",
                            pname=f"{mname0}_{mname1}",
                            **exchange_kwargs,
                        )
                        d[f"{mname0}_{mname1}"] = exchg

                        if mvr_data:
                            mvr = modflow.ModflowGwfmvr(
                                exchg,
                                modelnames=True,
                                maxmvr=maxmvr,
                                maxpackages=len(packages),
                                packages=packages,
                                perioddata=mvr_data,
                                filename=f"{mname0}_{mname1}.mvr",
                            )

                            d[f"{mname0}_{mname1}_mvr"] = mvr

                built.append(m0)

        return d

    def create_multi_model_exchanges(self, mname0, mname1):
        """
        Method to create multi-model exchange packages, e.g., GWF-GWT

        Parameters
        ----------
        mname0 : str
            model name of first model in exchange
        mname1 : str
            model name of second model in exchange

        Returns
        -------
            None
        """
        exchange_classes = {
            "gwfgwt": modflow.ModflowGwfgwt,
            "gwfgwe": modflow.ModflowGwfgwe,
        }
        ml0 = self._new_sim.get_model(mname0)
        ml1 = self._new_sim.get_model(mname1)

        mtype0 = ml0.model_type[:3]
        mtype1 = ml1.model_type[:3]

        if mtype0.lower() != "gwf":
            raise AssertionError(
                f"GWF must be the first specified type for multimodel "
                f"exchanges: type supplied {mtype1.upper()} "
            )

        if mtype1.lower() not in ("gwt", "gwe"):
            raise NotImplementedError(
                f"Unsupported exchange type GWF-{mtype1.upper()}"
            )

        exchangecls = exchange_classes[f"{mtype0}{mtype1}"]
        filename = f"{mname0}_{mname1}.exg"
        exchangecls(
            self._new_sim,
            exgmnamea=mname0,
            exgmnameb=mname1,
            filename=filename,
        )

    def split_model(self, array):
        """
        User method to split a model based on an array

        Parameters
        ----------
        array : np.ndarray
            integer array of new model numbers. Array must either be of
            dimension (NROW, NCOL), (NCPL), or (NNODES for unstructured grid
            models).

        Returns
        -------
            MFSimulation object
        """
        if not self._allow_splitting:
            raise AssertionError(
                "Mf6Splitter cannot split a model that "
                "is part of a split simulation"
            )

        # set number formatting string for file paths
        array = np.array(array).astype(int)
        s = str(np.max(array))
        self._fdigits = len(s)

        self._remap_nodes(array)

        if self._new_sim is None:
            self._new_sim = modflow.MFSimulation(
                version=self._sim.version, exe_name=self._sim.exe_name, sim_ws=self._sim.sim_path
            )
            self._create_sln_tdis()

        nam_options = {mkey: {} for mkey in self._new_ncpl.keys()}
        for item, value in self._model.name_file.blocks[
            "options"
        ].datasets.items():
            if item == "list":
                continue
            if value.array is None:
                continue
            if item.endswith("_filerecord"):
                self._remap_filerecords(item, value, nam_options, namfile=True)
            else:
                for mkey in self._new_ncpl.keys():
                    nam_options[mkey][item] = value.array
        self._model_dict = {}
        for mkey in self._new_ncpl.keys():
            mdl_cls = PackageContainer.model_factory(self._model_type)
            self._model_dict[mkey] = mdl_cls(
                self._new_sim,
                modelname=f"{self._modelname}_{mkey :0{self._fdigits}d}",
                **nam_options[mkey],
            )

        for package in self._model.packagelist:
            paks = self._remap_package(package)

        if self._mover:
            mover = self._model.mvr
            self._remap_package(mover, ismvr=True)

        epaks = self._create_exchanges()

        return self._new_sim

    def split_multi_model(self, array):
        """
        Method to split integrated models such as GWF-GWT or GWF-GWE models.
        Note: this method will not work to split multiple connected GWF models

        Parameters
        ----------
        array : np.ndarray
            integer array of new model numbers. Array must either be of
            dimension (NROW, NCOL), (NCPL), or (NNODES for unstructured grid
            models).

        Returns
        -------
            MFSimulation object
        """
        if not self._allow_splitting:
            raise AssertionError(
                "Mf6Splitter cannot split a model that "
                "is part of a split simulation"
            )

        # set number formatting string for file paths
        array = np.array(array).astype(int)
        s = str(np.max(array))
        self._fdigits = len(s)

        # get model names and types, assert that first model is a GWF model
        model_names = self._sim.model_names
        models = [self._sim.get_model(mn) for mn in model_names]
        model_types = [type(ml) for ml in models]

        ix = model_types.index(modflow.ModflowGwf)
        if ix != 0:
            idxs = [
                ix,
            ] + [idx for idx in range(len(model_names)) if idx != ix]
            model_names = [model_names[idx] for idx in idxs]
            models = [models[idx] for idx in idxs]
            model_types = [model_types[idx] for idx in idxs]
            self.switch_models(modelname=model_names[0], remap_nodes=True)

        #  assert consistent idomain and modelgrid shapes!
        shapes = [ml.modelgrid.shape for ml in models]
        idomains = [ml.modelgrid.idomain for ml in models]
        gwf_shape = shapes.pop(0)
        gwf_idomain = idomains.pop(0)

        for ix, shape in enumerate(shapes):
            idomain = idomains[ix]
            mname = model_names[ix + 1]
            if shape != gwf_shape:
                raise AssertionError(
                    f"Model {mname} shape {shape} is not consistent with GWF "
                    f"model shape {gwf_shape}"
                )

            gwf_inactive = np.where(gwf_idomain == 0)
            inactive = np.where(idomain == 0)
            if not np.allclose(inactive, gwf_inactive):
                raise AssertionError(
                    f"Model {mname} idomain is not consistent with GWF "
                    f"model idomain"
                )

        gwf_base = model_names[0]
        model_labels = [
            f"{i :0{self._fdigits}d}" for i in sorted(np.unique(array))
        ]

        self._multimodel_exchange_gwf_names = {
            int(i): f"{gwf_base}_{i}" for i in model_labels
        }

        new_sim = self.split_model(array)
        for mname in model_names[1:]:
            self.switch_models(modelname=mname, remap_nodes=False)
            new_sim = self.split_model(array)

        for mbase in model_names[1:]:
            for label in model_labels:
                mname0 = f"{gwf_base}_{label}"
                mname1 = f"{mbase}_{label}"
                self.create_multi_model_exchanges(mname0, mname1)

        # register models to correct IMS package
        solution_recarray = self._sim.name_file.solutiongroup.data[0]
        sln_mname_cols = [
            i for i in solution_recarray.dtype.names if "slnmnames" in i
        ]
        if len(solution_recarray) > 1:
            # need to associate solutions with solution groups
            imspkgs = []
            imspkg_names = []
            for pkg in new_sim.sim_package_list:
                if isinstance(pkg, modflow.ModflowIms):
                    imspkgs.append(pkg)
                    imspkg_names.append(pkg.filename)

            for record in solution_recarray:
                fname = record.slnfname
                ims_ix = imspkg_names.index(fname)
                ims_obj = imspkgs[ims_ix]
                mnames = []
                for col in sln_mname_cols:
                    mbase = record[col]
                    if mbase is None:
                        continue

                    for label in model_labels:
                        mnames.append(f"{mbase}_{label}")

                new_sim.register_ims_package(ims_obj, mnames)

        return self._new_sim
