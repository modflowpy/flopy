"""
The vtk module provides functionality for exporting model inputs and
outputs to VTK.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from ..datbase import DataInterface, DataType
from ..utils import Util3d, import_optional_dependency

warnings.simplefilter("always", DeprecationWarning)


VTKIGNORE = (
    "vkcb",
    "perlen",
    "steady",
    "tsmult",
    "nstp",
    "iac",
    "ja",
    "ihc",
    "cl12",
    "hwva",
    "angldegx",
    "angledegx",
)


class Pvd:
    """
    Simple class to build a Paraview Data File (PVD)

    """

    def __init__(self):
        self.__data = [
            '<?xml version="1.0"?>\n',
            '<VTKFile type="Collection" version="0.1" '
            'byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n',
            "<Collection>\n",
        ]

    def add_timevalue(self, file, timevalue):
        """
        Method to add a Dataset record to the pvd file

        Parameters
        ----------
        file : os.PathLike or str
            vtu file name
        timevalue : float
            time step value in model time
        """
        file = Path(file)
        if file.suffix != ".vtu":
            file = file.with_suffix(".vtu")

        record = (
            f'<DataSet timestep="{timevalue}" group="" '
            f'part="0" file="{file.name}"/>\n'
        )
        self.__data.append(record)

    def write(self, f):
        """
        Method to write a pvd file from the PVD object.

        Parameters
        ----------
        f : os.PathLike or str
            PVD file name

        """
        f = Path(f)
        if f.suffix != ".pvd":
            f = f.with_suffix(".pvd")

        with f.open("w") as foo:
            foo.writelines(self.__data)
            foo.write("</Collection>\n")
            foo.write("</VTKFile>")


class Vtk:
    """
    Class that builds VTK objects and exports models to VTK files


    Parameters
    ----------
    model : flopy.ModelInterface object
        any flopy model object, example flopy.modflow.Modflow() object
    modelgrid : flopy.discretization.Grid object
        any flopy modelgrid object, example. VertexGrid
    vertical_exageration : float
        floating point value to scale vertical exageration of the vtk points
        default is 1.
    binary : bool
        flag that indicates if Vtk will write a binary or text file. Binary
        is prefered as paraview has a bug (8/4/2021) where is cannot represent
        NaN values from ASCII (non xml) files. In this case no-data values
        are set to 1e+30.
    xml : bool
        flag to write xml based VTK files
    pvd : bool
        boolean flag to write a paraview pvd file for transient data. This
        file maps vtk files to a model time.
    shared_points : bool
        boolean flag to share points in grid polyhedron construction. Default
        is False, as paraview has a bug (8/4/2021) where some polyhedrons will
        not properly close when using shared points. If shared_points is True
        file size will be slightly smaller.
    smooth : bool
        boolean flag to interpolate vertex elevations based on shared cell
        elevations. Default is False.
    point_scalars : bool
        boolen flag to write interpolated data at each point based "shared
        vertices".

    """

    def __init__(
        self,
        model=None,
        modelgrid=None,
        vertical_exageration=1,
        binary=True,
        xml=False,
        pvd=False,
        shared_points=False,
        smooth=False,
        point_scalars=False,
    ):

        vtk = import_optional_dependency("vtk")

        if model is None and modelgrid is None:
            raise AssertionError(
                "A model or modelgrid must be provided to use Vtk"
            )

        elif model is not None:
            self.modelgrid = model.modelgrid
            self.modeltime = model.modeltime
        else:
            self.modelgrid = modelgrid
            self.modeltime = None

        self.binary = binary
        self.xml = xml
        self.pvd = pvd

        if self.pvd and not self.xml:
            print(
                "Switching to xml, ASCII and standard binary are not "
                "supported by Paraview's PVD reader"
            )
            self.xml = True

        self.vertical_exageration = vertical_exageration
        self.shared_points = shared_points
        self.smooth = smooth
        self.point_scalars = point_scalars

        self.verts = self.modelgrid.verts
        self.nnodes = self.modelgrid.nnodes

        # check if iverts is has closed verts
        self.iverts = []
        for iv in self.modelgrid.iverts:
            if iv[0] == iv[-1]:
                iv = iv[:-1]
            self.iverts.append(iv)

        nvpl = 0
        for iv in self.iverts:
            nvpl += len(iv)

        self.nvpl = nvpl

        # method to accomodate DISU grids, do not use modelgrid.ncpl!
        self.ncpl = len(self.iverts)
        if self.nnodes == len(self.iverts):
            self.nlay = 1
        else:
            self.nlay = self.modelgrid.nlay

        self._laycbd = self.modelgrid.laycbd
        if self._laycbd is None:
            self._laycbd = np.zeros((self.nlay,), dtype=int)

        self._active = []
        self._ncbd = 0
        for i in range(self.nlay):
            self._active.append(1)
            if self._laycbd[i] != 0:
                self._active.append(0)
                self._ncbd += 1

        # USG trap
        if self.modelgrid.top.size == self.nnodes:
            self.top = self.modelgrid.top.reshape(self.nnodes)
        else:
            self.top = self.modelgrid.top.reshape(self.ncpl)
        self.botm = self.modelgrid.botm.reshape(-1, self.ncpl)

        if self.modelgrid.idomain is not None:
            self.idomain = self.modelgrid.idomain.reshape(self.nnodes)
        else:
            self.idomain = np.ones((self.nnodes,), dtype=int)

        if self.modeltime is not None:
            perlen = self.modeltime.perlen
            self._totim = np.add.accumulate(perlen)

        self.points = []
        self.faces = []
        self.vtk_grid = None
        self.vtk_polygons = None
        self.vtk_pathlines = None
        self._pathline_points = []
        self._point_scalar_graph = None
        self._point_scalar_numpy_graph = None
        self._idw_weight_graph = None
        self._idw_total_weight_graph = None

        self._vtk_geometry_set = False
        self.__transient_output_data = False
        self.__transient_data = {}
        self.__transient_vector = {}
        self.__pathline_transient_data = {}
        self.__vtk = vtk

        if self.point_scalars:
            self._create_point_scalar_graph()

    def _create_smoothed_elevation_graph(self, adjk, top_elev=True):
        """
        Method to create a dictionary of shared point elevations

        Parameters
        ----------
        adjk : int
            confining bed adjusted layer

        Returns
        -------
        dict
            Key is vertex number, value is elevation

        """
        elevations = {}
        for i, iv in enumerate(self.iverts):
            for v in iv:
                if v is None:
                    continue
                if not top_elev:
                    zv = self.botm[adjk][i] * self.vertical_exageration
                elif adjk == 0:
                    zv = self.top[i] * self.vertical_exageration
                else:
                    if self.top.size == self.nnodes:
                        adji = (adjk * self.ncpl) + i
                        zv = self.top[adji] * self.vertical_exageration
                    else:
                        zv = self.botm[adjk - 1][i] * self.vertical_exageration

                if v in elevations:
                    elevations[v].append(zv)
                else:
                    elevations[v] = [zv]

        for key in elevations:
            elevations[key] = np.mean(elevations[key])

        return elevations

    def _create_point_scalar_graph(self):
        """
        Method to create a point scalar graph to map cells to points

        """
        graph = {}
        v0 = 0
        v1 = 0
        nvert = len(self.verts)
        shared_points = self.shared_points
        if len(self._active) != self.nlay:
            shared_points = False

        for k in range(self.nlay):
            for i, iv in enumerate(self.iverts):
                adji = (k * self.ncpl) + i
                for v in iv:
                    if v is None:
                        continue
                    xvert = self.verts[v, 0]
                    yvert = self.verts[v, 1]
                    adjv = v + v1
                    if adjv not in graph:
                        graph[adjv] = {
                            "vtk_points": [v0],
                            "idx": [adji],
                            "xv": [xvert],
                            "yv": [yvert],
                        }
                    else:
                        graph[adjv]["vtk_points"].append(v0)
                        graph[adjv]["idx"].append(adji)
                        graph[adjv]["xv"].append(xvert)
                        graph[adjv]["yv"].append(yvert)
                    v0 += 1

            v1 += nvert

            if k == self.nlay - 1 or not shared_points:
                for i, iv in enumerate(self.iverts):
                    adji = (k * self.ncpl) + i
                    for v in iv:
                        if v is None:
                            continue
                        xvert = self.verts[v, 0]
                        yvert = self.verts[v, 1]
                        adjv = v + v1
                        if adjv not in graph:
                            graph[adjv] = {
                                "vtk_points": [v0],
                                "idx": [adji],
                                "xv": [xvert],
                                "yv": [yvert],
                            }
                        else:
                            graph[adjv]["vtk_points"].append(v0)
                            graph[adjv]["idx"].append(adji)
                            graph[adjv]["xv"].append(xvert)
                            graph[adjv]["yv"].append(yvert)
                        v0 += 1
                v1 += nvert

        # convert this to a numpy representation
        max_shared = 0
        num_points = v0
        for k, d in graph.items():
            if len(d["vtk_points"]) > max_shared:
                max_shared = len(d["vtk_points"])

        numpy_graph = np.ones((max_shared, num_points), dtype=int) * -1
        xvert = np.ones((max_shared, num_points)) * np.nan
        yvert = np.ones((max_shared, num_points)) * np.nan
        for k, d in graph.items():
            for _, pt in enumerate(d["vtk_points"]):
                for ixx, value in enumerate(d["idx"]):
                    if self.idomain[value] > 0:
                        numpy_graph[ixx, pt] = value
                        xvert[ixx, pt] = d["xv"][ixx]
                        yvert[ixx, pt] = d["yv"][ixx]

        # now create the IDW weights for point scalars
        xc = np.ravel(self.modelgrid.xcellcenters)
        yc = np.ravel(self.modelgrid.ycellcenters)

        if xc.size != self.nnodes:
            xc = np.tile(xc, self.nlay)
            yc = np.tile(yc, self.nlay)

        xc = np.where(numpy_graph != -1, xc[numpy_graph], np.nan)
        yc = np.where(numpy_graph != -1, yc[numpy_graph], np.nan)

        asq = (xvert - xc) ** 2
        bsq = (yvert - yc) ** 2

        weights = np.sqrt(asq + bsq)
        tot_weights = np.nansum(weights, axis=0)

        self._point_scalar_graph = graph
        self._point_scalar_numpy_graph = numpy_graph
        self._idw_weight_graph = weights
        self._idw_total_weight_graph = tot_weights

    def _build_grid_geometry(self):
        """
        Method that creates lists of vertex points and cell faces

        """
        points = []
        faces = []
        v0 = 0
        v1 = 0
        ncb = 0
        shared_points = self.shared_points
        if len(self._active) != self.nlay:
            shared_points = False

        for k in range(self.nlay):
            adjk = k + ncb
            if k != self.nlay - 1:
                if self._active[adjk + 1] == 0:
                    ncb += 1

            if self.smooth:
                elevations = self._create_smoothed_elevation_graph(adjk)

            for i, iv in enumerate(self.iverts):
                for v in iv:
                    if v is None:
                        continue
                    xv = self.verts[v, 0]
                    yv = self.verts[v, 1]
                    if self.smooth:
                        zv = elevations[v]
                    elif k == 0:
                        zv = self.top[i] * self.vertical_exageration
                    else:
                        if self.top.size == self.nnodes:
                            adji = (adjk * self.ncpl) + i
                            zv = self.top[adji] * self.vertical_exageration
                        else:
                            zv = (
                                self.botm[adjk - 1][i]
                                * self.vertical_exageration
                            )

                    points.append([xv, yv, zv])
                    v1 += 1

                cell_faces = [
                    [v for v in range(v0, v1)],
                    [v + self.nvpl for v in range(v0, v1)],
                ]

                for v in range(v0, v1):
                    if v != v1 - 1:
                        cell_faces.append(
                            [v + 1, v, v + self.nvpl, v + self.nvpl + 1]
                        )
                    else:
                        cell_faces.append(
                            [v0, v, v + self.nvpl, v0 + self.nvpl]
                        )

                v0 = v1
                faces.append(cell_faces)

            if k == self.nlay - 1 or not shared_points:
                if self.smooth:
                    elevations = self._create_smoothed_elevation_graph(
                        adjk, top_elev=False
                    )

                for i, iv in enumerate(self.iverts):
                    for v in iv:
                        if v is None:
                            continue
                        xv = self.verts[v, 0]
                        yv = self.verts[v, 1]
                        if self.smooth:
                            zv = elevations[v]
                        else:
                            zv = self.botm[adjk][i] * self.vertical_exageration

                        points.append([xv, yv, zv])
                        v1 += 1

                v0 = v1

        self.points = points
        self.faces = faces

    def _set_vtk_grid_geometry(self):
        """
        Method to set vtk's geometry and add it to the vtk grid object

        """
        if self._vtk_geometry_set:
            return

        if not self.faces:
            self._build_grid_geometry()

        self.vtk_grid = self.__vtk.vtkUnstructuredGrid()

        points = self.__vtk.vtkPoints()
        for point in self.points:
            points.InsertNextPoint(point)

        self.vtk_grid.SetPoints(points)

        for node in range(self.nnodes):
            cell_faces = self.faces[node]
            nface = len(cell_faces)
            fid_list = self.__vtk.vtkIdList()
            fid_list.InsertNextId(nface)
            for face in cell_faces:
                fid_list.InsertNextId(len(face))
                [fid_list.InsertNextId(i) for i in face]

            self.vtk_grid.InsertNextCell(self.__vtk.VTK_POLYHEDRON, fid_list)

        self._vtk_geometry_set = True

    def _build_hfbs(self, pkg):
        """
        Method to add hfb planes to the vtk object

        Parameters
        ----------
        pkg : object
            flopy hfb object

        """
        from vtk.util import numpy_support

        # check if modflow 6 or modflow 2005
        if hasattr(pkg, "hfb_data"):
            mf6 = False
            hfb_data = pkg.hfb_data
        else:
            # asssume that there is no transient hfb data for now
            hfb_data = pkg.stress_period_data.array[0]
            mf6 = True

        points = []
        faces = []
        array = []
        cnt = 0
        verts = self.modelgrid.cross_section_vertices
        verts = np.dstack((verts[0], verts[1]))
        botm = np.ravel(self.botm)
        for hfb in hfb_data:
            if self.modelgrid.grid_type == "structured":
                if not mf6:
                    k = hfb["k"]
                    cell1 = list(hfb[["k", "irow1", "icol1"]])
                    cell2 = list(hfb[["k", "irow2", "icol2"]])
                else:
                    cell1 = hfb["cellid1"]
                    cell2 = hfb["cellid2"]
                    k = cell1[0]
                n0, n1 = self.modelgrid.get_node([cell1, cell2])

            else:
                cell1 = hfb["cellid1"]
                cell2 = hfb["cellid2"]
                if len(cell1) == 2:
                    k, n0 = cell1
                    _, n1 = cell2
                else:
                    n0 = cell1[0]
                    n1 = cell1[1]
                    k = 0

            array.append(hfb["hydchr"])
            adjn0 = n0 - (k * self.ncpl)
            adjn1 = n1 - (k * self.ncpl)
            v1 = verts[adjn0]
            v2 = verts[adjn1]

            # get top and botm elevations, use max and min:
            if k == 0 or self.top.size == self.nnodes:
                tv = np.max([self.top[n0], self.top[n1]])
                bv = np.min([botm[n0], botm[n1]])
            else:
                tv = np.max([botm[n0 - self.ncpl], botm[n1 - self.ncpl]])
                bv = np.min([botm[n0], botm[n1]])

            tv *= self.vertical_exageration
            bv *= self.vertical_exageration

            pts = []
            for v in v1:
                # ix = np.where(v2 == v)
                ix = np.where((v2.T[0] == v[0]) & (v2.T[1] == v[1]))
                if len(ix[0]) > 0 and len(pts) < 2:
                    pts.append(v2[ix[0][0]])

            pts = np.sort(pts)[::-1]

            # create plane, counter-clockwise order
            pts = [
                (pts[0, 0], pts[0, 1], tv),
                (pts[1, 0], pts[1, 1], tv),
                (pts[1, 0], pts[1, 1], bv),
                (pts[0, 0], pts[0, 1], bv),
            ]

            # add to points and faces
            plane_face = []
            for pt in pts:
                points.append(pt)
                plane_face.append(cnt)
                cnt += 1

            faces.append(plane_face)

        # now create the vtk geometry
        vtk_points = self.__vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point)

        # now create an UnstructuredGrid object
        polydata = self.__vtk.vtkUnstructuredGrid()
        polydata.SetPoints(vtk_points)

        # create the vtk polygons
        for face in faces:
            polygon = self.__vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(4)
            for ix, iv in enumerate(face):
                polygon.GetPointIds().SetId(ix, iv)
            polydata.InsertNextCell(
                polygon.GetCellType(), polygon.GetPointIds()
            )

        # and then set the hydchr data
        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=np.array(array), array_type=self.__vtk.VTK_FLOAT
        )
        vtk_arr.SetName("hydchr")
        polydata.GetCellData().SetScalars(vtk_arr)
        self.vtk_polygons = polydata

    def _build_point_scalar_array(self, array):
        """
        Method to build a point scalar array using an inverse distance
        weighting scheme

        Parameters
        ----------
        array : np.ndarray
            ndarray of shape nnodes

        Returns
        -------
        np.ndarray
            array with shape n vtk points
        """
        isint = False
        if np.issubdtype(array[0], np.dtype(int)):
            isint = True

        if isint:
            # maybe think of a way to do ints with the numpy graph,
            #  looping through the dict is very inefficient
            ps_array = np.zeros((len(self.points),), dtype=array.dtype)
            for _, value in self._point_scalar_graph.items():
                for ix, pt in enumerate(value["vtk_points"]):
                    ps_array[pt] = array[value["idx"][ix]]
        else:
            ps_graph = self._point_scalar_numpy_graph.copy()
            idxs = np.where(np.isnan(array))
            not_graphed = np.isin(ps_graph, idxs[0])
            ps_graph[not_graphed] = -1
            ps_array = np.where(ps_graph >= 0, array[ps_graph], np.nan)

            # do inverse distance weighting and apply mask to retain
            # nan valued cells because numpy returns 0 when all vals are nan
            weight_graph = self._idw_weight_graph.copy()
            weight_graph[not_graphed] = np.nan
            weighted_vals = weight_graph * ps_array
            mask = np.isnan(weighted_vals).all(axis=0)
            weighted_vals = np.nansum(weighted_vals, axis=0)
            weighted_vals[mask] = np.nan
            total_weight_graph = np.nansum(weight_graph, axis=0)
            ps_array = weighted_vals / total_weight_graph

        return ps_array

    def _add_timevalue(self, index, fname):
        """
        Method to add a TimeValue to a vtk object, used with transient arrays

        Parameters
        ----------
        index : int, tuple
            integer representing kper or a tuple of (kstp, kper)
        fname : os.PathLike or str
            path to the vtu file

        """
        if not self.pvd:
            return

        try:
            timeval = self._totim[index]
        except (IndexError, KeyError):
            return

        self.pvd.add_timevalue(fname, timeval)

    def _mask_values(self, array, masked_values):
        """
        Method to mask values with nan

        Parameters
        ----------
        array : np.ndarray
            numpy array of values
        masked_values : list
            values to convert to nan

        Returns
        -------
        np.ndarray
        """
        if masked_values is not None:
            try:
                for mv in masked_values:
                    array[array == mv] = np.nan
            except ValueError:
                pass

        return array

    def add_array(self, array, name, masked_values=None, dtype=None):
        """
        Method to set an array to the vtk grid

        Parameters
        ----------
        array : np.ndarray, list
            list of array values to set to the vtk_array
        name : str
            array name for vtk
        masked_values : list, None
            list of values to set equal to nan
        dtype : vtk datatype object
            method to supply and force a vtk datatype

        """
        from vtk.util import numpy_support

        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        array = np.ravel(array)

        if array.size != self.nnodes:
            raise AssertionError("array must be the size as the modelgrid")

        if self.idomain is not None:
            try:
                array[self.idomain == 0] = np.nan
            except ValueError:
                pass

        array = self._mask_values(array, masked_values)

        if not self.binary and not self.xml:
            # ascii does not properly render nan values
            array = np.nan_to_num(array, nan=1e30)

        if self.point_scalars:
            array = self._build_point_scalar_array(array)

        if dtype is None:
            dtype = self.__vtk.VTK_FLOAT
            if np.issubdtype(array[0], np.dtype(int)):
                dtype = self.__vtk.VTK_INT

        vtk_arr = numpy_support.numpy_to_vtk(num_array=array, array_type=dtype)
        vtk_arr.SetName(name)

        if self.point_scalars:
            self.vtk_grid.GetPointData().AddArray(vtk_arr)
        else:
            self.vtk_grid.GetCellData().AddArray(vtk_arr)

    def add_transient_array(self, d, name=None, masked_values=None):
        """
        Method to add transient array data to the vtk object

        Parameters
        ----------
        d: dict
            dictionary of array2d, arry3d data or numpy array data
        name : str, None
            parameter name, required when user provides a dictionary
            of numpy arrays
        masked_values : list, None
            list of values to set equal to nan

        Returns
        -------
        None
        """
        if self.__transient_output_data:
            raise AssertionError(
                "Transient arrays cannot be mixed with transient output, "
                "Please create a seperate vtk object for transient package "
                "data"
            )

        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        k = list(d.keys())[0]
        transient = dict()
        if isinstance(d[k], DataInterface):
            if d[k].data_type in (DataType.array2d, DataType.array3d):
                if name is None:
                    name = d[k].name
                    if isinstance(name, list):
                        name = name[0]

                for kper, value in d.items():
                    if value.array.size != self.nnodes:
                        array = np.zeros(self.nnodes) * np.nan
                        array[: value.array.size] = np.ravel(value.array)
                    else:
                        array = value.array

                    array = self._mask_values(array, masked_values)
                    transient[kper] = array
        else:
            if name is None:
                raise ValueError(
                    "name must be specified when providing numpy arrays"
                )
            for kper, trarray in d.items():
                if trarray.size != self.nnodes:
                    array = np.zeros(self.nnodes) * np.nan
                    array[: trarray.size] = np.ravel(trarray)
                else:
                    array = trarray

                array = self._mask_values(array, masked_values)
                transient[kper] = array

        for k, v in transient.items():
            if k not in self.__transient_data:
                self.__transient_data[k] = {name: v}
            else:
                self.__transient_data[k][name] = v

    def add_transient_list(self, mflist, masked_values=None):
        """
        Method to add transient list data to a vtk object

        Parameters
        ----------
        mflist : flopy.utils.MfList object
        masked_values : list, None
            list of values to set equal to nan

        """
        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        pkg_name = mflist.package.name[0]
        mfl = mflist.array
        if isinstance(mfl, dict):
            for arr_name, arr4d in mflist.array.items():
                d = {kper: array for kper, array in enumerate(arr4d)}
                name = f"{pkg_name}_{arr_name}"
                self.add_transient_array(d, name)
        else:
            export = {}
            for kper in range(mflist.package.parent.nper):
                try:
                    arr_dict = mflist.to_array(kper, mask=True)
                except ValueError:
                    continue

                if arr_dict is None:
                    continue

                for arr_name, array in arr_dict.items():
                    if arr_name not in export:
                        export[arr_name] = {kper: array}
                    else:
                        export[arr_name][kper] = array

            for arr_name, d in export.items():
                name = f"{pkg_name}_{arr_name}"
                self.add_transient_array(d, name, masked_values=masked_values)

    def add_vector(self, vector, name, masked_values=None):
        """
        Method to add vector data to vtk

        Parameters
        ----------
        vector : array
            array of dimension (3, nnodes)
        name : str
            name of the vector to be displayed in vtk
        masked_values : list, None
            list of values to set equal to nan

        """
        from vtk.util import numpy_support

        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        if isinstance(vector, (tuple, list)):
            vector = np.array(vector)

        if vector.size != 3 * self.nnodes:
            if vector.size == 3 * self.ncpl:
                vector = np.reshape(vector, (3, self.ncpl))
                tv = np.full((3, self.nnodes), np.nan)
                for ix, q in enumerate(vector):
                    tv[ix, : self.ncpl] = q
                vector = tv
            else:
                raise AssertionError(
                    "Size of vector must be 3 * nnodes or 3 * ncpl"
                )
        else:
            vector = np.reshape(vector, (3, self.nnodes))

        if self.point_scalars:
            tmp = []
            for v in vector:
                tmp.append(self._build_point_scalar_array(v))
            vector = np.array(tmp)

        vector = self._mask_values(vector, masked_values)

        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=vector, array_type=self.__vtk.VTK_FLOAT
        )
        vtk_arr.SetName(name)
        vtk_arr.SetNumberOfComponents(3)

        if self.point_scalars:
            self.vtk_grid.GetPointData().SetVectors(vtk_arr)
        else:
            self.vtk_grid.GetCellData().SetVectors(vtk_arr)

    def add_transient_vector(self, d, name, masked_values=None):
        """
        Method to add transient vector data to vtk

        Parameters
        ----------
        d : dict
            dictionary of vectors
        name : str
            name of vector to be displayed in vtk
        masked_values : list, None
            list of values to set equal to nan

        """
        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        if self.__transient_data:
            k = list(self.__transient_data.keys())[0]
            if len(d) != len(self.__transient_data[k]):
                print(
                    "Transient vector not same size as transient arrays time "
                    "stamp may be unreliable for vector data in VTK file"
                )

        if isinstance(d, dict):
            cnt = 0
            for key, value in d.items():
                if not isinstance(value, np.ndarray):
                    value = np.array(value)

                if (
                    value.size != 3 * self.ncpl
                    or value.size != 3 * self.nnodes
                ):
                    raise AssertionError(
                        "Size of vector must be 3 * nnodes or 3 * ncpl"
                    )

                value = self._mask_values(value, masked_values)
                self.__transient_vector[cnt] = {name: value}
                cnt += 1

    def add_package(self, pkg, masked_values=None):
        """
        Method to set all arrays within a package to VTK arrays

        Parameters
        ----------
        pkg : flopy.pakbase.Package object
            flopy package object, example ModflowWel
        masked_values : list, None
            list of values to set equal to nan

        """
        if not self._vtk_geometry_set:
            self._set_vtk_grid_geometry()

        if "hfb" in pkg.name[0].lower():
            self._build_hfbs(pkg)
            return

        for item, value in pkg.__dict__.items():
            if item in VTKIGNORE:
                continue

            if isinstance(value, list):
                for v in value:
                    if isinstance(v, Util3d):
                        if value.array.size != self.nnodes:
                            continue
                        self.add_array(v.array, item, masked_values)

            if isinstance(value, DataInterface):
                if value.data_type in (DataType.array2d, DataType.array3d):
                    if value.array is not None:
                        if value.array.size < self.nnodes:
                            if value.array.size < self.ncpl:
                                continue

                            array = np.zeros(self.nnodes) * np.nan
                            array[: value.array.size] = np.ravel(value.array)

                        elif value.array.size > self.nnodes and self._ncbd > 0:
                            # deal with confining beds
                            array = np.array(
                                [
                                    value.array[ix]
                                    for ix, i in enumerate(self._active)
                                    if i != 0
                                ]
                            )

                        else:
                            array = value.array

                        self.add_array(array, item, masked_values)

                elif value.data_type == DataType.transient2d:
                    if value.array is not None:
                        if hasattr(value, "transient_2ds"):
                            self.add_transient_array(
                                value.transient_2ds, item, masked_values
                            )
                        else:
                            d = {ix: i for ix, i in enumerate(value.array)}
                            self.add_transient_array(d, item, masked_values)

                elif value.data_type == DataType.transient3d:
                    if value.array is not None:
                        self.add_transient_array(
                            value.transient_3ds, item, masked_values
                        )

                elif value.data_type == DataType.transientlist:
                    self.add_transient_list(value, masked_values)

                else:
                    pass

    def add_model(self, model, selpaklist=None, masked_values=None):
        """
        Method to add all array and transient list data from a modflow model
        to a timeseries of vtk files

        Parameters
        ----------
        model : fp.modflow.ModelInterface
            any flopy model object
        selpaklist : list, None
            optional parameter where the user can supply a list of packages
            to export.
        masked_values : list, None
            list of values to set equal to nan

        """
        for package in model.packagelist:
            if selpaklist is not None:
                if package.name[0] not in selpaklist:
                    continue

            self.add_package(package, masked_values)

    def add_pathline_points(self, pathlines, timeseries=False):
        """
        Method to add Modpath output from a pathline or timeseries file
        to the grid. Colors will be representative of totim.

        Parameters
        ----------
        pathlines : np.recarray or list
            pathlines accepts a numpy recarray of a particle pathline or
            a list of numpy reccarrays associated with pathlines
        timeseries : bool
            method to plot data as a series of vtk timeseries files for
            animation or as a single static vtk file. Default is false

        """
        if isinstance(pathlines, (np.recarray, np.ndarray)):
            pathlines = [pathlines]

        keys = ["particleid", "time"]
        if not timeseries:
            arrays = {key: [] for key in keys}
            points = []
            for recarray in pathlines:
                recarray["z"] *= self.vertical_exageration
                for rec in recarray:
                    points.append(tuple(rec[["x", "y", "z"]]))
                    for key in keys:
                        arrays[key].append(rec[key])

            self._set_modpath_point_data(points, arrays)

        else:
            self.vtk_pathlines = self.__vtk.vtkUnstructuredGrid()
            timeseries_data = {}
            points = {}
            for recarray in pathlines:
                recarray["z"] *= self.vertical_exageration
                for rec in recarray:
                    time = rec["time"]
                    if time not in points:
                        points[time] = [tuple(rec[["x", "y", "z"]])]
                        t = {key: [] for key in keys}
                        timeseries_data[time] = t

                    else:
                        points[time].append(tuple(rec[["x", "y", "z"]]))

                    for key in keys:
                        timeseries_data[time][key].append(rec[key])

            self.__pathline_transient_data = timeseries_data
            self._pathline_points = points

    def add_heads(self, hds, kstpkper=None, masked_values=None):
        """
        Method to add head data to a vtk file

        Parameters
        ----------
        hds : flopy.utils.LayerFile object
            Binary or Formatted HeadFile type object
        kstpkper : tuple, list of tuples, None
            tuple or list of tuples of kstpkper, if kstpkper=None all
            records are selected
        masked_values : list, None
            list of values to set equal to nan

        """
        if not self.__transient_output_data and self.__transient_data:
            raise AssertionError(
                "Head data cannot be mixed with transient package data, "
                "Please create a seperate vtk object for transient head data"
            )

        if kstpkper is None:
            kstpkper = hds.get_kstpkper()
        elif isinstance(kstpkper, (list, tuple)):
            if not isinstance(kstpkper[0], (list, tuple)):
                kstpkper = [kstpkper]
        else:
            pass

        # reset totim based on values read from head file
        times = hds.get_times()
        kstpkpers = hds.get_kstpkper()
        self._totim = {ki: time for (ki, time) in zip(kstpkpers, times)}

        text = hds.text.decode()

        d = dict()
        for ki in kstpkper:
            d[ki] = hds.get_data(ki)

        self.__transient_output_data = False
        self.add_transient_array(d, name=text, masked_values=masked_values)
        self.__transient_output_data = True

    def add_cell_budget(
        self, cbc, text=None, kstpkper=None, masked_values=None
    ):
        """
        Method to add cell budget data to vtk

        Parameters
        ----------
        cbc : flopy.utils.CellBudget object
            flopy binary CellBudget object
        text : str or None
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc. If text=None
            all compatible records are exported
        kstpkper : tuple, list of tuples, None
            tuple or list of tuples of kstpkper, if kstpkper=None all
            records are selected
        masked_values : list, None
            list of values to set equal to nan

        """
        if not self.__transient_output_data and self.__transient_data:
            raise AssertionError(
                "Binary data cannot be mixed with transient package data, "
                "Please create a seperate vtk object for transient head data"
            )

        records = cbc.get_unique_record_names(decode=True)
        imeth_dict = {
            record: imeth for (record, imeth) in zip(records, cbc.imethlist)
        }
        if text is None:
            keylist = records
        else:
            if not isinstance(text, list):
                keylist = [text]
            else:
                keylist = text

        if kstpkper is None:
            kstpkper = cbc.get_kstpkper()
        elif isinstance(kstpkper, tuple):
            if not isinstance(kstpkper[0], (list, tuple)):
                kstpkper = [kstpkper]
        else:
            pass

        # reset totim based on values read from budget file
        times = cbc.get_times()
        kstpkpers = cbc.get_kstpkper()
        self._totim = {ki: time for (ki, time) in zip(kstpkpers, times)}

        for name in keylist:
            d = {}
            for i, ki in enumerate(kstpkper):
                try:
                    array = cbc.get_data(kstpkper=ki, text=name, full3D=True)
                    if len(array) == 0:
                        continue

                    array = np.ma.filled(array, np.nan)
                    if array.size < self.nnodes:
                        if array.size < self.ncpl:
                            raise AssertionError(
                                "Array size must be equal to "
                                "either ncpl or nnodes"
                            )

                        array = np.zeros(self.nnodes) * np.nan
                        array[: array.size] = np.ravel(array)

                except ValueError:
                    if imeth_dict[name] == 6:
                        array = np.full((self.nnodes,), np.nan)
                        rec = cbc.get_data(kstpkper=ki, text=name)[0]
                        for [node, q] in zip(rec["node"], rec["q"]):
                            array[node] = q
                    else:
                        continue

                d[ki] = array

            self.__transient_output_data = False
            self.add_transient_array(d, name, masked_values)
            self.__transient_output_data = True

    def _set_modpath_point_data(self, points, d):
        """
        Method to build the vtk point geometry and set arrays for
        modpath pathlines

        Parameters
        ----------
        points : list
            list of (x, y, z) points
        d : dict
            dictionary of numpy arrays to add to vtk

        """
        from vtk.util import numpy_support

        nverts = len(points)

        self.vtk_pathlines = self.__vtk.vtkUnstructuredGrid()

        vtk_points = self.__vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point)

        self.vtk_pathlines.SetPoints(vtk_points)

        # create a Vertex instance for each point data add to grid
        for i in range(nverts):
            vertex = self.__vtk.vtkPolyVertex()
            vertex.GetPointIds().SetNumberOfIds(1)
            vertex.GetPointIds().SetId(0, i)

            # set data to the pathline grid
            self.vtk_pathlines.InsertNextCell(
                vertex.GetCellType(), vertex.GetPointIds()
            )

        # process arrays and add arrays to grid.
        for name, array in d.items():
            array = np.array(array)
            vtk_array = numpy_support.numpy_to_vtk(
                num_array=array, array_type=self.__vtk.VTK_FLOAT
            )
            vtk_array.SetName(name)
            self.vtk_pathlines.GetCellData().AddArray(vtk_array)

    def write(self, f, kper=None):
        """
        Method to write a vtk file from the VTK object

        Parameters
        ----------
        f : os.PathLike or str
            vtk file name
        kpers : int, list, tuple
            stress period or list of stress periods to write to vtk. This
            parameter only applies to transient package data.

        """
        grids = [self.vtk_grid, self.vtk_polygons, self.vtk_pathlines]
        suffix = ["", "_hfb", "_pathline"]

        extension = ".vtk"
        if self.pvd:
            self.pvd = Pvd()
            extension = ".vtu"

        f = Path(f)
        f.parent.mkdir(exist_ok=True, parents=True)

        if kper is not None:
            if isinstance(kper, (int, float)):
                kper = [int(kper)]

        for ix, grid in enumerate(grids):
            if grid is None:
                continue

            if f.suffix not in (".vtk", ".vtu"):
                foo = f.parent / f"{f.name}{suffix[ix]}{extension}"
            else:
                foo = f.parent / f"{f.stem}{suffix[ix]}{f.suffix}"

            if not self.xml:
                w = self.__vtk.vtkUnstructuredGridWriter()
                if self.binary:
                    w.SetFileTypeToBinary()
            else:
                w = self.__vtk.vtkXMLUnstructuredGridWriter()
                if not self.binary:
                    w.SetDataModeToAscii()

            if self.__pathline_transient_data and ix == 2:
                stp = 0
                for time, d in self.__pathline_transient_data.items():
                    tf = self.__create_transient_vtk_path(foo, stp)
                    points = self._pathline_points[time]
                    self._set_modpath_point_data(points, d)

                    w.SetInputData(self.vtk_pathlines)
                    w.SetFileName(str(tf))
                    w.Update()
                    stp += 1

            else:
                w.SetInputData(grid)

                if (
                    self.__transient_data or self.__transient_vector
                ) and ix == 0:
                    if self.__transient_data:
                        cnt = 0
                        for per, d in self.__transient_data.items():
                            if kper is not None:
                                if per not in kper:
                                    continue

                            if self.__transient_output_data:
                                tf = self.__create_transient_vtk_path(foo, cnt)
                            else:
                                tf = self.__create_transient_vtk_path(foo, per)
                            self._add_timevalue(per, tf)
                            for name, array in d.items():
                                self.add_array(array, name)

                            if per in self.__transient_vector:
                                d = self.__transient_vector[d]
                                for name, vector in d.items():
                                    self.add_vector(vector, name)

                            w.SetFileName(str(tf))
                            w.Update()
                            cnt += 1
                    else:
                        cnt = 0
                        for per, d in self.__transient_vector.items():
                            if kper is not None:
                                if per not in kper:
                                    continue

                            if self.__transient_output_data:
                                tf = self.__create_transient_vtk_path(foo, cnt)
                            else:
                                tf = self.__create_transient_vtk_path(foo, per)
                            self._add_timevalue(per)
                            for name, vector in d.items():
                                self.add_vector(vector, name)

                            w.SetFileName(str(tf))
                            w.update()
                            cnt += 1
                else:
                    w.SetFileName(str(foo))
                    w.Update()

        if not type(self.pvd) == bool:
            if f.suffix not in (".vtk", ".vtu"):
                pvdfile = f.parent / f"{f.name}.pvd"
            else:
                pvdfile = f.with_suffix(".pvd")

            self.pvd.write(pvdfile)

    def __create_transient_vtk_path(self, path, kper):
        """
        Method to set naming convention for transient vtk file series

        Parameters
        ----------
        path : Path
            vtk file path
        kper : int
            zero based stress period number

        Returns
        -------
        Path
            updated vtk file path of format <filebase>_{:06d}.vtk where
            {:06d} represents the six zero padded stress period time
        """
        return path.parent / f"{path.stem.rstrip('_')}_{kper:06d}{path.suffix}"


def export_model(
    model,
    otfolder,
    package_names=None,
    nanval=-1e20,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
    kpers=None,
):
    """
    Export model to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_model`

    Parameters
    ----------
    model : flopy model instance
        flopy model
    otfolder : str
        output folder
    package_names : list
        list of package names to be exported
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        True if array is 2d, default is False
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """
    warnings.warn("export_model is deprecated, please use Vtk.add_model()")

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")

    vtk = Vtk(
        model,
        vertical_exageration=1,
        binary=binary,
        smooth=smooth,
        point_scalars=point_scalars,
    )

    vtk.add_model(model, package_names)

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    name = model.name
    vtk.write(os.path.join(otfolder, name), kper=kpers)


def export_package(
    pak_model,
    pak_name,
    otfolder,
    vtkobj=None,
    nanval=-1e20,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
    kpers=None,
):
    """
    Export package to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_package`

    Parameters
    ----------
    pak_model : flopy model instance
        the model of the package
    pak_name : str
        the name of the package
    otfolder : str
        output folder to write the data
    vtkobj : VTK instance
        a vtk object (allows export_package to be called from
        export_model)
    nanval : scalar
        no data value, default value is -1e20
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """
    warnings.warn("export_package is deprecated, use Vtk.add_package()")

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")

    if not vtkobj:
        vtk = Vtk(
            pak_model,
            binary=binary,
            smooth=smooth,
            point_scalars=point_scalars,
        )
    else:
        vtk = vtkobj

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    if isinstance(pak_name, list):
        pak_name = pak_name[0]

    p = pak_model.get_package(pak_name)
    vtk.add_package(p)

    vtk.write(os.path.join(otfolder, pak_name), kper=kpers)


def export_transient(
    model,
    array,
    output_folder,
    name,
    nanval=-1e20,
    array2d=False,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
    kpers=None,
):
    """
    Export transient arrays and lists to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_transient_array` or :meth:`Vtk.add_transient_list`

    Parameters
    ----------
    model : MFModel
        the flopy model instance
    array : Transient instance
        flopy transient array
    output_folder : str
        output folder to write the data
    name : str
        name of array
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        True if array is 2d, default is False
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """
    warnings.warn(
        "export_transient is deprecated, use Vtk.add_transient_array() or "
        "Vtk.add_transient_list()"
    )

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")
    if array2d:
        warnings.warn(
            "array2d parameter is deprecated, 2d arrays are "
            "handled automatically"
        )

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vtk = Vtk(model, binary=binary, smooth=smooth, point_scalars=point_scalars)

    if array.data_type == DataType.transient2d:
        if array.array is not None:
            if hasattr(array, "transient_2ds"):
                vtk.add_transient_array(array.transient_2ds, name)
            else:
                d = {ix: i for ix, i in enumerate(array.array)}
                vtk.add_transient_array(d, name)

    elif array.data_type == DataType.transient3d:
        if array.array is None:
            vtk.add_transient_array(array.transient_3ds, name)

    elif array.data_type == DataType.transientlist:
        vtk.add_transient_list(array)

    else:
        raise TypeError(f"type {type(array)} not valid for export_transient")

    vtk.write(os.path.join(output_folder, name), kper=kpers)


def export_array(
    model,
    array,
    output_folder,
    name,
    nanval=-1e20,
    array2d=False,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
):
    """
    Export array to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_array`

    Parameters
    ----------
    model : flopy model instance
        the flopy model instance
    array : flopy array
        flopy 2d or 3d array
    output_folder : str
        output folder to write the data
    name : str
        name of array
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        true if the array is 2d and represents the first layer, default is
        False
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """
    warnings.warn("export_array is deprecated, please use Vtk.add_array()")

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")
    if array2d:
        warnings.warn(
            "array2d parameter is deprecated, 2d arrays are "
            "handled automatically"
        )

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if array.size < model.modelgrid.nnodes:
        if array.size < model.modelgrid.ncpl:
            raise AssertionError(
                "Array size must be equal to either ncpl or nnodes"
            )

        array = np.zeros(model.modelgrid.nnodes) * np.nan
        array[: array.size] = np.ravel(array)

    vtk = Vtk(model, binary=binary, smooth=smooth, point_scalars=point_scalars)

    vtk.add_array(array, name)
    vtk.write(os.path.join(output_folder, name))


def export_heads(
    model,
    hdsfile,
    otfolder,
    text="head",
    precision="auto",
    verbose=False,
    nanval=-1e20,
    kstpkper=None,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
):
    """
    Exports binary head file to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_heads`

    Parameters
    ----------
    model : MFModel
        the flopy model instance
    hdsfile : str, HeadFile object
        binary head file path or object
    otfolder : str
        output folder to write the data
    text : string
        Name of the text string in the head file.  Default is 'head'.
    precision : str
        Precision of data in the head file: 'auto', 'single' or 'double'.
        Default is 'auto'.
    verbose : bool
        If True, write information to the screen. Default is False.
    nanval : scalar
        no data value, default value is -1e20
    kstpkper : tuple of ints or list of tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """
    warnings.warn("export_heads is deprecated, use Vtk.add_heads()")

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")

    from ..utils import HeadFile

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    if not isinstance(hdsfile, HeadFile):
        hds = HeadFile(
            hdsfile, text=text, precision=precision, verbose=verbose
        )
    else:
        hds = hdsfile

    vtk = Vtk(model, binary=binary, smooth=smooth, point_scalars=point_scalars)

    vtk.add_heads(hds, kstpkper)
    name = f"{model.name}_{text}"
    vtk.write(os.path.join(otfolder, name))


def export_cbc(
    model,
    cbcfile,
    otfolder,
    precision="single",
    verbose=False,
    nanval=-1e20,
    kstpkper=None,
    text=None,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=True,
):
    """
    Exports cell by cell file to vtk

    .. deprecated:: 3.3.5
        Use :meth:`Vtk.add_cell_budget`

    Parameters
    ----------
    model : flopy model instance
        the flopy model instance
    cbcfile : str
        the cell by cell file
    otfolder : str
        output folder to write the data
    precision : str
        Precision of data in the cell by cell file: 'single' or 'double'.
        Default is 'single'.
    verbose : bool
        If True, write information to the screen. Default is False.
    nanval : scalar
        no data value
    kstpkper : tuple of ints or list of tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    text : str or list of str
        The text identifier for the record.  Examples include
        'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """
    warnings.warn("export_cbc is deprecated, use Vtk.add_cell_budget()")

    if nanval != -1e20:
        warnings.warn("nanval is deprecated, setting to np.nan")
    if true2d:
        warnings.warn("true2d is no longer supported, setting to False")
    if vtk_grid_type != "auto":
        warnings.warn("vtk_grid_type is deprecated, setting to binary")

    from ..utils import CellBudgetFile

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    if not isinstance(cbcfile, CellBudgetFile):
        cbc = CellBudgetFile(cbcfile, precision=precision, verbose=verbose)
    else:
        cbc = cbcfile

    vtk = Vtk(model, binary=binary, smooth=smooth, point_scalars=point_scalars)

    vtk.add_cell_budget(cbc, text, kstpkper)
    fname = f"{model.name}_CBC"
    vtk.write(os.path.join(otfolder, fname))
