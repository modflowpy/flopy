import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pandas import DataFrame

from .geometry import transform
from .geospatial_utils import GeoSpatialUtil
from .utl_import import import_optional_dependency

shapely = import_optional_dependency("shapely", errors="silent")


def parse_shapely_ix_result(collection, ix_result, shptyps=None):
    """Recursive function for parsing shapely intersection results. Returns a
    list of shapely shapes matching shptyps.

    Parameters
    ----------
    collection : list
        state variable for storing result, generally
        an empty list
    ix_result : shapely.geometry type
        any shapely intersection result
    shptyps : str, list of str, or None, optional
        if None (default), return all types of shapes.
        if str, return shapes of that type, if list of str,
        return all types in list

    Returns
    -------
    list
        list containing shapely geometries of type shptyps
    """
    # convert shptyps to list if needed
    if isinstance(shptyps, str):
        shptyps = [shptyps]
    elif shptyps is None:
        shptyps = [None]

    # if empty
    if ix_result.is_empty:
        return collection
    # base case: geom_type is partial or exact match to shptyp
    elif ix_result.geom_type in shptyps:
        collection.append(ix_result)
        return collection
    # recursion for collections
    elif hasattr(ix_result, "geoms"):
        for ishp in ix_result.geoms:
            parse_shapely_ix_result(collection, ishp, shptyps=shptyps)
    # if collecting all types
    elif shptyps[0] is None:
        return collection.append(ix_result)
    return collection


class GridIntersect:
    """Class for intersecting shapely geometries with MODFLOW grids.

    Notes
    -----
     - The STR-tree query is based on the bounding box of the shape or
       collection, if the bounding box of the shape covers nearly the entire
       grid, the query won't be able to limit the search space much, resulting
       in slower performance. Therefore, it can sometimes be faster to
       intersect each individual shape in a collection than it is to intersect
       with the whole collection at once.
    """

    def __init__(self, mfgrid, rtree=True, local=False):
        """Intersect shapes (Point, Linestring, Polygon) with a modflow grid.

        Parameters
        ----------
        mfgrid : flopy modflowgrid
            MODFLOW grid as implemented in flopy
        rtree : bool, optional
            build an STR-Tree if True (default). If False no STR-tree
            is built, but intersects will filter and loop through candidate model
            gridcells (which is generally slower and not recommended).
        local : bool, optional
            use local model coordinates from model grid to build grid geometries,
            default is False and uses real-world coordinates (with offset and rotation).
        """
        import_optional_dependency(
            "shapely", error_message="GridIntersect requires shapely"
        )
        self.mfgrid = mfgrid
        self.local = local
        self.rtree = rtree

        # build arrays of geoms and cellids
        if self.mfgrid.grid_type == "structured":
            self.geoms, self.cellids = self._rect_grid_to_geoms_cellids()
        elif self.mfgrid.grid_type == "vertex":
            self.geoms, self.cellids = self._vtx_grid_to_geoms_cellids()
        elif self.mfgrid.grid_type == "unstructured":
            raise NotImplementedError()
            self.geoms, self.cellids = self._usg_grid_to_geoms_cellids()
        else:
            raise NotImplementedError(
                f"Grid type {self.mfgrid.grid_type} not supported"
            )

        # build STR-tree if specified
        if self.rtree:
            strtree = import_optional_dependency(
                "shapely.strtree",
                error_message="STRTree requires shapely",
            )
            self.strtree = strtree.STRtree(self.geoms)

    def _parse_input_shape(self, shp, shapetype=None):
        """Internal method to parse input shape.

        Allows numpy arrays containing shapely geometries, otherwise delegates to
        GeoSpatialUtil.

        Parameters
        ----------
        shp : shapely.geometry, geojson object, shapefile.Shape, np.ndarray,
              or flopy geometry object
            shape to intersect with the grid
        shapetype : str, optional
            type of shape (i.e. "point", "linestring", "polygon" or their
            multi-variants), used by GeoSpatialUtil if shp is passed as a list
            of vertices, default is None

        Returns
        -------
        shp : shapely.geometry or np.ndarray
            shapely geometry or array of shapely geometries
        """
        if isinstance(shp, np.ndarray) and isinstance(shp[0], shapely.Geometry):
            shapetypes = shapely.get_type_id(shp)
            assert len(np.unique(shapetypes)) == 1, (
                "If passing an array of shapely geometries, all geometries must be "
                "of the same type."
            )
            shapetype = shapely.GeometryType(shapetypes[0])
        else:
            gu = GeoSpatialUtil(shp, shapetype=shapetype)
            shp = gu.shapely
            shapetype = gu.shapetype
        return shp, shapetype

    def intersect(
        self,
        shp,
        shapetype=None,
        sort_by_cellid=True,
        return_all_intersections=False,
        contains_centroid=False,
        min_area_fraction=None,
        geo_dataframe=False,
    ):
        """Method to intersect a shape with a model grid.

        Parameters
        ----------
        shp : shapely.geometry, geojson object, shapefile.Shape,
              or flopy geometry object
        shapetype : str, optional
            type of shape (i.e. "point", "linestring", "polygon" or their
            multi-variants), used by GeoSpatialUtil if shp is passed as a list
            of vertices, default is None
        sort_by_cellid : bool
            sort results by cellid, ensures cell with lowest cellid is returned
            for boundary cases when using vertex methods, default is True
        return_all_intersections :  bool, optional
            if True, return multiple intersection results for points or
            linestrings on grid cell boundaries (e.g. returns 2 intersection
            results if a point lies on the boundary between two grid cells).
            The default is False. Only used if shape type is "point" or
            "linestring".
        contains_centroid :  bool, optional
            if True, only store intersection result if cell centroid is
            contained within intersection shape, only used if shape type is
            "polygon"
        min_area_fraction : float, optional
            float defining minimum intersection area threshold, if intersection
            area is smaller than min_frac_area * cell_area, do not store
            intersection result, only used if shape type is "polygon"
        geo_dataframe : bool, optional
            if True, return a geopandas GeoDataFrame, default is False

        Returns
        -------
        numpy.recarray or gepandas.GeoDataFrame
            a record array containing information about the intersection or
            a geopandas.GeoDataFrame if geo_dataframe=True
        """
        shp, shapetype = self._parse_input_shape(shp, shapetype=shapetype)

        # if array, only accept length 1
        if isinstance(shp, np.ndarray) and len(shp) > 1:
            raise ValueError(
                "intersect() only accepts arrays containing one "
                f"{shapetype.name.lower()} at a time."
            )

        if shapetype in {
            "Point",
            "MultiPoint",
            shapely.GeometryType.POINT,
            shapely.GeometryType.MULTIPOINT,
        }:
            rec = self._intersect_point(
                shp,
                sort_by_cellid=sort_by_cellid,
                return_all_intersections=return_all_intersections,
            )
        elif shapetype in {
            "LineString",
            "MultiLineString",
            shapely.GeometryType.LINESTRING,
            shapely.GeometryType.MULTILINESTRING,
            shapely.GeometryType.LINEARRING,
        }:
            rec = self._intersect_linestring(
                shp,
                sort_by_cellid=sort_by_cellid,
                return_all_intersections=return_all_intersections,
            )
        elif shapetype in {
            "Polygon",
            "MultiPolygon",
            shapely.GeometryType.POLYGON,
            shapely.GeometryType.MULTIPOLYGON,
        }:
            rec = self._intersect_polygon(
                shp,
                sort_by_cellid=sort_by_cellid,
                contains_centroid=contains_centroid,
                min_area_fraction=min_area_fraction,
            )
        else:
            raise TypeError(f"Shapetype {shapetype} is not supported")

        if geo_dataframe:
            gpd = import_optional_dependency("geopandas")
            gdf = (
                gpd.GeoDataFrame(rec)
                .rename(columns={"ixshapes": "geometry"})
                .set_geometry("geometry")
            )
            if self.mfgrid.crs is not None:
                gdf = gdf.set_crs(self.mfgrid.crs)
            return gdf

        return rec

    def _rect_grid_to_geoms_cellids(self):
        """internal method, return shapely polygons and cellids for structured
        grid cells.

        Returns
        -------
        geoms : array_like
            array of shapely Polygons
        cellids : array_like
            array of cellids
        """
        shapely = import_optional_dependency("shapely")

        nrow = self.mfgrid.nrow
        ncol = self.mfgrid.ncol
        ncells = nrow * ncol
        cellids = np.arange(ncells)
        if self.local:
            xvertices, yvertices = np.meshgrid(*self.mfgrid.xyedges)
        else:
            xvertices = self.mfgrid.xvertices
            yvertices = self.mfgrid.yvertices

        # arrays of coordinates for rectangle cells
        I, J = np.ogrid[0:nrow, 0:ncol]
        xverts = np.stack(
            [
                xvertices[I, J],
                xvertices[I, J + 1],
                xvertices[I + 1, J + 1],
                xvertices[I + 1, J],
            ]
        ).transpose((1, 2, 0))
        yverts = np.stack(
            [
                yvertices[I, J],
                yvertices[I, J + 1],
                yvertices[I + 1, J + 1],
                yvertices[I + 1, J],
            ]
        ).transpose((1, 2, 0))

        # use array-based methods for speed
        geoms = shapely.polygons(
            shapely.linearrings(
                xverts.flatten(),
                y=yverts.flatten(),
                indices=np.repeat(cellids, 4),
            )
        )
        return geoms, cellids

    def _usg_grid_to_geoms_cellids(self):
        """internal method, return shapely polygons and cellids for
        unstructured grids.

        Returns
        -------
        geoms : array_like
            array of shapely Polygons
        cellids : array_like
            array of cellids
        """
        raise NotImplementedError()

    def _vtx_grid_to_geoms_cellids(self):
        """internal method, return shapely polygons and cellids for vertex
        grids.

        Returns
        -------
        geoms : array_like
            array of shapely Polygons
        cellids : array_like
            array of cellids
        """
        shapely = import_optional_dependency("shapely")
        if self.local:
            geoms = [
                shapely.polygons(
                    list(
                        zip(
                            *self.mfgrid.get_local_coords(
                                *np.array(self.mfgrid.get_cell_vertices(node)).T
                            )
                        )
                    )
                )
                for node in range(self.mfgrid.ncpl)
            ]
        else:
            geoms = [
                shapely.polygons(self.mfgrid.get_cell_vertices(node))
                for node in range(self.mfgrid.ncpl)
            ]
        return np.array(geoms), np.arange(self.mfgrid.ncpl)

    def query_grid(self, shp, predicate=None):
        """Perform spatial query on grid with shapely geometry. If no spatial
        query is possible returns all grid cells.

        Parameters
        ----------
        shp : shapely.geometry
            shapely geometry
        predicate : str, optional
            spatial predicate to use for query, default is None. See
            documentation of self.strtree.query for options.

        Returns
        -------
        array_like
            array containing cellids of grid cells in query result
        """
        if self.rtree:
            result = self.strtree.query(shp, predicate=predicate)
        else:
            # no spatial query
            result = self.cellids
        return result

    def filter_query_result(self, shp, cellids):
        """Filter array of geometries to obtain grid cells that intersect with
        shape.

        Used to (further) reduce query result to cells that intersect with
        shape.

        Parameters
        ----------
        shp : shapely.geometry
            shapely geometry that is prepared and used to filter
            query result
        cellids : iterable
            iterable of cellids, query result

        Returns
        -------
        array_like
            filter or generator containing polygons that intersect with shape
        """
        # flipped arguments to be consistent with all other methods in class
        msg = (
            "the cellids and shp arguments were flipped, please"
            " pass them as filter_query_result(shp, cellids)"
        )
        if isinstance(cellids, np.ndarray):
            if isinstance(cellids[0], shapely.Geometry):
                warnings.warn(msg)
                cellids, shp = shp, cellids
        elif isinstance(cellids, shapely.Geometry):
            warnings.warn(msg)
            cellids, shp = shp, cellids

        # get only gridcells that intersect
        if not shapely.is_prepared(shp).all():
            shapely.prepare(shp)
        qcellids = cellids[shapely.intersects(self.geoms[cellids], shp)]
        return qcellids

    def _intersect_point_shapely(self, *args, **kwargs):
        """Deprecated method, use _intersect_point instead."""
        import warnings

        warnings.warn(
            "_intersect_point_shapely is deprecated, use _intersect_point instead.",
            DeprecationWarning,
        )
        return self._intersect_point(*args, **kwargs)

    def _intersect_point(
        self,
        shp,
        sort_by_cellid=True,
        return_all_intersections=False,
    ):
        if self.rtree:
            qcellids = self.strtree.query(shp, predicate="intersects")
        else:
            qcellids = self.filter_query_result(self.cellids, shp)

        if sort_by_cellid:
            qcellids = np.sort(qcellids)

        ixresult = shapely.intersection(shp, self.geoms[qcellids])
        # discard empty intersection results
        mask_empty = shapely.is_empty(ixresult)
        # keep only Point and MultiPoint
        mask_type = np.isin(shapely.get_type_id(ixresult), [0, 4])
        ixresult = ixresult[~mask_empty & mask_type]
        qcellids = qcellids[~mask_empty & mask_type]

        if not return_all_intersections:
            keep_cid = []
            keep_pts = []
            parsed = []
            for ishp, cid in zip(ixresult, qcellids):
                points = []
                for pnt in shapely.get_parts(ishp):
                    next_pnt = next(iter(pnt.coords))
                    if next_pnt not in parsed:
                        points.append(pnt)
                    parsed.append(next_pnt)

                if len(points) > 1:
                    keep_pts.append(shapely.MultiPoint(points))
                    keep_cid.append(cid)
                elif len(points) == 1:
                    keep_pts.append(points[0])
                    keep_cid.append(cid)
        else:
            keep_pts = ixresult
            keep_cid = qcellids

        names = ["cellids", "ixshapes"]
        formats = ["O", "O"]
        rec = np.recarray(len(keep_pts), names=names, formats=formats)

        # if structured calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids = list(
                zip(*self.mfgrid.get_lrc([self.cellids[keep_cid]])[0][1:])
            )
        else:
            rec.cellids = self.cellids[keep_cid]
        rec.ixshapes = keep_pts

        return rec

    def _intersect_linestring_shapely(self, *args, **kwargs):
        """Deprecated method, use _intersect_linestring instead."""

        warnings.warn(
            "_intersect_linestring_shapely is deprecated, "
            "use _intersect_linestring instead.",
            DeprecationWarning,
        )
        return self._intersect_linestring(*args, **kwargs)

    def _intersect_linestring(
        self,
        shp,
        sort_by_cellid=True,
        return_all_intersections=False,
    ):
        if self.rtree:
            qcellids = self.strtree.query(shp, predicate="intersects")
        else:
            qcellids = self.filter_query_result(shp, self.cellids)

        if sort_by_cellid:
            qcellids = np.sort(qcellids)

        ixresult = shapely.intersection(shp, self.geoms[qcellids])
        # discard empty intersection results
        mask_empty = shapely.is_empty(ixresult)
        # keep only Linestring and MultiLineString
        geomtype_ids = shapely.get_type_id(ixresult)
        all_ids = [
            shapely.GeometryType.LINESTRING,
            shapely.GeometryType.MULTILINESTRING,
            shapely.GeometryType.GEOMETRYCOLLECTION,
        ]
        line_ids = [
            shapely.GeometryType.LINESTRING,
            shapely.GeometryType.MULTILINESTRING,
        ]
        mask_type = np.isin(geomtype_ids, all_ids)
        ixresult = ixresult[~mask_empty & mask_type]
        qcellids = qcellids[~mask_empty & mask_type]

        # parse geometry collections (i.e. when part of linestring touches a cell edge,
        # resulting in a point intersection result)
        if shapely.GeometryType.GEOMETRYCOLLECTION in geomtype_ids:

            def parse_linestrings_in_geom_collection(gc):
                parts = shapely.get_parts(gc)
                parts = parts[np.isin(shapely.get_type_id(parts), line_ids)]
                if len(parts) > 1:
                    p = shapely.multilinestrings(parts)
                elif len(parts) == 0:
                    p = shapely.LineString()
                else:
                    p = parts[0]
                return p

            mask_gc = (
                geomtype_ids[~mask_empty & mask_type]
                == shapely.GeometryType.GEOMETRYCOLLECTION
            )
            # NOTE: not working for multiple geometry collections, result is reduced
            # to a single multilinestring, which causes doubles in the result
            # ixresult[mask_gc] = np.apply_along_axis(
            #     parse_linestrings_in_geom_collection,
            #     axis=0,
            #     arr=ixresult[mask_gc],
            # )
            ixresult[mask_gc] = [
                parse_linestrings_in_geom_collection(gc) for gc in ixresult[mask_gc]
            ]

        if not return_all_intersections:
            # intersection with grid cell boundaries
            ixbounds = shapely.intersection(
                shp, shapely.get_exterior_ring(self.geoms[qcellids])
            )
            mask_bnds_empty = shapely.is_empty(ixbounds)
            mask_bnds_type = np.isin(shapely.get_type_id(ixbounds), all_ids)
            # get ids of boundary intersections
            idxs = np.nonzero(~mask_bnds_empty & mask_bnds_type)[0]

            # loop through results, starting with highest cellid
            jdxs = idxs[::-1]
            for jx, i in enumerate(jdxs):
                # calculate intersection with results w potential boundary
                # intersections
                isect = ixresult[i].intersection(ixresult[idxs])

                # masks to obtain overlapping intersection result
                mask_self = idxs == i  # select not self
                mask_bnds_empty = shapely.is_empty(isect)  # select boundary ix result
                mask_overlap = np.isin(shapely.get_type_id(isect), all_ids)

                # calculate difference between self and overlapping result
                diff = shapely.difference(
                    ixresult[i],
                    isect[mask_overlap & ~mask_self & ~mask_bnds_empty],
                )
                # update intersection result if necessary
                if len(diff) > 0:
                    ixresult[jdxs[jx]] = diff[0]

            # mask out empty results
            mask_keep = ~shapely.is_empty(ixresult)
            ixresult = ixresult[mask_keep]
            qcellids = qcellids[mask_keep]

        names = ["cellids", "ixshapes", "lengths"]
        formats = ["O", "O", "f8"]

        rec = np.recarray(len(ixresult), names=names, formats=formats)
        # if structured grid calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids = list(
                zip(*self.mfgrid.get_lrc([self.cellids[qcellids]])[0][1:])
            )
        else:
            rec.cellids = self.cellids[qcellids]
        rec.ixshapes = ixresult
        rec.lengths = shapely.length(ixresult)

        return rec

    def _intersect_polygon_shapely(self, *args, **kwargs):
        """Deprecated method, use _intersect_polygon instead."""
        import warnings

        warnings.warn(
            "_intersect_polygon_shapely is deprecated, use _intersect_polygon instead.",
            DeprecationWarning,
        )
        return self._intersect_polygon(*args, **kwargs)

    def _intersect_polygon(
        self,
        shp,
        sort_by_cellid=True,
        contains_centroid=False,
        min_area_fraction=None,
    ):
        if self.rtree:
            qcellids = self.strtree.query(shp, predicate="intersects")
        else:
            qcellids = self.filter_query_result(shp, self.cellids)

        if sort_by_cellid:
            qcellids = np.sort(qcellids)

        ixresult = shapely.intersection(shp, self.geoms[qcellids])
        # discard empty intersection results
        mask_empty = shapely.is_empty(ixresult)
        # keep only Polygons and MultiPolygons
        geomtype_ids = shapely.get_type_id(ixresult)
        mask_type = np.isin(geomtype_ids, [3, 6, 7])
        ixresult = ixresult[~mask_empty & mask_type]
        qcellids = qcellids[~mask_empty & mask_type]

        # parse geometry collections (i.e. when part of polygon lies on cell edge,
        # resulting in a linestring intersection result)
        if 7 in geomtype_ids:

            def parse_polygons_in_geom_collection(gc):
                parts = shapely.get_parts(gc)
                parts = parts[np.isin(shapely.get_type_id(parts), [3, 6])]
                if len(parts) > 1:
                    p = shapely.multipolygons(parts)
                elif len(parts) == 0:
                    p = shapely.Polygon()
                else:
                    p = parts[0]
                return p

            mask_gc = geomtype_ids[~mask_empty & mask_type] == 7
            ixresult[mask_gc] = np.apply_along_axis(
                parse_polygons_in_geom_collection, axis=0, arr=ixresult[mask_gc]
            )

        # check centroids
        if contains_centroid:
            centroids = shapely.centroid(self.geoms[qcellids])
            mask_centroid = shapely.contains(ixresult, centroids) | shapely.touches(
                ixresult, centroids
            )
            ixresult = ixresult[mask_centroid]
            qcellids = qcellids[mask_centroid]

        # check intersection area
        if min_area_fraction:
            ix_areas = shapely.area(ixresult)
            cell_areas = shapely.area(self.geoms[qcellids])
            mask_area_frac = (ix_areas / cell_areas) >= min_area_fraction
            ixresult = ixresult[mask_area_frac]
            qcellids = qcellids[mask_area_frac]

        # fill rec array
        names = ["cellids", "ixshapes", "areas"]
        formats = ["O", "O", "f8"]
        rec = np.recarray(len(ixresult), names=names, formats=formats)
        # if structured calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids = list(
                zip(*self.mfgrid.get_lrc([self.cellids[qcellids]])[0][1:])
            )
        else:
            rec.cellids = self.cellids[qcellids]
        rec.ixshapes = ixresult
        rec.areas = shapely.area(ixresult)

        return rec

    def intersects(
        self,
        shp,
        shapetype=None,
        dataframe=False,
        return_cellids=True,
    ):
        """Return cellids for grid cells that intersect with shape.

        Parameters
        ----------
        shp : shapely.geometry, geojson geometry, shapefile.shape,
              or flopy geometry object
            shape to intersect with the grid
        shapetype : str, optional
            type of shape (i.e. "point", "linestring", "polygon" or
            their multi-variants), used by GeoSpatialUtil if shp is
            passed as a list of vertices, default is None
        dataframe : bool, optional
            if True, return a pandas.DataFrame, default is False
        return_all_intersections : bool, optional
            if True (default), return multiple intersection results for points on grid
            cell boundaries (e.g. returns 2 intersection results if a point lies on the
            boundary between two grid cells).
        return_cellids : bool, optional
            if True (default), return cellids of intersected grid cells.
            If False, only return grid node numbers, i.e. index of entry in
            ``GridIntersect.geoms``.

        Returns
        -------
        numpy.recarray or pandas.DataFrame
            a record array or pandas.DataFrame containing cell IDs of the gridcells
            the shape intersects with.
        """
        shp, shapetype = self._parse_input_shape(shp, shapetype=shapetype)

        # query grid or strtree
        qcellids = self.query_grid(shp, predicate="intersects")
        if not self.rtree:
            if isinstance(shp, np.ndarray) and len(shp) > 1:
                raise ValueError(
                    "intersects() only accepts arrays containing one "
                    f"{shapetype.name.lower()} at a time when rtree=False."
                )
            qfiltered = self.filter_query_result(shp, qcellids)
        else:
            qfiltered = qcellids

        # sort cellids
        if qfiltered.ndim == 1:
            qfiltered = np.sort(qfiltered)
        else:
            qfiltered = qfiltered[:, np.lexsort((qfiltered[1], qfiltered[0]))]

        # determine size of output array
        nr = len(qfiltered) if qfiltered.ndim == 1 else qfiltered.shape[1]

        # build rec-array
        rec = np.recarray(
            nr,
            names=["shp_ids", "cellids"],
            formats=[
                int,
                "O"
                if (return_cellids and self.mfgrid.grid_type == "structured")
                else float,
            ],
        )
        # shp was passed as single geometry
        if qfiltered.ndim == 1:
            rec.shp_ids[:] = 0
            rec.cellids = qfiltered
        # shape passed as array of geometries
        else:
            rec.shp_ids = qfiltered[0]
            rec.cellids = qfiltered[1]

        if self.mfgrid.grid_type == "structured" and return_cellids:
            rec.cellids = self._nodenumber_to_rowcol(rec.cellids)

        if dataframe:
            return DataFrame(rec).set_index("shp_ids")
        return rec

    def _nodenumber_to_rowcol(self, nodes):
        """Convert node number to (row, col) tuple.

        Parameters
        ----------
        nodes : array_like
            array of cellids to convert

        Returns
        -------
        array_like
            array of (row, col) tuples
        """
        # cast to float and allow nans
        idx = np.nonzero(~np.isnan(nodes.astype(float)))
        rc = np.full_like(nodes, np.nan, dtype=object)
        rc[idx] = list(zip(*self.mfgrid.get_lrc([nodes[idx].astype(int)])[0][1:]))
        return rc
        return rec

    @staticmethod
    def plot_polygon(result, ax=None, **kwargs):
        """method to plot the polygon intersection results from the resulting
        numpy.recarray.

        Note: only works when recarray has 'ixshapes' column!

        Parameters
        ----------
        result : numpy.recarray or geopandas.GeoDataFrame
            record array or GeoDataFrame containing intersection results
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the plot function

        Returns
        -------
        matplotlib.pyplot.axes
            returns the axes handle
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")
            autoscale = True
        else:
            autoscale = False

        patches = []
        if "facecolor" in kwargs:
            use_facecolor = True
            fc = kwargs.pop("facecolor")
        else:
            use_facecolor = None

        def add_poly_patch(poly):
            if not use_facecolor:
                fc = f"C{i % 10}"
            ppi = _polygon_patch(poly, facecolor=fc, **kwargs)
            patches.append(ppi)

        # allow for result to be geodataframe
        geoms = (
            result.ixshapes if isinstance(result, np.rec.recarray) else result.geometry
        )
        for i, ishp in enumerate(geoms):
            if hasattr(ishp, "geoms"):
                for geom in ishp.geoms:
                    add_poly_patch(geom)
            else:
                add_poly_patch(ishp)

        pc = PatchCollection(patches, match_original=True)
        ax.add_collection(pc)

        if autoscale:
            ax.autoscale_view()

        return ax

    @staticmethod
    def plot_linestring(result, ax=None, cmap=None, **kwargs):
        """method to plot the linestring intersection results from the
        resulting numpy.recarray.

        Note: only works when recarray has 'ixshapes' column!

        Parameters
        ----------
        result : numpy.recarray or geopandas.GeoDataFrame
            record array or GeoDataFrame containing intersection results
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        cmap : str
            matplotlib colormap
        **kwargs:
            passed to the plot function

        Returns
        -------
        matplotlib.pyplot.axes
            returns the axes handle
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")

        specified_color = True
        if "c" in kwargs:
            c = kwargs.pop("c")
        elif "color" in kwargs:
            c = kwargs.pop("color")
        else:
            specified_color = False

        if cmap is not None:
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, result.shape[0]))

        # allow for result to be geodataframe
        geoms = (
            result.ixshapes if isinstance(result, np.rec.recarray) else result.geometry
        )
        for i, ishp in enumerate(geoms):
            if not specified_color:
                if cmap is None:
                    c = f"C{i % 10}"
                else:
                    c = colors[i]
            if ishp.geom_type == "MultiLineString":
                for part in ishp.geoms:
                    ax.plot(part.xy[0], part.xy[1], ls="-", c=c, **kwargs)
            else:
                ax.plot(ishp.xy[0], ishp.xy[1], ls="-", c=c, **kwargs)

        return ax

    @staticmethod
    def plot_point(result, ax=None, **kwargs):
        """method to plot the point intersection results from the resulting
        numpy.recarray.

        Note: only works when recarray has 'ixshapes' column!

        Parameters
        ----------
        result : numpy.recarray or geopandas.GeoDataFrame
            record array or GeoDataFrame containing intersection results
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the scatter function

        Returns
        -------
        matplotlib.pyplot.axes
            returns the axes handle
        """
        import matplotlib.pyplot as plt

        shapely_geo = import_optional_dependency("shapely.geometry")

        if ax is None:
            _, ax = plt.subplots()

        x, y = [], []
        # allow for result to be geodataframe
        geoms = (
            result.ixshapes if isinstance(result, np.rec.recarray) else result.geometry
        )
        geo_coll = shapely_geo.GeometryCollection(list(geoms))
        collection = parse_shapely_ix_result([], geo_coll, ["Point"])
        for c in collection:
            x.append(c.x)
            y.append(c.y)
        ax.scatter(x, y, **kwargs)

        return ax

    def plot_intersection_result(self, result, plot_grid=True, ax=None, **kwargs):
        """Plot intersection result.

        Parameters
        ----------
        result : numpy.rec.recarray or geopandas.GeoDataFrame
            result of intersect()
        plot_grid : bool, optional
            plot model grid, by default True
        ax : matplotlib.Axes, optional
            axes to plot on, by default None which creates a new axis

        Returns
        -------
        ax : matplotlib.Axes
            returns axes handle
        """
        shapely = import_optional_dependency("shapely")

        if plot_grid:
            self.mfgrid.plot(ax=ax)

        geoms = (
            result["ixshapes"]
            if isinstance(result, np.rec.recarray)
            else result["geometry"]
        )
        if np.isin(
            shapely.get_type_id(geoms),
            [shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT],
        ).all():
            ax = GridIntersect.plot_point(result, ax=ax, **kwargs)
        elif np.isin(
            shapely.get_type_id(geoms),
            [
                shapely.GeometryType.LINESTRING,
                shapely.GeometryType.MULTILINESTRING,
            ],
        ).all():
            ax = GridIntersect.plot_linestring(result, ax=ax, **kwargs)
        elif np.isin(
            shapely.get_type_id(geoms),
            [shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON],
        ).all():
            ax = GridIntersect.plot_polygon(result, ax=ax, **kwargs)

        return ax


def _polygon_patch(polygon, **kwargs):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    patch = PathPatch(
        Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
        ),
        **kwargs,
    )
    return patch
