import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy.lib import recfunctions as nprecfns
from pandas import NA, DataFrame

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
     - Supports structured and vertex grids. No support for unstructured grids.
       If grid is layered-unstructured, creating a single layer vertex-grid can be
       used as a workaround.
     - For linestrings and polygons only 2D intersection operations are supported.
     - Point intersections can optionally return layer position based on the
       z-coordinate.
     - The STR-tree can be disabled, but this is generally not recommended as some
       functions will not work without it.
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

        Allows numpy arrays containing shapely geometries; otherwise delegates to
        ``GeoSpatialUtil`` for parsing.

        Parameters
        ----------
        shp : shapely geometry, geojson object, shapefile.Shape, np.ndarray,
              or a FloPy geometry object
            Shape to intersect with the grid. If an ``np.ndarray`` is provided, all
            elements must be shapely geometries of the same type.
        shapetype : str, optional
            Type of shape ("point", "linestring", "polygon" or their
            multi-variants). Used by ``GeoSpatialUtil`` if shp is passed as a list
            of vertices. Default is None.

        Returns
        -------
        tuple
            (geom, gtype) where geom is a shapely geometry or an array of
            shapely geometries, and gtype is the corresponding shapely
            GeometryType (or a matching string value).
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
        handle_z=False,
        geo_dataframe=None,
    ):
        """Intersect a shape with a model grid.

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
        handle_z : bool, optional
            Method for handling z-dimension in intersection results for point
            intersections. Default is False which ignores z-dimension. If True
            returns the layer index for each point. Points below/above the grid
            are returned as masked values or pd.NA if returned as dataframe.
        geo_dataframe : bool, optional
            If True, return a geopandas ``GeoDataFrame``, otherwise return a
            numpy recarray. Default will be set to True in the future;
            currently, None triggers a deprecation warning and returns a
            recarray (legacy behavior).

        Returns
        -------
        numpy.recarray or geopandas.GeoDataFrame
            Intersection results. Result contains the following columns:
            - cellid: cellid of intersected grid cell
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            - ixshapes or geometry: shapely geometry of the intersection result
            And optionally:
            - layer: layer index of for points with z-dimension when handle_z is True
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
            - lengths: length of intersection results for linestrings
            - areas: areas of intersection results for polygons

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

            # handle elevation data for points
            # if handle_z is True
            # if shp has z information
            # if there are intersection results
            if handle_z and shapely.has_z(shp).any() and len(rec.cellid) > 0:
                laypos = self.get_layer_from_z(shp, rec.cellid)
                # copy data to new array to include layer position
                rec = nprecfns.append_fields(
                    rec,
                    names="layer",
                    data=laypos,
                    dtypes=int,
                    usemask=False,
                    asrecarray=True,
                )
                rec = np.ma.masked_where(rec.layer < 0, rec)

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

        if geo_dataframe is None:
            warnings.warn(
                "In the future this function will return a GeoDataFrame by default. "
                "Set geo_dataframe=True to adopt future behavior and silence this "
                "warning. Set geo_dataframe=False to silence this warning and maintain "
                "old behavior",
                DeprecationWarning,
            )
        if geo_dataframe:
            gpd = import_optional_dependency("geopandas")
            if gpd is None:
                raise ModuleNotFoundError(
                    "GeoDataFrame output requires geopandas to be installed."
                )
            gdf = (
                gpd.GeoDataFrame(rec)
                .rename(columns={"ixshapes": "geometry"})
                .set_geometry("geometry")
                .replace(999999, NA)
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
        """Perform spatial query on the grid using a shapely geometry.

        If no spatial query is possible (``rtree=False``), returns all grid cells.

        Parameters
        ----------
        shp : shapely.geometry
            shapely geometry
        predicate : str, optional
            spatial predicate to use for query, default is None. See
            documentation of self.strtree.query for options.

        Returns
        -------
        numpy.ndarray
            For a single geometry, a 1-D array of cellids. For an array of geometries,
            a 2xN array where the first row is the input geometry index (``shp_id``)
            and the second row contains the matching cellids.
        """
        if self.rtree:
            result = self.strtree.query(shp, predicate=predicate)
        else:
            # no spatial query
            result = self.cellids
        return result

    def filter_query_result(self, shp, cellids):
        """Filter a query result to cells that truly intersect a shape.

        Used to (further) reduce the query result to cells that intersect with
        shp. This is primarily used when ``rtree=False``.

        Parameters
        ----------
        shp : shapely.geometry
            shapely geometry that is prepared and used to filter
            query result
        cellids : iterable
            iterable of cellids, query result

        Returns
        -------
        numpy.ndarray
            Array of cellids that intersect shp.
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
        """Intersect a Point or MultiPoint with the grid.

        Parameters
        ----------
        shp : shapely Point or MultiPoint (or array with a single point)
            Geometry to intersect.
        sort_by_cellid : bool, optional
            If True (default), sort candidate cells by cellid
        return_all_intersections : bool, optional
            If True, return multiple intersection results for points on cell boundaries.
            Default is False.

        Returns
        -------
        numpy.recarray
            Intersection results. Result contains the following columns:
            - cellid: cellid of intersected grid cell
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            - ixshapes or geometry: shapely geometry of the intersection result
            And optionally:
            - layer: layer index of for points with z-dimension when handle_z is True
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
        """
        r = self.intersects(shp, dataframe=False)
        qcellids = r.cellid[r.cellid >= 0]

        if sort_by_cellid:
            qcellids = np.sort(qcellids)

        ixresult = shapely.intersection(shp, self.geoms[qcellids])
        # discard empty intersection results
        mask_empty = shapely.is_empty(ixresult)
        # keep only Point and MultiPoint
        mask_type = np.isin(
            shapely.get_type_id(ixresult),
            [shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT],
        )
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

        if self.mfgrid.grid_type == "structured":
            names = ["cellids", "cellid", "row", "col", "ixshapes"]
            formats = ["O", int, int, int, "O"]
        else:
            names = ["cellids", "cellid", "ixshapes"]
            formats = [int, int, "O"]
        rec = np.recarray(len(keep_pts), names=names, formats=formats)
        rec.cellid = self.cellids[keep_cid]
        rec.ixshapes = keep_pts

        # if structured calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids, rec.row, rec.col = self._cellid_to_rowcol(
                self.cellids[keep_cid]
            )
        else:
            rec.cellids = rec.cellid  # NOTE: legacy support for cellids column
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
        """Intersect a LineString or MultiLineString with the grid.

        Parameters
        ----------
        shp : shapely LineString or MultiLineString
            Geometry to intersect.
        sort_by_cellid : bool, optional
            If True (default), sort candidate cells by cellid before
            intersecting.
        return_all_intersections : bool, optional
            If True, keep overlapping boundary segments as separate results.
            Default is False.

        Returns
        -------
        numpy.recarray
            Intersection results. Result contains the following columns:
            - cellid: cellid of intersected grid cell
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            - ixshapes or geometry: shapely geometry of the intersection result
            And optionally:
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
            - lengths: length of intersection results for linestrings
        """
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

        if self.mfgrid.grid_type == "structured":
            names = ["cellids", "cellid", "row", "col", "ixshapes", "lengths"]
            formats = ["O", int, int, int, "O", float]
        else:
            names = ["cellids", "cellid", "ixshapes", "lengths"]
            formats = ["O", int, "O", float]

        rec = np.recarray(len(ixresult), names=names, formats=formats)
        rec.cellid = self.cellids[qcellids]
        rec.ixshapes = ixresult
        rec.lengths = shapely.length(ixresult)

        # if structured grid calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids, rec.row, rec.col = self._cellid_to_rowcol(
                self.cellids[qcellids]
            )
        else:
            rec.cellids = rec.cellid  # NOTE: legacy support for cellids column

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
        """Intersect a Polygon or MultiPolygon with the grid.

        Parameters
        ----------
        shp : shapely Polygon or MultiPolygon
            Geometry to intersect.
        sort_by_cellid : bool, optional
            If True (default), sort candidate cells by cellid before
            intersecting.
        contains_centroid : bool, optional
            If True, only keep results where the cell centroid is contained in
            (or touches) the intersection. Default is False.
        min_area_fraction : float, optional
            Minimum fractional area threshold relative to the cell area. Cells with
            an intersection area smaller than ``min_area_fraction * cell_area`` are
            discarded. Default is None (no threshold).

        Returns
        -------
        numpy.recarray
            Intersection results. Result contains the following columns:
            - cellid: cellid of intersected grid cell
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            - ixshapes or geometry: shapely geometry of the intersection result
            And optionally:
            - layer: layer index of for points with z-dimension when handle_z is True
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
            - areas: areas of intersection results for polygons
        """
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
        if self.mfgrid.grid_type == "structured":
            names = ["cellids", "cellid", "row", "col", "ixshapes", "areas"]
            formats = ["O", int, int, int, "O", float]
        else:
            names = ["cellids", "cellid", "ixshapes", "areas"]
            formats = ["O", int, "O", float]
        rec = np.recarray(len(ixresult), names=names, formats=formats)
        rec.cellid = self.cellids[qcellids]
        rec.ixshapes = ixresult
        rec.areas = shapely.area(ixresult)

        # if structured calculate (i, j) cell address
        if self.mfgrid.grid_type == "structured":
            rec.cellids, rec.row, rec.col = self._cellid_to_rowcol(
                self.cellids[qcellids]
            )
        else:
            rec.cellids = rec.cellid  # NOTE: legacy support for cellids column

        return rec

    def intersects(
        self,
        shp,
        shapetype=None,
        dataframe=None,
    ):
        """Return candidate grid cellids that intersect with shape(s).

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
            If True, return a ``pandas.DataFrame``; otherwise return a numpy
            recarray. Default will be set to True in the future; currently,
            None triggers a deprecation warning and returns a recarray (legacy
            behavior).

        Returns
        -------
        numpy.recarray or pandas.DataFrame
            Grid cell candidates for intersections. Result contains the following
            columns:
            - cellid: cellid of candidate grid cell
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            And optionally:
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
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
        # use float dtype to allow nans in row/col/cellid
        if self.mfgrid.grid_type == "structured":
            names = ["shp_id", "cellids", "cellid", "row", "col"]
            formats = [int, "O", int, int, int]
        else:
            names = ["shp_id", "cellids", "cellid"]
            formats = [int, int, int]
        rec = np.recarray(nr, names=names, formats=formats)

        # shp was passed as single geometry
        if qfiltered.ndim == 1:
            rec.shp_id[:] = 0
            rec.cellid = qfiltered
        # shape passed as array of geometries
        else:
            rec.shp_id = qfiltered[0]
            rec.cellid = qfiltered[1]

        if self.mfgrid.grid_type == "structured":
            rec.cellids, rec.row, rec.col = self._cellid_to_rowcol(rec.cellid)
        else:
            rec.cellids = rec.cellid  # NOTE: legacy support for cellids column
        if dataframe is None:
            warnings.warn(
                "In the future this function will return a dataframe by default. "
                "Set dataframe=True to adopt future behavior and silence this warning. "
                "Set dataframe=False to silence this warning and maintain old behavior",
                DeprecationWarning,
            )
        if dataframe:
            return DataFrame(rec).set_index("shp_id")
        return rec

    def _cellid_to_rowcol(self, cellids):
        """Convert cellid (node number) to row, col.

        Parameters
        ----------
        cellids : array_like
            array of cellids to convert

        Returns
        -------
        tuple of array_like
            array of (row, col) tuples, array of rows, array of columns
        """
        idx = np.nonzero(cellids >= 0)
        row = np.full_like(cellids, -1, dtype=int)  # -1 is invalid
        col = np.full_like(cellids, -1, dtype=int)  # -1 is invalid
        row[idx], col[idx] = self.mfgrid.get_lrc([cellids[idx]])[0][1:]
        # NOTE: build tuple for backward compatibility
        rctuple = np.full_like(cellids, -1, dtype=object)
        rctuple[idx] = list(zip(row[idx], col[idx]))
        return rctuple, row, col

    def points_to_cellids(
        self,
        pts,
        handle_z=False,
        dataframe=True,
    ):
        """Return cellids for points intersecting the grid.

        Parameters
        ----------
        pts : shapely geometry, geojson geometry, shapefile.Shape, np.ndarray,
              or FloPy geometry object
            Point(s) to intersect with the grid. array inputs must contain
            shapely point (or multipoint) geometries.
        handle_z : bool, optional
            Handle z-dimension for points. If True, returns a "layer" column with
            the computed layer index for each point (NA is returned where the
            points lie above/below the model grid). Default is False.
        dataframe : bool, optional
            If True, return a ``pandas.DataFrame``; otherwise return a numpy
            recarray. Default is True.

        Returns
        -------
        pandas.DataFrame or numpy.recarray
            Grid cell indices per point. Result contains the following
            columns:
            - cellid: cellid of grid cell containing point
            - cellids: legacy column containing (row, col) tuples for structured grids,
              or cellids for vertex grids (deprecated, use "cellid" column instead)
            And optionally:
            - row: row index of intersected grid cell, for structured grids
            - col: column index of intersected grid cell, for structured grids
            - layer: layer index of for points with z-dimension when handle_z is True

        Notes
        -----
        Requires ``rtree=True`` when initializing ``GridIntersect``.
        """
        if not self.rtree:
            raise ValueError(
                "points_to_cellids() requires rtree=True when"
                " initializing GridIntersect"
            )
        if not isinstance(pts, np.ndarray):
            if shapely.get_type_id(pts) == shapely.GeometryType.MULTIPOINT:
                pts = np.array(shapely.get_parts(pts))
            else:
                pts = np.array([pts])

        # query grid or strtree
        qfiltered = self.query_grid(pts, predicate="intersects")

        # sort cellids
        if qfiltered.ndim == 1:
            qfiltered = np.sort(qfiltered)
        else:
            qfiltered = qfiltered[:, np.lexsort((qfiltered[1], qfiltered[0]))]

        # determine size of output array
        if isinstance(pts, np.ndarray):
            # one result per point
            nr = len(pts)
        else:
            # single point
            nr = 1 if len(qfiltered) > 0 else 0  # 1 if intersects, else 0

        # build rec-array
        if self.mfgrid.grid_type == "structured":
            names = ["shp_id", "cellids", "cellid", "row", "col"]
            formats = [int, "O", int, int, int]  # float to allow nans in row/col
        else:
            names = ["shp_id", "cellids", "cellid"]
            formats = [int, int, int]  # float to allow nans
        rec = np.recarray(nr, names=names, formats=formats)

        rec.cellid = -1  # invalid value by default
        # return only 1 gr id cell intersection result for each point
        uniques, idx = np.unique(qfiltered[0], return_index=True)
        rec.shp_id = np.arange(nr)
        rec.cellid[uniques] = qfiltered[1, idx]

        if self.mfgrid.grid_type == "structured":
            rec.cellids, rec.row, rec.col = self._cellid_to_rowcol(rec.cellid)
        else:
            rec.cellids = rec.cellid  # NOTE: legacy support for cellids column

        if handle_z and shapely.has_z(pts).any() and len(rec.cellid) > 0:
            laypos = self.get_layer_from_z(pts, rec.cellid)
            # copy data to new array to include layer position
            rec = nprecfns.append_fields(
                rec,
                names="layer",
                data=laypos,
                dtypes=int,
                usemask=False,
                asrecarray=True,
            )

        if dataframe:
            # replace invalid indices with NA, replace invalid layer with NA
            return (
                DataFrame(rec).replace(-1, NA).replace(999_999, NA).set_index("shp_id")
            )
        return np.ma.masked_where(rec.cellid < 0, rec)

    @staticmethod
    def plot_polygon(polys, ax=None, **kwargs):
        """Plot polygons.

        Parameters
        ----------
        polys : collection of polygons
            list, array or GeoSeries containing polygons
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

        if isinstance(polys, (shapely.Polygon, shapely.MultiPolygon)):
            polys = [polys]

        for i, ishp in enumerate(polys):
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
    def plot_linestring(ls, ax=None, cmap=None, **kwargs):
        """Plot linestrings.

        Parameters
        ----------
        ls : collection of linestrings
            list, array or GeoSeries containing linestrings
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

        shapely_plot = import_optional_dependency("shapely.plotting")

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

        if isinstance(ls, (shapely.LineString, shapely.MultiLineString)):
            ls = [ls]

        if cmap is not None:
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(ls)))

        for i, ishp in enumerate(ls):
            if not specified_color:
                if cmap is None:
                    c = f"C{i % 10}"
                else:
                    c = colors[i]
            shapely_plot.plot_line(ishp, ax=ax, color=c, **kwargs)

        return ax

    @staticmethod
    def plot_point(pts, ax=None, **kwargs):
        """Plot points.

        Parameters
        ----------
        pts : collection of points
            array or GeoSeries containing point geometries
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the scatter function

        Returns
        -------
        matplotlib.pyplot.axes
            returns the axes handle
        """
        shapely = import_optional_dependency("shapely")
        shapely_plot = import_optional_dependency("shapely.plotting")

        if ax is None:
            _, ax = plt.subplots()
        # allow for result to be geodataframe
        if isinstance(pts, (shapely.Point, shapely.MultiPoint)):
            pts = [pts]

        maskpts = np.isin(
            shapely.get_type_id(pts),
            [shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT],
        )
        shapely_plot.plot_points(pts[maskpts], ax=ax, **kwargs)

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
            ax = GridIntersect.plot_point(geoms, ax=ax, **kwargs)
        elif np.isin(
            shapely.get_type_id(geoms),
            [
                shapely.GeometryType.LINESTRING,
                shapely.GeometryType.MULTILINESTRING,
                shapely.GeometryType.LINEARRING,
            ],
        ).all():
            ax = GridIntersect.plot_linestring(geoms, ax=ax, **kwargs)
        elif np.isin(
            shapely.get_type_id(geoms),
            [shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON],
        ).all():
            ax = GridIntersect.plot_polygon(geoms, ax=ax, **kwargs)

        return ax

    def get_layer_from_z(self, pts, cellids):
        """Compute layer indices from point z-values.

        Parameters
        ----------
        pts : shapely geometry
            Points geometry (single or array-like) with z-values.
        cellids : array_like
            Array of candidate cellids for the points.

        Returns
        -------
        numpy.ndarray
            layer index for each point. Points below/above the grid are returned
            with index -1.
        """
        z_arr = np.atleast_1d(shapely.get_z(pts))
        mask_valid = cellids >= 0
        if self.mfgrid.grid_type == "structured":
            row, col = self.mfgrid.get_lrc([cellids[mask_valid].astype(int)])[0][1:]
            surface_elevations = self.mfgrid.top_botm[:, row, col]
        elif self.mfgrid.grid_type == "vertex":
            surface_elevations = self.mfgrid.top_botm[:, cellids[mask_valid]]
        else:
            raise NotImplementedError(
                "get_layer_from_z() is only implemented for "
                "structured and vertex grids."
            )
        zb = surface_elevations < z_arr[mask_valid]
        mask_above = zb.all(axis=0)
        mask_below = (~zb).all(axis=0)
        laypos = (np.nanargmax(zb, axis=0) - 1).astype(int)
        laypos[mask_above] = -1
        laypos[mask_below] = -1
        laypos_full = np.full_like(z_arr, -1, dtype=int)
        laypos_full[mask_valid] = laypos
        return laypos_full


def _polygon_patch(polygon, **kwargs):
    patch = PathPatch(
        Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
        ),
        **kwargs,
    )
    return patch
