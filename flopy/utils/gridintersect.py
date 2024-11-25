import warnings

import numpy as np
from pandas import DataFrame

from .geometry import transform
from .geospatial_utils import GeoSpatialUtil
from .utl_import import import_optional_dependency

shapely = import_optional_dependency("shapely", errors="silent")

# TODO: remove the following methods and classes in version 3.10.0
# - ModflowGridIndices
# - GridIntersect:
#   - remove method kwarg from __init__
#   - remove structured methods from intersect() and intersects()
#   - _intersect_point_structured()
#   - _intersect_linestring_structured()
#   - _get_nodes_intersecting_linestring()
#   - _check_adjacent_cells_intersecting_line()
#   - _intersect_rectangle_structured()
#   - _intersect_polygon_structured()
#   - _transform_geo_interface_polygon()


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
     - Building the STR-tree can take a while for large grids. Once built the
       intersect routines (for individual shapes) should be pretty fast.
    """

    def __init__(self, mfgrid, method=None, rtree=True, local=False):
        """Intersect shapes (Point, Linestring, Polygon) with a modflow grid.

        Parameters
        ----------
        mfgrid : flopy modflowgrid
            MODFLOW grid as implemented in flopy
        method : str, optional
            Method to use for intersection shapes with the grid. Method 'vertex'
            will be the only option in the future. Method 'structured' is deprecated.
            This keyword argument will be removed in a future release.

            .. deprecated:: 3.9.0
                method="vertex" will be the only option from 3.10.0

        rtree : bool, optional
            whether to build an STR-Tree, default is True. If False no STR-tree
            is built, but intersects will loop through all model gridcells
            (which is generally slower).
        local : bool, optional
            use local model coordinates from model grid to build grid geometries,
            default is False and uses real-world coordinates (with offset and rotation).
        """
        import_optional_dependency(
            "shapely", error_message="GridIntersect requires shapely"
        )
        self.mfgrid = mfgrid
        self.local = local
        # TODO: remove method kwarg in version v3.10.0
        # keep default behavior for v3.9.0, but warn if method is not vertex
        # allow silencing of warning with method="vertex" in v3.9.0
        if method is None:
            # determine method from grid_type
            self.method = self.mfgrid.grid_type
        else:
            # set method
            self.method = method
        if self.method != "vertex":
            warnings.warn(
                (
                    'Note `method="structured"` is deprecated. '
                    'Pass `method="vertex"` to silence this warning. '
                    "This will be the new default in a future release and this "
                    "keyword argument will be removed."
                ),
                category=DeprecationWarning,
            )
        self.rtree = rtree

        # really only necessary for method=='vertex' as structured methods
        # do not require a full list of shapely geometries, but useful to be
        # able to obtain the grid shapes nonetheless
        self._set_method_get_gridshapes()

        if self.method == "vertex":
            # build arrays of geoms and cellids
            self.geoms, self.cellids = self._get_gridshapes()

            # build STR-tree if specified
            if self.rtree:
                strtree = import_optional_dependency(
                    "shapely.strtree",
                    error_message="STRTree requires shapely",
                )
                self.strtree = strtree.STRtree(self.geoms)

        elif self.method == "structured" and mfgrid.grid_type == "structured":
            # geoms and cellids do not need to be assigned for structured
            # methods
            self.geoms = None
            self.cellids = None

        else:
            raise ValueError(
                f"Method '{self.method}' not recognized or not supported "
                f"for grid_type '{self.mfgrid.grid_type}'!"
            )

    def intersect(
        self,
        shp,
        shapetype=None,
        sort_by_cellid=True,
        keepzerolengths=False,
        return_all_intersections=False,
        contains_centroid=False,
        min_area_fraction=None,
        geo_dataframe=False,
        shapely2=None,
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
        keepzerolengths : bool
            boolean method to keep zero length intersections for linestring
            intersection, only used if shape type is "linestring"
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
        if shapely2 is not None:
            warnings.warn(
                "The shapely2 keyword argument is deprecated. "
                "Shapely<2 support was dropped in flopy version 3.9.0."
            )
        gu = GeoSpatialUtil(shp, shapetype=shapetype)
        shp = gu.shapely

        if gu.shapetype in ("Point", "MultiPoint"):
            if self.method == "structured" and self.mfgrid.grid_type == "structured":
                rec = self._intersect_point_structured(
                    shp, return_all_intersections=return_all_intersections
                )
            else:
                rec = self._intersect_point_shapely(
                    shp,
                    sort_by_cellid=sort_by_cellid,
                    return_all_intersections=return_all_intersections,
                )
        elif gu.shapetype in ("LineString", "MultiLineString"):
            if self.method == "structured" and self.mfgrid.grid_type == "structured":
                rec = self._intersect_linestring_structured(
                    shp,
                    keepzerolengths,
                    return_all_intersections=return_all_intersections,
                )
            else:
                rec = self._intersect_linestring_shapely(
                    shp,
                    keepzerolengths,
                    sort_by_cellid=sort_by_cellid,
                    return_all_intersections=return_all_intersections,
                )
        elif gu.shapetype in ("Polygon", "MultiPolygon"):
            if self.method == "structured" and self.mfgrid.grid_type == "structured":
                rec = self._intersect_polygon_structured(
                    shp,
                    contains_centroid=contains_centroid,
                    min_area_fraction=min_area_fraction,
                )
            else:
                rec = self._intersect_polygon_shapely(
                    shp,
                    sort_by_cellid=sort_by_cellid,
                    contains_centroid=contains_centroid,
                    min_area_fraction=min_area_fraction,
                )
        else:
            raise TypeError(f"Shapetype {gu.shapetype} is not supported")

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

    def _set_method_get_gridshapes(self):
        """internal method, set self._get_gridshapes to the appropriate method
        for obtaining grid cell geometries."""
        # Set method for obtaining grid shapes
        if self.mfgrid.grid_type == "structured":
            self._get_gridshapes = self._rect_grid_to_geoms_cellids
        elif self.mfgrid.grid_type == "vertex":
            self._get_gridshapes = self._vtx_grid_to_geoms_cellids
        elif self.mfgrid.grid_type == "unstructured":
            raise NotImplementedError()

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

    def query_grid(self, shp):
        """Perform spatial query on grid with shapely geometry. If no spatial
        query is possible returns all grid cells.

        Parameters
        ----------
        shp : shapely.geometry
            shapely geometry

        Returns
        -------
        array_like
            array containing cellids of grid cells in query result
        """
        if self.rtree:
            result = self.strtree.query(shp)
        else:
            # no spatial query
            result = self.cellids
        return result

    def filter_query_result(self, cellids, shp):
        """Filter array of geometries to obtain grid cells that intersect with
        shape.

        Used to (further) reduce query result to cells that intersect with
        shape.

        Parameters
        ----------
        cellids : iterable
            iterable of cellids, query result
        shp : shapely.geometry
            shapely geometry that is prepared and used to filter
            query result

        Returns
        -------
        array_like
            filter or generator containing polygons that intersect with shape
        """
        # get only gridcells that intersect
        if not shapely.is_prepared(shp):
            shapely.prepare(shp)
        qcellids = cellids[shapely.intersects(self.geoms[cellids], shp)]
        return qcellids

    def _intersect_point_shapely(
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

    def _intersect_linestring_shapely(
        self,
        shp,
        keepzerolengths=False,
        sort_by_cellid=True,
        return_all_intersections=False,
    ):
        if keepzerolengths:
            warnings.warn(
                "`keepzerolengths` is deprecated. For obtaining all cellids that "
                "intersect with a LineString, use `intersects()`.",
                DeprecationWarning,
            )

        if self.rtree:
            qcellids = self.strtree.query(shp, predicate="intersects")
        else:
            qcellids = self.filter_query_result(self.cellids, shp)

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

    def _intersect_polygon_shapely(
        self,
        shp,
        sort_by_cellid=True,
        contains_centroid=False,
        min_area_fraction=None,
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

    def intersects(self, shp, shapetype=None, dataframe=False):
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

        Returns
        -------
        numpy.recarray or pandas.DataFrame
            a record array or pandas.DataFrame containing cell IDs of the gridcells
            the shape intersects with.
        """
        shp = GeoSpatialUtil(shp, shapetype=shapetype).shapely
        qfiltered = self.strtree.query(shp, predicate="intersects")

        # build rec-array
        rec = np.recarray(len(qfiltered), names=["cellids"], formats=["O"])
        if self.mfgrid.grid_type == "structured":
            rec.cellids = list(zip(*self.mfgrid.get_lrc([qfiltered])[0][1:]))
        else:
            rec.cellids = qfiltered

        if dataframe:
            return DataFrame(rec)
        return rec

    def _intersect_point_structured(self, shp, return_all_intersections=False):
        """intersection method for intersecting points with structured grids.

        .. deprecated:: 3.9.0
            use _intersect_point_shapely() or set method="vertex" in GridIntersect.

        Parameters
        ----------
        shp : shapely.geometry.Point or MultiPoint
            point shape to intersect with grid
        return_all_intersections :  bool, optional
            if True, return multiple intersection results for points on grid
            cell boundaries (e.g. returns 2 intersection results if a point
            lies on the boundary between two grid cells). The default is False,
            which will return a single intersection result for boundary cases.

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection
        """
        shapely_geo = import_optional_dependency("shapely.geometry")

        nodelist = []

        Xe, Ye = self.mfgrid.xyedges

        if isinstance(shp, shapely_geo.Point):
            shp = [shp]
        elif isinstance(shp, shapely_geo.MultiPoint):
            shp = list(shp.geoms)
        else:
            raise ValueError("expected Point or MultiPoint")

        ixshapes = []
        for p in shp:
            # if grid is rotated or offset transform point to local coords
            if (
                self.mfgrid.angrot != 0.0
                or self.mfgrid.xoffset != 0.0
                or self.mfgrid.yoffset != 0.0
            ) and not self.local:
                rx, ry = transform(
                    p.x,
                    p.y,
                    self.mfgrid.xoffset,
                    self.mfgrid.yoffset,
                    self.mfgrid.angrot_radians,
                    inverse=True,
                )
            else:
                rx = p.x
                ry = p.y

            # two dimensional point
            jpos = ModflowGridIndices.find_position_in_array(Xe, rx)
            ipos = ModflowGridIndices.find_position_in_array(Ye, ry)

            if jpos is not None and ipos is not None:
                # use only first idx if return_all_intersections is False
                if not return_all_intersections:
                    if isinstance(jpos, list):
                        jpos = jpos[0]
                    if isinstance(ipos, list):
                        ipos = ipos[0]
                # three dimensional point
                if p._ndim == 3:
                    # find k, if ipos or jpos on boundary, use first entry
                    if isinstance(jpos, list):
                        jj = jpos[0]
                    else:
                        jj = jpos
                    if isinstance(ipos, list):
                        ii = ipos[0]
                    else:
                        ii = ipos
                    kpos = ModflowGridIndices.find_position_in_array(
                        self.mfgrid.botm[:, ii, jj], p.z
                    )
                    # if z-position on boundary, use first k
                    if isinstance(kpos, list):
                        kpos = kpos[0]
                    if kpos is not None:
                        # point on boundary, either jpos or ipos has len > 1
                        if isinstance(ipos, list) or isinstance(jpos, list):
                            # convert to list if needed for loop
                            if not isinstance(ipos, list):
                                ipos = [ipos]
                            if not isinstance(jpos, list):
                                jpos = [jpos]
                            for ii in ipos:
                                for jj in jpos:
                                    nodelist.append((kpos, ii, jj))
                                    ixshapes.append(p)
                        # point not on boundary
                        else:
                            nodelist.append((kpos, ipos, jpos))
                            ixshapes.append(p)
                else:
                    # point on boundary, either jpos or ipos has len > 1
                    if isinstance(ipos, list) or isinstance(jpos, list):
                        # convert to list if needed for loop
                        if not isinstance(ipos, list):
                            ipos = [ipos]
                        if not isinstance(jpos, list):
                            jpos = [jpos]
                        for ii in ipos:
                            for jj in jpos:
                                nodelist.append((ii, jj))
                                ixshapes.append(p)
                    else:
                        nodelist.append((ipos, jpos))
                        ixshapes.append(p)

        # remove duplicates
        if not return_all_intersections:
            tempnodes = []
            tempshapes = []
            for node, ixs in zip(nodelist, ixshapes):
                if node not in tempnodes:
                    tempnodes.append(node)
                    tempshapes.append(ixs)
                else:
                    tempshapes[-1] = shapely_geo.MultiPoint([tempshapes[-1], ixs])

            ixshapes = tempshapes
            nodelist = tempnodes

        rec = np.recarray(
            len(nodelist), names=["cellids", "ixshapes"], formats=["O", "O"]
        )
        rec.cellids = nodelist
        rec.ixshapes = ixshapes
        return rec

    def _intersect_linestring_structured(
        self, shp, keepzerolengths=False, return_all_intersections=False
    ):
        """method for intersecting linestrings with structured grids.

        .. deprecated:: 3.9.0
            use _intersect_point_shapely() or set method="vertex" in GridIntersect.

        Parameters
        ----------
        shp : shapely.geometry.Linestring or MultiLineString
            linestring to intersect with grid
        keepzerolengths : bool, optional
            if True keep intersection results with length=0, in
            other words, grid cells the linestring does not cross
            but does touch, by default False
        return_all_intersections :  bool, optional
            if True, return multiple intersection results for linestrings on
            grid cell boundaries (e.g. returns 2 intersection results if a
            linestring lies on the boundary between two grid cells). The
            default is False, which will return a single intersection result
            for boundary cases.

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection
        """
        shapely = import_optional_dependency("shapely")
        shapely_geo = import_optional_dependency("shapely.geometry")
        affinity_loc = import_optional_dependency("shapely.affinity")

        # get local extent of grid
        if (
            self.mfgrid.angrot != 0.0
            or self.mfgrid.xoffset != 0.0
            or self.mfgrid.yoffset != 0.0
        ):
            xmin = np.min(self.mfgrid.xyedges[0])
            xmax = np.max(self.mfgrid.xyedges[0])
            ymin = np.min(self.mfgrid.xyedges[1])
            ymax = np.max(self.mfgrid.xyedges[1])
        else:
            xmin, xmax, ymin, ymax = self.mfgrid.extent
        pl = shapely_geo.box(xmin, ymin, xmax, ymax)

        # rotate and translate linestring to local coords
        if (
            self.mfgrid.xoffset != 0.0 or self.mfgrid.yoffset != 0.0
        ) and not self.local:
            shp = affinity_loc.translate(
                shp, xoff=-self.mfgrid.xoffset, yoff=-self.mfgrid.yoffset
            )
        if self.mfgrid.angrot != 0.0 and not self.local:
            shp = affinity_loc.rotate(shp, -self.mfgrid.angrot, origin=(0.0, 0.0))

        # clip line to mfgrid bbox
        lineclip = shp.intersection(pl)

        if lineclip.length == 0.0:  # linestring does not intersect modelgrid
            return np.recarray(
                0,
                names=["cellids", "vertices", "lengths", "ixshapes"],
                formats=["O", "O", "f8", "O"],
            )
        if lineclip.geom_type == "MultiLineString":  # there are multiple lines
            nodelist, lengths, vertices = [], [], []
            ixshapes = []
            for ls in lineclip.geoms:
                n, l, v, ixs = self._get_nodes_intersecting_linestring(
                    ls, return_all_intersections=return_all_intersections
                )
                nodelist += n
                lengths += l
                # if necessary, transform coordinates back to real
                # world coordinates
                if (
                    self.mfgrid.angrot != 0.0
                    or self.mfgrid.xoffset != 0.0
                    or self.mfgrid.yoffset != 0.0
                ) and not self.local:
                    v_realworld = []
                    for pt in v:
                        pt = np.array(pt)
                        rx, ry = transform(
                            pt[:, 0],
                            pt[:, 1],
                            self.mfgrid.xoffset,
                            self.mfgrid.yoffset,
                            self.mfgrid.angrot_radians,
                            inverse=False,
                        )
                        v_realworld.append(list(zip(rx, ry)))
                    ixs_realworld = []
                    for ix in ixs:
                        ix_realworld = affinity_loc.rotate(
                            ix, self.mfgrid.angrot, origin=(0.0, 0.0)
                        )
                        ix_realworld = affinity_loc.translate(
                            ix_realworld, self.mfgrid.xoffset, self.mfgrid.yoffset
                        )
                        ixs_realworld.append(ix_realworld)
                else:
                    v_realworld = v
                    ixs_realworld = ixs
                vertices += v_realworld
                ixshapes += ixs_realworld
        else:  # linestring is fully within grid
            (nodelist, lengths, vertices, ixshapes) = (
                self._get_nodes_intersecting_linestring(
                    lineclip, return_all_intersections=return_all_intersections
                )
            )
            # if necessary, transform coordinates back to real
            # world coordinates
            if (
                self.mfgrid.angrot != 0.0
                or self.mfgrid.xoffset != 0.0
                or self.mfgrid.yoffset != 0.0
            ) and not self.local:
                v_realworld = []
                for pt in vertices:
                    pt = np.array(pt)
                    rx, ry = transform(
                        pt[:, 0],
                        pt[:, 1],
                        self.mfgrid.xoffset,
                        self.mfgrid.yoffset,
                        self.mfgrid.angrot_radians,
                        inverse=False,
                    )
                    v_realworld.append(list(zip(rx, ry)))
                vertices = v_realworld

                ix_shapes_realworld = []
                for ixs in ixshapes:
                    ixs = affinity_loc.rotate(
                        ixs, self.mfgrid.angrot, origin=(0.0, 0.0)
                    )
                    ixs = affinity_loc.translate(
                        ixs, self.mfgrid.xoffset, self.mfgrid.yoffset
                    )
                    ix_shapes_realworld.append(ixs)
                ixshapes = ix_shapes_realworld

        # bundle linestrings in same cell
        tempnodes = []
        templengths = []
        tempverts = []
        tempshapes = []
        unique_nodes = list(set(nodelist))
        parsed_nodes = []
        if len(unique_nodes) < len(nodelist):
            for inode in nodelist:
                # maintain order of nodes by keeping track of parsed nodes
                if inode in parsed_nodes:
                    continue
                templengths.append(
                    sum([l for l, i in zip(lengths, nodelist) if i == inode])
                )
                tempverts.append([v for v, i in zip(vertices, nodelist) if i == inode])
                tempshapes.append(
                    [ix for ix, i in zip(ixshapes, nodelist) if i == inode]
                )
                parsed_nodes.append(inode)

            nodelist = parsed_nodes
            lengths = templengths
            vertices = tempverts
            ixshapes = tempshapes

        # eliminate any nodes that have a zero length
        if not keepzerolengths:
            tempnodes = []
            templengths = []
            tempverts = []
            tempshapes = []
            for i, _ in enumerate(nodelist):
                if lengths[i] > 0:
                    tempnodes.append(nodelist[i])
                    templengths.append(lengths[i])
                    tempverts.append(vertices[i])
                    ishp = ixshapes[i]
                    if isinstance(ishp, list):
                        ishp = shapely.unary_union(ishp)
                    tempshapes.append(ishp)
            nodelist = tempnodes
            lengths = templengths
            vertices = tempverts
            ixshapes = tempshapes

        rec = np.recarray(
            len(nodelist),
            names=["cellids", "vertices", "lengths", "ixshapes"],
            formats=["O", "O", "f8", "O"],
        )
        rec.vertices = vertices
        rec.lengths = lengths
        rec.cellids = nodelist
        rec.ixshapes = ixshapes

        return rec

    def _get_nodes_intersecting_linestring(
        self, linestring, return_all_intersections=False
    ):
        """helper function, intersect the linestring with the a structured grid
        and return a list of node indices and the length of the line in that
        node.

        .. deprecated:: 3.9.0
            method="structured" is deprecated.

        Parameters
        ----------
        linestring: shapely.geometry.LineString or MultiLineString
            shape to intersect with the grid

        Returns
        -------
        nodelist, lengths, vertices: lists
            lists containing node ids, lengths of intersects and the
            start and end points of the intersects
        """
        shapely_geo = import_optional_dependency("shapely.geometry")

        nodelist = []
        lengths = []
        vertices = []
        ixshapes = []

        # start at the beginning of the line
        x, y = linestring.xy

        # linestring already in local coords but
        # because intersect_point does transform again
        # we transform back to real world here if necessary
        if (
            self.mfgrid.angrot != 0.0
            or self.mfgrid.xoffset != 0.0
            or self.mfgrid.yoffset != 0.0
        ) and not self.local:
            x0, y0 = transform(
                [x[0]],
                [y[0]],
                self.mfgrid.xoffset,
                self.mfgrid.yoffset,
                self.mfgrid.angrot_radians,
                inverse=False,
            )
        else:
            x0 = [x[0]]
            y0 = [y[0]]

        (i, j) = self.intersect(shapely_geo.Point(x0[0], y0[0])).cellids[0]
        Xe, Ye = self.mfgrid.xyedges
        xmin = Xe[j]
        xmax = Xe[j + 1]
        ymax = Ye[i]
        ymin = Ye[i + 1]
        pl = shapely_geo.box(xmin, ymin, xmax, ymax)
        intersect = linestring.intersection(pl)
        # if linestring starts in cell, exits, and re-enters
        # a MultiLineString is returned.
        ixshapes.append(intersect)
        length = intersect.length
        lengths.append(length)
        if hasattr(intersect, "geoms"):
            x, y = [], []
            for igeom in intersect.geoms:
                x.append(igeom.xy[0])
                y.append(igeom.xy[1])
            x = np.concatenate(x)
            y = np.concatenate(y)
        else:
            x = intersect.xy[0]
            y = intersect.xy[1]
        verts = [(ixy[0], ixy[1]) for ixy in zip(x, y)]
        vertices.append(verts)
        nodelist.append((i, j))

        n = 0
        while True:
            (i, j) = nodelist[n]
            (node, length, verts, ixshape) = (
                self._check_adjacent_cells_intersecting_line(
                    linestring, (i, j), nodelist
                )
            )

            for inode, ilength, ivert, ix in zip(node, length, verts, ixshape):
                if inode is not None:
                    if not return_all_intersections:
                        if ivert not in vertices:
                            nodelist.append(inode)
                            lengths.append(ilength)
                            vertices.append(ivert)
                            ixshapes.append(ix)
                    else:
                        nodelist.append(inode)
                        lengths.append(ilength)
                        vertices.append(ivert)
                        ixshapes.append(ix)

            if n == len(nodelist) - 1:
                break
            n += 1

        return nodelist, lengths, vertices, ixshapes

    def _check_adjacent_cells_intersecting_line(self, linestring, i_j, nodelist):
        """helper method that follows a line through a structured grid.

        .. deprecated:: 3.9.0
            method="structured" is deprecated.

        Parameters
        ----------
        linestring : shapely.geometry.LineString
            shape to intersect with the grid
        i_j : tuple
            tuple containing (nrow, ncol)
        nodelist : list of tuples
            list of node ids that have already been added
            as intersections

        Returns
        -------
        node, length, verts: lists
            lists containing nodes, lengths and vertices of
            intersections with adjacent cells relative to the
            current cell (i, j)
        """
        shapely_geo = import_optional_dependency("shapely.geometry")

        i, j = i_j

        Xe, Ye = self.mfgrid.xyedges

        node = []
        length = []
        verts = []
        ixshape = []

        # check to left
        if j > 0:
            ii = i
            jj = j - 1
            if (ii, jj) not in nodelist:
                xmin = Xe[jj]
                xmax = Xe[jj + 1]
                ymax = Ye[ii]
                ymin = Ye[ii + 1]
                pl = shapely_geo.box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if hasattr(intersect, "geoms"):
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1]) for ixy in zip(x, y)])
                    node.append((ii, jj))

        # check to right
        if j < self.mfgrid.ncol - 1:
            ii = i
            jj = j + 1
            if (ii, jj) not in nodelist:
                xmin = Xe[jj]
                xmax = Xe[jj + 1]
                ymax = Ye[ii]
                ymin = Ye[ii + 1]
                pl = shapely_geo.box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if hasattr(intersect, "geoms"):
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1]) for ixy in zip(x, y)])
                    node.append((ii, jj))

        # check to back
        if i > 0:
            ii = i - 1
            jj = j
            if (ii, jj) not in nodelist:
                xmin = Xe[jj]
                xmax = Xe[jj + 1]
                ymax = Ye[ii]
                ymin = Ye[ii + 1]
                pl = shapely_geo.box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if hasattr(intersect, "geoms"):
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1]) for ixy in zip(x, y)])
                    node.append((ii, jj))

        # check to front
        if i < self.mfgrid.nrow - 1:
            ii = i + 1
            jj = j
            if (ii, jj) not in nodelist:
                xmin = Xe[jj]
                xmax = Xe[jj + 1]
                ymax = Ye[ii]
                ymin = Ye[ii + 1]
                pl = shapely_geo.box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if hasattr(intersect, "geoms"):
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1]) for ixy in zip(x, y)])
                    node.append((ii, jj))

        # special case for linestrings intersecting in vertex and continuing
        # towards bottom right, check diagonally to front-right, if no other
        # neighbours found
        if np.sum(length) == 0:
            if (i < self.mfgrid.nrow - 1) and (j < self.mfgrid.ncol - 1):
                ii = i + 1
                jj = j + 1
                if (ii, jj) not in nodelist:
                    xmin = Xe[jj]
                    xmax = Xe[jj + 1]
                    ymax = Ye[ii]
                    ymin = Ye[ii + 1]
                    pl = shapely_geo.box(xmin, ymin, xmax, ymax)
                    if linestring.intersects(pl):
                        intersect = linestring.intersection(pl)
                        ixshape.append(intersect)
                        length.append(intersect.length)
                        if hasattr(intersect, "geoms"):
                            x, y = [], []
                            for igeom in intersect.geoms:
                                x.append(igeom.xy[0])
                                y.append(igeom.xy[1])
                            x = np.concatenate(x)
                            y = np.concatenate(y)
                        else:
                            x = intersect.xy[0]
                            y = intersect.xy[1]
                        verts.append([(ixy[0], ixy[1]) for ixy in zip(x, y)])
                        node.append((ii, jj))

        return node, length, verts, ixshape

    def _intersect_rectangle_structured(self, rectangle):
        """intersect a rectangle with a structured grid to retrieve node ids of
        intersecting grid cells.

        .. deprecated:: 3.9.0
            method="structured" is deprecated.

        Note: only works in local coordinates (i.e. non-rotated grid
        with origin at (0, 0))

        Parameters
        ----------
        rectangle : list of tuples
            list of lower-left coordinate and upper-right
            coordinate: [(xmin, ymin), (xmax, ymax)]

        Returns
        -------
        nodelist: list of tuples
            list of tuples containing node ids with which
            the rectangle intersects
        """

        shapely_geo = import_optional_dependency("shapely.geometry")

        nodelist = []

        # return if rectangle does not contain any cells
        if (
            self.mfgrid.angrot != 0.0
            or self.mfgrid.xoffset != 0.0
            or self.mfgrid.yoffset != 0.0
        ):
            minx = np.min(self.mfgrid.xyedges[0])
            maxx = np.max(self.mfgrid.xyedges[0])
            miny = np.min(self.mfgrid.xyedges[1])
            maxy = np.max(self.mfgrid.xyedges[1])
            local_extent = [minx, maxx, miny, maxy]
        else:
            local_extent = self.mfgrid.extent

        xmin, xmax, ymin, ymax = local_extent
        bgrid = shapely_geo.box(xmin, ymin, xmax, ymax)
        (rxmin, rymin), (rxmax, rymax) = rectangle
        b = shapely_geo.box(rxmin, rymin, rxmax, rymax)

        if not b.intersects(bgrid):
            # return with nodelist as an empty list
            return []

        Xe, Ye = self.mfgrid.xyedges

        jmin = ModflowGridIndices.find_position_in_array(Xe, xmin)
        if jmin is None:
            if xmin <= Xe[0]:
                jmin = 0
            elif xmin >= Xe[-1]:
                jmin = self.mfgrid.ncol - 1

        jmax = ModflowGridIndices.find_position_in_array(Xe, xmax)
        if jmax is None:
            if xmax <= Xe[0]:
                jmax = 0
            elif xmax >= Xe[-1]:
                jmax = self.mfgrid.ncol - 1

        imin = ModflowGridIndices.find_position_in_array(Ye, ymax)
        if imin is None:
            if ymax >= Ye[0]:
                imin = 0
            elif ymax <= Ye[-1]:
                imin = self.mfgrid.nrow - 1

        imax = ModflowGridIndices.find_position_in_array(Ye, ymin)
        if imax is None:
            if ymin >= Ye[0]:
                imax = 0
            elif ymin <= Ye[-1]:
                imax = self.mfgrid.nrow - 1

        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                nodelist.append((i, j))

        return nodelist

    def _intersect_polygon_structured(
        self, shp, contains_centroid=False, min_area_fraction=None
    ):
        """intersect polygon with a structured grid. Uses bounding box of the
        Polygon to limit search space.

        .. deprecated:: 3.9.0
            method="structured" is deprecated. Use `_intersect_polygon_shapely()`.


        Notes
        -----
        If performance is slow, try setting the method to 'vertex'
        in the GridIntersect object. For polygons this is often
        faster.

        Parameters
        ----------
        shp : shapely.geometry.Polygon
            polygon to intersect with the grid
        contains_centroid :  bool, optional
            if True, only store intersection result if cell centroid is
            contained within intersection shape
        min_area_fraction : float, optional
            float defining minimum intersection area threshold, if
            intersection area is smaller than min_frac_area * cell_area, do
            not store intersection result

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection
        """
        shapely_geo = import_optional_dependency("shapely.geometry")
        affinity_loc = import_optional_dependency("shapely.affinity")

        # initialize the result lists
        nodelist = []
        areas = []
        vertices = []
        ixshapes = []

        # transform polygon to local grid coordinates
        if (
            self.mfgrid.xoffset != 0.0 or self.mfgrid.yoffset != 0.0
        ) and not self.local:
            shp = affinity_loc.translate(
                shp, xoff=-self.mfgrid.xoffset, yoff=-self.mfgrid.yoffset
            )
        if self.mfgrid.angrot != 0.0 and not self.local:
            shp = affinity_loc.rotate(shp, -self.mfgrid.angrot, origin=(0.0, 0.0))

        # use the bounds of the polygon to restrict the cell search
        minx, miny, maxx, maxy = shp.bounds
        rectangle = ((minx, miny), (maxx, maxy))
        nodes = self._intersect_rectangle_structured(rectangle)

        for i, j in nodes:
            if (
                self.mfgrid.angrot != 0.0
                or self.mfgrid.xoffset != 0.0
                or self.mfgrid.yoffset != 0.0
            ):
                cell_coords = [
                    (self.mfgrid.xyedges[0][j], self.mfgrid.xyedges[1][i]),
                    (self.mfgrid.xyedges[0][j + 1], self.mfgrid.xyedges[1][i]),
                    (self.mfgrid.xyedges[0][j + 1], self.mfgrid.xyedges[1][i + 1]),
                    (self.mfgrid.xyedges[0][j], self.mfgrid.xyedges[1][i + 1]),
                ]
            else:
                cell_coords = self.mfgrid.get_cell_vertices(i, j)
            cell_polygon = shapely_geo.Polygon(cell_coords)
            if shp.intersects(cell_polygon):
                intersect = shp.intersection(cell_polygon)
                collection = parse_shapely_ix_result(
                    [], intersect, shptyps=["Polygon", "MultiPolygon"]
                )
                if len(collection) == 0:
                    continue
                if len(collection) > 1:
                    intersect = shapely_geo.MultiPolygon(collection)
                else:
                    intersect = collection[0]

                # only store results if area > 0.0
                if intersect.area == 0.0:
                    continue
                # option: only store result if cell centroid is contained
                # within intersection result
                if contains_centroid:
                    if not intersect.intersects(cell_polygon.centroid):
                        continue
                # option: min_area_fraction, only store if intersected area
                # is larger than fraction * cell_area
                if min_area_fraction:
                    if intersect.area < (min_area_fraction * cell_polygon.area):
                        continue

                nodelist.append((i, j))
                areas.append(intersect.area)

                # if necessary, transform coordinates back to real
                # world coordinates
                if (
                    self.mfgrid.angrot != 0.0
                    or self.mfgrid.xoffset != 0.0
                    or self.mfgrid.yoffset != 0.0
                ) and not self.local:
                    v_realworld = []
                    if intersect.geom_type.startswith("Multi"):
                        for ipoly in intersect.geoms:
                            v_realworld += self._transform_geo_interface_polygon(ipoly)
                    else:
                        v_realworld += self._transform_geo_interface_polygon(intersect)
                    intersect_realworld = affinity_loc.rotate(
                        intersect, self.mfgrid.angrot, origin=(0.0, 0.0)
                    )
                    intersect_realworld = affinity_loc.translate(
                        intersect_realworld, self.mfgrid.xoffset, self.mfgrid.yoffset
                    )
                else:
                    v_realworld = intersect.__geo_interface__["coordinates"]
                    intersect_realworld = intersect
                ixshapes.append(intersect_realworld)
                vertices.append(v_realworld)

        rec = np.recarray(
            len(nodelist),
            names=["cellids", "vertices", "areas", "ixshapes"],
            formats=["O", "O", "f8", "O"],
        )
        rec.vertices = vertices
        rec.areas = areas
        rec.cellids = nodelist
        rec.ixshapes = ixshapes

        return rec

    def _transform_geo_interface_polygon(self, polygon):
        """Internal method, helper function to transform geometry
        __geo_interface__.

        .. deprecated:: 3.9.0
            method="structured" is deprecated. Only used by
            `_intersect_polygon_structured()`

        Used for translating intersection result coordinates back into
        real-world coordinates.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            polygon to transform coordinates for

        Returns
        -------
        geom_list : list
            list containing transformed coordinates in same structure as
            the original __geo_interface__.
        """

        if polygon.geom_type.startswith("Multi"):
            raise TypeError("Does not support Multi geometries!")

        geom_list = []
        for coords in polygon.__geo_interface__["coordinates"]:
            geoms = []
            try:
                # test depth of list/tuple
                _ = coords[0][0][0]
                if len(coords) == 2:
                    shell, holes = coords
                else:
                    raise ValueError("Cannot parse __geo_interface__")
            except TypeError:
                shell = coords
                holes = None
            except Exception as e:
                raise e
            # transform shell coordinates
            shell_pts = []
            for pt in shell:
                rx, ry = transform(
                    [pt[0]],
                    [pt[1]],
                    self.mfgrid.xoffset,
                    self.mfgrid.yoffset,
                    self.mfgrid.angrot_radians,
                    inverse=False,
                )
                shell_pts.append((rx, ry))
            geoms.append(shell_pts)
            # transform holes coordinates if necessary
            if holes:
                holes_pts = []
                for pt in holes:
                    rx, ry = transform(
                        [pt[0]],
                        [pt[1]],
                        self.mfgrid.xoffset,
                        self.mfgrid.yoffset,
                        self.mfgrid.angrot_radians,
                        inverse=False,
                    )
            # append (shells, holes) to transformed coordinates list
            geom_list.append(tuple(geoms))
        return geom_list

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

        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

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


class ModflowGridIndices:
    """Collection of methods that can be used to find cell indices for a
    structured, but irregularly spaced MODFLOW grid.

    .. deprecated:: 3.9.0
        This class is deprecated and will be removed in version 3.10.0.
    """

    @staticmethod
    def find_position_in_array(arr, x):
        """If arr has x positions for the left edge of a cell, then return the
        cell index containing x.

        Parameters
        ----------
        arr : A one dimensional array (such as Xe) that contains
            coordinates for the left cell edge.

        x : float
            The x position to find in arr.
        """
        jpos = []

        if np.isclose(x, arr[-1]):
            return len(arr) - 2

        xmin = min(arr[0], arr[-1])
        xmax = max(arr[0], arr[-1])

        if np.isclose(x, xmin):
            x = xmin
        if np.isclose(x, xmax):
            x = xmax
        if not (xmin <= x <= xmax):
            return None

        # go through each position
        for j in range(len(arr) - 1):
            xl = arr[j]
            xr = arr[j + 1]
            frac = (x - xl) / (xr - xl)
            if 0.0 <= frac <= 1.0:
                jpos.append(j)
        if len(jpos) == 0:
            return None
        elif len(jpos) == 1:
            return jpos[0]
        else:
            return jpos

    @staticmethod
    def kij_from_nodenumber(nodenumber, nlay, nrow, ncol):
        """Convert the modflow node number to a zero-based layer, row and
        column format.  Return (k0, i0, j0).

        Parameters
        ----------
        nodenumber: int
            The cell nodenumber, ranging from 1 to number of
            nodes.
        nlay: int
            The number of layers.
        nrow: int
            The number of rows.
        ncol: int
            The number of columns.
        """
        if nodenumber > nlay * nrow * ncol:
            raise Exception("Error in function kij_from_nodenumber...")
        n = nodenumber - 1
        k = int(n / nrow / ncol)
        i = int((n - k * nrow * ncol) / ncol)
        j = n - k * nrow * ncol - i * ncol
        return (k, i, j)

    @staticmethod
    def nodenumber_from_kij(k, i, j, nrow, ncol):
        """Calculate the nodenumber using the zero-based layer, row, and column
        values.  The first node has a value of 1.

        Parameters
        ----------
        k : int
            The model layer number as a zero-based value.
        i : int
            The model row number as a zero-based value.
        j : int
            The model column number as a zero-based value.
        nrow : int
            The number of model rows.
        ncol : int
            The number of model columns.
        """
        return k * nrow * ncol + i * ncol + j + 1

    @staticmethod
    def nn0_from_kij(k, i, j, nrow, ncol):
        """Calculate the zero-based nodenumber using the zero-based layer, row,
        and column values.  The first node has a value of 0.

        Parameters
        ----------
        k : int
            The model layer number as a zero-based value.
        i : int
            The model row number as a zero-based value.
        j : int
            The model column number as a zero-based value.
        nrow : int
            The number of model rows.
        ncol : int
            The number of model columns.
        """
        return k * nrow * ncol + i * ncol + j

    @staticmethod
    def kij_from_nn0(n, nlay, nrow, ncol):
        """Convert the node number to a zero-based layer, row and column
        format.  Return (k0, i0, j0).

        Parameters
        ----------
        nodenumber : int
            The cell nodenumber, ranging from 0 to number of
            nodes - 1.
        nlay : int
            The number of layers.
        nrow : int
            The number of rows.
        ncol : int
            The number of columns.
        """
        if n > nlay * nrow * ncol:
            raise Exception("Error in function kij_from_nodenumber...")
        k = int(n / nrow / ncol)
        i = int((n - k * nrow * ncol) / ncol)
        j = n - k * nrow * ncol - i * ncol
        return (k, i, j)


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
