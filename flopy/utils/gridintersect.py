import matplotlib.pyplot as plt
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
    print("matplotlib is needed for grid intersect operations! Please " +
          "matplotlib if you need to use grid intersect functionality.")
from .geometry import transform
try:
    from shapely.geometry import (MultiPoint, Point, Polygon, box,
                                  GeometryCollection)
    from shapely.strtree import STRtree
    from shapely.affinity import translate, rotate
except ModuleNotFoundError:
    print("Shapely is needed for grid intersect operations! Please install " +
          "shapely if you need to use grid intersect functionality.")


def parse_shapely_ix_result(collection, ix_result, shptyps=None):
    """
    Recursive function for parsing shapely intersection results.
    Returns a list of shapely shapes matching shptyp

    Parameters
    ----------
    collection : list
        state variable for storing result, generally
        an empty list
    ix_result : shapely.geometry type
        any shapely intersection result
    shptyp : str, list of str, or None, optional
        if None (default), return all types of shapes.
        if str, return shapes of that type, if list of str,
        return all types in list

    Returns
    -------
    collection : list
        list containing shapely geometries of type shptyp

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
        for ishp in ix_result:
            parse_shapely_ix_result(collection, ishp, shptyps=shptyps)
    # if collecting all types
    elif shptyps[0] is None:
        return collection.append(ix_result)
    return collection


class GridIntersect:
    """
    Class for intersecting shapely shapes (Point, Linestring, Polygon,
    or their Multi variants) with MODFLOW grids. Contains optimized search
    routines for structured grids.

    Notes
    -----
     - The STR-tree query is based on the bounding box of the shape or
       collection, if the bounding box of the shape covers nearly the entire
       grid, the query won't be able to limit the search space much resulting
       in slower performance. Therefore, it is sometimes faster to intersect
       each individual shape in a collection than it is to intersect with the
       whole collection at once.
     - Building the STRtree can take a while for large grids. Once built the
       intersect routines (for individual shapes) should be pretty fast.
     - The optimized routines for structured grids will generally outperform
       the shapely routines because of the reduced overhead of building and
       parsing the queried STR-tree. For Polygons, shapely is sometimes faster
       than the optimized structured routines.

    """

    def __init__(self, mfgrid, method="strtree"):
        """
        Intersect shapes (Point, Linestring, Polygon) with a
        modflow grid.

        Parameters
        ----------
        mfgrid : flopy modflowgrid
            MODFLOW grid as implemented in flopy
        method : str, optional
            either "strtree" which builds an STRTree (most flexible)
            or "structured" which uses optimized methods that only work
            for structured grids, by default "strtree"

        """

        self.mfgrid = mfgrid

        if method == "strtree":
            if mfgrid.grid_type == "structured":
                self.gridshapes = self._rect_grid_to_shape_list()
            elif mfgrid.grid_type == "unstructured":
                raise NotImplementedError()
            elif mfgrid.grid_type == "vertex":
                self.gridshapes = self._vtx_grid_to_shape_list()

            self.strtree = STRtree(self.gridshapes)

            self.intersect_point = self._intersect_point_shapely
            self.intersect_linestring = self._intersect_linestring_shapely
            self.intersect_polygon = self._intersect_polygon_shapely

        elif method == "structured" and mfgrid.grid_type == "structured":
            self.strtree = None
            self.intersect_point = self._intersect_point_structured
            self.intersect_linestring = self._intersect_linestring_structured
            self.intersect_polygon = self._intersect_polygon_structured

        else:
            raise NotImplementedError(
                "Method 'structured' only works for structured grids.")

    def _rect_grid_to_shape_list(self):
        """
        internal method, convert structured grid to list of
        shapely polygons

        Returns
        -------
        list
            list of shapely Polygons

        """
        shplist = []
        for i in range(self.mfgrid.nrow):
            for j in range(self.mfgrid.ncol):
                xy = self.mfgrid.get_cell_vertices(i, j)
                p = Polygon(xy)
                p.name = (i, j)
                shplist.append(p)
        return shplist

    def _usg_grid_to_shape_list(self):
        """
        internal method, convert unstructred grid to list of shapely
        polygons

        Returns
        -------
        list
            list of shapely Polygons
        """
        raise NotImplementedError()

    def _vtx_grid_to_shape_list(self):
        """
        internal method, convert vertex grid to list of shapely polygons

        Returns
        -------
        list
            list of shapely Polygons

        """

        shplist = []
        if isinstance(self.mfgrid._cell2d, np.recarray):
            for icell in self.mfgrid._cell2d.icell2d:
                points = []
                icverts = ["icvert_{}".format(i) for i in
                           range(self.mfgrid._cell2d["ncvert"][icell])]
                for iv in self.mfgrid._cell2d[icverts][icell]:
                    points.append((self.mfgrid._vertices.xv[iv],
                                   self.mfgrid._vertices.yv[iv]))
                # close the polygon, if necessary
                if points[0] != points[-1]:
                    points.append(points[0])
                p = Polygon(points)
                p.name = icell
                shplist.append(p)
        elif isinstance(self.mfgrid._cell2d, list):
            for icell in range(len(self.mfgrid._cell2d)):
                points = []
                for iv in self.mfgrid._cell2d[icell][-3:]:
                    points.append((self.mfgrid._vertices[iv][1],
                                   self.mfgrid._vertices[iv][2]))
                # close the polygon, if necessary
                if points[0] != points[-1]:
                    points.append(points[0])
                p = Polygon(points)
                p.name = icell
                shplist.append(p)
        return shplist

    @staticmethod
    def _sort_strtree_result(shapelist):
        """
        internal method, sort strtree query result by node id

        Parameters
        ----------
        shapelist : list
            list of shapely Polygons

        Returns
        -------
        list
            sorted list of Polygons

        """
        def sort_key(o):
            return o.name
        shapelist.sort(key=sort_key)
        return shapelist

    def _intersect_point_shapely(self, shp, sort_by_cellid=True):
        """
        intersect grid with Point or MultiPoint

        Parameters
        ----------
        shp : Point or MultiPoint
            shapely Point or MultiPoint to intersect with grid. Note,
            it is generally faster to loop over a MultiPoint and intersect
            per point than to intersect a MultiPoint directly.
        sort_by_cellid : bool, optional
            flag whether to sort cells by id, used to ensure node
            with lowest id is returned, by default True

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """
        ixshapes = self.strtree.query(shp)
        if sort_by_cellid:
            ixshapes = self._sort_strtree_result(ixshapes)

        isectshp = []
        cellids = []
        vertices = []
        parsed_points = []  # for keeping track of points

        # loop over cells returned by spatial query
        for r in ixshapes:
            # do intersection
            intersect = shp.intersection(r)
            # parse result per Point
            collection = parse_shapely_ix_result(
                [], intersect, shptyps=["Point"])
            # loop over intersection result and store information
            cell_verts = []
            cell_shps = []
            for c in collection:
                verts = c.__geo_interface__["coordinates"]
                # avoid returning multiple cells for points on boundaries
                if verts in parsed_points:
                    continue
                parsed_points.append(verts)
                cell_shps.append(c)  # collect only new points
                cell_verts.append(verts)
            # if any new ix found
            if len(cell_shps) > 0:
                # combine new points in MultiPoint
                isectshp.append(MultiPoint(cell_shps) if len(cell_shps) > 1
                                else cell_shps[0])
                vertices.append(tuple(cell_verts))
                cellids.append(r.name)

        rec = np.recarray(len(isectshp),
                          names=["cellids", "vertices", "ixshapes"],
                          formats=["O", "O", "O"])
        rec.ixshapes = isectshp
        rec.vertices = vertices
        rec.cellids = cellids

        return rec

    def _intersect_linestring_shapely(self, shp, keepzerolengths=False,
                                      sort_by_cellid=True):
        """
        intersect with LineString or MultiLineString

        Parameters
        ----------
        shp : shapely.geometry.LineString or MultiLineString
            LineString to intersect with the grid
        keepzerolengths : bool, optional
            keep linestrings with length zero, default is False
        sort_by_cellid : bool, optional
            flag whether to sort cells by id, used to ensure node
            with lowest id is returned, by default True

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """
        result = self.strtree.query(shp)
        if sort_by_cellid:
            result = self._sort_strtree_result(result)

        # initialize empty lists for storing results
        isectshp = []
        cellids = []
        vertices = []
        lengths = []

        # loop over cells returned by spatial query
        for r in result:
            # do intersection
            intersect = shp.intersection(r)
            # parse result
            collection = parse_shapely_ix_result(
                [], intersect, shptyps=["LineString", "MultiLineString"])
            # loop over intersection result and store information
            for c in collection:
                verts = c.__geo_interface__["coordinates"]
                # test if linestring was already processed (if on boundary)
                if verts in vertices:
                    continue
                # if keep zero don't check length
                if not keepzerolengths:
                    if c.length == 0.:
                        continue
                isectshp.append(c)
                lengths.append(c.length)
                vertices.append(verts)
                cellids.append(r.name)

        rec = np.recarray(len(isectshp),
                          names=["cellids", "vertices", "lengths", "ixshapes"],
                          formats=["O", "O", "f8", "O"])
        rec.ixshapes = isectshp
        rec.vertices = vertices
        rec.lengths = lengths
        rec.cellids = cellids

        return rec

    def _intersect_polygon_shapely(self, shp, sort_by_cellid=True):
        """
        intersect with Polygon or MultiPolygon

        Parameters
        ----------
        shp : shapely.geometry.Polygon or MultiPolygon
            shape to intersect with the grid
        sort_by_cellid : bool, optional
            flag whether to sort cells by id, used to ensure node
            with lowest id is returned, by default True

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """
        ixshapes = self.strtree.query(shp)
        if sort_by_cellid:
            ixshapes = self._sort_strtree_result(ixshapes)

        isectshp = []
        cellids = []
        vertices = []
        areas = []

        # loop over cells returned by spatial query
        for r in ixshapes:
            # do intersection
            intersect = shp.intersection(r)
            # parse result
            collection = parse_shapely_ix_result(
                [], intersect, shptyps=["Polygon", "MultiPolygon"])
            # loop over intersection result and store information
            for c in collection:
                # don't store intersections with 0 area
                if c.area == 0.:
                    continue
                verts = c.__geo_interface__["coordinates"]
                isectshp.append(c)
                areas.append(c.area)
                vertices.append(verts)
                cellids.append(r.name)

        rec = np.recarray(len(isectshp),
                          names=["cellids", "vertices", "areas", "ixshapes"],
                          formats=["O", "O", "f8", "O"])
        rec.ixshapes = isectshp
        rec.vertices = vertices
        rec.areas = areas
        rec.cellids = cellids

        return rec

    def _intersect_point_structured(self, shp):
        """
        intersection method for intersecting points with structured grids

        Parameters
        ----------
        shp : shapely.geometry.Point or MultiPoint
            point shape to intersect with grid

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """
        nodelist = []

        Xe, Ye = self.mfgrid.xyedges

        try:
            iter(shp)
        except TypeError:
            shp = [shp]

        ixshapes = []
        for p in shp:
            # if grid is rotated or offset transform point to local coords
            if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                    or self.mfgrid.yoffset != 0.):
                rx, ry = transform(p.x, p.y, self.mfgrid.xoffset,
                                   self.mfgrid.yoffset,
                                   self.mfgrid.angrot_radians,
                                   inverse=True)
            else:
                rx = p.x
                ry = p.y

            # two dimensional point
            jpos = ModflowGridIndices.find_position_in_array(Xe, rx)
            ipos = ModflowGridIndices.find_position_in_array(Ye, ry)

            if jpos is not None and ipos is not None:
                nodelist.append((ipos, jpos))
                ixshapes.append(p)

            # three dimensional point
            if p._ndim == 3:
                # find k
                kpos = ModflowGridIndices.find_position_in_array(
                    self.mfgrid.botm[:, ipos, jpos], p.z)
                if kpos is not None:
                    nodelist.append((kpos, ipos, jpos))

        # remove duplicates
        tempnodes = []
        tempshapes = []
        for node, ixs in zip(nodelist, ixshapes):
            if node not in tempnodes:
                tempnodes.append(node)
                tempshapes.append(ixs)
            else:
                # TODO: not sure if this is correct
                tempshapes[-1] = MultiPoint([tempshapes[-1], ixs])

        ixshapes = tempshapes
        nodelist = tempnodes

        rec = np.recarray(len(nodelist), names=["cellids", "ixshapes"],
                          formats=["O", "O"])
        rec.cellids = nodelist
        rec.ixshapes = ixshapes
        return rec

    def _intersect_linestring_structured(self, shp, keepzerolengths=False):
        """
        method for intersecting linestrings with structured grids

        Parameters
        ----------
        shp : shapely.geometry.Linestring or MultiLineString
            linestring to intersect with grid
        keepzerolengths : bool, optional
            if True keep intersection results with length=0, in
            other words, grid cells the linestring does not cross
            but does touch, by default False

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """
        # get local extent of grid
        if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                or self.mfgrid.yoffset != 0.):
            xmin = np.min(self.mfgrid.xyedges[0])
            xmax = np.max(self.mfgrid.xyedges[0])
            ymin = np.min(self.mfgrid.xyedges[1])
            ymax = np.max(self.mfgrid.xyedges[1])
        else:
            xmin, xmax, ymin, ymax = self.mfgrid.extent
        pl = box(xmin, ymin, xmax, ymax)

        # rotate and translate linestring to local coords
        if (self.mfgrid.xoffset != 0. or self.mfgrid.yoffset != 0.):
            shp = translate(shp, xoff=-self.mfgrid.xoffset,
                            yoff=-self.mfgrid.yoffset)
        if self.mfgrid.angrot != 0.:
            shp = rotate(shp, -self.mfgrid.angrot, origin=(0., 0.))

        # clip line to mfgrid bbox
        lineclip = shp.intersection(pl)

        if lineclip.length == 0.:  # linestring does not intersect modelgrid
            return np.recarray(0, names=["cellids", "vertices",
                                         "lengths", "ixshapes"],
                               formats=["O", "O", "f8", "O"])
        if lineclip.geom_type is 'MultiLineString':  # there are multiple lines
            nodelist, lengths, vertices = [], [], []
            ixshapes = []
            for ls in lineclip:
                n, l, v, ix = self._get_nodes_intersecting_linestring(ls)
                nodelist += n
                lengths += l
                # if necessary, transform coordinates back to real
                # world coordinates
                if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                        or self.mfgrid.yoffset != 0.):
                    v_realworld = []
                    for pt in v:
                        rx, ry = transform([pt[0]], [pt[1]],
                                           self.mfgrid.xoffset,
                                           self.mfgrid.yoffset,
                                           self.mfgrid.angrot_radians,
                                           inverse=False)
                        v_realworld.append([rx, ry])
                    ix_realworld = rotate(
                        ix, self.mfgrid.angrot, origin=(0., 0.))
                    ix_realworld = translate(
                        ix_realworld, self.mfgrid.xoffset, self.mfgrid.yoffset)
                else:
                    v_realworld = v
                    ix_realworld = ix
                vertices += v_realworld
                ixshapes += ix_realworld
        else:  # linestring is fully within grid
            nodelist, lengths, vertices, ixshapes = \
                self._get_nodes_intersecting_linestring(
                    lineclip)
            # if necessary, transform coordinates back to real
            # world coordinates
            if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                    or self.mfgrid.yoffset != 0.):
                v_realworld = []
                for pt in vertices:
                    rx, ry = transform([pt[0]], [pt[1]], self.mfgrid.xoffset,
                                       self.mfgrid.yoffset,
                                       self.mfgrid.angrot_radians,
                                       inverse=False)
                    v_realworld.append([rx, ry])
                vertices = v_realworld

                ix_shapes_realworld = []
                for ixs in ixshapes:
                    ixs = rotate(ixs, self.mfgrid.angrot, origin=(0., 0.))
                    ixs = translate(ixs, self.mfgrid.xoffset,
                                    self.mfgrid.yoffset)
                    ix_shapes_realworld.append(ixs)
                ixshapes = ix_shapes_realworld

        # bundle linestrings in same cell
        tempnodes = []
        templengths = []
        tempverts = []
        tempshapes = []
        unique_nodes = list(set(nodelist))
        if len(unique_nodes) < len(nodelist):
            for inode in unique_nodes:
                templengths.append(
                    sum([l for l, i in zip(lengths, nodelist) if i == inode]))
                tempverts.append(
                    [v for v, i in zip(vertices, nodelist) if i == inode])
                tempshapes.append(
                    [ix for ix, i in zip(ixshapes, nodelist) if i == inode])

            nodelist = unique_nodes
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
                    tempshapes.append(ixshapes[i])
            nodelist = tempnodes
            lengths = templengths
            vertices = tempverts
            ixshapes = tempshapes

        rec = np.recarray(len(nodelist),
                          names=["cellids", "vertices", "lengths", "ixshapes"],
                          formats=["O", "O", "f8", "O"])
        rec.vertices = vertices
        rec.lengths = lengths
        rec.cellids = nodelist
        rec.ixshapes = ixshapes

        return rec

    def _get_nodes_intersecting_linestring(self, linestring):
        """
        helper function, intersect the linestring with the a structured
        grid and return a list of node indices and the length of the
        line in that node.

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
        nodelist = []
        lengths = []
        vertices = []
        ixshapes = []

        # start at the beginning of the line
        x, y = linestring.xy

        # linestring already in local coords but
        # because intersect_point does transform again
        # we transform back to real world here if necessary
        if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                or self.mfgrid.yoffset != 0.):
            x0, y0 = transform([x[0]], [y[0]], self.mfgrid.xoffset,
                               self.mfgrid.yoffset, self.mfgrid.angrot_radians,
                               inverse=False)
        else:
            x0 = [x[0]]
            y0 = [y[0]]

        (i, j) = self.intersect_point(Point(x0[0], y0[0])).cellids[0]
        Xe, Ye = self.mfgrid.xyedges
        xmin = Xe[j]
        xmax = Xe[j + 1]
        ymax = Ye[i]
        ymin = Ye[i + 1]
        pl = box(xmin, ymin, xmax, ymax)
        intersect = linestring.intersection(pl)
        # if linestring starts in cell, exits, and re-enters
        # a MultiLineString is returned.
        ixshapes.append(intersect)
        length = intersect.length
        lengths.append(length)
        if intersect.geom_type == "MultiLineString":
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
            node, length, verts, ixshape = \
                self._check_adjacent_cells_intersecting_line(
                    linestring, (i, j), nodelist)

            for inode, ilength, ivert, ix in zip(node, length, verts, ixshape):
                if inode is not None:
                    if ivert not in vertices:
                        nodelist.append(inode)
                        lengths.append(ilength)
                        vertices.append(ivert)
                        ixshapes.append(ix)

            if n == len(nodelist) - 1:
                break
            n += 1

        return nodelist, lengths, vertices, ixshapes

    def _check_adjacent_cells_intersecting_line(self, linestring, i_j,
                                                nodelist):
        """
        helper method that follows a line through a structured grid

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
                pl = box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if intersect.geom_type == "MultiLineString":
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1])
                                  for ixy in zip(*intersect.xy)])
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
                pl = box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if intersect.geom_type == "MultiLineString":
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1])
                                  for ixy in zip(*intersect.xy)])
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
                pl = box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if intersect.geom_type == "MultiLineString":
                        x, y = [], []
                        for igeom in intersect.geoms:
                            x.append(igeom.xy[0])
                            y.append(igeom.xy[1])
                        x = np.concatenate(x)
                        y = np.concatenate(y)
                    else:
                        x = intersect.xy[0]
                        y = intersect.xy[1]
                    verts.append([(ixy[0], ixy[1]) for ixy in
                                  zip(*intersect.xy)])
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
                pl = box(xmin, ymin, xmax, ymax)
                if linestring.intersects(pl):
                    intersect = linestring.intersection(pl)
                    ixshape.append(intersect)
                    length.append(intersect.length)
                    if intersect.geom_type == "MultiLineString":
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
        """
        intersect a rectangle with a structured grid to retrieve
        node ids of intersecting grid cells.

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

        nodelist = []

        # return if rectangle does not contain any cells
        if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                or self.mfgrid.yoffset != 0.):
            minx = np.min(self.mfgrid.xyedges[0])
            maxx = np.max(self.mfgrid.xyedges[0])
            miny = np.min(self.mfgrid.xyedges[1])
            maxy = np.max(self.mfgrid.xyedges[1])
            local_extent = [minx, maxx, miny, maxy]
        else:
            local_extent = self.mfgrid.extent

        xmin, xmax, ymin, ymax = local_extent
        bgrid = box(xmin, ymin, xmax, ymax)
        (rxmin, rymin), (rxmax, rymax) = rectangle
        b = box(rxmin, rymin, rxmax, rymax)

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

    def _intersect_polygon_structured(self, shp):
        """
        intersect polygon with a structured grid. Uses
        bounding box of the Polygon to limit search space.

        Parameters
        ----------
        shp : shapely.geometry.Polygon
            polygon to intersect with the grid

        Returns
        -------
        numpy.recarray
            a record array containing information about the intersection

        """

        # initialize the result lists
        nodelist = []
        areas = []
        vertices = []
        ixshapes = []

        # transform polygon to local grid coordinates
        if (self.mfgrid.xoffset != 0. or self.mfgrid.yoffset != 0.):
            shp = translate(shp, xoff=-self.mfgrid.xoffset,
                            yoff=-self.mfgrid.yoffset)
        if self.mfgrid.angrot != 0.:
            shp = rotate(shp, -self.mfgrid.angrot, origin=(0., 0.))

        # use the bounds of the polygon to restrict the cell search
        minx, miny, maxx, maxy = shp.bounds
        rectangle = ((minx, miny), (maxx, maxy))
        nodes = self._intersect_rectangle_structured(rectangle)

        for (i, j) in nodes:
            if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                    or self.mfgrid.yoffset != 0.):
                cell_coords = [(self.mfgrid.xyedges[0][j],
                                self.mfgrid.xyedges[1][i]),
                               (self.mfgrid.xyedges[0][j + 1],
                                self.mfgrid.xyedges[1][i]),
                               (self.mfgrid.xyedges[0][j + 1],
                                self.mfgrid.xyedges[1][i + 1]),
                               (self.mfgrid.xyedges[0][j],
                                self.mfgrid.xyedges[1][i + 1])]
            else:
                cell_coords = self.mfgrid.get_cell_vertices(i, j)
            node_polygon = Polygon(cell_coords)
            if shp.intersects(node_polygon):
                intersect = shp.intersection(node_polygon)
                if intersect.area > 0.:
                    nodelist.append((i, j))
                    areas.append(intersect.area)

                    # if necessary, transform coordinates back to real
                    # world coordinates
                    if (self.mfgrid.angrot != 0. or self.mfgrid.xoffset != 0.
                            or self.mfgrid.yoffset != 0.):
                        v_realworld = []
                        for pt in intersect.__geo_interface__["coordinates"]:
                            rx, ry = transform([pt[0]], [pt[1]],
                                               self.mfgrid.xoffset,
                                               self.mfgrid.yoffset,
                                               self.mfgrid.angrot_radians,
                                               inverse=False)
                            v_realworld.append([rx, ry])
                        intersect_realworld = rotate(intersect,
                                                     self.mfgrid.angrot,
                                                     origin=(0., 0.))
                        intersect_realworld = translate(intersect_realworld,
                                                        self.mfgrid.xoffset,
                                                        self.mfgrid.yoffset)
                    else:
                        v_realworld = intersect.__geo_interface__[
                            "coordinates"]
                        intersect_realworld = intersect
                    ixshapes.append(intersect_realworld)
                    vertices.append(v_realworld)

        rec = np.recarray(len(nodelist),
                          names=["cellids", "vertices", "areas", "ixshapes"],
                          formats=["O", "O", "f8", "O"])
        rec.vertices = vertices
        rec.areas = areas
        rec.cellids = nodelist
        rec.ixshapes = ixshapes

        return rec

    @staticmethod
    def plot_polygon(rec, ax=None, **kwargs):
        """
        method to plot the polygon intersection results from
        the resulting numpy.recarray.

        Note: only works when recarray has 'intersects' column!

        Parameters
        ----------
        rec : numpy.recarray
            record array containing intersection results
            (the resulting shapes)
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the plot function

        Returns
        -------
        ax: matplotlib.pyplot.axes
            returns the axes handle

        """
        try:
            from descartes import PolygonPatch
        except ModuleNotFoundError:
            msg = 'descartes package needed for plotting polygons'
            if plt is None:
                msg = 'matplotlib and descartes packages needed for ' + \
                      'plotting polygons'
            raise ModuleNotFoundError(msg)

        if plt is None:
            msg = 'matplotlib package needed for plotting polygons'
            raise ModuleNotFoundError(msg)

        if ax is None:
            _, ax = plt.subplots()

        for i, ishp in enumerate(rec.ixshapes):
            ppi = PolygonPatch(ishp, facecolor="C{}".format(i % 10), **kwargs)
            ax.add_patch(ppi)

        return ax

    @staticmethod
    def plot_linestring(rec, ax=None, **kwargs):
        """
        method to plot the linestring intersection results from
        the resulting numpy.recarray.

        Note: only works when recarray has 'intersects' column!

        Parameters
        ----------
        rec : numpy.recarray
            record array containing intersection results
            (the resulting shapes)
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the plot function

        Returns
        -------
        ax: matplotlib.pyplot.axes
            returns the axes handle

        """
        if plt is None:
            msg = 'matplotlib package needed for plotting polygons'
            raise ModuleNotFoundError(msg)

        if ax is None:
            _, ax = plt.subplots()

        for i, ishp in enumerate(rec.ixshapes):
            if ishp.type == "MultiLineString":
                for part in ishp:
                    ax.plot(part.xy[0], part.xy[1], ls="-",
                            c="C{}".format(i % 10), **kwargs)
            else:
                ax.plot(ishp.xy[0], ishp.xy[1], ls="-",
                        c="C{}".format(i % 10), **kwargs)

        return ax

    @staticmethod
    def plot_point(rec, ax=None, **kwargs):
        """
        method to plot the point intersection results from
        the resulting numpy.recarray.

        Note: only works when recarray has 'intersects' column!

        Parameters
        ----------
        rec : numpy.recarray
            record array containing intersection results
        ax : matplotlib.pyplot.axes, optional
            axes to plot onto, if not provided, creates a new figure
        **kwargs:
            passed to the scatter function

        Returns
        -------
        ax: matplotlib.pyplot.axes
            returns the axes handle

        """
        if plt is None:
            msg = 'matplotlib package needed for plotting polygons'
            raise ModuleNotFoundError(msg)

        if ax is None:
            _, ax = plt.subplots()

        x, y = [], []
        geo_coll = GeometryCollection(list(rec.ixshapes))
        collection = parse_shapely_ix_result([], geo_coll, ["Point"])
        for c in collection:
            x.append(c.x)
            y.append(c.y)
        ax.scatter(x, y, **kwargs)

        return ax


class ModflowGridIndices:
    """
    Collection of methods that can be used to find cell indices for a
    structured, but irregularly spaced MODFLOW grid.
    """

    @staticmethod
    def find_position_in_array(arr, x):
        """
        If arr has x positions for the left edge of a cell, then
        return the cell index containing x.

        Parameters
        ----------
        arr : A one dimensional array (such as Xe) that contains
            coordinates for the left cell edge.

        x : float
            The x position to find in arr.

        """
        jpos = None

        if x == arr[-1]:
            return len(arr) - 2

        if x < min(arr[0], arr[-1]):
            return None

        if x > max(arr[0], arr[-1]):
            return None

        # go through each position
        for j in range(len(arr) - 1):
            xl = arr[j]
            xr = arr[j + 1]
            frac = (x - xl) / (xr - xl)
            if 0. <= frac <= 1.0:
                # if min(xl, xr) <= x < max(xl, xr):
                jpos = j
                return jpos

        return jpos

    @staticmethod
    def kij_from_nodenumber(nodenumber, nlay, nrow, ncol):
        """
        Convert the modflow node number to a zero-based layer, row and column
        format.  Return (k0, i0, j0).

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
            raise Exception('Error in function kij_from_nodenumber...')
        n = nodenumber - 1
        k = int(n / nrow / ncol)
        i = int((n - k * nrow * ncol) / ncol)
        j = n - k * nrow * ncol - i * ncol
        return (k, i, j)

    @staticmethod
    def nodenumber_from_kij(k, i, j, nrow, ncol):
        """
        Calculate the nodenumber using the zero-based layer, row, and column
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
        """
        Calculate the zero-based nodenumber using the zero-based layer, row,
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
        """
        Convert the node number to a zero-based layer, row and column
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
            raise Exception('Error in function kij_from_nodenumber...')
        k = int(n / nrow / ncol)
        i = int((n - k * nrow * ncol) / ncol)
        j = n - k * nrow * ncol - i * ncol
        return (k, i, j)
