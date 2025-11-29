"""
Geospatial indexing for FloPy vertex and unstructured grids.

Provides efficient spatial queries using KD-tree with cell centers
AND vertices for robust edge case handling, plus pre-computed ConvexHull
equations for fast point-in-polygon testing.

Note: StructuredGrid has its own optimized spatial methods and does not
use this index.
"""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


class GeospatialIndex:
    """
    Geospatial index for efficient geometric queries on vertex/unstructured grids.

    Uses KD-tree indexing with cell centers + vertices to find candidate cells,
    then pre-computed ConvexHull hyperplane equations for fast vectorized
    point-in-polygon testing.

    The cell center + vertices approach ensures edge cases are handled:
    - Points near cell boundaries
    - Points in thin/sliver cells where the cell center may be far from the query

    Note
    ----
    This index uses the grid's ``xcellcenters`` and ``ycellcenters`` properties,
    which represent user-provided or computed cell center coordinates. These
    are not necessarily true geometric centroids (center of mass). For convex
    polygons like triangles and rectangles, the difference is negligible. For
    concave or irregular cells, the cell center may fall outside the cell
    boundary. The index handles this by also indexing all cell vertices,
    ensuring robust spatial queries regardless of cell center placement.

    Note: StructuredGrid has its own optimized spatial methods and should not
    use this index.

    Parameters
    ----------
    grid : VertexGrid or UnstructuredGrid
        A FloPy vertex or unstructured grid object
    epsilon : float, optional
        Tolerance for point-in-cell tests. Used for both bounding box
        expansion and ConvexHull hyperplane distance tests. Ensures boundary
        points are included in adjacent cells.

    Attributes
    ----------
    grid : Grid
        The grid object this index was built for
    epsilon : float
        Tolerance for geometric tests
    is_3d : bool
        True if index uses 3D coordinates (x,y,z), False for 2D (x,y only).
        Automatically True when grid has grid_varies_by_layer=True.
    tree : scipy.spatial.cKDTree
        KD-tree of cell centers + vertices for fast spatial queries.
        Uses 2D (x,y) or 3D (x,y,z) depending on is_3d.
    point_to_cell : ndarray
        Mapping from KD-tree point index to cell index
    hull_equations : list
        Pre-computed ConvexHull equations for each cell (2D cells only)
    bounding_boxes : ndarray
        Pre-computed bounding boxes for each cell

    Examples
    --------
    >>> from flopy.discretization import VertexGrid
    >>> from flopy.utils.geospatial_index import GeospatialIndex
    >>>
    >>> # Create a simple triangular grid
    >>> grid = VertexGrid(vertices, cell2d)
    >>> index = GeospatialIndex(grid)
    >>>
    >>> # Single point query
    >>> cellid = index.query_point(x=5.5, y=5.5)
    >>>
    >>> # Multiple points (vectorized)
    >>> cellids = index.query_points(x=[1.5, 5.5, 9.5], y=[1.5, 5.5, 9.5])
    """

    def __init__(self, grid, epsilon=1e-3):
        """
        Build geospatial index for a vertex or unstructured grid.

        Parameters
        ----------
        grid : VertexGrid or UnstructuredGrid
            A FloPy vertex or unstructured grid
        epsilon : float, optional
            Tolerance for point-in-cell tests.
            Used for bounding box expansion and ConvexHull tests.
        """
        self.grid = grid
        self.epsilon = epsilon

        # Determine if we need 3D indexing
        # Use 3D when grid geometry varies by layer (different cells per layer)
        self.is_3d = hasattr(grid, "grid_varies_by_layer") and grid.grid_varies_by_layer

        self._build_index()

    def _build_index(self):
        """
        Build KD-tree with centroids + vertices and pre-compute
        ConvexHull equations.

        For 3D grids (grid_varies_by_layer=True), indexes all nnodes with x,y,z.
        For 2D grids, indexes 2D cells with x,y only.
        """
        points = []
        point_to_cell = []

        # Get grid dimensions for vertex/unstructured grids
        if self.grid.grid_type not in ("vertex", "unstructured"):
            raise ValueError(
                f"GeospatialIndex only supports vertex and unstructured grids, "
                f"got: {self.grid.grid_type}"
            )

        if self.is_3d:
            # 3D indexing: index all nnodes with x,y,z coordinates
            self.ncells = self.grid.nnodes
            self._build_3d_index(points, point_to_cell)
        else:
            # 2D indexing: index only 2D cells with x,y coordinates
            if hasattr(self.grid, "ncpl"):
                ncpl = self.grid.ncpl
                if ncpl is None:
                    # ncpl not set, fall back to xcellcenters
                    self.ncells = len(self.grid.xcellcenters)
                elif np.isscalar(ncpl):
                    # VertexGrid: ncpl is scalar
                    self.ncells = ncpl
                else:
                    # UnstructuredGrid: ncpl is array
                    # Use first layer's cell count for 2D spatial indexing
                    if len(ncpl) > 0:
                        self.ncells = ncpl[0]
                    else:
                        self.ncells = len(self.grid.xcellcenters)
            else:
                self.ncells = len(self.grid.xcellcenters)
            self._build_vertex_index(points, point_to_cell)

        # Build KD-tree from centroids + vertices
        self.points = np.array(points)
        self.point_to_cell = np.array(point_to_cell, dtype=int)
        self.tree = cKDTree(self.points)

        # Pre-compute ConvexHull equations and bounding boxes (2D only)
        if not self.is_3d:
            self._precompute_hulls()
        else:
            # For 3D, store z-bounds for each cell
            self._precompute_3d_bounds()

    def _build_vertex_index(self, points, point_to_cell):
        """Build index for vertex/unstructured grid - centroids + vertices."""
        # Disable copy cache to avoid expensive deepcopy during index build
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        xc = self.grid.xcellcenters
        yc = self.grid.ycellcenters
        xv, yv, _ = self.grid.xyzvertices

        for cellid in range(self.ncells):
            # Add centroid
            points.append([xc[cellid], yc[cellid]])
            point_to_cell.append(cellid)

            # Add all cell vertices
            cell_xv = xv[cellid]
            cell_yv = yv[cellid]
            for vi in range(len(cell_xv)):
                points.append([cell_xv[vi], cell_yv[vi]])
                point_to_cell.append(cellid)

        # Restore copy cache setting
        self.grid._copy_cache = original_copy_cache

    def _build_3d_index(self, points, point_to_cell):
        """Build 3D index for grid_varies_by_layer grids.

        Includes centroids + vertices with z-coordinates.
        """
        # Disable copy cache to avoid expensive deepcopy during index build
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        xc = self.grid.xcellcenters
        yc = self.grid.ycellcenters
        xv, yv, zv = self.grid.xyzvertices

        # Get z-coordinates for cell centroids (use mid-point of top/bottom)
        zc = (zv[0] + zv[1]) / 2.0

        for cellid in range(self.ncells):
            # Add centroid with z-coordinate
            points.append([xc[cellid], yc[cellid], zc[cellid]])
            point_to_cell.append(cellid)

            # Add cell vertices with z-coordinates
            cell_xv = np.atleast_1d(xv[cellid])
            cell_yv = np.atleast_1d(yv[cellid])
            # Use top z for vertices (could also use bottom or average)
            cell_zv_top = zv[0, cellid]
            cell_zv_bot = zv[1, cellid]
            cell_zv_mid = (cell_zv_top + cell_zv_bot) / 2.0

            for vi in range(len(cell_xv)):
                # Add vertex at mid-z (compromise between top and bottom)
                points.append([cell_xv[vi], cell_yv[vi], cell_zv_mid])
                point_to_cell.append(cellid)

        # Restore copy cache setting
        self.grid._copy_cache = original_copy_cache

    def _precompute_hulls(self):
        """Pre-compute ConvexHull equations and bounding boxes for all cells."""
        # Disable copy cache to avoid expensive deepcopy during precomputation
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        self.hull_equations = []
        self.bounding_boxes = np.zeros((self.ncells, 4))  # xmin, xmax, ymin, ymax

        for cellid in range(self.ncells):
            verts = self._get_cell_vertices(cellid)

            # Handle empty or degenerate cells
            if len(verts) < 3:
                self.bounding_boxes[cellid] = [np.inf, -np.inf, np.inf, -np.inf]
                self.hull_equations.append(None)
                continue

            # Compute bounding box
            self.bounding_boxes[cellid] = [
                verts[:, 0].min(),
                verts[:, 0].max(),
                verts[:, 1].min(),
                verts[:, 1].max(),
            ]

            # Compute ConvexHull equations
            try:
                hull = ConvexHull(verts)
                self.hull_equations.append(hull.equations)
            except Exception:
                # Degenerate geometry
                self.hull_equations.append(None)

        # Restore copy cache setting
        self.grid._copy_cache = original_copy_cache

    def _precompute_3d_bounds(self):
        """Pre-compute 3D bounding boxes (x,y,z bounds) for all cells."""
        xv, yv, zv = self.grid.xyzvertices

        # Store 3D bounding boxes: [xmin, xmax, ymin, ymax, zmin, zmax]
        self.bounding_boxes_3d = np.zeros((self.ncells, 6))

        for cellid in range(self.ncells):
            cell_xv = np.atleast_1d(xv[cellid])
            cell_yv = np.atleast_1d(yv[cellid])
            cell_z_top = zv[0, cellid]
            cell_z_bot = zv[1, cellid]

            self.bounding_boxes_3d[cellid] = [
                np.min(cell_xv),
                np.max(cell_xv),
                np.min(cell_yv),
                np.max(cell_yv),
                min(cell_z_top, cell_z_bot),
                max(cell_z_top, cell_z_bot),
            ]

    def _get_cell_vertices(self, cellid):
        """Get vertices for a cell."""
        xv, yv, _ = self.grid.xyzvertices
        return np.column_stack([xv[cellid], yv[cellid]])

    def query_point(self, x, y, z=None, k=None):
        """
        Find cell containing a single point.

        Parameters
        ----------
        x, y : float
            Point coordinates
        z : float, optional
            Z-coordinate. Required for 3D grids (grid_varies_by_layer=True).
        k : int, optional
            Number of unique cells to check (default 30)

        Returns
        -------
        cellid : int or np.nan
            Cell index containing the point, or np.nan if outside grid
        """
        z_array = np.array([z]) if z is not None else None
        result = self.query_points(np.array([x]), np.array([y]), z=z_array, k=k)
        cellid = result[0]
        return int(cellid) if not np.isnan(cellid) else np.nan

    def query_points(self, x, y, z=None, k=None):
        """
        Find cells containing multiple points (vectorized).

        Uses KD-tree to find k nearest unique cells, then tests for containment.
        For 2D grids: uses ConvexHull testing.
        For 3D grids: uses 3D bounding box testing.

        Parameters
        ----------
        x, y : array-like
            Point coordinates (must have same length)
        z : array-like, optional
            Z-coordinates. Required for 3D grids (grid_varies_by_layer=True).
            For 2D grids with layers, z-search is handled internally.
        k : int, optional
            Number of unique candidate cells to check per point (default 30)

        Returns
        -------
        cellids : ndarray
            Array of cell indices (np.nan for points outside grid).
            For 3D grids, returns 3D cell index (nnodes).
            For 2D grids, returns 2D cell index (ncpl).
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        # For 3D grids, z is required
        if self.is_3d:
            if z is None:
                raise ValueError(
                    "Z-coordinate required for 3D grids (grid_varies_by_layer=True)"
                )
            z = np.atleast_1d(z)
            if len(z) != len(x):
                raise ValueError("z must have the same length as x and y")
        else:
            if z is not None:
                z = np.atleast_1d(z)
                if len(z) != len(x):
                    raise ValueError("z must have the same length as x and y")

        if k is None:
            k = 30  # Default: check 30 unique cells

        # Build query points (2D or 3D)
        if self.is_3d:
            points = np.column_stack([x, y, z])
        else:
            points = np.column_stack([x, y])

        n_points = len(points)
        results = np.full(n_points, np.nan, dtype=float)

        # For each point, query KD-tree to get k unique candidate cells
        for i in range(n_points):
            point = points[i]

            # Query KD-tree adaptively to get k unique cells
            candidates = self._get_k_unique_cells(point, k)

            # Collect all matching cells for tie-breaking
            matching_cells = []

            # Test candidates in order (nearest first)
            for cellid in candidates:
                if self.is_3d:
                    # 3D: check 3D bounding box
                    if self._point_in_cell_3d(point, cellid):
                        matching_cells.append(cellid)
                else:
                    # 2D: use ConvexHull test
                    if self._point_in_cell_vectorized(point[:2], cellid):
                        # Found 2D cell, now check z if provided
                        if z is None:
                            matching_cells.append(cellid)
                        else:
                            # Search through layers to find the right z
                            cell_3d = self._find_layer_for_z(cellid, z[i])
                            if cell_3d is not None:
                                matching_cells.append(cell_3d)

            # Apply grid-specific tie-breaking when multiple cells match
            if len(matching_cells) > 0:
                results[i] = self._apply_tiebreaker(matching_cells)

        # Return int array if all points found, float array otherwise (to preserve nan)
        valid_mask = ~np.isnan(results)
        if np.all(valid_mask):
            return results.astype(int)
        return results

    def _get_k_unique_cells(self, point, k):
        """
        Query KD-tree to get k unique candidate cells.

        Since KD-tree contains centroids + vertices, we may need to query
        more than k points to get k unique cells.

        Parameters
        ----------
        point : ndarray
            Query point [x, y]
        k : int
            Number of unique cells desired

        Returns
        -------
        unique_cells : ndarray
            Array of up to k unique cell indices, ordered by distance
        """
        # Query k*5 points to get k unique cells (accounting for vertices)
        k_points = min(k * 5, len(self.points))

        distances, indices = self.tree.query(point, k=k_points)

        # Handle scalar result
        if np.isscalar(indices):
            indices = [indices]

        # Extract unique cells while preserving order
        seen = set()
        unique_cells = []
        for idx in indices:
            cellid = self.point_to_cell[idx]
            if cellid not in seen:
                seen.add(cellid)
                unique_cells.append(cellid)
                if len(unique_cells) >= k:
                    break

        # If we still don't have k unique cells, query more points
        while len(unique_cells) < k and k_points < len(self.points):
            k_points = min(k_points * 2, len(self.points))
            distances, indices = self.tree.query(point, k=k_points)

            if np.isscalar(indices):
                indices = [indices]

            for idx in indices:
                cellid = self.point_to_cell[idx]
                if cellid not in seen:
                    seen.add(cellid)
                    unique_cells.append(cellid)
                    if len(unique_cells) >= k:
                        break

            if k_points >= len(self.points):
                break  # Can't query any more points

        return np.array(unique_cells[:k], dtype=int)

    def _point_in_cell_vectorized(self, point, cellid):
        """
        Test if point is inside cell using pre-computed bounding box + ConvexHull.

        Parameters
        ----------
        point : ndarray
            Point coordinates [x, y]
        cellid : int
            Cell index

        Returns
        -------
        bool
            True if point is inside cell
        """
        # Fast bounding box rejection with epsilon tolerance for edge cases
        # Expand bounding box slightly to handle points on boundaries
        bbox = self.bounding_boxes[cellid]
        if not (
            bbox[0] - self.epsilon <= point[0] <= bbox[1] + self.epsilon
            and bbox[2] - self.epsilon <= point[1] <= bbox[3] + self.epsilon
        ):
            return False

        # Precise ConvexHull test
        hull_eq = self.hull_equations[cellid]
        if hull_eq is not None:
            # Vectorized hyperplane test: Ax + By + C <= epsilon
            # Tolerance allows points on edges to be included
            distances = point @ hull_eq[:, :-1].T + hull_eq[:, -1]
            return np.all(distances <= self.epsilon)
        else:
            # Degenerate geometry - should rarely happen
            return False

    def _point_in_cell_3d(self, point, cellid):
        """
        Test if 3D point is inside cell using 3D bounding box.

        Parameters
        ----------
        point : ndarray
            Point coordinates [x, y, z]
        cellid : int
            Cell index

        Returns
        -------
        bool
            True if point is inside cell's 3D bounding box
        """
        # Use 3D bounding box test with epsilon tolerance
        bbox = self.bounding_boxes_3d[cellid]
        return (
            bbox[0] - self.epsilon <= point[0] <= bbox[1] + self.epsilon
            and bbox[2] - self.epsilon <= point[1] <= bbox[3] + self.epsilon
            and bbox[4] - self.epsilon <= point[2] <= bbox[5] + self.epsilon
        )

    def _apply_tiebreaker(self, matching_cells):
        """
        Apply tie-breaking when multiple cells match.

        For vertex/unstructured grids: choose cell with lowest cell ID.

        Parameters
        ----------
        matching_cells : list
            List of cell indices that all contain the point

        Returns
        -------
        cellid : int
            The selected cell after tie-breaking
        """
        if len(matching_cells) == 1:
            return matching_cells[0]

        return min(matching_cells)

    def _find_layer_for_z(self, icell2d, z):
        """
        Find 3D cell index for a 2D cell and z-coordinate.

        Searches through layers to find which layer contains the z-coordinate.

        Parameters
        ----------
        icell2d : int
            2D cell index (from layer 0)
        z : float
            Z-coordinate

        Returns
        -------
        cell_3d : int or None
            3D cell index, or None if z not found in any layer
        """
        # Get z-bounds for all cells
        _, _, zv = self.grid.xyzvertices

        # For VertexGrid: same 2D geometry for all layers
        if self.grid.grid_type == "vertex":
            # Search through all layers to find which contains z
            # For VertexGrid: zv[lay, icell2d] = top, zv[lay+1, icell2d] = bottom
            for lay in range(self.grid.nlay):
                z_top = zv[lay, icell2d]
                z_bot = zv[lay + 1, icell2d]
                z_min = min(z_top, z_bot)
                z_max = max(z_top, z_bot)

                if z_min <= z <= z_max:
                    # Found the layer! Return the layer index
                    return lay

            # z not found in any layer
            return None

        # For UnstructuredGrid: compute 3D cell index using vectorized search
        elif self.grid.grid_type == "unstructured":
            # Build array of 3D cell indices for this 2D cell across all layers
            if hasattr(self.grid, "ncpl") and isinstance(self.grid.ncpl, np.ndarray):
                # Compute cumulative sum to get 3D cell indices
                ncpl_cumsum = np.concatenate([[0], np.cumsum(self.grid.ncpl[:-1])])
                cell_indices_3d = icell2d + ncpl_cumsum
            else:
                # Fallback: assume constant ncpl
                cell_indices_3d = (
                    icell2d + np.arange(self.grid.nlay) * self.grid.ncpl[0]
                )

            # Get z-bounds for all layers of this 2D cell
            z_tops = zv[0, cell_indices_3d]
            z_bots = zv[1, cell_indices_3d]
            z_mins = np.minimum(z_tops, z_bots)
            z_maxs = np.maximum(z_tops, z_bots)

            # Find layers containing z (vectorized)
            mask = (z_mins <= z) & (z <= z_maxs)
            matching_layers = np.where(mask)[0]

            if len(matching_layers) > 0:
                # Return first matching layer's 3D cell index
                return cell_indices_3d[matching_layers[0]]

        return None

    def __repr__(self):
        """String representation."""
        return (
            f"GeospatialIndex({self.grid.grid_type} grid, "
            f"{self.ncells} cells, "
            f"{len(self.points)} indexed points)"
        )
