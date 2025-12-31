"""
Geospatial indexing for FloPy grids.

Provides efficient spatial queries for all grid types:
- StructuredGrid: Uses searchsorted for O(log n) row/column finding
- VertexGrid/UnstructuredGrid: Uses KD-tree with cell centers + vertices,
  plus pre-computed ConvexHull equations for fast point-in-polygon testing.

This module provides a unified spatial query interface for all grid types,
with each type using the optimal algorithm for its geometry.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree


class GeospatialIndex:
    """
    Geospatial index for efficient spatial queries on any FloPy grid.

    Provides a unified interface for point intersection queries across all
    grid types, using the optimal algorithm for each:

    - **StructuredGrid**: Uses numpy searchsorted for O(log n) row/column
      finding. No index structure is built; queries operate directly on
      the grid's edge arrays.

    - **VertexGrid/UnstructuredGrid**: Uses KD-tree indexing with cell
      centers + vertices to find candidate cells, then pre-computed
      ConvexHull hyperplane equations for fast vectorized point-in-polygon
      testing.

    The cell center + vertices approach for unstructured grids ensures
    edge cases are handled:
    - Points near cell boundaries
    - Points in thin/sliver cells where the cell center may be far from the query

    Note
    ----
    For vertex/unstructured grids, this index uses the grid's ``xcellcenters``
    and ``ycellcenters`` properties, which represent user-provided or computed
    cell center coordinates. These are not necessarily true geometric centroids
    (center of mass). The index handles this by also indexing all cell vertices,
    ensuring robust spatial queries regardless of cell center placement.

    Parameters
    ----------
    grid : StructuredGrid, VertexGrid, or UnstructuredGrid
        A FloPy grid object
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
    tree : scipy.spatial.cKDTree or None
        KD-tree of cell centers + vertices for fast spatial queries
        (vertex/unstructured grids only). None for structured grids.
    point_to_cell : ndarray or None
        Mapping from KD-tree point index to cell index
        (vertex/unstructured grids only). None for structured grids.
    hull_equations : list or None
        Pre-computed ConvexHull equations for each cell (2D vertex/unstructured
        grids only). None for structured grids.
    bounding_boxes : ndarray or None
        Pre-computed bounding boxes for each cell (vertex/unstructured
        grids only). None for structured grids.

    Examples
    --------
    >>> from flopy.discretization import StructuredGrid, VertexGrid
    >>> from flopy.utils.geospatial_index import GeospatialIndex
    >>>
    >>> # Works with any grid type
    >>> index = GeospatialIndex(grid)
    >>>
    >>> # Single point query
    >>> cellid = index.query_point(x=5.5, y=5.5)
    >>>
    >>> # Multiple points (vectorized)
    >>> cellids = index.query_points(x=[1.5, 5.5, 9.5], y=[1.5, 5.5, 9.5])
    >>>
    >>> # For structured grids, get (row, col) tuples
    >>> row, col = index.intersect(x=5.5, y=5.5)
    """

    def __init__(self, grid, epsilon=1e-3):
        """
        Build geospatial index for any FloPy grid type.

        Parameters
        ----------
        grid : StructuredGrid, VertexGrid, or UnstructuredGrid
            A FloPy grid object
        epsilon : float, optional
            Tolerance for point-in-cell tests.
            Used for bounding box expansion and ConvexHull tests.
        """
        self.grid = grid
        self.epsilon = epsilon

        # Initialize attributes that may not be set for all grid types
        self.tree = None
        self.points = None
        self.point_to_cell = None
        self.hull_equations = None
        self.bounding_boxes = None
        self.bounding_boxes_3d = None

        self._build_index()

    @property
    def is_3d(self):
        """True if grid geometry varies by layer (requires 3D indexing)."""
        return getattr(self.grid, "grid_varies_by_layer", False)

    def _build_index(self):
        """
        Build spatial index appropriate for the grid type.

        For structured grids: No index structure needed; uses searchsorted
        directly on edge arrays.

        For vertex/unstructured grids:
        - 2D: KD-tree with cell centers + vertices, ConvexHull equations
        - 3D (grid_varies_by_layer=True): 3D KD-tree with bounding boxes
        """
        # Structured grids don't need an index structure
        if self.grid.grid_type == "structured":
            self.ncells = self.grid.nnodes
            # Cache edge arrays for faster queries
            self._xe, self._ye = self.grid.xyedges
            self._ye_flipped = self._ye[::-1]
            return

        # Vertex/unstructured grids use KD-tree indexing
        points = []
        point_to_cell = []

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

        # Build KD-tree from cell centers + vertices
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
        """Build index for vertex/unstructured grid - cell centers + vertices."""
        # Disable copy cache to avoid expensive deepcopy during index build
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        xc = self.grid.xcellcenters
        yc = self.grid.ycellcenters
        xv, yv, _ = self.grid.xyzvertices

        for cellid in range(self.ncells):
            # Add cell center
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

        Includes cell centers + vertices with z-coordinates.
        """
        # Disable copy cache to avoid expensive deepcopy during index build
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        xc = np.asarray(self.grid.xcellcenters)
        yc = np.asarray(self.grid.ycellcenters)
        xv, yv, zv = self.grid.xyzvertices

        # Get z-coordinates for cell centers (use mid-point of top/bottom)
        zc = (zv[0] + zv[1]) / 2.0

        # For multi-layer unstructured grids, xc/yc may be per-layer (ncpl values)
        # while we iterate over all nnodes. Build mapping from node to layer-local cell.
        ncpl = getattr(self.grid, "ncpl", None)
        if ncpl is not None and not np.isscalar(ncpl):
            ncpl = np.atleast_1d(ncpl)
            xc_is_per_layer = len(xc) < self.ncells
        else:
            xc_is_per_layer = False

        # Build xc_idx mapping vectorized
        cellids = np.arange(self.ncells)
        if xc_is_per_layer:
            cumulative_ncpl = np.concatenate([[0], np.cumsum(ncpl[:-1])])
            layers = np.searchsorted(np.cumsum(ncpl), cellids, side="right")
            xc_idx = cellids - cumulative_ncpl[layers]
        else:
            xc_idx = cellids

        # Add cell centers
        centers = np.column_stack([xc[xc_idx], yc[xc_idx], zc])
        center_cells = cellids

        # Add vertices - must loop due to variable vertex count per cell
        vertex_points = []
        vertex_cells = []
        zc_mid = (zv[0] + zv[1]) / 2.0
        for cellid in range(self.ncells):
            cell_xv = np.atleast_1d(xv[cellid])
            cell_yv = np.atleast_1d(yv[cellid])
            cell_z = zc_mid[cellid]
            n_verts = len(cell_xv)
            vertex_points.append(
                np.column_stack([cell_xv, cell_yv, np.full(n_verts, cell_z)])
            )
            vertex_cells.append(np.full(n_verts, cellid, dtype=int))

        # Combine cell centers and vertices
        all_points = np.vstack([centers] + vertex_points)
        all_cells = np.concatenate([center_cells] + vertex_cells)

        points.extend(all_points.tolist())
        point_to_cell.extend(all_cells.tolist())

        # Restore copy cache setting
        self.grid._copy_cache = original_copy_cache

    def _precompute_hulls(self):
        """Pre-compute edge equations and bounding boxes for all cells.

        Uses vectorized edge equation computation instead of scipy ConvexHull
        for ~100x faster precomputation. Edge equations define half-planes
        for point-in-polygon testing.
        """
        # Disable copy cache to avoid expensive deepcopy during precomputation
        original_copy_cache = getattr(self.grid, "_copy_cache", True)
        self.grid._copy_cache = False

        xv, yv, _ = self.grid.xyzvertices

        # Group cells by vertex count for vectorized processing
        cell_groups = defaultdict(list)
        for cellid in range(self.ncells):
            n_verts = len(xv[cellid])
            cell_groups[n_verts].append(cellid)

        # Initialize storage
        self.edge_equations = [None] * self.ncells  # List of (n_edges, 3) arrays
        self.bounding_boxes = np.zeros((self.ncells, 4))  # xmin, xmax, ymin, ymax

        # Process each group with vectorized operations
        for n_verts, cellids in cell_groups.items():
            if n_verts < 3:
                # Degenerate cells
                for cellid in cellids:
                    self.bounding_boxes[cellid] = [np.inf, -np.inf, np.inf, -np.inf]
                continue

            cellids = np.array(cellids)
            n_cells = len(cellids)

            # Gather vertices for this group: (n_cells, n_verts, 2)
            verts = np.zeros((n_cells, n_verts, 2))
            for i, cellid in enumerate(cellids):
                verts[i, :, 0] = xv[cellid]
                verts[i, :, 1] = yv[cellid]

            # Vectorized bounding boxes
            self.bounding_boxes[cellids, 0] = verts[:, :, 0].min(axis=1)
            self.bounding_boxes[cellids, 1] = verts[:, :, 0].max(axis=1)
            self.bounding_boxes[cellids, 2] = verts[:, :, 1].min(axis=1)
            self.bounding_boxes[cellids, 3] = verts[:, :, 1].max(axis=1)

            # Vectorized edge equations: half-plane representation
            # For edge from v[i] to v[i+1], compute inward-pointing normal
            v0 = verts  # (n_cells, n_verts, 2)
            v1 = np.roll(verts, -1, axis=1)  # Next vertex

            dx = v1[:, :, 0] - v0[:, :, 0]  # (n_cells, n_verts)
            dy = v1[:, :, 1] - v0[:, :, 1]

            # Edge length (avoid division by zero)
            length = np.sqrt(dx * dx + dy * dy)
            length = np.where(length > 0, length, 1.0)

            # Perpendicular normal: rotate edge vector 90 degrees
            # (-dy, dx) gives CCW normal, but we need to verify orientation
            nx = -dy / length
            ny = dx / length

            # Plane equation: nx*x + ny*y + c = 0, where c = -(nx*x0 + ny*y0)
            c = -(nx * v0[:, :, 0] + ny * v0[:, :, 1])

            # Check orientation: centroid should be on negative side (inside)
            centroids = verts.mean(axis=1)  # (n_cells, 2)
            cx = centroids[:, 0:1]  # (n_cells, 1)
            cy = centroids[:, 1:2]

            # Distance from centroid to each edge plane
            dist = cx * nx + cy * ny + c  # (n_cells, n_verts)

            # Flip normals where centroid is on positive side
            flip_mask = dist > 0
            nx = np.where(flip_mask, -nx, nx)
            ny = np.where(flip_mask, -ny, ny)
            c = np.where(flip_mask, -c, c)

            # Stack into equations array: (n_cells, n_verts, 3)
            equations = np.stack([nx, ny, c], axis=-1)

            # Store equations for each cell
            for i, cellid in enumerate(cellids):
                self.edge_equations[cellid] = equations[i]

        # Restore copy cache setting
        self.grid._copy_cache = original_copy_cache

        # For compatibility, also set hull_equations as alias
        self.hull_equations = self.edge_equations

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
        For 2D grids: uses vectorized edge equation testing.
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

        n_points = len(x)

        # Use vectorized path for 2D queries without z
        if not self.is_3d and z is None:
            return self._query_points_vectorized_2d(x, y, k)

        # Fall back to loop-based approach for 3D or z-layer queries
        # Build query points (2D or 3D)
        if self.is_3d:
            points = np.column_stack([x, y, z])
        else:
            points = np.column_stack([x, y])

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
                    # 2D: use edge equation test
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

    def _query_points_vectorized_2d(self, x, y, k):
        """
        Vectorized 2D point query - batch KD-tree and containment tests.

        Uses batch KD-tree query for all points, then tests candidates
        with vectorized edge equation checks.
        """
        n_points = len(x)
        points = np.column_stack([x, y])

        # Batch KD-tree query for all points
        k_query = min(k * 5, len(self.points))
        _, indices = self.tree.query(points, k=k_query)

        # Handle single point case
        if n_points == 1:
            indices = indices.reshape(1, -1)

        # Map KD-tree indices to cell IDs
        candidate_cells = self.point_to_cell[indices]  # (n_points, k_query)

        results = np.full(n_points, np.nan, dtype=float)

        # Process each point - vectorize the candidate testing
        for i in range(n_points):
            # Get unique candidate cells (first k unique)
            seen = set()
            unique_candidates = []
            for c in candidate_cells[i]:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)
                    if len(unique_candidates) >= k:
                        break

            if not unique_candidates:
                continue

            # Test all candidates and collect matches for tie-breaking
            px, py = x[i], y[i]
            matching_cells = []
            for cellid in unique_candidates:
                eq = self.edge_equations[cellid]
                if eq is None:
                    continue

                # Vectorized edge test for this cell
                distances = px * eq[:, 0] + py * eq[:, 1] + eq[:, 2]
                if np.all(distances <= self.epsilon):
                    matching_cells.append(cellid)

            # Apply tie-breaking if multiple cells match
            if matching_cells:
                results[i] = self._apply_tiebreaker(matching_cells)

        # Return int array if all points found
        valid_mask = ~np.isnan(results)
        if np.all(valid_mask):
            return results.astype(int)
        return results

    def _get_k_unique_cells(self, point, k):
        """
        Query KD-tree to get k unique candidate cells.

        Since KD-tree contains cell centers + vertices, we may need to query
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
        Test if point is inside cell using pre-computed bounding box + edge equations.

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

        # Precise edge equation test (half-plane intersection)
        edge_eq = self.edge_equations[cellid]
        if edge_eq is not None:
            # Vectorized half-plane test: nx*x + ny*y + c <= epsilon
            # Tolerance allows points on edges to be included
            distances = point @ edge_eq[:, :-1].T + edge_eq[:, -1]
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
        if self.grid.grid_type == "structured":
            return (
                f"GeospatialIndex({self.grid.grid_type} grid, "
                f"{self.grid.nrow} rows x {self.grid.ncol} cols)"
            )
        return (
            f"GeospatialIndex({self.grid.grid_type} grid, "
            f"{self.ncells} cells, "
            f"{len(self.points)} indexed points)"
        )

    # =========================================================================
    # Unified intersect interface
    # =========================================================================

    def intersect(self, x, y, z=None, local=False, forgive=False):
        """
        Find the cell(s) containing the given point(s).

        This is the unified interface for spatial queries across all grid types.
        Dispatches to the optimal algorithm based on grid type:
        - StructuredGrid: searchsorted (O(log n))
        - VertexGrid/UnstructuredGrid: KD-tree + ConvexHull

        Parameters
        ----------
        x : float or array-like
            The x-coordinate(s) of the query point(s)
        y : float or array-like
            The y-coordinate(s) of the query point(s)
        z : float, array-like, or None
            Optional z-coordinate(s). If provided, returns layer information.
        local : bool, optional
            If True, x and y are in local coordinates (default False)
        forgive : bool, optional
            If True, return NaN for points outside the grid instead of
            raising an error (default False)

        Returns
        -------
        For StructuredGrid:
            row, col : int or ndarray
                Row and column indices. If z is provided, returns (lay, row, col).
        For VertexGrid:
            cellid : int or ndarray
                Cell index (icell2d). If z is provided, returns (lay, cellid).
        For UnstructuredGrid:
            cellid : int or ndarray
                Cell index. If z is provided and grid_varies_by_layer is False,
                returns (lay, cellid).

        Raises
        ------
        ValueError
            If point is outside grid and forgive=False
        """
        if self.grid.grid_type == "structured":
            return self._intersect_structured(x, y, z, local, forgive)
        else:
            return self._intersect_unstructured(x, y, z, local, forgive)

    def _intersect_structured(self, x, y, z=None, local=False, forgive=False):
        """
        Find row/col for structured grid using searchsorted.

        Uses vectorized binary search for O(log n) performance.
        """
        # Check if inputs are scalar
        x_is_scalar = np.isscalar(x)
        y_is_scalar = np.isscalar(y)
        z_is_scalar = z is None or np.isscalar(z)
        is_scalar_input = x_is_scalar and y_is_scalar and z_is_scalar

        # Convert to arrays for uniform processing
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if z is not None:
            z = np.atleast_1d(z)

        # Validate array shapes
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if z is not None and len(z) != len(x):
            raise ValueError("z must have the same length as x and y")

        # Transform to local coordinates if needed
        if not local:
            x, y = self.grid.get_local_coords(x, y)

        # Get cached edge arrays
        xe = self._xe
        ye_flipped = self._ye_flipped

        # Vectorized row/col calculation
        n_points = len(x)
        rows = np.full(n_points, np.nan, dtype=float)
        cols = np.full(n_points, np.nan, dtype=float)

        # Vectorized column finding using searchsorted
        # side="left" ensures x==edge goes to the cell on the left (tie-breaking)
        cols_valid = np.searchsorted(xe, x, side="left") - 1
        cols_mask = (cols_valid >= 0) & (cols_valid < self.grid.ncol)
        cols[cols_mask] = cols_valid[cols_mask]

        # Vectorized row finding using searchsorted on flipped ye
        # side="right" on flipped array ensures y==edge goes to lower row
        rows_flipped = np.searchsorted(ye_flipped, y, side="right")
        rows_valid = len(self._ye) - 1 - rows_flipped
        rows_mask = (rows_valid >= 0) & (rows_valid < self.grid.nrow)
        rows[rows_mask] = rows_valid[rows_mask]

        # Check for errors if not forgiving
        if not forgive:
            invalid_mask = np.isnan(rows) | np.isnan(cols)
            if np.any(invalid_mask):
                idx = np.where(invalid_mask)[0][0]
                raise ValueError(
                    f"x, y point given is outside of the model area: "
                    f"({x[idx]}, {y[idx]})"
                )

        # If either row or col is NaN, set both to NaN
        invalid_mask = np.isnan(rows) | np.isnan(cols)
        rows[invalid_mask] = np.nan
        cols[invalid_mask] = np.nan

        # Convert to int where valid
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            rows[valid_mask] = rows[valid_mask].astype(int)
            cols[valid_mask] = cols[valid_mask].astype(int)

        if z is None:
            # Return 2D results
            if is_scalar_input:
                row, col = rows[0], cols[0]
                if not np.isnan(row) and not np.isnan(col):
                    row, col = int(row), int(col)
                return row, col
            else:
                return (
                    rows.astype(int) if np.all(valid_mask) else rows,
                    cols.astype(int) if np.all(valid_mask) else cols,
                )

        # Handle z-coordinate - vectorized layer finding
        lays = np.full(n_points, np.nan, dtype=float)

        # Only process points that have valid row/col
        valid_mask = ~(np.isnan(rows) | np.isnan(cols))
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            valid_rows = rows[valid_indices].astype(int)
            valid_cols = cols[valid_indices].astype(int)
            valid_z = z[valid_indices]

            # Get top/bottom elevations for all valid points and all layers
            tops_bottoms = self.grid.top_botm[:, valid_rows, valid_cols].T

            # Check which layer each point belongs to
            in_layer = (tops_bottoms[:, :-1] >= valid_z[:, np.newaxis]) & (
                valid_z[:, np.newaxis] >= tops_bottoms[:, 1:]
            )

            # Find the first (topmost) layer for each point
            layer_indices = np.argmax(in_layer, axis=1)

            # Set layer values only where a valid layer was found
            n_valid = len(valid_indices)
            found_layer = in_layer[np.arange(n_valid), layer_indices]
            lays[valid_indices[found_layer]] = layer_indices[found_layer]

            # Check for errors if not forgiving
            if not forgive:
                not_found = ~found_layer
                if np.any(not_found):
                    idx = valid_indices[not_found][0]
                    raise ValueError(
                        f"point given is outside the model area: "
                        f"({x[idx]}, {y[idx]}, {z[idx]})"
                    )

        # Return 3D results
        if is_scalar_input:
            lay, row, col = lays[0], rows[0], cols[0]
            if not np.isnan(lay):
                lay, row, col = int(lay), int(row), int(col)
            return lay, row, col
        else:
            valid_3d = ~np.isnan(lays) & ~np.isnan(rows) & ~np.isnan(cols)
            return (
                lays.astype(int) if np.all(valid_3d) else lays,
                rows.astype(int) if np.all(valid_3d) else rows,
                cols.astype(int) if np.all(valid_3d) else cols,
            )

    def _intersect_unstructured(self, x, y, z=None, local=False, forgive=False):
        """
        Find cell(s) for vertex/unstructured grid using KD-tree.

        Uses KD-tree nearest neighbor search + ConvexHull point-in-polygon.
        """
        # Check if inputs are scalar
        x_is_scalar = np.isscalar(x)
        y_is_scalar = np.isscalar(y)
        z_is_scalar = z is None or np.isscalar(z)
        is_scalar_input = x_is_scalar and y_is_scalar and z_is_scalar

        # Convert to arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if z is not None:
            z = np.atleast_1d(z)

        # Validate array shapes
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if z is not None and len(z) != len(x):
            raise ValueError("z must have the same length as x and y")

        # Transform to world coordinates if local
        if local:
            x, y = self.grid.get_coords(x, y)

        # Use existing query_points for the spatial search
        if self.is_3d:
            # 3D grid requires z
            if z is None:
                raise ValueError(
                    "Z-coordinate required for 3D grids (grid_varies_by_layer=True)"
                )
            cellids = self.query_points(x, y, z=z)
        else:
            # 2D search, then layer search if z provided
            cellids = self.query_points(x, y, z=z)

        # Check for errors if not forgiving
        if not forgive:
            invalid_mask = np.isnan(cellids)
            if np.any(invalid_mask):
                idx = np.where(invalid_mask)[0][0]
                if z is not None:
                    raise ValueError(
                        f"point given is outside the model area: "
                        f"({x[idx]}, {y[idx]}, {z[idx]})"
                    )
                else:
                    raise ValueError(
                        f"x, y point given is outside of the model area: "
                        f"({x[idx]}, {y[idx]})"
                    )

        # Handle return format based on grid type and z
        if self.grid.grid_type == "vertex" and z is not None and not self.is_3d:
            # For VertexGrid with z, return (lay, icell2d)
            # The layer was already found in query_points via _find_layer_for_z
            # We need to extract layer and icell2d from the result
            n_points = len(x)
            lays = np.full(n_points, np.nan, dtype=float)
            icell2ds = np.full(n_points, np.nan, dtype=float)

            valid_mask = ~np.isnan(cellids)
            if np.any(valid_mask):
                # Re-query 2D to get icell2d, then find layer separately
                for i in np.where(valid_mask)[0]:
                    # Re-do the 2D query to get icell2d
                    icell2d_result = self.query_points(
                        np.array([x[i]]), np.array([y[i]])
                    )[0]
                    if not np.isnan(icell2d_result):
                        icell2ds[i] = icell2d_result
                        # Find layer for this cell
                        lay = self._find_layer_for_z(int(icell2d_result), z[i])
                        if lay is not None:
                            lays[i] = lay

            if is_scalar_input:
                lay, icell2d = lays[0], icell2ds[0]
                if not np.isnan(lay) and not np.isnan(icell2d):
                    return int(lay), int(icell2d)
                return lay, icell2d
            else:
                valid_3d = ~np.isnan(lays) & ~np.isnan(icell2ds)
                return (
                    lays.astype(int) if np.all(valid_3d) else lays,
                    icell2ds.astype(int) if np.all(valid_3d) else icell2ds,
                )

        # Simple case: return cellid(s) directly
        if is_scalar_input:
            cellid = cellids[0]
            if not np.isnan(cellid):
                return int(cellid)
            return cellid
        else:
            valid_mask = ~np.isnan(cellids)
            if np.all(valid_mask):
                return cellids.astype(int)
            return cellids
