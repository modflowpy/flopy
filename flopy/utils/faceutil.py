"""
Face utilities for finding shared faces between grid cells
"""

import numpy as np

from ..discretization.grid import Grid


def get_shared_face_indices(mg: Grid, node1: int, node2: int) -> tuple[int, int] | None:
    """
    Find a shared face between two cells and return its vertex indices.

    Parameters
    ----------
    mg : Grid
        Model grid
    node1 : int
        First node number (2D)
    node2 : int
        Second node number (2D)

    Returns
    -------
    tuple or None
        Tuple of two vertex indices representing the shared edge,
        or None if cells don't share an edge
    """

    iverts = mg.iverts
    iv0 = iverts[node1]
    iv1 = iverts[node2]

    # collect first cell's faces
    faces = []
    for ix in range(len(iv0)):
        faces.append(tuple(sorted((iv0[ix - 1], iv0[ix]))))

    # find matching face in second cell
    for ix in range(len(iv1)):
        face = tuple(sorted((iv1[ix - 1], iv1[ix])))
        if face in faces:
            return face

    return None


def get_shared_face(
    mg: Grid, cellid1: tuple[int, ...], cellid2: tuple[int, ...]
) -> list[tuple[float, float]] | None:
    """
    Get the 2D coordinates of the shared face between two cells.

    Parameters
    ----------
    mg : Grid
        Model grid
    cellid1 : tuple of int
        First cell ID
    cellid2 : tuple of int
        Second cell ID

    Returns
    -------
    list or None
        List of two (x, y) tuples representing the shared face endpoints,
        or None if cells don't share a face
    """
    if cellid1 == cellid2:
        raise ValueError("cellid1 and cellid2 must be different")

    node1 = mg.get_node([cellid1], node2d=True)[0]
    node2 = mg.get_node([cellid2], node2d=True)[0]
    if face := get_shared_face_indices(mg, node1, node2):
        return [tuple(mg.verts[face[0]]), tuple(mg.verts[face[1]])]
    return None


def get_shared_face_3d(
    mg: Grid, cellid1: tuple[int, ...], cellid2: tuple[int, ...]
) -> list[tuple[float, float, float]] | None:
    """
    Get the 3D coordinates of the shared face between two cells.

    Parameters
    ----------
    mg : Grid
        Model grid
    cellid1 : tuple
        First cell ID
    cellid2 : tuple
        Second cell ID

    Returns
    -------
    list or None
        List of (x, y, z) tuples representing the shared face vertices,
        or None if cells don't share a face
    """
    if cellid1 == cellid2:
        raise ValueError("cellid1 and cellid2 must be different")

    if not mg.is_valid:
        raise ValueError("Grid is not valid")

    # For DISU grids with explicit 3D vertices
    if mg.grid_type == "unstructured":
        node1 = cellid1[0]
        node2 = cellid2[0]
        face = set(mg.iverts[node1]) & set(mg.iverts[node2])
        if len(face) < 2:
            return None
        # Check if vertices are in list format [idx, x, y, z] or record array format
        if isinstance(mg._vertices, list):
            # _vertices format is [idx, x, y, z], so skip the first element
            return [tuple(mg._vertices[idx][1:]) for idx in face]
        else:
            # _vertices is a numpy record array with only 2D coordinates (no z)
            # Cannot construct 3D faces without explicit 3D vertex data
            return None

    # Find shared lateral face
    node1 = mg.get_node([cellid1], node2d=True)[0]
    node2 = mg.get_node([cellid2], node2d=True)[0]
    shared_lat_face = get_shared_face_indices(mg, node1, node2)
    if shared_lat_face is None:
        return None

    verts = mg.verts
    is_vert = is_vertical(mg, cellid1, cellid2)

    if not is_vert:
        # horizontal face between layers
        if len(cellid1) == 3:
            lower_layer = max(cellid1[0], cellid2[0])
            row, col = cellid1[1], cellid1[2]
            z = mg.botm[lower_layer - 1, row, col]
        elif len(cellid1) == 2:
            lower_layer = max(cellid1[0], cellid2[0])
            cell2d = cellid1[1]
            if mg.botm.ndim == 1:
                z = mg.botm[cell2d]
            else:
                z = mg.botm[lower_layer - 1, cell2d]
        else:
            return None

        return [
            (verts[shared_lat_face[0]][0], verts[shared_lat_face[0]][1], z),
            (verts[shared_lat_face[1]][0], verts[shared_lat_face[1]][1], z),
        ]
    else:
        # vertical face between laterally adjacent cells
        if len(cellid1) == 3:
            layer = cellid1[0]
            row1, col1 = cellid1[1], cellid1[2]
            row2, col2 = cellid2[1], cellid2[2]
            if layer == 0:
                z_top = min(mg.top[row1, col1], mg.top[row2, col2])
            else:
                z_top = min(
                    mg.botm[layer - 1, row1, col1], mg.botm[layer - 1, row2, col2]
                )
            z_bot = max(mg.botm[layer, row1, col1], mg.botm[layer, row2, col2])

        elif len(cellid1) == 2:
            layer = cellid1[0]
            cell2d1 = cellid1[1]
            cell2d2 = cellid2[1]
            if layer == 0:
                z_top = min(mg.top[cell2d1], mg.top[cell2d2])
            else:
                if mg.botm.ndim == 1:
                    z_top = mg.top[cell2d1]
                else:
                    z_top = min(
                        mg.botm[layer - 1, cell2d1], mg.botm[layer - 1, cell2d2]
                    )
            if mg.botm.ndim == 1:
                z_bot = max(mg.botm[cell2d1], mg.botm[cell2d2])
            else:
                z_bot = max(mg.botm[layer, cell2d1], mg.botm[layer, cell2d2])
        else:
            return None

        return [
            (verts[shared_lat_face[0]][0], verts[shared_lat_face[0]][1], z_top),
            (verts[shared_lat_face[1]][0], verts[shared_lat_face[1]][1], z_top),
            (verts[shared_lat_face[1]][0], verts[shared_lat_face[1]][1], z_bot),
            (verts[shared_lat_face[0]][0], verts[shared_lat_face[0]][1], z_bot),
        ]


def is_vertical(mg: Grid, cellid1: tuple[int, ...], cellid2: tuple[int, ...]) -> bool:
    """
    Determine if a 3D face is horizontal (between vertically stacked cells) or
    vertical (between laterally adjacent cells).

    For structured (DIS) and vertex (DISV) grids, uses cellid structure to
    determine orientation. For unstructured (DISU) grids, prefers to use the
    ihc (horizontal connection indicator) array if available, falling back to
    geometric analysis of shared face z-coordinates.

    Parameters
    ----------
    mg : Grid
        Model grid
    cellid1 : tuple
        First cell ID
    cellid2 : tuple
        Second cell ID

    Returns
    -------
    bool
        True if face is vertical (laterally adjacent cells),
        False if horizontal (vertically stacked cells)
    """
    if len(cellid1) == 3:
        # Structured grid: (layer, row, col)
        # Vertical face if same layer but different row or col
        return cellid1[0] == cellid2[0] and (
            cellid1[1] != cellid2[1] or cellid1[2] != cellid2[2]
        )
    elif len(cellid1) == 2:
        # Vertex grid: (layer, cell2d)
        # Vertical face if same layer but different cell2d
        return cellid1[0] == cellid2[0] and cellid1[1] != cellid2[1]
    else:
        # Unstructured grid: (node,)
        if mg.grid_type != "unstructured":
            raise ValueError("Given 1-element node number, expected unstructured grid")

        # Use connectivity data (ihc) to determine face orientation
        if mg.ihc is None or mg.iac is None:
            raise ValueError(
                "Unstructured grid must have ihc (horizontal connection indicator) "
                "and iac (connectivity) data to determine face orientation"
            )

        node1 = cellid1[0]
        node2 = cellid2[0]
        ja = mg.ja
        ihc = mg.ihc
        iac = mg.iac

        # Calculate cumulative indices (iac may contain counts, not cumulative sums)
        # First, check if iac is already cumulative or if it's counts
        if iac[0] == 0:
            # Already cumulative, use directly
            iac_cumulative = iac
        else:
            # Contains counts, compute cumulative sum
            import numpy as np

            iac_cumulative = np.concatenate([[0], np.cumsum(iac)])

        # Check connections from node1 to node2
        if node1 < len(iac):
            idx0 = iac_cumulative[node1]
            idx1 = iac_cumulative[node1 + 1]
            for i in range(idx0 + 1, idx1):  # Skip first entry (diagonal)
                if ja[i] == node2:
                    # ihc == 0 means vertical connection
                    # ihc != 0 means horizontal connection
                    # Vertical face = horizontal connection (ihc != 0)
                    return ihc[i] != 0

        # Check connections from node2 to node1
        if node2 < len(iac):
            idx0 = iac_cumulative[node2]
            idx1 = iac_cumulative[node2 + 1]
            for i in range(idx0 + 1, idx1):  # Skip first entry (diagonal)
                if ja[i] == node1:
                    # ihc == 0 means vertical connection
                    # ihc != 0 means horizontal connection
                    # Vertical face = horizontal connection (ihc != 0)
                    return ihc[i] != 0

        # Connection not found between cells
        raise ValueError(f"No connection found between cells {cellid1} and {cellid2}")


def hfb_data_to_linework(
    recarray: np.recarray, modelgrid: Grid
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Convert HFB barrier data to line segments representing shared cell faces.

    Parameters
    ----------
    recarray : np.recarray
        recarray of hfb input data
    modelgrid : Grid
        modelgrid instance

    Returns
    -------
    list
        list of line segments, each as a tuple of two (x, y) coordinate tuples
    """
    verts = modelgrid.verts
    nodes = []
    if modelgrid.grid_type == "structured":
        if "k" in recarray.dtype.names:
            for rec in recarray:
                node1 = modelgrid.get_node([(0, rec["irow1"], rec["icol1"])])[0]
                node2 = modelgrid.get_node([(0, rec["irow2"], rec["icol2"])])[0]
                nodes.append((node1, node2))
        else:
            for rec in recarray:
                node1 = modelgrid.get_node([(0,) + rec["cellid1"][1:]])[0]
                node2 = modelgrid.get_node([(0,) + rec["cellid2"][1:]])[0]
                nodes.append((node1, node2))

    elif modelgrid.grid_type == "vertex":
        for rec in recarray:
            nodes.append((rec["cellid1"][-1], rec["cellid2"][-1]))

    else:
        if "node1" in recarray.dtype.names:
            nodes = list(zip(recarray["node1"], recarray["node2"]))
        else:
            for rec in recarray:
                nodes.append((rec["cellid1"][0], rec["cellid2"][0]))

    shared_faces = []
    for node0, node1 in nodes:
        face = get_shared_face_indices(modelgrid, node0, node1)
        if face is None:
            raise AssertionError(
                f"No shared cell faces found. Cannot represent HFB "
                f"for nodes {node0} and {node1}"
            )
        shared_faces.append(face)

    lines = []
    for face in shared_faces:
        lines.append((tuple(verts[face[0]]), tuple(verts[face[1]])))

    return lines
