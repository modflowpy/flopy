import numpy as np

from .binarygrid_util import MfGrdFile


def get_structured_connectivity(nlay, nrow, ncol, idomain=None):
    """
    Build IA and JA connectivity arrays for a structured (DIS) grid.

    Parameters
    ----------
    nlay : int
        Number of layers
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    idomain : np.ndarray, optional
        Domain array indicating active (>0) and inactive (<=0) cells.
        Shape: (nlay, nrow, ncol). If None, all cells are active.

    Returns
    -------
    ia : np.ndarray
        Index array (CSR format), shape (ncells + 1,), dtype int32.
        ia[n] is the starting position in ja for cell n's connections.
        ia[ncells] is the total number of connections.
    ja : np.ndarray
        Connection array (CSR format), shape (nja,), dtype int32.
        Contains cell numbers for each connection (0-based).
    nja : int
        Total number of connections

    Notes
    -----
    Connectivity order for structured grids (upper triangle only):
    1. Diagonal (self connection)
    2. Right (+1 in j, same k, i)
    3. Front (+1 in i, same k, j)
    4. Lower (+1 in k, same i, j)

    The IA/JA arrays use 0-based indexing (Python convention).
    When writing to MF6 binary files, add 1 to convert to Fortran 1-based indexing.

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.mf6.utils import build_structured_connectivity
    >>> nlay, nrow, ncol = 2, 3, 3
    >>> ia, ja, nja = build_structured_connectivity(nlay, nrow, ncol)
    >>> print(f"Total cells: {nlay * nrow * ncol}, connections: {nja}")
    Total cells: 18, connections: 42
    """
    ncells = nlay * nrow * ncol

    # Default to all active cells if idomain not provided
    if idomain is None:
        idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
    else:
        idomain = np.asarray(idomain, dtype=np.int32)
        if idomain.shape != (nlay, nrow, ncol):
            raise ValueError(
                f"idomain shape {idomain.shape} does not match grid shape "
                f"({nlay}, {nrow}, {ncol})"
            )

    ia = np.zeros(ncells + 1, dtype=np.int32)
    ja_list = []
    nja = 0

    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j

                # Skip inactive cells - they still get an entry in IA
                if idomain[k, i, j] <= 0:
                    ia[node + 1] = nja
                    continue

                # Add diagonal (self connection)
                ja_list.append(node)
                nja += 1

                # Add connections to neighbors (upper triangle only)
                # Right neighbor (j+1)
                if j + 1 < ncol and idomain[k, i, j + 1] > 0:
                    m = k * nrow * ncol + i * ncol + (j + 1)
                    ja_list.append(m)
                    nja += 1

                # Front neighbor (i+1)
                if i + 1 < nrow and idomain[k, i + 1, j] > 0:
                    m = k * nrow * ncol + (i + 1) * ncol + j
                    ja_list.append(m)
                    nja += 1

                # Lower neighbor (k+1)
                if k + 1 < nlay and idomain[k + 1, i, j] > 0:
                    m = (k + 1) * nrow * ncol + i * ncol + j
                    ja_list.append(m)
                    nja += 1

                ia[node + 1] = nja

    ja = np.array(ja_list, dtype=np.int32)

    return ia, ja, nja


def get_structured_faceflows(
    flowja,
    grb_file=None,
    ia=None,
    ja=None,
    nlay=None,
    nrow=None,
    ncol=None,
    verbose=False,
):
    """
    Get the face flows for the flow right face, flow front face, and
    flow lower face from the MODFLOW 6 flowja flows. This method can
    be useful for building face flow arrays for MT3DMS, MT3D-USGS, and
    RT3D. This method only works for a structured MODFLOW 6 model.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    ia : list or ndarray
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray
        CRS column pointers. Only required if grb_file is not provided.
    nlay : int
        number of layers in the grid. Only required if grb_file is not provided.
    nrow : int
        number of rows in the grid. Only required if grb_file is not provided.
    ncol : int
        number of columns in the grid. Only required if grb_file is not provided.
    verbose: bool
        Write information to standard output

    Returns
    -------
    frf : ndarray
        right face flows
    fff : ndarray
        front face flows
    flf : ndarray
        lower face flows

    """
    if grb_file is not None:
        grb = MfGrdFile(grb_file, verbose=verbose)
        if grb.grid_type != "DIS":
            raise ValueError(
                "get_structured_faceflows method is only for structured DIS grids"
            )
        ia, ja = grb.ia, grb.ja
        nlay, nrow, ncol = grb.nlay, grb.nrow, grb.ncol
    else:
        if ia is None or ja is None or nlay is None or nrow is None or ncol is None:
            raise ValueError(
                "ia, ja, nlay, nrow, and ncol must be"
                "specified if a MODFLOW 6 binary grid"
                "file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = flowja.flatten()

    # evaluate size of flowja relative to ja
    _check_flowja_size(flowja, ja)

    # create empty flat face flow arrays
    shape = (nlay, nrow, ncol)
    frf = np.zeros(shape, dtype=float).flatten()  # right
    fff = np.zeros(shape, dtype=float).flatten()  # front
    flf = np.zeros(shape, dtype=float).flatten()  # lower

    def get_face(m, n, nlay, nrow, ncol):
        """
        Determine connection direction at (m, n)
        in a connection or intercell flow matrix.

        Notes
        -----
        For visual intuition in 2 dimensions
        https://stackoverflow.com/a/16330162/6514033
        helps. MODFLOW uses the left-side scheme in 3D.

        Parameters
        ----------
        m : int
            row index
        n : int
            column index
        nlay : int
            number of layers in the grid
        nrow : int
            number of rows in the grid
        ncol : int
            number of columns in the grid

        Returns
        -------
        face : int
            0: right, 1: front, 2: lower
        """

        d = m - n
        if d == 1:
            # handle 1D cases
            if nrow == 1 and ncol == 1:
                return 2
            elif nlay == 1 and ncol == 1:
                return 1
            elif nlay == 1 and nrow == 1:
                return 0
            else:
                # handle 2D layers/rows case
                return 1 if ncol == 1 else 0
        elif d % (nrow * ncol) == 0:
            return 2
        else:
            return 1

    # fill right, front and lower face flows
    # (below main diagonal)
    flows = [frf, fff, flf]
    nodes = nlay * nrow * ncol
    for n in range(nodes):
        for i in range(ia[n] + 1, ia[n + 1]):
            m = ja[i]
            if m <= n:
                continue
            face = get_face(m, n, nlay, nrow, ncol)
            flows[face][n] = -1 * flowja[i]

    # reshape and return
    return frf.reshape(shape), fff.reshape(shape), flf.reshape(shape)


def get_structured_flowja(
    faceflows,
    grb_file=None,
    ia=None,
    ja=None,
    nlay=None,
    nrow=None,
    ncol=None,
    idomain=None,
    verbose=False,
):
    """
    Get connection flows (flowja) from face flows for a structured grid.

    This is the inverse of get_structured_faceflows(). Converts MODFLOW-2005/NWT
    style face flows (flow right face, flow front face, flow lower face) to
    MODFLOW 6 style connection flows for the FLOW-JA-FACE budget term.

    Parameters
    ----------
    faceflows : tuple of ndarray
        Tuple of (frf, fff, flf) where:
        - frf : flow right face, shape (nlay, nrow, ncol)
        - fff : flow front face, shape (nlay, nrow, ncol)
        - flf : flow lower face, shape (nlay, nrow, ncol)
    grb_file : str, optional
        MODFLOW 6 binary grid file path
    ia : list or ndarray, optional
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray, optional
        CRS column pointers. Only required if grb_file is not provided.
    nlay : int, optional
        Number of layers. Only required if grb_file is not provided.
    nrow : int, optional
        Number of rows. Only required if grb_file is not provided.
    ncol : int, optional
        Number of columns. Only required if grb_file is not provided.
    idomain : ndarray, optional
        Domain array, shape (nlay, nrow, ncol)
    verbose : bool, optional
        Write information to standard output (default False)

    Returns
    -------
    flowja : ndarray
        Connection flows, size (nja,)

    See Also
    --------
    get_structured_faceflows : Inverse operation (flowja to face flows)

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.mf6.utils import get_structured_flowja
    >>> nlay, nrow, ncol = 1, 3, 3
    >>> frf = np.ones((nlay, nrow, ncol)) * 1.0
    >>> fff = np.ones((nlay, nrow, ncol)) * 2.0
    >>> flf = np.ones((nlay, nrow, ncol)) * 3.0
    >>> flowja = get_structured_flowja((frf, fff, flf),
    ...     nlay=nlay, nrow=nrow, ncol=ncol)
    """
    # Unpack face flows
    qright, qfront, qlower = faceflows

    # Get grid information
    if grb_file is not None:
        grb = MfGrdFile(grb_file, verbose=verbose)
        if grb.grid_type != "DIS":
            raise ValueError(
                "get_structured_flowja method is only for structured DIS grids"
            )
        ia, ja = grb.ia, grb.ja
        nlay, nrow, ncol = grb.nlay, grb.nrow, grb.ncol
    else:
        if ia is None or ja is None or nlay is None or nrow is None or ncol is None:
            raise ValueError(
                "ia, ja, nlay, nrow, and ncol must be specified if grb_file is not provided"
            )

    # Validate input shapes
    expected_shape = (nlay, nrow, ncol)
    for name, arr in [("qright", qright), ("qfront", qfront), ("qlower", qlower)]:
        arr = np.asarray(arr)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match grid shape {expected_shape}"
            )

    # Convert to arrays
    qright = np.asarray(qright, dtype=np.float64)
    qfront = np.asarray(qfront, dtype=np.float64)
    qlower = np.asarray(qlower, dtype=np.float64)

    # Default to all active if idomain not provided
    if idomain is None:
        idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
    else:
        idomain = np.asarray(idomain, dtype=np.int32)

    ncells = nlay * nrow * ncol
    nja = len(ja)
    flowja = np.zeros(nja, dtype=np.float64)

    for n in range(ncells):
        # Skip inactive cells
        k, i, j = np.unravel_index(n, (nlay, nrow, ncol))
        if idomain[k, i, j] <= 0:
            continue

        # Get connections for this cell
        istart = ia[n]
        iend = ia[n + 1]

        for ipos in range(istart, iend):
            m = ja[ipos]

            # Diagonal - no self flow
            if m == n:
                flowja[ipos] = 0.0
                continue

            # Determine connection type by comparing node numbers
            km, im, jm = np.unravel_index(m, (nlay, nrow, ncol))

            # Right connection (j increases by 1)
            if km == k and im == i and jm == j + 1:
                flowja[ipos] = qright[k, i, j]

            # Front connection (i increases by 1)
            elif km == k and im == i + 1 and jm == j:
                flowja[ipos] = qfront[k, i, j]

            # Lower connection (k increases by 1)
            elif km == k + 1 and im == i and jm == j:
                flowja[ipos] = qlower[k, i, j]

    return flowja


def get_residuals(flowja, grb_file=None, ia=None, ja=None, shape=None, verbose=False):
    """
    Get the residual from the MODFLOW 6 flowja flows. The residual is stored
    in the diagonal position of the flowja vector.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    ia : list or ndarray
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray
        CRS column pointers. Only required if grb_file is not provided.
    shape : tuple
        shape of returned residual. A flat array is returned if shape is None
        and grbfile is None.
    verbose: bool
        Write information to standard output

    Returns
    -------
    residual : ndarray
        Residual for each cell

    """
    if grb_file is not None:
        grb = MfGrdFile(grb_file, verbose=verbose)
        shape = grb.shape
        ia, ja = grb.ia, grb.ja
    else:
        if ia is None or ja is None:
            raise ValueError(
                "ia and ja arrays must be specified if the MODFLOW 6 "
                "binary grid file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = flowja.flatten()

    # evaluate size of flowja relative to ja
    _check_flowja_size(flowja, ja)

    # create residual
    nodes = grb.nodes
    residual = np.zeros(nodes, dtype=float)

    # fill flow terms
    for n in range(nodes):
        i0, i1 = ia[n], ia[n + 1]
        if i0 < i1:
            residual[n] = flowja[i0]
        else:
            residual[n] = np.nan

    # reshape residual terms
    if shape is not None:
        residual = residual.reshape(shape)
    return residual


def _check_flowja_size(flowja, ja):
    """
    Check the shape of flowja relative to ja.
    """
    if flowja.shape != ja.shape:
        raise ValueError(f"size of flowja ({flowja.shape}) not equal to {ja.shape}")
