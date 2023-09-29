import numpy as np

from .binarygrid_util import MfGrdFile


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
                "get_structured_faceflows method "
                "is only for structured DIS grids"
            )
        ia, ja = grb.ia, grb.ja
        nlay, nrow, ncol = grb.nlay, grb.nrow, grb.ncol
    else:
        if (
            ia is None
            or ja is None
            or nlay is None
            or nrow is None
            or ncol is None
        ):
            raise ValueError(
                "ia, ja, nlay, nrow, and ncol must be"
                "specified if a MODFLOW 6 binary grid"
                "file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = flowja.flatten()

    # evaluate size of flowja relative to ja
    __check_flowja_size(flowja, ja)

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
        elif d == nrow * ncol:
            return 2
        else:
            return 1

    # fill right, front and lower face flows
    # (below main diagonal)
    flows = [frf, fff, flf]
    for n in range(grb.nodes):
        for i in range(ia[n] + 1, ia[n + 1]):
            m = ja[i]
            if m <= n:
                continue
            face = get_face(m, n, nlay, nrow, ncol)
            flows[face][n] = -1 * flowja[i]

    # reshape and return
    return frf.reshape(shape), fff.reshape(shape), flf.reshape(shape)


def get_residuals(
    flowja, grb_file=None, ia=None, ja=None, shape=None, verbose=False
):
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
    __check_flowja_size(flowja, ja)

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


# internal
def __check_flowja_size(flowja, ja):
    """
    Check the shape of flowja relative to ja.
    """
    if flowja.shape != ja.shape:
        raise ValueError(
            f"size of flowja ({flowja.shape}) not equal to {ja.shape}"
        )
