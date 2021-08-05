import numpy as np


def get_lak_connections(modelgrid, lake_map, idomain=None, bedleak=0.1):
    """
    Function to create lake package connection data from a zero-based
    integer array of lake numbers. If the shape of lake number array is
    equal to (nrow, ncol) or (ncpl) then the lakes are on top of the model
    and are vertically connected to cells at the top of the model. Otherwise
    the lakes are embedded in the grid.

    TODO: implement embedded lakes for VertexGrid

    TODO: add support for UnstructuredGrid

    Parameters
    ----------
    modelgrid : StructuredGrid, VertexGrid
        model grid
    lake_map : MaskedArray, ndarray, list, tuple
        location and zero-based lake number for lakes in the model domain.
        If lake_map is of size (nrow, ncol) or (ncpl) lakes are located on
        top of the model and vertically connected to cells in model layer 1.
        If lake_map is of size (nlay, nrow, ncol) or (nlay, ncpl) lakes
        are embedded in the model domain and horizontal and vertical lake
        connections are defined.
    idomain : int or ndarray
        location of inactive cells, which are defined with a zero value. If a
        ndarray is passed it must be of size (nlay, nrow, ncol) or
        (nlay, ncpl).
    bedleak : ndarray, list, tuple, float
        bed leakance for lakes in the model domain. If bedleak is a float the
        same bed leakance is applied to each lake connection in the model.
        If bedleak is of size (nrow, ncol) or (ncpl) then all lake
        connections for the cellid are given the same bed leakance value.

    Returns
    -------
    idomain : ndarry
        idomain adjusted to inactivate cells with lakes
    connection_dict : dict
        dictionary with the zero-based lake number keys and number of
        connections in a lake values
    connectiondata : list of lists
        connectiondata block for the lake package

    """

    if modelgrid.grid_type in ("unstructured",):
        raise ValueError(
            "unstructured grids not supported in get_lak_connections()"
        )

    embedded = True
    shape3d = modelgrid.shape
    shape2d = shape3d[1:]

    # convert to numpy array if necessary
    if isinstance(lake_map, (list, tuple)):
        lake_map = np.array(lake_map, dtype=np.int32)
    elif isinstance(lake_map, (int, float)):
        raise TypeError(
            "lake_map must be a Masked Array, ndarray, list, or tuple"
        )

    # evaluate lake_map shape
    shape_map = lake_map.shape
    if shape_map != shape3d:
        if shape_map != shape2d:
            raise ValueError(
                "lake_map shape ({}) must be equal to the grid shape for "
                "each layer ({})".format(shape_map, shape2d)
            )
        else:
            embedded = False

    # process idomain
    if idomain is None:
        idomain = np.ones(shape3d, dtype=np.int32)
    elif isinstance(idomain, int):
        idomain = np.ones(shape3d, dtype=np.int32) * idomain
    elif isinstance(idomain, (float, bool)):
        raise ValueError("idomain must be a integer")

    # check dimensions of idomain
    if idomain.shape != shape3d:
        raise ValueError(
            "shape of idomain "
            "({}) not equal to {}".format(idomain.shape, shape3d)
        )

    # convert bedleak to numpy array if necessary
    if isinstance(bedleak, (float, int)):
        bedleak = np.ones(shape2d, dtype=float) * float(bedleak)
    elif isinstance(bedleak, (list, tuple)):
        bedleak = np.array(bedleak, dtype=float)

    # get the model grid elevations and reset lake_map using idomain
    # if lake is embedded and in an inactive cell
    if embedded:
        elevations = modelgrid.top_botm
        lake_map[idomain < 1] = -1
    else:
        elevations = None

    # determine if masked array, in not convert to masked array
    if not np.ma.is_masked(lake_map):
        lake_map = np.ma.masked_where(lake_map < 0, lake_map)

    connection_dict = {}
    connectiondata = []

    # find unique lake numbers
    unique = np.unique(lake_map)

    # exclude lakes with lake numbers less than 0
    idx = np.where(unique > -1)
    unique = unique[idx]

    dx, dy = None, None

    # embedded lakes
    for lake_number in unique:
        iconn = 0
        indices = np.argwhere(lake_map == lake_number)
        for index in indices:
            cell_index = tuple(index.tolist())
            if embedded:
                leak_value = bedleak[cell_index[1:]]
                if modelgrid.grid_type == "structured":
                    if dx is None:
                        xv, yv = modelgrid.xvertices, modelgrid.yvertices
                        dx = xv[0, 1:] - xv[0, :-1]
                        dy = yv[:-1, 0] - yv[1:, 0]
                    (
                        cellids,
                        claktypes,
                        belevs,
                        televs,
                        connlens,
                        connwidths,
                    ) = __structured_lake_connections(
                        lake_map, idomain, cell_index, dx, dy, elevations
                    )
                elif modelgrid.grid_type == "vertex":
                    raise NotImplementedError(
                        "embedded lakes have not been implemented"
                    )
            else:
                cellid = (0,) + cell_index
                leak_value = bedleak[cell_index]
                if idomain[cellid] > 0:
                    cellids = [cellid]
                    claktypes = ["vertical"]
                    belevs = [0.0]
                    televs = [0.0]
                    connlens = [0.0]
                    connwidths = [0.0]
                else:
                    cellids = []
                    claktypes = []
                    belevs = []
                    televs = []
                    connlens = []
                    connwidths = []

            # iterate through each cellid
            for (cellid, claktype, belev, telev, connlen, connwidth) in zip(
                cellids, claktypes, belevs, televs, connlens, connwidths
            ):
                connectiondata.append(
                    [
                        lake_number,
                        iconn,
                        cellid[:],
                        claktype,
                        leak_value,
                        belev,
                        telev,
                        connlen,
                        connwidth,
                    ]
                )
                iconn += 1

        # set number of connections for lake
        connection_dict[lake_number] = iconn

        # reset idomain for lake
        if iconn > 0:
            idx = np.where((lake_map == lake_number) & (idomain > 0))
            idomain[idx] = 0

    return idomain, connection_dict, connectiondata


def __structured_lake_connections(
    lake_map, idomain, cell_index, dx, dy, elevations
):
    nlay, nrow, ncol = lake_map.shape
    cellids = []
    claktypes = []
    belevs = []
    televs = []
    connlens = []
    connwidths = []

    k, i, j = cell_index

    if idomain[cell_index] > 0:
        # back face
        if i > 0:
            ci = (k, i - 1, j)
            cit = (k + 1, i - 1, j)
            if np.ma.is_masked(lake_map[ci]) and idomain[ci] > 0:
                cellids.append(ci)
                claktypes.append("horizontal")
                belevs.append(elevations[cit])
                televs.append(elevations[ci])
                connlens.append(0.5 * dy[i - 1])
                connwidths.append(dx[j])

        # left face
        if j > 0:
            ci = (k, i, j - 1)
            cit = (k + 1, i, j - 1)
            if np.ma.is_masked(lake_map[ci]) and idomain[ci] > 0:
                cellids.append(ci)
                claktypes.append("horizontal")
                belevs.append(elevations[cit])
                televs.append(elevations[ci])
                connlens.append(0.5 * dx[j - 1])
                connwidths.append(dy[i])

        # right face
        if j < ncol - 1:
            ci = (k, i, j + 1)
            cit = (k + 1, i, j + 1)
            if np.ma.is_masked(lake_map[ci]) and idomain[ci] > 0:
                cellids.append(ci)
                claktypes.append("horizontal")
                belevs.append(elevations[cit])
                televs.append(elevations[ci])
                connlens.append(0.5 * dx[j + 1])
                connwidths.append(dy[i])

        # front face
        if i < nrow - 1:
            ci = (k, i + 1, j)
            cit = (k + 1, i + 1, j)
            if np.ma.is_masked(lake_map[ci]) and idomain[ci] > 0:
                cellids.append(ci)
                claktypes.append("horizontal")
                belevs.append(elevations[cit])
                televs.append(elevations[ci])
                connlens.append(0.5 * dy[i + 1])
                connwidths.append(dx[j])

        # lower face
        if k < nlay - 1:
            ci = (k + 1, i, j)
            if np.ma.is_masked(lake_map[ci]) and idomain[ci] > 0:
                cellids.append(ci)
                claktypes.append("vertical")
                belevs.append(0.0)
                televs.append(0.0)
                connlens.append(0.0)
                connwidths.append(0.0)

    return cellids, claktypes, belevs, televs, connlens, connwidths
