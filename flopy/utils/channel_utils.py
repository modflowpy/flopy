from typing import Optional, Union

import numpy as np

# power for Manning's hydraulic radius term
_mpow = 2.0 / 3.0


def get_segment_wetted_station(
    x0,
    x1,
    h0,
    h1,
    depth,
):
    """
    Computes the start and end stations of the wetted portion of a channel segment.
    If the segment is fully submerged the start and end stations are the segment's
    x-coordinates, unchanged. If the segment is partially submerged, the start/end
    x-coordinates of the wetted subsegment are obtained using the segment's slope.
    If the segment is wholly above the water surface, the end coordinate is set to
    the value of the start coordinate (zero length in effect discards the segment).

    Parameters
    ----------
    x0: x-coordinate of the segment's first point
    x1: x-coordinate of the segment's second point
    h0: elevation of the segment's first point
    h1: elevation of the segment's second point
    depth: elevation of the channel water surface

    Returns
    -------
        The start and end stations (x-coordinates) of the wetted subsegment.
    """

    # calculate the minimum and maximum segment endpoint elevation
    hmin = min(h0, h1)
    hmax = max(h0, h1)

    # if the water surface elevation is less than or equal to the
    # minimum segment endpoint elevation, station length is zero.
    if depth <= hmin:
        x1 = x0

    # if water surface is between hmin and hmax, station length is less than x1 - x0
    elif depth < hmax:
        xlen = x1 - x0
        dlen = h1 - h0
        if abs(dlen) > 0.0:
            slope = xlen / dlen
        else:
            slope = 0.0
        if h0 > h1:
            dx = (depth - h1) * slope
            xt = x1 + dx
            xt0 = xt
            xt1 = x1
        else:
            dx = (depth - h0) * slope
            xt = x0 + dx
            xt0 = x0
            xt1 = xt
        x0 = xt0
        x1 = xt1

    return x0, x1


def get_segment_wetted_perimeter(
    x0: float,
    x1: float,
    h0: float,
    h1: float,
    depth: float,
):
    """
    Computes the wetted length of the channel cross-section segment given
    the channel depth. See https://en.wikipedia.org/wiki/Wetted_perimeter.

    Parameters
    ----------
    x0: x-coordinate of the segment's first point
    x1: x-coordinate of the segment's second point
    h0: elevation of the segment's first point
    h1: elevation of the segment's second point
    depth: elevation of the channel water surface

    Returns
    -------
        The wetted length of the segment
    """

    # -- calculate the minimum and maximum elevation
    hmin = min(h0, h1)
    hmax = max(h0, h1)

    if depth <= hmin:
        return 0

    # -- calculate the wetted perimeter for the segment
    xlen = x1 - x0
    if xlen < 0:
        raise ValueError("x1 must be greater than x0")

    if xlen == 0:
        return 0

    if xlen > 0.0:
        if depth > hmax:
            hlen = hmax - hmin
        else:
            hlen = depth - hmin
    else:
        if depth > hmin:
            hlen = min(depth, hmax) - hmin
        else:
            hlen = 0.0

    return np.sqrt(xlen**2.0 + hlen**2.0)


def get_segment_wetted_area(
    x0: float, x1: float, h0: float, h1: float, depth: float
):
    """
    Computes the cross-sectional area above a segment of a channel with the
    given geometry and filled to the given depth. Area above the segment is
    defined as that of the quadrilateral formed by the channel segment, two
    vertical sides, and the water surface.

    Parameters
    ----------
    x0: x-coordinate of the segment's first point
    x1: x-coordinate of the segment's second point
    h0: elevation of the segment's first point
    h1: elevation of the segment's second point
    depth: elevation of the channel water surface

    Returns
    -------
        The wetted area above the segment
    """

    # -- calculate the minimum and maximum elevation
    hmin = min(h0, h1)
    hmax = max(h0, h1)

    if depth <= hmin:
        return 0

    # calculate the wetted area for the segment
    xlen = x1 - x0
    area = 0.0
    if xlen > 0.0:

        # add the area above hmax
        if depth > hmax:
            area += xlen * (depth - hmax)

        # add the area below hmax
        area += (0.5 if hmax != hmin else 1) * (hmax - hmin)

    return area


def get_wetted_area(
    x: np.ndarray,
    h: np.ndarray,
    depth: float,
    verbose=False,
):
    """
    Computes the cross-sectional wetted area of an open channel filled to a given depth.
    The channel's geometry is provided as pairs of coordinates (x, h) defining boundary
    segments. The segments are assumed to be connected in the order provided and arrays
    for x and h must be the same length. See https://en.wikipedia.org/wiki/Wetted_area.

    Parameters
    ----------
    x: x-coordinates of the channel's boundary segments
    h: elevations of the channel's boundary segments
    depth: elevation of the channel water surface
    verbose: if True, print debugging information

    Returns
    -------
        The wetted area of the channel
    """

    area = 0.0
    if x.shape[0] == 1:
        area = x[0] * depth
    else:
        for idx in range(0, x.shape[0] - 1):
            x0, x1 = x[idx], x[idx + 1]
            h0, h1 = h[idx], h[idx + 1]

            # get station data
            x0, x1 = get_segment_wetted_station(x0, x1, h0, h1, depth)

            # get wetted area
            a = get_segment_wetted_area(x0, x1, h0, h1, depth)
            area += a

            if verbose:
                print(
                    f"{idx}->{idx + 1} ({x0},{x1}) - "
                    f"perimeter={x1 - x0} - area={a}"
                )

    return area


def get_wetted_perimeter(
    x: np.ndarray,
    h: np.ndarray,
    depth: float,
    verbose=False,
):
    """
    Computes the cross-sectional wetted perimeter of an open channel filled to a given depth.
    The channel's geometry is provided as pairs of coordinates (x, h) defining segments along
    the channel's boundary. The segments are assumed connected in the order given, and arrays
    x and h must share the same length. See https://en.wikipedia.org/wiki/Wetted_perimeter.

    Parameters
    ----------
    x: x-coordinates of the channel's boundary segments
    h: elevations of the channel's boundary segments
    depth: elevation of the channel water surface
    verbose: if True, print debugging information

    Returns
    -------
        The wetted perimeter of the channel
    """

    perimeter = 0.0
    if x.shape[0] == 1:
        perimeter = x[0]
    else:
        for idx in range(0, x.shape[0] - 1):
            x0, x1 = x[idx], x[idx + 1]
            h0, h1 = h[idx], h[idx + 1]

            # get station data
            x0, x1 = get_segment_wetted_station(x0, x1, h0, h1, depth)

            # get wetted perimeter
            perimeter += get_segment_wetted_perimeter(x0, x1, h0, h1, depth)

            # write to screen
            if verbose:
                print(f"{idx}->{idx + 1} ({x0},{x1}) - perimeter={x1 - x0}")

    return perimeter


def get_discharge_rect(
    width: float,
    depth: float,
    roughness: float,
    slope: float,
    conv: float = 1.0,
):
    """
    Calculates the discharge (volumetric flow rate) for a rectangular channel.

    Parameters
    ----------
    width: the channel's width
    depth: the channel's water depth
    roughness: Manning's roughness coefficient
    slope: the channel's slope
    conv: the conversion factor

    Returns
    -------
        The channel's discharge
    """

    if depth <= 0.0:
        raise ValueError("Depth must be positive.")

    if width <= 0.0:
        raise ValueError("Width must be positive.")

    if roughness <= 0:
        raise ValueError(f"Roughness coefficient must be positive")

    if slope < 0:
        raise ValueError(f"Slope must be non-negative")

    area = width * depth
    perim = 2.0 * depth + width
    radius = area / perim
    # previously depth**_mpow was used below
    return conv * area * radius**_mpow * slope**0.5 / roughness


def get_discharge(
    x: Union[int, float, np.ndarray],
    h: Union[int, float, np.ndarray],
    depth: Optional[float] = None,
    roughness: float = 0.01,
    slope: float = 0.001,
    conv: float = 1.0,
):
    """
    Calculates the discharge according to Manning's equation for a
    channel with the provided cross-sectional geometry, slope, and
    roughness. See https://en.wikipedia.org/wiki/Manning_formula.

    Parameters
    ----------
    x: x-coordinates of the channel's boundary segments
    h: elevations of the channel's boundary segments
    depth: elevation of the channel water surface
    roughness: Manning's roughness coefficient
    slope: the channel's slope
    conv: the conversion factor

    Returns
    -------
        The channel's discharge
    """

    if roughness <= 0:
        raise ValueError(f"Roughness coefficient must be positive")

    if slope < 0:
        raise ValueError(f"Slope must be non-negative")

    if isinstance(x, int) or isinstance(x, float):
        if not (isinstance(h, int) or isinstance(h, float)):
            raise ValueError(
                "Channel width and height must be integers or floats"
            )

        if x <= 0 or h <= 0:
            raise ValueError(f"Channel width and height must be positive")

        if depth is None:
            depth = h

        q = get_discharge_rect(x, depth, roughness, slope, conv)

    elif isinstance(x, np.ndarray):
        if not isinstance(h, np.ndarray):
            raise ValueError("If x is an array, h must be an array")

        if x.shape != h.shape:
            raise ValueError(f"Arrays x and h must be the same shape")

        if len(x) < 2:
            raise ValueError(f"Arrays x and h must have at least 2 elements")

        if isinstance(roughness, float):
            roughness = (
                np.ones(
                    x.shape if isinstance(x, np.ndarray) else (1,), dtype=float
                )
                * roughness
            )

        q = 0.0
        for i0 in range(x.shape[0] - 1):
            in1 = i0 - 1
            i1 = i0 + 1

            perim = get_segment_wetted_perimeter(
                x[i0], x[i1], h[i0], h[i1], depth
            )
            if perim == 0:
                continue

            # if left neighbor is vertical and descending, add its length
            if i0 > 0:
                left_len = x[in1] - x[i0]
            else:
                left_len = None
            if left_len == 0 and h[in1] > h[i0]:
                perim += min(depth, h[in1]) - h[i0]

            # if right neighbor is vertical and ascending, add its length
            if i0 < x.shape[0] - 2:
                right_len = x[i1 + 1] - x[i1]
                if right_len == 0 and h[i1 + 1] > h[i1]:
                    perim += min(depth, h[i1 + 1]) - h[i1]

            area = get_segment_wetted_area(x[i0], x[i1], h[i0], h[i1], depth)
            radius = area / perim
            q += conv * area * radius**_mpow * slope**0.5 / roughness[i0]

    else:
        raise ValueError(f"Invalid type for x: {type(x)}")

    return q


def get_depth(
    x: np.ndarray,
    h: np.ndarray,
    q: float,
    roughness: float = 0.01,
    slope: float = 0.001,
    conv: float = 1.0,
    dd: float = 1e-4,
    max_iter: int = 100,
    verbose=False,
):
    """
    Uses Manning's equation to approximate depth of flow for an open channel
    given its cross-sectional geometry, a roughness coefficient, the rate of
    flow, and the channel slope. An optional conversion factor may be given.
    A single roughness coefficient or distinct coefficients for each segment
    can also be provided. An iterative method is used to solve for the depth
    and the number of iterations can be capped (by default, the max is 100).

    Parameters
    ----------
    x: x-coordinates of the channel's boundary segments
    h: elevations of the channel's boundary segments
    q: the flow rate
    roughness: Manning's roughness coefficient
    slope: the channel's slope
    conv: the conversion factor
    dd: the increment used for approximating the derivative
    verbose: if True, print debugging information

    Returns
    -------
        The depth of flow
    """

    d0 = 0.0
    q0 = get_discharge(
        x,
        h,
        d0,
        roughness=roughness,
        slope=slope,
        conv=conv,
    )
    r = q0 - q

    iter = 0
    if verbose:
        print(f"iteration {iter:>2d} - residual={r}")
    while abs(r) > 1e-12:
        q1 = get_discharge(
            x,
            h,
            d0 + dd,
            roughness=roughness,
            slope=slope,
            conv=conv,
        )
        dq = q1 - q0
        if dq != 0.0:
            derv = dd / (q1 - q0)
        else:
            derv = 0.0
        d0 -= derv * r
        q0 = get_discharge(
            x,
            h,
            d0,
            roughness=roughness,
            slope=slope,
            conv=conv,
        )
        r = q0 - q

        iter += 1
        if verbose:
            print(f"iteration {iter:>2d} - residual={r}")
        if iter > max_iter:
            break
    return d0


def get_depths(
    flows: Union[float, np.ndarray],
    x: np.ndarray,
    h: np.ndarray,
    roughness: Union[float, np.ndarray] = 0.01,
    slope: float = 0.001,
    conv: float = 1.0,
    dd: float = 1e-4,
):
    """
    Uses Manning's equation to approximate depth of flow for an open channel
    given its cross-sectional geometry, a roughness coefficient, one or more
    flow rates, and the channel's slope. An optional conversion factor can be
    provided, which is used to convert units of length from metric system to
    another measurement system (e.g. for English units specify factor 1.49).
    A single flow rate can be provided, or an array of flow rates. An array
    is returned containing the depth of flow for each provided flow rate, in
    the order given. A single roughness coefficient can be provided, or else
    distinct values for each segment of the channel boundary can be provided.

    Parameters
    ----------
    flows: the flow rate(s) to calculate depths for
    x: x-coordinates of the channel's boundary segments
    h: elevations of the channel's boundary segments
    roughness: Manning's roughness coefficient
    slope: the channel's slope
    conv: the conversion factor for converting from metric to another unit
    dd: the increment used for approximating the derivative

    Returns
    -------
        An array of depths of flow (one element for each provided in flows)
    """

    if isinstance(flows, float):
        flows = np.array([flows], dtype=float)
    if isinstance(roughness, float):
        roughness = np.ones(x.shape, dtype=float) * roughness
    depths = np.zeros(flows.shape, dtype=float)
    for idx, q in enumerate(flows):
        depths[idx] = get_depth(
            x,
            h,
            q,
            roughness=roughness,
            slope=slope,
            conv=conv,
            dd=dd,
            verbose=False,
        )

    return depths
