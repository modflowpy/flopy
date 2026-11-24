import math

import numpy as np
import pytest

from flopy.utils.channel_utils import (
    get_depth,
    get_discharge,
    get_discharge_rect,
    get_segment_wetted_area,
    get_segment_wetted_perimeter,
    get_segment_wetted_station,
    get_wetted_area,
    get_wetted_perimeter,
)


def test_get_segment_wetted_station_all_dry():
    depth = 10
    p0 = (0, 12)
    p1 = (10, 11)
    x0, x1 = get_segment_wetted_station(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )
    # zero-length segment at x0
    assert x0 == x1 == p0[0]


def test_get_segment_wetted_station_partial():
    depth = 10

    # left bank (sloping downwards to the right)
    p0 = (0, 12)
    p1 = (10, 8)
    x0, x1 = get_segment_wetted_station(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )
    # left endpoint should be moved to the right
    assert x0 != x1
    assert x0 != p0[0]
    assert x1 == p1[0]
    assert x0 == (p1[0] - p0[0]) / 2

    # right bank (sloping upwards to the right)
    p0 = (0, 8)
    p1 = (10, 12)
    x0, x1 = get_segment_wetted_station(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )
    # right endpoint should be moved to the left
    assert x0 != x1
    assert x0 == p0[0]
    assert x1 != p1[0]
    assert x1 == (p1[0] - p0[0]) / 2


def test_get_segment_wetted_station_submerged():
    depth = 13
    p0 = (0, 12)
    p1 = (10, 11)
    x0, x1 = get_segment_wetted_station(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )
    # entire segment should be returned
    assert x0 == p0[0]
    assert x1 == p1[0]


def test_get_segment_wetted_perimeter_dry():
    depth = 10
    p0 = (0, 12)
    p1 = (10, 11)
    perim = get_segment_wetted_perimeter(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )
    assert perim == 0


point_lists = [
    # single segments
    [(0, -1), (1, -1)],
    [(0, 0), (1, 1)],
    [(0, 1), (2, -1)],
    [(0, -1), (2, 1)],
    [(0, 1), (1, 0)],
    # channels with multiple segments
    [(0, -1), (1, -1), (2, -1)],  # flat
    [(0, 0), (1, -1), (2, 0)],  # triangular
    [(0, -1), (1, -2), (2, -2), (3, -1)],  # trapezoidal
    [(0, -1), (1, -2), (2, -4), (3, -4), (4, -1)],  # complex
]


@pytest.mark.parametrize(
    "depth, p0, p1",
    [
        (0, point_lists[0][0], point_lists[0][1]),
        (0, point_lists[1][0], point_lists[1][1]),
        (0, point_lists[2][0], point_lists[2][1]),
        (3, point_lists[3][0], point_lists[3][1]),
    ],
)
def test_get_segment_wetted_perimeter(depth, p0, p1):
    wp = get_segment_wetted_perimeter(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )

    xlen = abs(p1[0] - p0[0])
    hlen = abs(p1[1] - p0[1])
    hmax = max([p0[1], p1[1]])
    hmin = min([p0[1], p1[1]])
    seg_len = math.sqrt(hlen**2 + xlen**2)

    # if segment is fully wetted, wetted perimeter is just its length
    if depth >= hmax:
        # expect perimeter 0 if the water surface is level with a flat segment
        if depth == hmin == hmax:
            assert wp == 0
        else:
            assert wp == seg_len

    # if segment is partially submerged, wetted perimeter should be
    # less than the length of the segment but greater than zero
    elif depth > hmin:
        assert wp > 0
        assert wp < seg_len

    # if segment is completely dry, wetted perimeter should be zero
    else:
        assert wp == 0


@pytest.mark.parametrize(
    "depth, p0, p1",
    [
        (0, point_lists[0][0], point_lists[0][1]),
        (0, point_lists[1][0], point_lists[1][1]),
        (0, point_lists[2][0], point_lists[2][1]),
        (3, point_lists[3][0], point_lists[3][1]),
    ],
)
def test_get_segment_wetted_area(depth, p0, p1):
    wa = get_segment_wetted_area(
        x0=p0[0], x1=p1[0], h0=p0[1], h1=p1[1], depth=depth
    )

    xlen = abs(p1[0] - p0[0])
    hlen = abs(p1[1] - p0[1])
    hmax = max([p0[1], p1[1]])
    hmin = min([p0[1], p1[1]])
    seg_len = math.sqrt(hlen**2 + xlen**2)
    tri_area = 0 if hlen == 0 else (0.5 * hlen * xlen)

    # if segment is submerged wetted area is that of quadrilateral
    # formed by the segment, the water surface, and vertical sides
    if depth > hmax:
        rect_area = xlen * (depth - hmax)
        expected = rect_area + tri_area * (0.5 if p0[1] != p1[1] else 1)
        assert wa == expected

    # if segment is partially submerged, wetted area should be
    # less than the area of the triangle formed by the segment
    # and the water surface, with one vertical side
    elif depth > hmin:
        assert wa > 0
        assert wa < tri_area

    # if segment is completely dry, wetted area should be zero
    else:
        assert wa == 0


def test_get_wetted_perimeter_rectangular():
    depth = 0
    points = np.array([[0, -0.2], [0, -1.4], [1.5, -1.4], [1.5, -0.2]])
    perim = get_wetted_perimeter(points[:, 0], points[:, 1], depth)
    assert perim == 1.5


@pytest.mark.parametrize("points", [np.array(pts) for pts in point_lists])
@pytest.mark.parametrize("depth", [0, 1, -1, -2])
def test_get_wetted_perimeter(depth, points):
    def total_perim(pts):
        return sum(
            [
                math.sqrt(
                    (pts[i][0] - pts[i - 1][0]) ** 2
                    + (pts[i][1] - pts[i - 1][1]) ** 2
                )
                for i in range(1, len(pts))
            ]
        )

    wp = get_wetted_perimeter(
        x=points[:, 0], h=points[:, 1], depth=depth, verbose=True
    )

    hmax = max(points[:, 1])
    hmin = min(points[:, 1])
    total_perim = total_perim(points)

    # if all segments are submerged, wetted perimeter is total perimeter
    if depth >= hmax:
        # expect perimeter 0 if the water surface is level with a flat channel
        if all(p == depth for p in points[:, 1]):
            assert wp == 0
        else:
            assert wp == total_perim

    # if at least some segments are at least partially submerged...
    elif depth > hmin:
        assert wp > 0
        assert wp < total_perim

        def patch(x0, x1, h0, h1, depth):
            # TODO: refactor get_segment_wetted_perimeter() to handle partial wetting
            #  internally so separate get_segment_wetted_station() is no longer needed?

            x0, x1 = get_segment_wetted_station(
                x0=x0, x1=x1, h0=h0, h1=h1, depth=depth
            )
            return get_segment_wetted_perimeter(
                x0=x0, x1=x1, h0=h0, h1=h1, depth=depth
            )

        assert np.isclose(
            wp,
            sum(
                [
                    patch(
                        x0=points[i][0],
                        x1=points[i + 1][0],
                        h0=points[i][1],
                        h1=points[i + 1][1],
                        depth=depth,
                    )
                    for i in range(0, len(points) - 1)
                ]
            ),
        )

    # if all segments are dry, wetted perimeter is zero
    else:
        assert wp == 0


def test_get_wetted_area_rectangular():
    depth = 0
    points = np.array([[0, -0.2], [0, -1.4], [1.5, -1.4], [1.5, -0.2]])
    expected = 1.5 * 1.4
    area = get_wetted_area(points[:, 0], points[:, 1], depth)
    assert area == expected


def test_get_discharge_rect():
    # adapted from https://www.youtube.com/watch?v=R-Vhs_AH8mA

    width = 0.5
    depth = 0.25
    n = 0.022
    s = 0.005
    k = 1.0
    expected = 0.1
    q = get_discharge_rect(width, depth, n, s, k)
    assert np.isclose(expected, q, rtol=1e-2)


def test_get_discharge_rectangular():
    # adapted from https://www.youtube.com/watch?v=R-Vhs_AH8mA

    # x and h as ints (width and height)
    width = 0.5
    depth = 0.25
    n = 0.022
    s = 0.005
    k = 1.0
    expected = 0.1
    q = get_discharge(width, depth, depth, n, s, k)
    assert np.isclose(expected, q, rtol=1e-2)

    # x and h as arrays
    x = np.array([0, 0, width, width])
    h = np.array([depth, 0, 0, depth])
    q = get_discharge(x, h, depth, n, s, k)
    assert np.isclose(expected, q, rtol=1e-2)


def test_get_discharge_trapezoidal():
    # adapted from https://www.youtube.com/watch?v=ucLa9_DDWPA

    x = np.array([0, 2, 4, 8, 10, 12])
    h = np.array([2, 1, 0, 0, 1, 2])
    n = 0.015
    s = 0.002
    k = 1.0
    depth = 1.3
    expected = 23.4
    q = get_discharge(x, h, depth, n, s, k)
    assert np.isclose(expected, q, rtol=0.1)


@pytest.mark.xfail(reason="debug complex array input case")
def test_get_discharge_compound_rectangular():
    # adapted from https://www.youtube.com/watch?v=BJZ73WWEc3M

    # manually sum over discharge for each rectangle
    n = 0.016
    s = 0.001
    k = 1.0
    expected = 3.3
    q0 = get_discharge(3, 0.2, roughness=n, slope=s, conv=k)
    q1 = get_discharge(1.5, 1.4, roughness=n, slope=s, conv=k)
    q2 = get_discharge(3, 0.2, roughness=n, slope=s, conv=k)
    sm = q0 + q1 + q2
    assert np.isclose(expected, sm, rtol=1e-2)

    # check single rectangular segment with array input for x and h
    x = np.array([0, 0, 3, 3])
    h = np.array([0, -0.2, -0.2, 0])
    q = get_discharge(x, h, 0, n, s, k)
    assert np.isclose(q, q0, rtol=1e-3)

    # check multiple rectangles with array input
    x = np.array([0, 0, 3, 3, 4.5, 4.5, 7.5, 7.5])
    h = np.array([0, -0.2, -0.2, -1.4, -1.4, -0.2, -0.2, 0])
    q = get_discharge(x, h, 0, n, s, k)
    assert np.isclose(expected, q, rtol=1e-2)


def test_get_depth():
    # adapted from https://www.youtube.com/watch?v=t9ywTXEcScE

    width = 5
    x = np.array([0, 0, width, width])
    h = np.array([2, 0, 0, 2])
    q = 6.5
    n = 0.02
    s = 0.0005
    k = 1.0
    expected = 1.3
    depth = get_depth(x, h, q, n, s, k)
    assert np.isclose(expected, depth, rtol=1e-1)


@pytest.mark.skip(reason="todo")
def test_get_depths():
    pass
