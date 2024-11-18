from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid


class GridCases:
    @staticmethod
    def structured_small():
        nlay, nrow, ncol = 3, 2, 3
        delc = 1.0 * np.ones(nrow, dtype=float)
        delr = 1.0 * np.ones(ncol, dtype=float)
        top = 10.0 * np.ones((nrow, ncol), dtype=float)
        idomain = np.ones((nlay, nrow, ncol), dtype=int)
        botm = np.zeros((nlay, nrow, ncol), dtype=float)
        botm[0, :, :] = 5.0
        botm[1, :, :] = 0.0
        botm[2, :, :] = -5.0
        return StructuredGrid(
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delc=delc,
            delr=delr,
            top=top,
            botm=botm,
            idomain=idomain,
        )

    @staticmethod
    def structured_cbd_small():
        nlay = 3
        nrow = ncol = 15
        laycbd = np.array([1, 2, 0], dtype=int)
        ncb = np.count_nonzero(laycbd)
        dx = dy = 150
        delc = np.array(
            [
                dy,
            ]
            * nrow
        )
        delr = np.array(
            [
                dx,
            ]
            * ncol
        )
        top = np.ones((15, 15))
        botm = np.ones((nlay + ncb, nrow, ncol))
        elevations = np.array([-10, -20, -40, -50, -70])[:, np.newaxis]
        botm *= elevations[:, None]

        return StructuredGrid(
            delc=delc,
            delr=delr,
            top=top,
            botm=botm,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            laycbd=laycbd,
        )

    @staticmethod
    def vertex_small():
        nlay, ncpl = 3, 5
        vertices = [
            [0, 0.0, 3.0],
            [1, 1.0, 3.0],
            [2, 2.0, 3.0],
            [3, 0.0, 2.0],
            [4, 1.0, 2.0],
            [5, 2.0, 2.0],
            [6, 0.0, 1.0],
            [7, 1.0, 1.0],
            [8, 2.0, 1.0],
            [9, 0.0, 0.0],
            [10, 1.0, 0.0],
        ]
        cell2d = [
            [0, 0.5, 2.5, 4, 0, 1, 4, 3],
            [1, 1.5, 2.5, 4, 1, 2, 5, 4],
            [2, 0.5, 1.5, 4, 3, 4, 7, 6],
            [3, 1.5, 1.5, 4, 4, 5, 8, 7],
            [4, 0.5, 0.5, 4, 6, 7, 10, 9],
        ]
        idomain = np.ones((nlay, ncpl), dtype=int)
        top = np.ones(ncpl, dtype=float) * 10.0
        botm = np.zeros((nlay, ncpl), dtype=float)
        botm[0, :] = 5.0
        botm[1, :] = 0.0
        botm[2, :] = -5.0
        return VertexGrid(
            nlay=nlay,
            ncpl=ncpl,
            vertices=vertices,
            cell2d=cell2d,
            top=top,
            botm=botm,
            idomain=idomain,
        )

    @staticmethod
    def unstructured_small():
        nlay = 3
        ncpl = [5, 5, 5]
        vertices = [
            [0, 0.0, 3.0],
            [1, 1.0, 3.0],
            [2, 2.0, 3.0],
            [3, 0.0, 2.0],
            [4, 1.0, 2.0],
            [5, 2.0, 2.0],
            [6, 0.0, 1.0],
            [7, 1.0, 1.0],
            [8, 2.0, 1.0],
            [9, 0.0, 0.0],
            [10, 1.0, 0.0],
        ]
        iverts = [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
            [6, 7, 10, 9],
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
            [6, 7, 10, 9],
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
            [6, 7, 10, 9],
        ]
        xcenters = [
            0.5,
            1.5,
            0.5,
            1.5,
            0.5,
        ]
        ycenters = [
            2.5,
            2.5,
            1.5,
            1.5,
            0.5,
        ]
        idomain = np.ones((nlay, 5), dtype=int)
        top = np.ones((nlay, 5), dtype=float)
        top[0, :] = 10.0
        top[1, :] = 5.0
        top[2, :] = 0.0
        botm = np.zeros((nlay, 5), dtype=float)
        botm[0, :] = 5.0
        botm[1, :] = 0.0
        botm[2, :] = -5.0

        return UnstructuredGrid(
            vertices=vertices,
            iverts=iverts,
            xcenters=xcenters,
            ycenters=ycenters,
            ncpl=ncpl,
            top=top.flatten(),
            botm=botm.flatten(),
            idomain=idomain.flatten(),
        )

    @staticmethod
    def unstructured_medium():
        iverts = [
            [4, 3, 2, 1, 0, None],
            [7, 0, 1, 6, 5, None],
            [11, 10, 9, 8, 2, 3],
            [1, 6, 13, 12, 8, 2],
            [15, 14, 13, 6, 5, None],
            [10, 9, 18, 17, 16, None],
            [8, 12, 20, 19, 18, 9],
            [22, 14, 13, 12, 20, 21],
            [24, 17, 18, 19, 23, None],
            [21, 20, 19, 23, 25, None],
        ]
        verts = [
            [0.0, 22.5],
            [5.1072, 22.5],
            [7.5, 24.0324],
            [7.5, 30.0],
            [0.0, 30.0],
            [0.0, 7.5],
            [4.684, 7.5],
            [0.0, 15.0],
            [14.6582, 21.588],
            [22.5, 24.3766],
            [22.5, 30.0],
            [15.0, 30.0],
            [15.3597, 8.4135],
            [7.5, 5.6289],
            [7.5, 0.0],
            [0.0, 0.0],
            [30.0, 30.0],
            [30.0, 22.5],
            [25.3285, 22.5],
            [24.8977, 7.5],
            [22.5, 5.9676],
            [22.5, 0.0],
            [15.0, 0.0],
            [30.0, 7.5],
            [30.0, 15.0],
            [30.0, 0.0],
        ]

        return UnstructuredGrid(verts, iverts, ncpl=[len(iverts)])

    @staticmethod
    def voronoi_polygon():
        ncpl = 3803
        domain = [
            [1831.381546, 6335.543757],
            [4337.733475, 6851.136153],
            [6428.747084, 6707.916043],
            [8662.980804, 6493.085878],
            [9350.437333, 5891.561415],
            [9235.861245, 4717.156511],
            [8963.743036, 3685.971717],
            [8691.624826, 2783.685023],
            [8047.13433, 2038.94045],
            [7416.965845, 578.0953252],
            [6414.425073, 105.4689614],
            [5354.596258, 205.7230386],
            [4624.173696, 363.2651598],
            [3363.836725, 563.7733141],
            [1330.11116, 1809.788273],
            [399.1804436, 2998.515188],
            [914.7728404, 5132.494831],
        ]
        poly = np.array(domain)
        max_area = 100.0**2
        angle = 30

        with TemporaryDirectory() as tempdir:
            tri = Triangle(
                maximum_area=max_area, angle=angle, model_ws=str(Path(tempdir))
            )
            tri.add_polygon(poly)
            tri.build(verbose=False)
            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid

    @staticmethod
    def voronoi_rectangle():
        ncpl = 1679
        xmin = 0.0
        xmax = 2.0
        ymin = 0.0
        ymax = 1.0
        poly = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
        max_area = 0.001
        angle = 30

        with TemporaryDirectory() as tempdir:
            tri = Triangle(
                maximum_area=max_area, angle=angle, model_ws=str(Path(tempdir))
            )
            tri.add_polygon(poly)
            tri.build(verbose=False)
            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid

    @staticmethod
    def voronoi_circle():
        ncpl = 538
        theta = np.arange(0.0, 2 * np.pi, 0.2)
        radius = 100.0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        poly = list(zip(x, y))
        max_area = 50
        angle = 30

        with TemporaryDirectory() as tempdir:
            tri = Triangle(
                maximum_area=max_area, angle=angle, model_ws=str(Path(tempdir))
            )
            tri.add_polygon(poly)
            tri.build(verbose=False)
            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid

    @staticmethod
    def voronoi_nested_circles():
        ncpl = 300

        theta = np.arange(0.0, 2 * np.pi, 0.2)
        radius = 100.0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        circle_poly = list(zip(x, y))

        theta = np.arange(0.0, 2 * np.pi, 0.2)
        radius = 30.0
        x = radius * np.cos(theta) + 25.0
        y = radius * np.sin(theta) + 25.0
        inner_circle_poly = list(zip(x, y))

        polys = [circle_poly, inner_circle_poly]
        max_area = 100
        angle = 30

        with TemporaryDirectory() as tempdir:
            tri = Triangle(
                maximum_area=max_area, angle=angle, model_ws=str(Path(tempdir))
            )
            for poly in polys:
                tri.add_polygon(poly)
            tri.add_hole((25, 25))
            tri.build(verbose=False)
            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid

    @staticmethod
    def voronoi_polygons():
        ncpl = 410
        active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
        area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
        area2 = [(60, 60), (80, 60), (80, 80), (60, 80)]

        with TemporaryDirectory() as tempdir:
            tri = Triangle(angle=30, model_ws=str(Path(tempdir)))
            tri.add_polygon(active_domain)
            tri.add_polygon(area1)
            tri.add_polygon(area2)
            tri.add_region((1, 1), 0, maximum_area=100)  # point inside active domain
            tri.add_region((11, 11), 1, maximum_area=10)  # point inside area1
            tri.add_region((61, 61), 2, maximum_area=3)  # point inside area2
            tri.build(verbose=False)
            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid

    @staticmethod
    def voronoi_many_polygons():
        ncpl = 1305
        active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
        area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
        area2 = [(70, 70), (90, 70), (90, 90), (70, 90)]

        with TemporaryDirectory() as tempdir:
            tri = Triangle(angle=30, model_ws=str(Path(tempdir)))

            # requirement that active_domain is first polygon to be added
            tri.add_polygon(active_domain)

            # requirement that any holes be added next
            theta = np.arange(0.0, 2 * np.pi, 0.2)
            radius = 10.0
            x = radius * np.cos(theta) + 50.0
            y = radius * np.sin(theta) + 70.0
            circle_poly0 = list(zip(x, y))
            tri.add_polygon(circle_poly0)
            tri.add_hole((50, 70))

            # Add a polygon to force cells to conform to it
            theta = np.arange(0.0, 2 * np.pi, 0.2)
            radius = 10.0
            x = radius * np.cos(theta) + 70.0
            y = radius * np.sin(theta) + 20.0
            circle_poly1 = list(zip(x, y))
            tri.add_polygon(circle_poly1)

            # add line through domain to force conforming cells
            line = [(x, x) for x in np.linspace(11, 89, 100)]
            tri.add_polygon(line)

            # then regions and other polygons should follow
            tri.add_polygon(area1)
            tri.add_polygon(area2)
            tri.add_region((1, 1), 0, maximum_area=100)  # point inside active domain
            tri.add_region((11, 11), 1, maximum_area=10)  # point inside area1
            tri.add_region((70, 70), 2, maximum_area=1)  # point inside area2

            tri.build(verbose=False)

            vor = VoronoiGrid(tri)
            gridprops = vor.get_gridprops_vertexgrid()
            grid = VertexGrid(**gridprops, nlay=1)

        return ncpl, vor, gridprops, grid
