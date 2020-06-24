Additional dependencies to use optional FloPy helper methods are listed below.

| Method                                                                               | Python Package                                                     |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `.PlotMapView()` in `flopy.plot`                                                     | **matplotlib** >= 1.4                                              |
| `.PlotCrossSection()` in `flopy.plot`                                                | **matplotlib** >= 1.4                                              |
| `.plot()`                                                                            | **matplotlib** >= 1.4                                              |
| `.plot_shapefile()`                                                                  | **matplotlib** >= 1.4 and **Pyshp** >= 1.2                         |
| `.to_shapefile()`                                                                    | **Pyshp** >= 1.2                                                   |
| `.export(*.shp)`                                                                     | **Pyshp** >= 1.2                                                   |
| `.export(*.nc)`                                                                      | **netcdf4** >= 1.1, and **python-dateutil** >= 2.4                 |
| `.export(*.tif)`                                                                     | **rasterio**                                                       |
| `.export(*.asc)` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.ndimage**                                                  |
| `.interpolate()` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.interpolate**                                              |
| `.interpolate()` in `flopy.mf6.utils.reference` `StructuredSpatialReference` class   | **scipy.interpolate**                                              |
| `._parse_units_from_proj4()` in `flopy.utils.reference` `SpatialReference` class     | **pyproj**                                                         |
| `.get_dataframes()` in `flopy.utils.mflistfile` `ListBudget` class                   | **pandas** >= 0.15                                                 |
| `.get_dataframes()` in `flopy.utils.observationfile` `ObsFiles` class                | **pandas** >= 0.15                                                 |
| `.get_dataframes()` in `flopy.utils.sfroutputfile` `ModflowSfr2` class               | **pandas** >= 0.15                                                 |
| `.get_dataframes()` in `flopy.utils.util_list` `MfList` class                        | **pandas** >= 0.15                                                 |
| `.get_dataframes()` in `flopy.utils.zonebud` `ZoneBudget` class                      | **pandas** >= 0.15                                                 |
| `.pivot_keyarray()` in `flopy.mf6.utils.arrayutils` `AdvancedPackageUtil` class      | **pandas** >= 0.15                                                 |
| `._get_vertices()` in `flopy.mf6.utils.binaryfile_utils` `MFOutputRequester` class   | **pandas** >= 0.15                                                 |
| `.get_dataframe()` in `flopy.mf6.utils.mfobservation` `Observations` class           | **pandas** >= 0.15                                                 |
| `.df()` in `flopy.modflow.mfsfr2` `SfrFile` class                                    | **pandas** >= 0.15                                                 |
| `.time_coverage()` in `flopy.export.metadata` `acc` class - ***used if available***  | **pandas** >= 0.15                                                 |
| `.loadtxt()` in `flopy.utils.flopyio` - ***used if available***                      | **pandas** >= 0.15                                                 |
| `.generate_classes()` in `flopy.mf6.utils`                                           | [**pymake**](https://github.com/modflowpy/pymake)                  |
| `.intersect()` in `flopy.discretization.VertexGrid`                                  | **matplotlib** >= 1.4                                              |
| `GridIntersect()` in `flopy.utils.gridintersect`                                     | **shapely**                                                        |
| `GridIntersect().plot_polygon()` in `flopy.utils.gridintersect`                      | **shapely** and **descartes**                                      |
| `Raster()` in `flopy.utils.Raster`                                                   | **rasterio**, **affine**, and **scipy**                            |
| `Raster().sample_polygon()` in `flopy.utils.Raster`                                  | **shapely**                                                        |
| `Raster().crop()` in `flopy.utils.Raster`                                            | **shapely**                                                        |
| `.array_at_verts()` in `flopy.discretization.structuredgrid` `StructuredGrid` class  | **scipy.interpolate**                                              |
