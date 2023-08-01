Additional dependencies to use optional FloPy helper methods are listed below.

| Method                                                                               | Python Package                                                           |
| ------------------------------------------------------------------------------------ |--------------------------------------------------------------------------|
| `.plot_shapefile()`                                                                  | **Pyshp** >= 2.0.0                                                       |
| `.to_shapefile()`                                                                    | **Pyshp** >= 2.0.0                                                       |
| `.export(*.shp)`                                                                     | **Pyshp** >= 2.0.0                                                       |
| `.export(*.nc)`                                                                      | **netcdf4** >= 1.1, and **python-dateutil** >= 2.4.0                     |
| `.export(*.tif)`                                                                     | **rasterio**                                                             |
| `.export(*.asc)` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.ndimage**                                                        |
| `.interpolate()` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.interpolate**                                                    |
| `.interpolate()` in `flopy.mf6.utils.reference` `StructuredSpatialReference` class   | **scipy.interpolate**                                                    |
| `._parse_units_from_proj4()` in `flopy.utils.reference` `SpatialReference` class     | **pyproj**                                                               |
| `.generate_classes()` in `flopy.mf6.utils`                                           | [**modflow-devtools**](https://github.com/MODFLOW-USGS/modflow-devtools) |
| `GridIntersect()` in `flopy.utils.gridintersect`                                     | **shapely**                                                              |
| `GridIntersect().plot_polygon()` in `flopy.utils.gridintersect`                      | **shapely** and **descartes**                                            |
| `Raster()` in `flopy.utils.Raster`                                                   | **rasterio**, **rasterstats**, **affine**, and **scipy**                 |
| `Raster().sample_polygon()` in `flopy.utils.Raster`                                  | **shapely**                                                              |
| `Raster().crop()` in `flopy.utils.Raster`                                            | **shapely**                                                              |
| `.array_at_verts()` in `flopy.discretization.structuredgrid` `StructuredGrid` class  | **scipy.interpolate**                                                    |
| `get_sciencebase_xml_metadata()` in `flopy.export.metadata` `acdd` class             | **defusedxml**                                                           |
| `flopy.utils.geospatial_utils` `GeoSpatialUtil` class                                | **geojson**                                                              |
| `flopy.utils.geospatial_utils` `GeoSpatialCollection` class                          | **geojson**                                                              |
| `flopy.export.vtk` `Vtk` class                                                       | **vtk**, optionally **pyvista**                                          |
