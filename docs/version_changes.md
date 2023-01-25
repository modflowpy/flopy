### Version 3.3.6

#### New features

* [feat(time step length)](https://github.com/modflowpy/flopy/commit/ea6a0a190070f065d74824b421de70d4a66ebcc2): Added feature that returns time step lengths from listing file (#1435) (#1437). Committed by scottrp on 2022-06-30.
* [feat](https://github.com/modflowpy/flopy/commit/3e2b5fb0ec0fc4b58854fce6155eaa40c25603c6): Get modflow utility (#1465). Committed by Mike Taves on 2022-07-27.
* [feat(Gridintersect)](https://github.com/modflowpy/flopy/commit/4f86fcf70fcb0d4bb76af136d45fd6857730811b): New grid intersection options (#1468). Committed by Davíd Brakenhoff on 2022-07-27.
* [feat](https://github.com/modflowpy/flopy/commit/af42e7827fe053af911efef0f37dcb76dad7e9c0): Unstructured grid from specification file (#1524). Committed by w-bonelli on 2022-09-08.
* [feat(aux variable checking)](https://github.com/modflowpy/flopy/commit/c3cdc1323ed416eda7027a8d839fcc3b29d5cfaa): Check now performs aux variable checking (#1399) (#1536). Committed by spaulins-usgs on 2022-09-12.
* [feat(get/set data record)](https://github.com/modflowpy/flopy/commit/984227d8f5762a4687cbac0a8175a6b07f751aee): Updated get_data/set_data functionality and new get_record/set_record methods (#1568). Committed by spaulins-usgs on 2022-10-06.
* [feat(get_modflow)](https://github.com/modflowpy/flopy/commit/b8d471cd27b66bbb601b909043843bb7c59e8b40): Support modflow6 repo releases (#1573). Committed by w-bonelli on 2022-10-11.
* [feat(contours)](https://github.com/modflowpy/flopy/commit/00757a4dc7ea03ba3582242ba4d2590f407c0d39): Use standard matplotlib contours for StructuredGrid map view plots (#1615). Committed by w-bonelli on 2022-11-10.

#### Bug fixes

* [fix(geometry)](https://github.com/modflowpy/flopy/commit/3a1d94a62c21acef19da477267cd6fad81b47802): Is_clockwise() now works as expected for a disv problem (#1374). Committed by Eric Morway on 2022-03-16.
* [fix(packaging)](https://github.com/modflowpy/flopy/commit/830016cc0f1b78c0c4d74efe2dd83e2b00a58247): Include pyproject.toml to sdist, check package for PyPI (#1373). Committed by Mike Taves on 2022-03-21.
* [fix(url)](https://github.com/modflowpy/flopy/commit/78e42643ebc8eb5ca3109de5b335e18c3c6e8159): Use modern DOI and USGS prefixes; upgrade other HTTP->HTTPS (#1381). Committed by Mike Taves on 2022-03-23.
* [fix(packaging)](https://github.com/modflowpy/flopy/commit/5083663ed93c56e7ecbed031532166c25c01db46): Add docs/*.md to MANIFEST.in for sdist (#1391). Committed by Mike Taves on 2022-04-01.
* [fix(_plot_transient2d_helper)](https://github.com/modflowpy/flopy/commit/a0712be97867d1c3acc3a2bf7660c5af861727b3): Fix filename construction for saving plots (#1388). Committed by Joshua Larsen on 2022-04-01.
* [fix(simulation packages)](https://github.com/modflowpy/flopy/commit/cb8c21907ef35172c792419d59e163a86e6f0939): *** Breaks interface *** (#1394). Committed by spaulins-usgs on 2022-04-19.
* [fix(plot_pathline)](https://github.com/modflowpy/flopy/commit/85d4558cc28d75a87bd7892cd48a8d9c24ad578e): Split recarray into particle list when it contains multiple particle ids (#1400). Committed by Joshua Larsen on 2022-04-30.
* [fix(lgrutil)](https://github.com/modflowpy/flopy/commit/69d6b156e3a01e1682cf73c9e18d9afcd2ae8087): Child delr/delc not correct if parent has variable row/col spacings (#1403). Committed by langevin-usgs on 2022-05-03.
* [fix(ModflowUzf1)](https://github.com/modflowpy/flopy/commit/45a9e574cb18738f9ef618228efa408c89a58113): Fix for loading negative iuzfopt (#1408). Committed by Joshua Larsen on 2022-05-06.
* [fix(features_to_shapefile)](https://github.com/modflowpy/flopy/commit/cadd216a680330788d9034e5f3a450daaafefebc): Fix missing bracket around linestring for shapefile (#1410). Committed by Joshua Larsen on 2022-05-06.
* [fix(mp7)](https://github.com/modflowpy/flopy/commit/1cd4ba0f7c6859e0b2e12ab0858b88782e722ad5): Ensure shape in variables 'zones' and 'retardation' is 3d (#1415). Committed by Ruben Caljé on 2022-05-11.
* [fix(ModflowUzf1)](https://github.com/modflowpy/flopy/commit/0136b7fe6e7e06cc97dfccf65eacfd417a816fbe): Update load for iuzfopt = -1 (#1416). Committed by Joshua Larsen on 2022-05-17.
* [fix(recursion)](https://github.com/modflowpy/flopy/commit/bfcb74426554c3160a39ead1475855761fe92a8c): Infinite recursion fix for __getattr__.  Spelling error fix in notebook. (#1414). Committed by scottrp on 2022-05-19.
* [fix(get_structured_faceflows)](https://github.com/modflowpy/flopy/commit/e5530fbca3ad715f93ea41eaf635fb41e4167e1d): Fix index issue in lower right cell (#1417). Committed by jdhughes-usgs on 2022-05-19.
* [fix(package paths)](https://github.com/modflowpy/flopy/commit/d5fca7ae1a31068ac133890ef8e9402af058b725): Fixed auto-generated package paths to make them unique (#1401) (#1425). Committed by spaulins-usgs on 2022-05-31.
* [fix(get-modflow/ci)](https://github.com/modflowpy/flopy/commit/eb59e104bc403387c62649f47816e5a7896d792f): Use GITHUB_TOKEN to side-step ratelimit (#1473). Committed by Mike Taves on 2022-07-29.
* [fix(get-modflow/ci)](https://github.com/modflowpy/flopy/commit/5d977c661d961f542c7fffcb3148541d78d8b03b): Handle 404 error to retry request from GitHub (#1480). Committed by Mike Taves on 2022-08-04.
* [fix(intersect)](https://github.com/modflowpy/flopy/commit/7e690a2c7de5909d50da116b6b6f19549343de5d): Update to raise error only when interface is called improperly (#1489). Committed by Joshua Larsen on 2022-08-11.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/d546747b068d9b93029b774b7676b552ead93640): Fix DeprecationWarnings for Shapely2.0 (#1504). Committed by Davíd Brakenhoff on 2022-08-19.
* [fix](https://github.com/modflowpy/flopy/commit/58938b9eb40cd43a9d5b57cc0281b74dafbee060): CI, tests & modpathfile (#1495). Committed by w-bonelli on 2022-08-22.
* [fix(HeadUFile)](https://github.com/modflowpy/flopy/commit/8f1342025957c496db88f9efbb42cae107e5e29f): Fix #1503 (#1510). Committed by w-bonelli on 2022-08-26.
* [fix(setuptools)](https://github.com/modflowpy/flopy/commit/a54e75383f11a950a192f0ca30b4c59d665eee5b): Only include flopy and flopy.* packages (not autotest) (#1529). Committed by Mike Taves on 2022-09-01.
* [fix(mfpackage)](https://github.com/modflowpy/flopy/commit/265ee594f0380d09ce1dfc18ea44f4d6a11951ef): Modify maxbound evaluation (#1530). Committed by jdhughes-usgs on 2022-09-02.
* [fix(PlotCrossSection)](https://github.com/modflowpy/flopy/commit/5a8e68fcda7cabe8d4e028ba6c905ae4f84448d6): Update number of points check (#1533). Committed by Joshua Larsen on 2022-09-08.
* [fix(parsenamefile)](https://github.com/modflowpy/flopy/commit/d6e7f3fd4fbce44a0ca2a8f6c8f611922697d93a): Only do lowercase comparison when parent dir exists (#1554). Committed by Mike Taves on 2022-09-22.
* [fix(obs)](https://github.com/modflowpy/flopy/commit/06f08beb440e01209523efb2c144f8cf6e97bba4): Modify observations to load single time step files (#1559). Committed by jdhughes-usgs on 2022-09-29.
* [fix(PlotMapView notebook)](https://github.com/modflowpy/flopy/commit/4d4c68e471cd4b7bc2092d57cf593bf0bf21d65f): Remove exact contour count assertion (#1564). Committed by w-bonelli on 2022-10-03.
* [fix(test markers)](https://github.com/modflowpy/flopy/commit/aff35857e13dfc514b62db3dbc44e9a3b7abb5ab): Add missing markers for exes required (#1577). Committed by w-bonelli on 2022-10-07.
* [fix(csvfile)](https://github.com/modflowpy/flopy/commit/2eeefd4a0b00473ef13480ba7bb8741d3410bdfe): Default csvfile to retain all characters in column names (#1587). Committed by langevin-usgs on 2022-10-14.
* [fix(csvfile)](https://github.com/modflowpy/flopy/commit/a0a92ed07d87e0def97dfc4a87d291aff91cd0f8): Correction to read_csv args, close file handle (#1590). Committed by Mike Taves on 2022-10-16.
* [fix(data shape)](https://github.com/modflowpy/flopy/commit/09e3bc5552af5e40e207ca67808397df644a20c8): Fixed incorrect data shape for sfacrecord (#1584).  Fixed a case where time array series data did not load correctly from a file when the data shape could not be determined (#1594). (#1598). Committed by spaulins-usgs on 2022-10-21.
* [fix(write_shapefile)](https://github.com/modflowpy/flopy/commit/8a4e204eff0162021a895ee194feb2ccef4fe50d): Fix transform call for exporting (#1608). Committed by Joshua Larsen on 2022-10-27.
* [fix(multiple)](https://github.com/modflowpy/flopy/commit/e20ea054fdc1ab54bb0f1118a243b943c390ea4f): Miscellanous fixes/enhancements (#1614). Committed by w-bonelli on 2022-11-03.
* [fix](https://github.com/modflowpy/flopy/commit/9d328534aa272d8b4f8cfe77d30117f980794a8a): Not reading comma separated data correctly (#1634). Committed by Michael Ou on 2022-11-23.
* [fix(get-modflow)](https://github.com/modflowpy/flopy/commit/57d08623ea4c91a0ba3d1e6cfead44d203359ac4): Fix code.json handling (#1641). Committed by w-bonelli on 2022-12-08.
* [fix(quotes+exe_path+nam_file)](https://github.com/modflowpy/flopy/commit/3131e97fd85f5cfa57b8cbb5646b4a1884469989): Fixes for quoted strings, exe path, and nam file (#1645). Committed by spaulins-usgs on 2022-12-08.

#### Refactoring

* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/8b8221207e324fbd96bad8379fb234a0888efac8): Iverts compatibility updates for vtk (#1378). Committed by Joshua Larsen on 2022-03-19.
* [refactor(Raster)](https://github.com/modflowpy/flopy/commit/140d3e5882891dcd46c597721a2469c53b0c3bf6): Added "mode" resampling to resample_to_grid (#1390). Committed by Joshua Larsen on 2022-04-01.
* [refactor(utils)](https://github.com/modflowpy/flopy/commit/1757470fe0dd28ed6c68c95950de877bd0b4a94b): Updates to zonbudget and get_specific_discharge (#1457). Committed by Joshua Larsen on 2022-07-20.
* [refactor(line_intersect_grid, PlotCrossSection)](https://github.com/modflowpy/flopy/commit/c9e6f61e8c4880c8a0cba5940c3178abde062937): Fix cell artifact and collinear issues (#1505). Committed by Joshua Larsen on 2022-08-19.
* [refactor(get-modflow)](https://github.com/modflowpy/flopy/commit/8f93d12b07486b031efe707a865c036b01ec1c4f): Add option to have flopy bindir on PATH (#1511). Committed by Mike Taves on 2022-08-29.
* [refactor(shapefile_utils)](https://github.com/modflowpy/flopy/commit/ceab01db707409375bd01a5a945db77d13093a92): Remove appdirs dependency (#1523). Committed by Mike Taves on 2022-08-30.
* [refactor(export_contours)](https://github.com/modflowpy/flopy/commit/f45c8f94201e2045d2f394c84f74e9bce4e3e303): (#1528). Committed by w-bonelli on 2022-09-01.
* [refactor(CrossSectionPlot, GeoSpatialUtil)](https://github.com/modflowpy/flopy/commit/18a36795d334baa965bacab3b2aa547268548b33): (#1521). Committed by w-bonelli on 2022-09-02.
* [refactor(get_lni)](https://github.com/modflowpy/flopy/commit/d8f8dd0c82b2340047b57423a012c29884d77c92): Simplify get_lni signature & behavior (#1520). Committed by w-bonelli on 2022-09-02.
* [refactor(make-release)](https://github.com/modflowpy/flopy/commit/02eb23dd3ec2fa7b9e6049790f8384bf93e3cde6): Use CITATION.cff for author metadata (#1547). Committed by Mike Taves on 2022-09-19.
* [refactor(exe_name)](https://github.com/modflowpy/flopy/commit/f52011050aa8eec46e3a4c9a2b6b891799e50336): Remove unnecessary ".exe" suffix (#1563). Committed by Mike Taves on 2022-09-30.
* [refactor(contour_array)](https://github.com/modflowpy/flopy/commit/2d6db9a614791bed00a43060bea744c13a0936b1): Added routine to mask errant triangles (#1562). Committed by Joshua Larsen on 2022-10-02.
* [refactor(GeoSpatialCollection, Gridgen)](https://github.com/modflowpy/flopy/commit/a0ca3440037341ca77e430d46dea5d1cc4a7aa6d): Added support for lists of shapely objects (#1565). Committed by Joshua Larsen on 2022-10-05.
* [refactor(rasters.py, map.py)](https://github.com/modflowpy/flopy/commit/9d845b3179e3d7dbc03bac618fa74595ea10754f): Speed improvement updates for `resample_to_grid()` (#1571). Committed by Joshua Larsen on 2022-10-07.
* [refactor(shapefile_utils)](https://github.com/modflowpy/flopy/commit/67077194437d0de771cfc09edfa222abff20037a): Pathlib compatibility, other improvements (#1583). Committed by Mike Taves on 2022-10-13.
* [refactor(tests)](https://github.com/modflowpy/flopy/commit/7973c70f523706da87c591de87680620e28ab293): Simplify test utilities per convention that tests are run from autotest folder (#1586). Committed by w-bonelli on 2022-10-14.
* [refactor(gridintersect)](https://github.com/modflowpy/flopy/commit/586151fb18471336caf9e33fdecb7450f37d4167): Faster __init__ and better shapely 2.0 compat (#1593). Committed by Mike Taves on 2022-10-19.
* [refactor(get_lrc, get_node)](https://github.com/modflowpy/flopy/commit/02749f09b1edface66dc822b91bf1fcd905c6c71): Use numpy methods, add examples (#1591). Committed by Mike Taves on 2022-10-19.
* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/f9068756311cf76f5e8f435f7ef9291af9c88eec): Updates for errant interpolation in point_scalar routines (#1601). Committed by Joshua Larsen on 2022-10-21.
* [refactor(docs/examples)](https://github.com/modflowpy/flopy/commit/798cf0d8b53ef61248a8b6d8da570eaca5483540): Correction to authors, use Path.cwd() (#1602). Committed by Mike Taves on 2022-10-26.
* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/2f71612197efedb08b0874d494b4658eb3f1d5c2): Use pathlib, corrections to docstrings (#1603). Committed by Mike Taves on 2022-10-26.
* [refactor(docs)](https://github.com/modflowpy/flopy/commit/66a28c50043d8dcaec4d6b6ee4906e4519bfacd6): Add ORCID icon and link to authors (#1606). Committed by Mike Taves on 2022-10-27.
* [refactor(plotting)](https://github.com/modflowpy/flopy/commit/b5d64e034504d962ebff1ea08dafef1abe00396d): Added default masked values  (#1610). Committed by Joshua Larsen on 2022-10-31.

### Version 3.3.5

#### New features

* [feat(mvt)](https://github.com/modflowpy/flopy/commit/aa0baac73db0ef65ac975511f6182148708e75be): Add simulation-level support for mover transport (#1357). Committed by langevin-usgs on 2022-02-18.
* [feat(gwtgwt-mvt)](https://github.com/modflowpy/flopy/commit/8663a3474dd42211be2b8124316f16ea2ae8de18): Add support for gwt-gwt with mover transport (#1356). Committed by langevin-usgs on 2022-02-18.
* [feat(inspect cells)](https://github.com/modflowpy/flopy/commit/ee009fd2826c15bd91b81b84d1b7daaba0b8fde3): New feature that returns model data associated with specified model cells (#1140) (#1325). Committed by spaulins-usgs on 2022-01-11.
* [feat(multiple package instances)](https://github.com/modflowpy/flopy/commit/2eeaf1db72150806bf0cee0aa30c813e02256741): Flopy support for multiple instances of the same package stored in dfn files (#1239) (#1321). Committed by spaulins-usgs on 2022-01-07.
* [feat(Gridintersect)](https://github.com/modflowpy/flopy/commit/806d60149c4b3d821fa409d71dd0592a94070c51): Add shapetype kwarg (#1301). Committed by Davíd Brakenhoff on 2021-12-03.
* [feat(get_ts)](https://github.com/modflowpy/flopy/commit/376b3234a7931785518e478b422c4987b2d8fd52): Added support to get_ts for headufile (#1260). Committed by Ross Kushnereit on 2021-10-09.
* [feat(CellBudget)](https://github.com/modflowpy/flopy/commit/45df13abfd0611cc423b01bff341b450064e3b01): Add support for full3d keyword (#1254). Committed by jdhughes-usgs on 2021-10-05.
* [feat(mf6)](https://github.com/modflowpy/flopy/commit/6c0df005bd1f61d7ce54f160fea82cc34916165b): Allow multi-package for stress package concentrations (spc) (#1242). Committed by langevin-usgs on 2021-09-16.
* [feat(lak6)](https://github.com/modflowpy/flopy/commit/5974d8044f404a5221960e837dcce73d0db140d2): Support none lake bedleak values (#1189). Committed by jdhughes-usgs on 2021-08-16.


#### Bug fixes

* [fix(exchange obs)](https://github.com/modflowpy/flopy/commit/14d461588e1dbdb7bbe396e89e2a9cfc9366c8f3): Fixed building of obs package for exchange packages (through mfsimulation) (#1363). Committed by spaulins-usgs on 2022-02-28.
* [fix(tab files)](https://github.com/modflowpy/flopy/commit/b59f1fef153a235b0e11d40107c1e30d5eae4f84): Fixed searching for tab file packages (#1337) (#1344). Committed by scottrp on 2022-02-16.
* [fix(ModflowUtllaktab)](https://github.com/modflowpy/flopy/commit/0ee7c8849a14bc1a0736d5680c761b887b9c04db): Utl-lak-tab.dfn is redundant to utl-laktab.dfn (#1339). Committed by Mike Taves on 2022-01-27.
* [fix(cellid)](https://github.com/modflowpy/flopy/commit/33e61b8ccd2556b65d33d25c5a3fb27114f5ff25): Fixes some issues with flopy properly identifying cell ids (#1335) (#1336). Committed by spaulins-usgs on 2022-01-25.
* [fix(postprocessing)](https://github.com/modflowpy/flopy/commit/0616db5f158a97d1b38544d993d866c19a672a80): Get_structured_faceflows fix to support 3d models (#1333). Committed by langevin-usgs on 2022-01-21.
* [fix(Raster)](https://github.com/modflowpy/flopy/commit/dced106e5f9bd9a5234a2cbcd74c67b959ca950c): Resample_to_grid failure no data masking failure with int dtype (#1328). Committed by Joshua Larsen on 2022-01-21.
* [fix(voronoi)](https://github.com/modflowpy/flopy/commit/24f142ab4e02ba96e04d9c5040c87cea35ad8fd9): Clean up voronoi examples and add error check (#1323). Committed by langevin-usgs on 2022-01-03.
* [fix(filenames)](https://github.com/modflowpy/flopy/commit/fb0955c174cd47e41cd15eac6b2fc5e4cc587935): Fixed how spaces in filenames are handled (#1236) (#1318). Committed by spaulins-usgs on 2021-12-16.
* [fix(paths)](https://github.com/modflowpy/flopy/commit/0432ce96a0a5eec4d20adb4d384505632a2db3dc): Path code made more robust so that non-standard model folder structures are supported (#1311) (#1316). Committed by scottrp on 2021-12-09.
* [fix(UnstructuredGrid)](https://github.com/modflowpy/flopy/commit/8ab2e8d1a869de14ee76b11d0c53f439560c7a72): Load vertices for unstructured grids (#1312). Committed by Chris Nicol on 2021-12-09.
* [fix(array)](https://github.com/modflowpy/flopy/commit/fad091540e3f75efa38668e74a304e6d7f6b715f): Getting array data (#1028) (#1290). Committed by spaulins-usgs on 2021-12-03.
* [fix(plot_pathline)](https://github.com/modflowpy/flopy/commit/849f68e96c151e64b63a351b50955d0d5e01f1f7): Sort projected pathline points by travel time instead of cell order (#1304). Committed by Joshua Larsen on 2021-12-03.
* [fix(autotests)](https://github.com/modflowpy/flopy/commit/bbdf0082749d04805af470bc1b0f197d73202e21): Added pytest.ini to declare test naming convention (#1307). Committed by Joshua Larsen on 2021-12-03.
* [fix(geospatial_utils.py)](https://github.com/modflowpy/flopy/commit/976ad8130614374392345ce8ee3adb766d378797): Added pyshp and shapely imports check to geospatial_utils.py (#1305). Committed by Joshua Larsen on 2021-12-03.
* [fix(path)](https://github.com/modflowpy/flopy/commit/220974ae98c8fbc9a278ce02eec71378fd84dc1a): Fix subdirectory path issues (#1298). Committed by Brioch Hemmings on 2021-11-18.
* [fix()](https://github.com/modflowpy/flopy/commit/0bf5a5e5fc7105651d0b661ec60667094bb84267): fix(mfusg/str) (#1296). Committed by Chris Nicol on 2021-11-11.
* [fix(io)](https://github.com/modflowpy/flopy/commit/1fa24026d890abc4508a39eddf9049399c1e4d3f): Read comma separated list (#1285). Committed by Michael Ou on 2021-11-02.
* [fixes(1247)](https://github.com/modflowpy/flopy/commit/71f6cc6b6391a460dd01e58b739372b35899f09c): Make flopy buildable with pyinstaller (#1248). Committed by Tim Mitchell on 2021-10-20.
* [fix(keystring records)](https://github.com/modflowpy/flopy/commit/0d21b925ebd4e2f31413ea67d28591d77a02d8c1): Fixed problem with keystring records containing multiple keywords (#1266). Committed by spaulins-usgs on 2021-10-15.
* [fix(ModflowFhb)](https://github.com/modflowpy/flopy/commit/fc094353a375b26f5afd6befb1bcdfef407c3a64): Update datasets 4, 5, 6, 7, 8 loading routine for multiline records (#1264). Committed by Joshua Larsen on 2021-10-15.
* [fix(MFFileMgmt.string_to_file_path)](https://github.com/modflowpy/flopy/commit/459aacbc0ead623c79bc4701b4c23666a280fbe8): Updated to support unc paths (#1256). Committed by Joshua Larsen on 2021-10-06.
* [fix(_mg_resync)](https://github.com/modflowpy/flopy/commit/fe913fe6e364501832194c87deb962d913a22a35): Added checks to reset the modelgrid resync (#1258). Committed by Joshua Larsen on 2021-10-06.
* [fix(MFPackage)](https://github.com/modflowpy/flopy/commit/658bf2ccbc9c6baeec4184e7c08aeb452f83c4ea): Fix mfsim.nam relative paths (#1252). Committed by Joshua Larsen on 2021-10-01.
* [fix(voronoi)](https://github.com/modflowpy/flopy/commit/ccdb36a8dd00347889e6048f7e6ddb7538e6308f): Voronoigrid class upgraded to better support irregular domains (#1253). Committed by langevin-usgs on 2021-10-01.
* [fix(writing tas)](https://github.com/modflowpy/flopy/commit/047c9e6fdb345c373b48b07fe0b81bc88105a543): Fixed writing non-constant tas (#1244) (#1245). Committed by spaulins-usgs on 2021-09-22.
* [fix(numpy elementwise comparison)](https://github.com/modflowpy/flopy/commit/779aa50afd9fb8105d0684b0ab50c64e0bc347de): Fixed numpy futurewarning. no elementwise compare is needed if user passes in a numpy recarray, this is only necessary for dictionaries. (#1235). Committed by spaulins-usgs on 2021-09-13.
* [fix()](https://github.com/modflowpy/flopy/commit/722224c4b438116528b83a77338e34d6f9807ef7): fix(shapefile_utils) unstructured shapefile export (#1220) (#1222). Committed by Chris Nicol on 2021-09-03.
* [fix(rename and comments)](https://github.com/modflowpy/flopy/commit/c2a01df3d49b0e06a21aa6aec532125b03992048): Package rename code and comment propagation (#1226). Committed by spaulins-usgs on 2021-09-03.
* [fix(numpy)](https://github.com/modflowpy/flopy/commit/8ec8d096afb63080d754a11d7d477ac4133c0cf6): Handle deprecation warnings from numpy 1.21 (#1211). Committed by Mike Taves on 2021-08-23.
* [fix(Grid.saturated_thick)](https://github.com/modflowpy/flopy/commit/e9dff6136c19631532de4a2f714092037fb81c98): Update saturated_thick to filter confining bed layers (#1197). Committed by Joshua Larsen on 2021-08-18.
* [fix()](https://github.com/modflowpy/flopy/commit/41b038c7e64e7c8fa05c0eeb4e1b7b147b2deb7f): fix(pakbase) unstructured storage check (#1187) (#1194). Committed by Chris Nicol on 2021-08-17.
* [fix(MF6Output)](https://github.com/modflowpy/flopy/commit/0d9faa34974c1e9cf32b9897e9a2a68f695b0936): Fix add new obs package issue (#1193). Committed by Joshua Larsen on 2021-08-17.
* [fix()](https://github.com/modflowpy/flopy/commit/e523256cfe7d2f17e0c16e377551601fa62aae38): fix(plot/plot_bc) plot_bc fails with unstructured models (#1185). Committed by Chris Nicol on 2021-08-16.
* [fix()](https://github.com/modflowpy/flopy/commit/a126f70797af14dff65951312f2c925ae5407b0a): fix(modflow/mfriv) unstructured mfusg RIV package load check fix (#1184). Committed by Chris Nicol on 2021-08-16.
* [fix(pakbase)](https://github.com/modflowpy/flopy/commit/74289dc397a3cd41861ddbf2001726d8489ba6af): Specify dtype=bool for 0-d active array used for usg (#1188). Committed by Mike Taves on 2021-08-16.
* [fix(raster)](https://github.com/modflowpy/flopy/commit/b893017995e01dd5c6dda5eb4347ebb707a16ece): Rework raster threads to work on osx (#1180). Committed by jdhughes-usgs on 2021-08-10.
* [fix()](https://github.com/modflowpy/flopy/commit/6d4e3b7358ffb29f321f7c7cf0b6736cfa157a40): fix(loading mflist from file) (#1179). Committed by J Dub on 2021-08-09.
* [fix(grid, plotting)](https://github.com/modflowpy/flopy/commit/fdc4608498884526d11079ecbd3c0c3548b11a61): Bugfixes for all grid instances and plotting code (#1174). Committed by Joshua Larsen on 2021-08-09.

### Version 3.3.4

#### New features

* [feat(mf6-lake)](https://github.com/modflowpy/flopy/commit/73a44059d18ee37f16d8e8fc90c231a0c9b16cf7): Add helper function to create lake connections (#1163). Committed by jdhughes-usgs on 2021-08-04.
* [feat(data storage)](https://github.com/modflowpy/flopy/commit/08b95ea0b4b582567bbc99ff89f9923910a33d70): Data automatically stored internally when simulation or model relative path changed (#1126) (#1157). Committed by spaulins-usgs on 2021-07-29.
* [feat(get_reduced_pumping)](https://github.com/modflowpy/flopy/commit/86b270d2cc085f92373c1623cb61a23d2befc70e): Update to read external pumping reduction files (#1162). Committed by Joshua Larsen on 2021-07-29.
* [feat(raster)](https://github.com/modflowpy/flopy/commit/3ffedb45c4fa017ff64a4d579314c12386696068): Add option to extrapolate using ``nearest`` method (#1159). Committed by jdhughes-usgs on 2021-07-26.
* [Feat(ZoneBudget6)](https://github.com/modflowpy/flopy/commit/04fe7e4a3c67cb068de444a2979a3528ccbc3e60): Added zonebudget6 class to zonbud.py (#1149). Committed by Joshua Larsen on 2021-07-12.
* [feat(grid)](https://github.com/modflowpy/flopy/commit/0560a4d444c1f0ecd90319c41a6942c8e78dac26): Add thickness method for all grid types (#1138). Committed by jdhughes-usgs on 2021-06-25.
* [feat(flopy for mf6 docs)](https://github.com/modflowpy/flopy/commit/fbba027a6c6c59553b0030f6bac2b9f68a14198c): Documentation added and improved for flopy  (#1121). Committed by spaulins-usgs on 2021-06-01.
* [Feat(.output)](https://github.com/modflowpy/flopy/commit/df843e00dc9df208e5b16e7a0523456f1e80c004): Added output property method to mf6 packages and models  (#1100). Committed by Joshua Larsen on 2021-04-23.


#### Bug fixes

* [fix(flopy_io)](https://github.com/modflowpy/flopy/commit/5098c56085ee8498654ecf76aa5c71f1067466b6): Limit fixed point format string to fixed column widths (#1172). Committed by jdhughes-usgs on 2021-08-06.
* [fix(modpath7)](https://github.com/modflowpy/flopy/commit/9020bf362ea6418a39c12def5c74eb98ecc51746): Address #993 and #1053 (#1170). Committed by jdhughes-usgs on 2021-08-05.
* [fix(lakpak_utils)](https://github.com/modflowpy/flopy/commit/08dfff9420a50e66994575344cfd75a9fb7b1173): Fix telev and belev for horizontal connections (#1168). Committed by jdhughes-usgs on 2021-08-05.
* [fix(mfsfr2.check)](https://github.com/modflowpy/flopy/commit/cf5ee1c885da2294a385cc3e436aa7a9c68cd143): Make reach_connection_gaps.chk.csv a normal csv file (#1151). Committed by Mike Taves on 2021-07-27.
* [fix(modflow/mfrch,mfevt)](https://github.com/modflowpy/flopy/commit/b780bfb9721156fbdd9a96d99cf7b084d99ab452): Mfusg rch,evt load and write fixes (#1148). Committed by Chris Nicol on 2021-07-13.
* [fix(Raster)](https://github.com/modflowpy/flopy/commit/679c8844704a4e9b337af06c1826984c2715371f): Update default dtypes to include unsigned integers (#1147). Committed by Joshua Larsen on 2021-07-12.
* [fix(utils/check)](https://github.com/modflowpy/flopy/commit/345e0fbfdc007b688982c09bfa82841d56ac5cf7): File check for mfusg unstructured models (#1145). Committed by Chris Nicol on 2021-07-09.
* [fix(list parsing)](https://github.com/modflowpy/flopy/commit/148087e2b6f989d8625a2d345fbe1069b58bd99b): Fixed errors caused by parsing certain lists with keywords in the middle (#1135). Committed by spaulins-usgs on 2021-06-25.
* [fix(plotutil.py, grid.py)](https://github.com/modflowpy/flopy/commit/e566845172380e3eae06981ca180923d2362ee56): Fix .plot() method using mflay parameter (#1134). Committed by Joshua Larsen on 2021-06-16.
* [fix(map.py)](https://github.com/modflowpy/flopy/commit/4cfb11db25c17f70708b47bec79f88479d35a3fe): Fix slow plotting routines  (#1118). Committed by Joshua Larsen on 2021-06-11.
* [fix(plot_pathline)](https://github.com/modflowpy/flopy/commit/fa98069de665fcbd45c4b468bef9ac484833c2f9): Update modpath intersection routines  (#1112). Committed by Joshua Larsen on 2021-06-11.
* [fix(gridgen)](https://github.com/modflowpy/flopy/commit/ce5dcf9254bc5714d3b4bd5e888bdf4860cd5ecb): Mfusg helper function not indexing correctly (#1124). Committed by langevin-usgs on 2021-05-28.
* [fix(numpy)](https://github.com/modflowpy/flopy/commit/6c328e1d57a07b452b08575806c1ef78c1277fe4): Aliases of builtin types is deprecated (#1105). Committed by Mike Taves on 2021-05-07.
* [fix(mfsfr2.py)](https://github.com/modflowpy/flopy/commit/870c5773307761af977934a0b6fdfa84e7f30e72): Dataset 6b and 6c write routine, remove blank lines (#1101). Committed by Joshua Larsen on 2021-04-26.
* [fix(performance)](https://github.com/modflowpy/flopy/commit/4104cf5e6a35e2a1fd6183442962ae5cb258fa7a): Implemented performance improvements including improvements suggested by @briochh (#1092) (#1097). Committed by spaulins-usgs on 2021-04-12.
* [fix(package write)](https://github.com/modflowpy/flopy/commit/58cb48e7a6240d2a01039b5a5ab3c67c5111662c): Allow user to define and write empty stress period blocks to package files (#1091) (#1093). Committed by scottrp on 2021-04-08.
* [fix(imports)](https://github.com/modflowpy/flopy/commit/36426534fae226570884f82a93a42374bd0019da): Fix shapely and geojson imports (#1083). Committed by jdhughes-usgs on 2021-03-20.
* [fix(shapely)](https://github.com/modflowpy/flopy/commit/ff7b8223a9184ffbe6460ae857be45533d5c874a): Handle deprecation warnings from shapely 1.8 (#1069). Committed by Mike Taves on 2021-03-19.
* [fix(mp7particledata)](https://github.com/modflowpy/flopy/commit/44ebec2cd6a4a79a2f3f7165eae8ced7da0340c8): Update dtype comparison for unstructured partlocs (#1071). Committed by rodrperezi on 2021-03-19.
* [fix(get_file_entry)](https://github.com/modflowpy/flopy/commit/5604ef78fb8b2366fc7d5d827e68413ac602c7fa): None text removed from empty stress period data with aux variable (#1080) (#1081). Committed by scottrp on 2021-03-19.
* [fix(data check)](https://github.com/modflowpy/flopy/commit/179fc4e06b6317f12adcc826597f9586317fc7c0): Minimum number of data columns check fixed (#1062) (#1067). Committed by spaulins-usgs on 2021-03-17.
* [fix(plotutil)](https://github.com/modflowpy/flopy/commit/605366bba794a7ca75b67ef956dc232e20c79c48): Number of plottable layers should be from util3d.shape[0] (#1077). Committed by Mike Taves on 2021-03-15.
* [fix(data check)](https://github.com/modflowpy/flopy/commit/cb04f9d2ac55f3cbc320ef212945f69045222366): Minimum number of data columns check fixed (#1062) (#1065). Committed by spaulins-usgs on 2021-02-19.

### Version 3.3.3

#### New features

* [feat(voronoi)](https://github.com/modflowpy/flopy/commit/71162aef76ad025c42753e82ddee14bdeef1fd22): Add voronoigrid class (#1034). Committed by langevin-usgs on 2021-01-05.
* [feat(unstructured)](https://github.com/modflowpy/flopy/commit/0f7a0f033ba0bdff19621e824e721d032a6d1313): Improve unstructured grid support for modflow 6 and modflow-usg (#1021). Committed by langevin-usgs on 2020-11-27.


#### Bug fixes

* [fix(createpackages)](https://github.com/modflowpy/flopy/commit/8a2cffb052d0c33a4699edb9554b039fa871f00f): Avoid creating invalid escape characters (#1055). Committed by Mike Taves on 2021-02-17.
* [fix(DeprecationWarning)](https://github.com/modflowpy/flopy/commit/7d36b1c7f67a9eddeda122644295cf23c0573c8d): Use collections.abc module instead of collections (#1057). Committed by Mike Taves on 2021-02-15.
* [fix(DeprecationWarning)](https://github.com/modflowpy/flopy/commit/39c0ecba20bd54f2eafd8350151e60efbcc5c162): Related to numpy (#1058). Committed by Mike Taves on 2021-02-15.
* [fix(numpy)](https://github.com/modflowpy/flopy/commit/943ace46c686cbdbfdc327a3016da5942211ab7e): Aliases of builtin types is deprecated as of numpy 1.20 (#1052). Committed by Mike Taves on 2021-02-11.
* [fix()](https://github.com/modflowpy/flopy/commit/3ccdd57e390b94c4f23daa3d46554efd39e8ec9d): fix(get_active) include the cbd layers when checking layer thickness before BAS is loaded (#1051). Committed by Michael Ou on 2021-02-08.
* [fix(dis)](https://github.com/modflowpy/flopy/commit/5bfdf715514bbefd3f9d9a2e79d5c486e48f1896): Fix for dis.get_lrc() and dis.get_node() (#1049). Committed by langevin-usgs on 2021-02-04.
* [fix()](https://github.com/modflowpy/flopy/commit/3088e51bb546afc5a68103a81821c750480263b7): fix(modflow/mflpf) mfusg unstructured lpf ikcflag addition (#1044). Committed by Chris Nicol on 2021-02-01.
* [fix(print MFArray)](https://github.com/modflowpy/flopy/commit/b8322ddc52241d3c2a49c4ebdf2e8865c6f36bb9): Fixed printing of layered arrays - issue #1043 (#1045). Committed by spaulins-usgs on 2021-01-29.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/369942c7ed3171927f50a20c27b951ed062eb0e1): Fix vertices for offset grids (#1037). Committed by Davíd Brakenhoff on 2021-01-13.
* [fix(PlotMapView.plot_array())](https://github.com/modflowpy/flopy/commit/7cb429f81a9206fa450cceff21c695faf01fce41): Fix value masking code for masked_values parameter (#1026). Committed by Joshua Larsen on 2020-12-03.
* [fix(sfr ic option)](https://github.com/modflowpy/flopy/commit/0ab929e01fdf2a2205d268e51364630d9bbaf9c5): Made ic optional and updated description (#1020). Committed by spaulins-usgs on 2020-11-27.

### Version 3.3.2

#### New features

* [feat(mf6)](https://github.com/modflowpy/flopy/commit/133f07fe2c0c418aa139ddae1429605ef5a1f253): Update t503 autotest to use mf6examples zip file (#1007). Committed by jdhughes-usgs on 2020-10-21.
* [feat(mf6)](https://github.com/modflowpy/flopy/commit/0bc91e4c2873ee2783968fd28b98c67c883457c0): Add modflow 6 gwt dfn and classes (#1006). Committed by jdhughes-usgs on 2020-10-21.
* [feat(geospatial_util)](https://github.com/modflowpy/flopy/commit/9ad6732c4b0c9cf26cf2deac0d95fee2d6859f97): Geospatial consolidation utilities (#1002). Committed by Joshua Larsen on 2020-10-19.
* [feat(cvfdutil)](https://github.com/modflowpy/flopy/commit/fdcc35ff5a7e7a809d7d10eff914881351a8f738): Capability to create disv nested grids (#997). Committed by langevin-usgs on 2020-10-02.
* [feat(cellid not as tuple)](https://github.com/modflowpy/flopy/commit/2763a1d5ef0a886582d682cc27f02f4a5cb52693): List data input now accepts cellids as separate integers (#976). Committed by spaulins-usgs on 2020-08-27.
* [feat(set_all_data_external) ](https://github.com/modflowpy/flopy/commit/46e041bc01e47b65ea8ba0420609cecc52ff74c0): New check_data option  (#966). Committed by spaulins-usgs on 2020-08-17.


#### Bug fixes

* [fix(gridintersect)](https://github.com/modflowpy/flopy/commit/5091182a7c71d30bf0b62e2e257cb13b4652d7cf): Use linestrings and structured grid (#996). Committed by Tom van Steijn on 2020-09-25.
* [fix()](https://github.com/modflowpy/flopy/commit/44d8b5b68c6624b5c20e78e2e7652c7429ba088a): fix(extraneous blocks and zero length data in a list)  (#990). Committed by spaulins-usgs on 2020-09-03.
* [fix(internal)](https://github.com/modflowpy/flopy/commit/af28ed8ffc7e6d6fd19d82a3f9a1d496514ef6b7): Flopy fixed to correctly reads in internal keyword when no multiplier supplied. also, header line added to package files. (#962). Committed by spaulins-usgs on 2020-08-13.
* [fix(mfsfr2.py](https://github.com/modflowpy/flopy/commit/573da13fd8a8069ea3d8fb189d8033eb20fc4eed): Find_path): wrap find_path() so that the routing dictionary (graph) can still be copied, but won't be copied with every call of find_path() as it recursively finds the routing connections along a path in the sfr network. this was severely impacting performance for large sfr networks. (#955). Committed by aleaf on 2020-08-11.
* [fix(#958)](https://github.com/modflowpy/flopy/commit/2fd9326534d2782428111ae47dfac1af0c21e04d): Fix gridintersect._vtx_grid_to_shape_generator rtree for cells with more than 3 vertices (#959). Committed by mkennard-aquaveo on 2020-08-07.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/a6b6c57150e9b28022bdc878090d0f1f9b0af261): Fix crash intersecting 3d points that are outside… (#957). Committed by mkennard-aquaveo on 2020-08-05.
* [fix(mvr)](https://github.com/modflowpy/flopy/commit/49b4aa27e84c8307e8ff15cc6d1a661031af6776): Added documentation explaining the difference between the two flopy mvr classes (#950). Committed by spaulins-usgs on 2020-07-31.
* [fix(check)](https://github.com/modflowpy/flopy/commit/cd81a0911374eb1af2a9fd5ab31a3c6df7baf675): Get_active() error for confining bed layers (#937). Committed by Michael Ou on 2020-07-29.
* [fix(rcha and evta)](https://github.com/modflowpy/flopy/commit/3640b1a15667b30b04a21a0ba55b6c063747a1b6): Irch and ievt arrays now zero based.  fix for issue #941.  also changed createpackages.py to always produce python scripts with unix-style line endings. (#949). Committed by spaulins-usgs on 2020-07-29.
* [fix(lgrutil)](https://github.com/modflowpy/flopy/commit/3e7e3679deb0d440a139a6d2a71ad174c4d65a12): Corrected bug in child to parent vertical exchanges (#948). Committed by langevin-usgs on 2020-07-27.
* [fix(time series slowness)](https://github.com/modflowpy/flopy/commit/47572f0ecbc024df1dfa57bcdff11175416f6f26): Speeds up performance when a large number of time series are used by a single stress package (#943). Committed by spaulins-usgs on 2020-07-17.
* [fix](https://github.com/modflowpy/flopy/commit/d16411dda8eae387da3bfbb8c2c9a32d069711e6): Restore python 3.5 compatibility (modulenotfounderror -> importerror) (#933). Committed by Mike Taves on 2020-07-09.
* [fix(read binary file)](https://github.com/modflowpy/flopy/commit/f437e8bcd135fee09e6fd4aa7d4691e589ab39c0): Fix for reading binary files with array data (#931). Committed by spaulins-usgs on 2020-06-30.
* [fix(idomain)](https://github.com/modflowpy/flopy/commit/27e4191875a59c09a2339a481d84f1052c89f0f3): Data checks now treat all positive idomain values as active  (#929). Committed by spaulins-usgs on 2020-06-29.
* [fix(Modpath7Sim)](https://github.com/modflowpy/flopy/commit/b27fccdee589c772e6e66160f27d60a8dd485645): Fix timepointinterval calculation for timepointoption 3 (#927). Committed by jdhughes-usgs on 2020-06-26.

### Version 3.3.1

#### New features

* [feat(ModflowAg)](https://github.com/modflowpy/flopy/commit/c89b2d83a81125c986c05df57ffc980fe10fd663): Add the modflowag package for nwt (pull request #922). Committed by Joshua Larsen on 2020-06-25.
* [feat(GridIntersect)](https://github.com/modflowpy/flopy/commit/176c9b45eb158217b1aeb0657d60ed06739e53e6): #902 (#903). Committed by Davíd Brakenhoff on 2020-06-10.
* [feat(mbase)](https://github.com/modflowpy/flopy/commit/4fc61d573c108be7601a2e8479a86c35040998c2): Suppress duplicate package warning if verbose is false (#908). Committed by Hughes, J.D on 2020-06-09.
* [feat(ModflowSms)](https://github.com/modflowpy/flopy/commit/f539d072e20cb5654fab8826ac3bbaf7d1f8b52e): Add support for simple, moderate, complex (#906). Committed by langevin-usgs on 2020-06-09.
* [feat(str)](https://github.com/modflowpy/flopy/commit/3182f57479c1a45d614151867f419a0d35f7f2c2): Add irdflg and iptflg control to str (#905). Committed by Hughes, J.D on 2020-06-09.
* [feat(Mf6ListBudget)](https://github.com/modflowpy/flopy/commit/04b74bd8fb4ed71cf0ed07f721204296b9fd4388): Add support for mf6 budgets with multiple packages with same name (#900). Committed by langevin-usgs on 2020-06-02.
* [feat(mfchd.py)](https://github.com/modflowpy/flopy/commit/88fbd4ccdf837dd4b2a70795522fcaa0a2470dfe): Prevent write chk-file always (#869). Committed by Ralf Junghanns on 2020-05-08.
* [feat(set all data external)](https://github.com/modflowpy/flopy/commit/0336b5c4f6c62994a7ffea26318240fbaa0ca1fb): Set all data external for simulation (#846). Committed by spaulins-usgs on 2020-04-08.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/e43dccc29d87e182aaa82687968d98391fa78e5d): Improve export at vertices (#844). Committed by Etienne Bresciani on 2020-04-06.
* [feat(netcdf)](https://github.com/modflowpy/flopy/commit/fb942b4675c335247b81f9287bdb8cdb3bcf360b): Use modern features from pyproj>=2.2.0 (#840). Committed by Mike Taves on 2020-03-31.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/b975368a9c5a25df616b024a08450056e92b6426): Export vectors to vtk. Committed by Etienne Bresciani on 2020-03-31.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/13c1dd28b9c470c70aac624b70888f5f6badf795): Export in .vti and .vtr when possible. Committed by Etienne Bresciani on 2020-03-28.
* [feat(vectors)](https://github.com/modflowpy/flopy/commit/1284871e5b8ed26c9d925e28295c2098a76b1bc9): Vector plots when stresses applied along boundaries (#817). Committed by langevin-usgs on 2020-03-05.
* [feat(plot_bc)](https://github.com/modflowpy/flopy/commit/207cd1ee5dde116e5b8912232d9b94a9c9304161): Updated plot_bc for maw, uzf, sfr, and multiple bc packages (#808). Committed by Joshua Larsen on 2020-02-11.
* [feat(disl grids)](https://github.com/modflowpy/flopy/commit/469727bcb695ee42611f3df227830ef3b25853f6): Support for 1d vertex grids. fix for writing gridlines to shapefile (#799). Committed by spaulins-usgs on 2020-02-04.
* [feat(zb netcdf)](https://github.com/modflowpy/flopy/commit/491f4fe3923a942e9a0d535a7da54173c4df1204): Zonebudget netcdf export support added  (#781). Committed by Joshua Larsen on 2020-01-17.
* [feat(mf6 checker)](https://github.com/modflowpy/flopy/commit/088f147f1d761875f3a1ffabdda82312c002305a): Input data check for mf6 (#779). Committed by spaulins-usgs on 2020-01-16.


#### Bug fixes

* [fix(#280, #835)](https://github.com/modflowpy/flopy/commit/6dae1b0ed5df6836ce6ea981de7050621a90477f): Set_all_data_external now includes constants, cellids in list data are checked (#920). Committed by spaulins-usgs on 2020-06-23.
* [fix(mfsfr2.check)](https://github.com/modflowpy/flopy/commit/796b7b15385de0181cabd33d3bc1915bfa76a83b): Negative segments for lakes no longer included in segment numbering order check (#915). Committed by aleaf on 2020-06-23.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/d5672f585faa31ead48973834d9b30cf70512cba): Fixes #916 and #917 (#918). Committed by Davíd Brakenhoff on 2020-06-22.
* [fix(ZoneBudget)](https://github.com/modflowpy/flopy/commit/0207372769c5389835c555820c11fc377a4370c7): Fix faulty logic in ZoneBudget (#911). Committed by Jason Bellino on 2020-06-22.
* [fix(SwtListBudget)](https://github.com/modflowpy/flopy/commit/eed5afdd03917256b097a797ec1137c2b80fb8a1): Totim was not being read correctly for seawat list file (#910). Committed by langevin-usgs on 2020-06-10.
* [fix(Seawat.modelgrid)](https://github.com/modflowpy/flopy/commit/7a31a1d7a210894d9970e2e7a3a6a6453518289b): Pass lenuni from dis file into modelgrid instance (#901). Committed by Joshua Larsen on 2020-06-02.
* [fix(mfsimulation)](https://github.com/modflowpy/flopy/commit/0958b28added4b572b168c35657f1d5fd968e952): Repair change for case insensitive model names (#897). Committed by langevin-usgs on 2020-06-01.
* [fix(_plot_util3d_helper, export_array)](https://github.com/modflowpy/flopy/commit/617b98dd0f5750c7cf4d1a9e8376e00fdb4e94bf): (#895). Committed by Joshua Larsen on 2020-06-01.
* [fix()](https://github.com/modflowpy/flopy/commit/8e70c9284490b220ef6fe759b0fafbb5d252f1ae): fix building in clean env (#894). Committed by Ritchie Vink on 2020-05-28.
* [fix(setup.py)](https://github.com/modflowpy/flopy/commit/b80a89c18821fe1bc7375c15e4c03d1e0be70916): Read package name, version, etc. from version.py (#893). Committed by Hughes, J.D on 2020-05-26.
* [fix(MFSimulation)](https://github.com/modflowpy/flopy/commit/3cec7929f323e7753ef80e95b31a64a6105dce41): Remove case sensitivity from register_ims_package() (#890). Committed by Joshua Larsen on 2020-05-25.
* [fix(modelgrid)](https://github.com/modflowpy/flopy/commit/bb861ac2f05e7e8d38a832d03ae0a45a2249d10d): Fix offset for xul and yul in read_usgs_model_reference_file and attribs_from_namfile_header (#889). Committed by Joshua Larsen on 2020-05-19.
* [fix(#886)](https://github.com/modflowpy/flopy/commit/f4e5ed39c5316e9bd2b7941c31f1b91253cd51e5): Mfarray aux variable now always returned as an numpy.ndarray. fixed problem with pylistutil not fully supporting numpy.ndarray type. (#887). Committed by spaulins-usgs on 2020-05-19.
* [fix(CellBudgetFile)](https://github.com/modflowpy/flopy/commit/29d7f849f8d2f160a537494661bd096a9e1c1253): Update for auto precision with imeth = 5 or 6 (#876). Committed by Joshua Larsen on 2020-05-14.
* [fix(#870)](https://github.com/modflowpy/flopy/commit/cbf1f65a8f3389f55bb55249ce8c358ac06cf75c): Update ims name file record after removing ims package (#880). Committed by Joshua Larsen on 2020-05-14.
* [fix](https://github.com/modflowpy/flopy/commit/e3a6c26fe8ad39c69cd5fec78064f12d2edbb521): #868, #874, #879 (#881). Committed by spaulins-usgs on 2020-05-14.
* [fix(#867)](https://github.com/modflowpy/flopy/commit/265fc7c64956b9165b9be2607680de086a9cf973): Fixed minimum record entry count for lists to never include keywords in the count (#875). Committed by spaulins-usgs on 2020-05-11.
* [fix(SfrFile)](https://github.com/modflowpy/flopy/commit/4100de2b80cdb542e820f3b41a88eaf9865fa63d): Update sfrfile for streambed elevation when present (#866). Committed by Joshua Larsen on 2020-05-04.
* [fix(#831,#858)](https://github.com/modflowpy/flopy/commit/a177d59c691367e568de07b340d53e8b1f2322fb): Data length check and case-insensitive lookup (#864). Committed by spaulins-usgs on 2020-05-01.
* [fix(#856)](https://github.com/modflowpy/flopy/commit/c45a5e5f1fb2f8224d87307564b68dc1c30063d9): Specifying period data with value none now only removes data from that period. also, fixed an unrelated problem by updating the multi-package list. (#857). Committed by spaulins-usgs on 2020-04-22.
* [fix(plot_array)](https://github.com/modflowpy/flopy/commit/f6545ff28004f4f6573e07aab939d774230df0a6): Update plot_array method to mask using np.ma.masked_values (#851). Committed by Joshua Larsen on 2020-04-15.
* [fix(ModflowDis)](https://github.com/modflowpy/flopy/commit/52ecd98a11fc4f459c9bd0b59cb854ff402eeb08): Zero based get_node() and get_lrc()… (#847). Committed by Joshua Larsen on 2020-04-10.
* [fix(binaryfile_utils, CellbudgetFile)](https://github.com/modflowpy/flopy/commit/f514a22a38264dc81c0db6dae62db4beae8df227): Update for imeth == 6 (#823). Committed by Joshua Larsen on 2020-04-10.
* [fix(utils.gridintersect)](https://github.com/modflowpy/flopy/commit/14040ff605dfe496d8ec88ca8eeb3f73daeee06e): Bug in gridintersect for vertex grids (#845). Committed by Davíd Brakenhoff on 2020-04-06.
* [fix(mflmt.py)](https://github.com/modflowpy/flopy/commit/1f322f0ef50be9dfaa2252cacb091aded4159113): Clean up docstring for package_flows argument in lmt. Committed by emorway-usgs on 2020-03-31.
* [fix(str)](https://github.com/modflowpy/flopy/commit/59cac5c0466b497c36e833784b436d4a5e47fcd5): Add consistent fixed and free format approach (#838). Committed by Hughes, J.D on 2020-03-31.
* [fix(vtk)](https://github.com/modflowpy/flopy/commit/aa9f3148008dd4c23d86db4e4dae5bae8c8bc564): Issues related to nan and type cast. Committed by Etienne Bresciani on 2020-03-31.
* [fix(pyproj)](https://github.com/modflowpy/flopy/commit/9a1b9e344aefad4ed222bd8d10eb6b9000c9c7d0): Pyproj.proj's errcheck keyword option not reliable (#837). Committed by Mike Taves on 2020-03-29.
* [fix](https://github.com/modflowpy/flopy/commit/4bfab916592784180d57df59b78d190e28c9099b): (#830): handling 'none' cellids in sfr package (#834). Committed by spaulins-usgs on 2020-03-25.
* [fix()](https://github.com/modflowpy/flopy/commit/90a498ea2356e0eca7ddbf1c89a0f5f12f14b5a3): Fix quasi3d plotting (#819). Committed by Ruben Caljé on 2020-03-02.
* [fix(ModflowSfr2.export)](https://github.com/modflowpy/flopy/commit/b6c9a3a6668ee8a35fda785da3709bd4842b7ba9): Update modflowsfr2.export() to use grid instead of spatialreference (#820). Committed by Joshua Larsen on 2020-02-28.
* [fix(flopy3_MT3DMS_examples.ipynb)](https://github.com/modflowpy/flopy/commit/65e8adaeea55a14a3d61c125801741c5b9ceb699): Wrong indices were indexed (#815). Committed by Eric Morway on 2020-02-20.
* [fix(build_exes.py)](https://github.com/modflowpy/flopy/commit/f3136a233274baa63ace7555d862c43bc67894f5): Fix bugs for windows (#810) (#813). Committed by Etienne Bresciani on 2020-02-18.
* [fix(mflist)](https://github.com/modflowpy/flopy/commit/4a885a107ee5ead5a116f95bb1f7f60549cba295): Default value not working (#812). Committed by langevin-usgs on 2020-02-13.
* [fix(flopy3_MT3DMS_examples.ipynb)](https://github.com/modflowpy/flopy/commit/0d152513d4ecb84fd474cb0495dd0676f05d5b72): Match constant heads to original problem (#809). Committed by Eric Morway on 2020-02-11.
* [fix(vtk)](https://github.com/modflowpy/flopy/commit/ab8120bc3dc1214fef90077ade226471900b89d5): Change in export_cbc output file name (#795). Committed by rodrperezi on 2020-01-30.
* [fix(mf6)](https://github.com/modflowpy/flopy/commit/84ee8a72b00fbcd8e7f396af9b356170634cfda6): Update create packages to support additional models (#790). Committed by Hughes, J.D on 2020-01-24.
* [fix(remove_package)](https://github.com/modflowpy/flopy/commit/102d25b23dc0281cccf65eb0a9d3cc8b05755484): Fixed remove_package to rebuild namefile recarray correctly after removing package - issue #776 (#784). Committed by spaulins-usgs on 2020-01-17.
* [fix(gridgen)](https://github.com/modflowpy/flopy/commit/061c72771ae4bc023e970bf9048a4c53bba2ba80): X, y center was not correct for rotated grids (#783). Committed by langevin-usgs on 2020-01-16.
* [fix(mtssm.py)](https://github.com/modflowpy/flopy/commit/f7886e2762464f69ae6b1e3ef1124bbee017b8a8): Handle 1st stress period with incrch equal to -1 (#780). Committed by Eric Morway on 2020-01-16.
* [fix(vtk)](https://github.com/modflowpy/flopy/commit/73f70518238ff383628ba25066c41647e875dc2e): Change in export_model when packages_names is none (#770). Committed by rodrperezi on 2019-12-30.
* [fix()](https://github.com/modflowpy/flopy/commit/ec71e6ce2d2f4bc3d05e3172cccba6c42bf26b87): fix(repeating blocks) (#771). Committed by spaulins-usgs on 2019-12-30.
* [fix(replace package)](https://github.com/modflowpy/flopy/commit/cf3586f0602acd20c9c672b2c763f09c6a252e00): When a second package of the same type is added to a model, the model now checks to see if the package type supports multiple packages. if it does not it automatically removes the first package before adding the second (#767). (#768). Committed by spaulins-usgs on 2019-12-18.
* [fix(Mflist)](https://github.com/modflowpy/flopy/commit/51d80641e81f844490306d2cddefacb21bda8302): Allow none as a list entry (#765). Committed by langevin-usgs on 2019-12-16.

### Version 3.3.0

* Dropped support for python 2.7
* Switched from [pangeo binder](https://aws-uswest2-binder.pangeo.io/) binder to [mybinder.org binder](https://mybinder.org)
* Added support for MODFLOW 6 Skeletal Compaction and Subsidence (CSUB) package

#### Bug fixes

* Fix issue in MNW2 when the input file had spaced between lines in Dataset 2. [#736](https://github.com/modflowpy/flopy/pull/736)
* Fix issue in MNW2 when the input file uses wellids with inconsistent cases in Dataset 2 and 4. Internally the MNW2 will convert all wellids to lower case strings. [#736](https://github.com/modflowpy/flopy/pull/736)
* Fix issue with VertexGrid plotting errors, squeeze proper dimension for head output, in `PlotMapView` and `PlotCrossSection`
* Fix issue in `PlotUtilities._plot_array_helper` mask MODFLOW-6 no flow and dry cells before plotting
* Removed assumption that transient SSM data appears in the first stress period [#754](https://github.com/modflowpy/flopy/issues/754) [#756](https://github.com/modflowpy/flopy/issues/754).  Fix includes a new autotest ([t068_test_ssm.py](https://github.com/modflowpy/flopy/blob/develop/autotest/t068_test_ssm.py)) that adds transient concentration data after the first stress period.
* Fix issues with add_record method for MfList [#758](https://github.com/modflowpy/flopy/pull/758)

### Version 3.2.13

* ModflowFlwob: Variable `irefsp` is now a zero-based integer (#596)
* ModflowFlwob: Added a load method and increased precision of `toffset` when writing to file (#598)
* New feature GridIntersect (#610): The GridIntersect object allows the user to intersect shapes (Points, LineStrings and Polygons) with a MODFLOW grid. These intersections return a numpy.recarray containing the intersection information, i.e. cell IDs, lengths or areas, and a shapely representation of the intersection results. Grids can be structured or vertex grids. Two intersections methods are implemented: `"structured"` and `"strtree"`: the former accelerates intersections with structured grids. The latter is more flexible and also works for vertex grids. The GridIntersect class is available through `flopy.utils.gridintersect`. The functionality requires the Shapely module. See the [example notebook](../examples/Notebooks/flopy3_grid_intersection_demo.ipynb) for an overview of this new feature.
* New feature `Raster` (#634): The Raster object allows the user to load raster files (Geotiff, arc ascii, .img) and sample points, sample polygons, create cross sections, crop, and resample raster data to a Grid.  Cropping has been implemented using a modified version of the ray casting algorithm for speed purposes. Resampling a raster can be performed with structured, vertex, and unstructured Grids. Rasters will return a numpy array of resampled data in the same shape as the Grid. The Raster class is available by calling `flopy.utils.Raster`. The functionality requires rasterio, affine, and scipy. See the ([example notebook](../examples/Notebooks/flopy3_raster_intersection.ipynb)) for an overview of this feature.
* Modify NAM and MFList output files to remove excessive whitespace (#622, #722)
* Deprecate `crs` class from `flopy.utils.reference` in favor of `CRS` class from `flopy.export.shapefile_utils` (#608)
* New feature in `PlotCrossSection` (#660). Added a `geographic_coords` parameter flag to `PlotCrossSection` which allows the user to plot cross sections with geographic coordinates on the x-axis. The default behavior is to plot distance along cross sectional line on the x-axis. See the ([example notebook](../examples/Notebooks/flopy3_PlotCrossSection_demo.ipynb)) for an overview of this feature.
* New feature with binaryfile readers, including `HeadFile` and `CellBudgetFile` (#669): [`with` statement](https://docs.python.org/3/reference/compound_stmts.html#with) is supported to open files for reading in a context manager, which automatically close when done or if an exception is raised.
* Improved the flopy list reader for MODFLOW stress packages (for mf2005, mfnwt, etc.), which may use SFAC to scale certain columns depending on the package.  The list reading now supports reading from external files, in addition to open/close.  The (binary) option is also supported for both open/close and external.  This new list reader is used for reading standard stress package lists and also lists used to create parameters.  The new list reader should be consistent with MODFLOW behavior.
* SfrFile detects additional columns (#708)
* Add a `default_float_format` property to mfsfr2, which is string formatted by NumPy versions > 1.14.0, or `{:.8g}` for older NumPy versions (#710)
* Support for pyshp 1.2.1 dropped, pyshp 2.1.0 support maintained
* Improved VTK export capabilities.  Added export for VTK at array level, package level, and model level.  Added binary head file export and cell by cell file export.  Added the ability to export point scalars in addition to cell scalars, and added smooth surface generation.  VTK export now supports writing transient data as well as exporting to binary .vtu files.
* Support for copying model and package instances with `copy.deepcopy()`
* Added link to Binder on [README](README.md) and [notebooks_examples](../examples/docs/notebook_examples.md) markdown documents. Binder provides an environment that runs and interactively serves the FloPy Jupyter notebooks.

#### Bug fixes

* When using the default `iuzfbnd=None` in the `__init__` routine of mtuzt.py, instantiation of IUZBND was generating a 3D array instead of a 2D array.  Now generates a 2D array
* ModflowSfr2 `__init__` was being slowed considerably by the `ModflowSfr2.all_segments` property method. Modified the `ModflowSfr2.graph` property method that describes routing connections between segments to handle cases where segments aren't listed in stress period 0.
* Ensure disordered fields in `reach_data` (Dataset 2) can be supported in `ModflowSfr2` and written to MODFLOW SFR input files.
* When loading a MF model with UZF active, item 8 ("`[IUZROW]` `[IUZCOL]` `IFTUNIT` `[IUZOPT]`") wasn't processed correctly when a user comment appeared at the end of the line
* MODFLOW-6 DISU JA arrays are now treated as zero-based cell IDs. JA, IHC, CL12 are outputted as jagged arrays.
* Models with multiple MODFLOW-6 WEL packages now load and save correctly.
* Exporting individual array and list data to a shapefile was producing an invalid attribute error. Attribute reference has been fixed.
* Fix UnboundLocalError and typo with `flopy.export.shapefile_utils.CRS` class (#608)
* Fix Python 2.7 issue with `flopy.export.utils.export_contourf` (#625)
* When loading a MT3D-USGS model, keyword options (if used) were ignored (#649)
* When loading a modflow model, spatial reference information was not being passed to the SpatialReference class (#659)
* Fix specifysurfk option in UZF, ModflowUZF1 read and write surfk variable
* Fix minor errors in flopy gridgen wrapper
* Close opened files after loading, to reduce `ResourceWarning` messages (#673)
* Fix bugs related to flake8's F821 "undefined name 'name'", which includes issues related to Mt3dPhc, ModflowSfr2, ModflowDe4, ListBudget, and ModflowSms (#686)
* Fix bugs related to flake8's F811 "redefinition of unused 'name'" (#688)
* Fix bugs related to flake8's W605 "invalid escape sequence '\\s'" (or similar) (#700)
* Fix EpsgReference class behavior with JSON user files (#702)
* Fix ModflowSfr2 read write logic for all combinations of isfropt and icalc
* IRCH array of the Recharge Package is now a zero-based variable, which means an IRCH value of 0 corresponds to the top model layer (#715)
* MODFLOW lists were not always read correctly if they used the SFAC or binary options or were used to define parameters (#683)
* Changed VDF Package density limiter defaults to zero (#646)

### Version 3.2.12

* Added a check method for OC package (#558)
* Change default map projection from EPSG:4326 to None (#535)
* Refactor warning message visibility and categories (#554, #575)
* Support for MODFLOW 6 external binary files added. Flopy can read/write binary files containing list and array data (#470, #553).
* Added silent option for MODFLOW 6 write_simulation (#552)
* Refactored MODFLOW-6 data classes. File writing operations moved from mfdata*.py to new classes created in mffileaccess.py. Data storage classes moved from mfdata.py to mfdatastorage.py. MFArray, MFList, and MFScalar interface classes simplified with most of the data processing code moved to mfdatastorage.py and mffileaccess.py.
* Added MODFLOW 6 quickstart example to front page.
* Added lgrutil test as autotest/t063_test_lgrutil.py and implemented a get_replicated_parent_array() method to the Lgr class so that the user can pass in a parent array and get back an array that is the size of the child model.
* Refactored much of the flopy code style to conform with Python conventions and those checked by Codacy.  Added an automated Codacy check as part of the pull request and commit checks.

#### Bug fixes

* Fixed bug in Mt3dms.load to show correct error message when loading non-existent NAM file (#545)
* Removed errant SFT parameter contained in Mt3dUzt.__init__ routine (#572)
* Fixed DISV shapefile export bug that applied layer 1 parameter values to all model layers during export (#508)
* Updated ModflowSfr2.load to store channel_geometry and channel_flow_data (6d, 6e) by nseg instead of itmp position (#546)
* Fixed bug in ModflowMnw2.make_node_data to be able to set multiple wells with different numbers of nodes (#556)
* Fixed bug reading MODFLOW 6 comma separated files (#509)
* Fixed bug constructing a grid class with MODFLOW-USG (#513)
* Optimized performance of grid class by minimizing redundant operations through use of data result caching (#520)
* Fixed bug passing multiple auxiliary variables for MODFLOW 6 array data (#533)
* Fixed bug in Mt3dUzt.__init__;  the variable ioutobs doesn't exist in the UZT package and was removed.
* Fixed MODFLOW-LGR bug in which ascii files were not able to be created for some output.  Added better testing of the MODFLOW-LGR capabilities to t035_test.py.
* Fixed multiple issues in mfdis that resulted in incorrect row column determination when using the method get_rc_from_node_coordinates (#560).  Added better testing of this to t007_test.py.
* Fixed the export_array_contours function as contours would not export in some cases (#577).  Added tests of export_array_contours and export_array to t007_test.py as these methods were not tested at all.

### Version 3.2.11

* Added support for the drain return package.
* Added support for pyshp version 2.x, which contains a different call signature for the writer than earlier versions.
* Added a new flopy3_MT3DMS_examples notebook, which uses Flopy to reproduce the example problems described in the MT3DMS documentation report by Zheng and Wang (1999).
* Pylint is now used on Travis for the Python 3.5 distribution to check for coding errors.
* Added testing with Python 3.7 on Travis, dropped testing Python 3.4.
* Added a new htop argument to the vtk writer, which allows cell tops to be defined by the simulated head.
* Generalized exporting and plotting to also work with MODFLOW 6. Added a new grid class and deprecated SpatialReference class. Added new plotting interfaces, `PlotMapView` and `PlotCrossSection`. Began deprecation of `ModelMap` and `ModelCrossSection` classes.
* Spatial reference system cache moved to epsgref.json in the user's data directory.
* Attempts to read empty files from flopy.utils raise a IOError exception.
* Changed interface for creating and accessing MODFLOW 6 observation, time series, and time array series packages. These packages can now be created and accessed directly from the package that references them.  These changes are not backward compatible, and will require existing scripts to be modified.  See the flopy3_mf6_obs_ts_tas.ipynb notebook for instructions.
* Changed the MODFLOW 6 fname argument to be filename.  This change is not backward compatible, and will require existing scripts to be modified if the fname argument was used in the package constructor.
* Added modflow-nwt options support for `ModflowWel`, `ModflowSfr2`, and `ModflowUzf1` via the `OptionBlock` class.

#### Bug fixes

* Removed variable MXUZCON from `mtuzt.py` that was present during the development of MT3D-USGS, but was not included in the release version of MT3D-USGS.
* Now account for UZT -> UZT2 changes with the release of MT3D-USGS 1.0.1.  Use of UZT is no longer supported.
* Fixed bug in `mfuzf1.py` when reading and writing `surfk` when `specifysurfk = True`.
* Fixed bug in `ModflowStr.load()`, utility would fail to load when comments were present.
* Fixed bug in MNW2 in which nodes were not sorted correctly.
* Ensure that external 1-D free arrays are written on one line.
* Typos corrected for various functions, keyword arguments, property names, input file options, and documentation.
* Fixed bug in `Mt3dUzt.__init__` that originated when copying code from mtsft.py to get started on mtuzt.py class.  The variable ioutobs doesn't exist in the UZT package and should never have appeared in the package to begin with.

### Version 3.2.10

* Added parameter_load variable to `mbase` that is set to true if parameter data are applied in the model (only used in models that support parameters). If this is set to `True` `free_format_input` is set to `True` (if currently `False`) when the `write_input()` method is called. This change preserves the precision of parameter data (which is free format data).
* MODFLOW 6 model and simulation packages can not be retrieved as a `MFSimulation` attribute
* Added support for multicomponent load in `mfsft.py`
* Added functionality to read esri-style epsg codes from [spatialreference.org](https://spatialreference.org).
* Added functionality to MODFLOW 6 that will automatically replace the existing package with the one being added if it has the same name as the existing package.
* Added separate MODFLOW 6 model classes for each model type. Model classes contain name file options.
* Added standard `run_model()` method arguments to mf6 `run_simulation()` method.
* some performance improvements to checking
* `SpatialReference.export_array()` now writes 3-D numpy arrays to multiband GeoTiffs
* Add load support to for MNW1; ModflowMnw1 now uses a `stress_period_data` `Mflist` to store MNW information, similar to other BC packages.
* Added a Triangle class that is a light wrapper for the Triangle program for generating triangular meshes.  Added a notebook called flopy3_triangle.ipynb that demonstrates how to use it and build a MODFLOW 6 model with a triangular mesh.  The current version of this Triangle class should be considered beta functionality as it is likely to change.
* Added support for MODPATH 7 (beta).
* Added support for MODPATH 3 and 5 pathline and endpoint output files.
* Added support for MODPATH timeseries output files (`flopy.utils.TimeseriesFile()`).
* Added support for plotting MODPATH timeseries output data (`plot_timeseries()`) with ModelMap.

#### Bug fixes

* Fixed issue in HOB when the same layer is specified in the `MLAY` data (dataset 4). If the layer exists the previous fraction value is added to the current value.
* Fixed bug in segment renumbering
* Changed default value for `ioutobs` `**kwargs` in `mtsft.py` from None to 0 to prevent failure.
* Fixed bug when passing extra components info from load to constructor in `mtsft.py` and `mtrct.py`.
* Fixed bug in `mt3ddsp` load - if `multidiffusion` is not found, should only read one 3d array.
* Fixed bug in `zonbud` utility that wasn't accumulating flow from constant heads.
* Fixed minor bug that precluded the passing of mass-balance record names (`TOTAL_IN`, `IN-OUT`, etc.).
* Fixed bug when writing shapefile projection (`.prj`) files using relative paths.
* Fixed bugs in `sfr.load()` -- `weight` and `flwtol` should be cast as floats, not integers.
* Fixed bug when `SpatialReference` supplied with geographic CRS.
* Fixed bug in `mfsfr.py` when writing kinematic data (`irtflg >0`).
* Fixed issue from change in MODFLOW 6 `inspect.getargspec()` method (for getting method arguments).
* Fixed MODFLOW 6 BINARY keyword for reading binary data from a file using  `OPEN/CLOSE` (needs parentheses around it).
* Fixed bug in `mtlkt.py` when initiating, loading, and/or writing lkt input file related to multi-species problems.


### Version 3.2.9

* Modified MODFLOW 5 OC stress_period_data=None default behaviour. If MODFLOW 5 OC stress_period_data is not provided then binary head output is saved for the last time step of each stress period.
* added multiple component support to ``mt3dusgs SFT`` module
* Optimized loading and saving of MODFLOW 6 files
* MODFLOW 6 identifiers are now zero based
* Added remove_package method in MFSimulation and MFModel that removes MODFLOW 6 packages from the existing simulation/model
* Changed some of the input argument names for MODFLOW 6 classes.  Note that this will break some existing user scripts.  For example, the stress period information was passed to the boundary package classes using the periodrecarray argument.  The argument is now called stress_period_data in order to be consistent with other Flopy functionality.
* Flopy code for MODFLOW 6 generalized to support different model types
* Flopy code for some MODFLOW 6 arguments now have default values in order to be consistent with other Flopy functionality
* Added `ModflowSfr2.export_transient_variable` method to export shapefiles of segment data variables, with stress period data as attributes
* Added support for UZF package gages

#### Bug fixes

* Fixed issue with default settings for MODFLOW 5 SUB package `dp` dataset.
* Fixed issue if an external BC list file has only one entry
* Some patching for recarray issues with latest ``numpy`` release (there are more of these lurking...)
* Fixed setting model relative path for MODFLOW 6 simulations
* Python 2.7 compatibility issues fixed for MODFLOW 6 simulations
* IMS file name conflicts now automatically resolved
* Fixed issue with passing in numpy ndarrays arrays as layered data
* Doc string formatting for MODFLOW 6 packages fixed to make doc strings easier to read
* UZF package: fixed issues with handling of finf, pet, extdp and extwc arrays.
* SFR package: fixed issue with reading stress period data where not all segments are listed for periods > 0.
* `SpatialReference.write_gridSpec` was not converting the model origin coordinates to model length units.
* shorted integer field lengths written to shapefiles to 18 characters; some readers may misinterpret longer field lengths as float dtypes.


### Version 3.2.8

* Added `has_package(name)` method to see if a package exists. This feature goes nicely with `get_package(name)` method.
* Added `set_model_units()` method to change model units for all files created by a model. This method can be useful when creating MODFLOW-LGR models from scratch.
* Added SFR2 package functionality
	* `export_inlets` method to write shapefile showing locations where external flows are entering the stream network.

#### Bug fixes

* Installation: Added dfn files required by MODFLOW 6 functionality to MANIFEST.in so that they are included in the distribution.
* SFR2 package: Fixed issue reading transient data when `ISFOPT` is 4 or 5 for the first stress period.


### Version 3.2.7

* Added beta support for MODFLOW 6 See [here](./mf6.md) for more information.
* Added support for retrieving time series from binary cell-by-cell files. Cell-by-cell time series are accessed in the same way they are accessed for heads and concentrations but a text string is required.
* Added support for FORTRAN free format array data using n*value where n is the number of times value is repeated.
* Added support for comma separators in 1D data in LPF and UPF files
* Added support for comma separators on non array data lines in DIS, BCF, LPF, UPW, HFB, and RCH Packages.
* Added `.reset_budgetunit()` method to OC package to facilitate saving cell-by-cell binary output to a single file for all packages that can save cell-by-cell output.
* Added a `.get_residual()` method to the `CellBudgetFile` class.
* Added support for binary stress period files (`OPEN/CLOSE filename (BINARY)`) in `wel` stress packages on load and instantiation. Will extend to other list-based MODFLOW stress packages.
* Added a new `flopy.utils.HeadUFile` Class (located in binaryfile.py) for reading unstructured head files from MODFLOW-USG.  The `.get_data()` method for this class returns a list of one-dimensional head arrays for each layer.
* Added metadata.acdd class to fetch model metadata from ScienceBase.gov and manage CF/ACDD-complaint metadata for NetCDF export
* Added sparse export option for boundary condition stress period data, where only cells for that B.C. are exported (for example, `package.stress_period_data.export('stuff.shp', sparse=True)`)
* Added additional SFR2 package functionality:
	*  `.export_linkages()` and `.export_outlets()` methods to export routing linkages and outlets
	*  sparse shapefile export, where only cells with SFR reaches are included
	*  `.plot_path()` method to plot streambed elevation profile along sequence of segments
	*  `.assign_layers()` method
	* additional error checks and bug fixes
* Added `SpatialReference` / GIS export functionality:
	*  GeoTiff export option to `SpatialReference.export_array`
	*  `SpatialReference.export_array_contours`: contours an array and then exports contours to shapefile
	*  inverse option added to `SpatialReference.transform`
	*  automatic reading of spatial reference info from .nam or usgs.model.reference files
* Modified node numbers in SFR package and `ModflowDis.get_node()` from one- to zero-based.
* Modified HYDMOD package `klay` variable from one- to zero-based.
* Added `.get_layer()` method to DIS package.
* Added `.get_saturated_thickness()` and `.get_gradients()` methods

#### Bug fixes

* OC package: Fixed bug when printing and saving data for select stress periods and timesteps. In previous versions, OC data was repeated until respecified.
* SUB package: Fixed bug if data set 15 is passed to preserved unit numbers (i.e., use unit numbers passed on load).
* SUB and SUB-WT packages: Fixed bugs `.load()` to pop original unit number.
* BTN package: Fixed bug in obs.
* LPF package: Fixed bug regarding when HANI is read and written.
* UZF package: added support for MODFLOW NWT options block; fixed issue with loading files with thti/thtr options
* SFR package: fixed bug with segment renumbering, issues with reading transient text file output,
* Fixed issues with dynamic setting of `SpatialReference` parameters
* NWT package: forgive missing value for MXITERXMD
* MNW2 package: fix bug where ztop and zbotm were written incorrectly in `get_allnode_data()`. This was not affecting writing of these variables, only their values in this summary array.
* PCGN package: fixed bug writing package.
* Fixed issue in `Util2d` when non-integer `cnstnt` passed.


### Version 3.2.6

* Added functionality to read binary grd file for unstructured grids.
* Additions to SpatialReference class:
	* xll, yll input option
	* transform method to convert model coordinates to real-world coordinates
	* epsg and length_multiplier arguments
* Export:
	* Added writing of prj files to shapefile export; prj information can be passed through spatial reference class, or given as an EPSG code or existing prj file path
	* Added NetCDF export to MNW2
* Added MODFLOW support for:
    * FHB Package - no support for flow or head auxiliary variables (datasets 2, 3, 6, and 8)
    * HOB Package
* New utilities:
	* `flopy.utils.get_transmissivities()` Computes transmissivity in each model layer at specified locations and open intervals. A saturated thickness is determined for each row, column or x, y location supplied, based on the well open interval (sctop, scbot), if supplied, otherwise the layer tops and bottoms
    and the water table are used.
* Added MODFLOW-LGR support - no support for model name files in different directories than the directory with the lgr control file.
* Additions to MODPATH:
	* shapefile export of MODPATH Pathline and Endpoint data
	* Modpath.create_mpsim() supports MNW2
	* creation of MODPATH StartingLocations files
	* Easy subsetting of endpoint and pathline results to destination cells of interest
* New ZoneBudget class provides ZONEBUDGET functionality:
    * reads a CellBudgetFile and accumulates flows by zone
    * pass `kstpkper` or `totim` keyword arguments to retrieve a subset of available times in the CellBudgetFile
    * includes a method to write the budget recarrays to a .csv file
    * ZoneBudget objects support numerical operators to facilitate conversion of units
    * utilities are included which read/write ZONEBUDGET-style zone files to and from numpy arrays
    * pass a dictionary of {zone: "alias"} to rename fields to more descriptive names (e.g. {1: 'New York', 2: 'Delmarva'}
* Added new precision='auto' option to flopy.utils.binaryfile for HeadFile and UcnFile readers.  This will automatically try and determine the float precision for head files created by single and double precision versions of MODFLOW.  'auto' is now the default.  Not implemented yet for cell by cell flow file.
* Modified MT3D-related packages to also support MT3D-USGS
  * BTN will support the use of keywords (e.g., 'MODFLOWStyleArrays', etc.) on the first line
  * DSP will support the use of keyword NOCROSS
  * Keyword FREE now added to MT3D name file when the flow-transport link (FTL) file is formatted.  Previously defaulted to unformatted only.
* Added 3 new packages:
  * SFT: Streamflow Transport, companion transport package for use with the SFR2 package in MODFLOW
  * LKT: Lake Transport, companion transport package for use with the LAK3 package in MODFLOW
  * UZT: Unsaturated-zone Transport, companion transport package for use with the UZF1 package in MODFLOW
* Modified LMT
  * load() functionality will now support optional PACKAGE_FLOWS line (last line of LMT input)
  * write_file() will will now insert PACKAGE_FLOWS line based on user input

#### Bug fixes

* Fixed bug in parsenamefile when file path in namefile is surrounded with quotes.
* Fixed bug in check routine when THICKSTRT is specified as an option in the LPF and UPW packages.
* Fixed bug in BinaryHeader.set_values method that prevented setting of entries based on passed kwargs.
* Fixed bugs in reading and writing SEAWAT Viscosity package.
* The DENSE and VISC arrays are now Transient3d objects, so they may change by stress period.
* MNW2: fixed bug with k, i, j node input option and issues with loading at model level
* Fixed bug in ModflowDis.get_cell_volumes().


### Version 3.2.5

* Added support for LAK and GAGE packages - full load and write functionality supported.
* Added support for MNW2 package. Load and write of .mnw2 package files supported. Support for .mnwi, or the results files (.qsu, .byn) not yet implemented.
* Improved support for changing the output format of arrays and variables written to MODFLOW input files.
* Restructured SEAWAT support so that packages can be added directly to the SEAWAT model, in addition to the approach of adding a modflow model and a mt3d model.  Can now load a SEAWAT model.
* Added load support for MT3DMS Reactions package
* Added multi-species support for MT3DMS Reactions package
* Added static method to Mt3dms().load_mas that reads an MT3D mass file and returns a recarray
* Added static method to Mt3dms().load_obs that reads an MT3D mass file and returns a recarray
* Added method to flopy.modpath.Modpath to create modpath simulation file from modflow model instance boundary conditions. Also added examples of creating modpath files and post-processing modpath pathline and endpoint files to the flopy3_MapExample notebook.

#### Bug fixes

* Fixed issue with VK parameters for LPF and UPW packages.
* Fixed issue with MT3D ADV load in cases where empty fields were present in the first line of the file.
* Fixed cross-section array plotting issues.
* BTN observation locations must now be entered in zero-based indices (a 1 is now added to the index values written to btn file)
* Uploaded supporting files for SFR example notebook; fixed issue with segment_data submitted as array (instead of dict) and as 0d array(s).
* Fixed CHD Package so that it now supports options, and therefore, auxiliary variables can be specified.
* Fixed loading BTN save times when numbers are touching.

### Version 3.2.4

* Added basic model checking functionality (`.check()`).
* Added support for reading SWR Process observation, stage, budget, flow, reach-aquifer exchanges, and structure flows.
* `flopy.utils.HydmodObs` returns a numpy recarray. Previously numpy arrays were returned except when the `slurp()` method was used. The slurp method has been deprecated but the same functionality is available using the `get_data()` method. The recarray returned from the `get_data()` method includes the `totim` value and one or all of the observations (`HYDLBL`).
* Added support for MODFLOW-USG DISU package for unstructured grids.
* Added class (`Gridgen`) for creating layered quadtree grids using GRIDGEN (`flopy.utils.gridgen`). See the flopy3_gridgen notebook for an example of how to use the `Gridgen` class.
* Added user-specified control on use of `OPEN/CLOSE` array options (see flopy3_external_file_handling notebook).
* Added user-specified control for array output formats (see flopy3_array_outputformat_options IPython notebook).
* Added shapefile as optional output format to `.export()` method and deprecated `.to_shapefile()` method.

#### Bug fixes

* Fixed issue with right justified format statement for array control record for MT3DMS.
* Fixed bug writing PHIRAMP for MODFLOW-NWT well files.
* Fixed bugs in NETCDF export methods.
* Fixed bugs in LMT and BTN classes.

### Version 3.2.3

* Added template creation support for several packages for used with PEST (and UCODE).
* Added support for the SEAWAT viscosity (VSC) package.
* Added support for the MODFLOW Stream (STR), Streamflow-Routing (SFR2), Subsidence (SUB), and Subsidence and Aquifer-System Compaction Package for Water-Table Aquifers (SWT) Packages.
* Mt3d model was redesigned based on recent changes to the Modflow model.  Mt3d packages rewritten to support multi-species.  Primary packages can be loaded (btn, adv, dsp, ssm, gcg).  Array utilities modified to read some MT3D RARRAY formats.
* Fixed array loading functionality for case when the CNSTNT value is zero.  If CNSTNT is zero and is used as an array multiplier, it is changed to 1 (as done in MODFLOW).
* Added support for the MODFLOW HYDMOD (HYD) Package and reading binary files created by the HYDMOD Package (`HydmodObs` Class) in the `flopy.utils` submodule.
* `flopy.utils.CellBudgetFile` returns a numpy recarray for list based budget data. Previously a dictionary with the `node` number and `q` were returned. The recarray will return the `node` number, `q`, and the `aux` variables for list based budget data.
* Added travis-ci automated testing.

### Version 3.2.2

* FloPy now supports some simple plotting capabilities for two- and three-dimensional model input data array classes  and transient two-dimensional stress period input data using the `.plot()` methods associated with the data array classes (`util_2d`, `util_3d`, and `transient_2d`). The model results reader classes (`HeadFile`, `UcnFile`, and `CellBudgetFile`) have also been extended to include a `.plot()` method that can be used to create simple plots of model output data. See the notebook [flopy3_PlotArrayExample](https://github.com/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_PlotArrayExample.ipynb).
* Added `.to_shapefile()` method to two- and three-dimensional model input data array classes (`util_2d` and `util_3d`), transient two-dimensional stress period input data classes (`transient_2d`), and model output data classes (`HeadFile`, `UcnFile`, and `CellBudgetFile`) that allows model data to be exported as polygon shapefiles with separate attribute columns for each model layer.
* Added support for ASCII model results files.
* Added support for reading MODPATH version 6 pathline and endpoint output files and plotting MODPATH results using mapping capabilities in `flopy.plot` submodule.
* Added `load()` method for MODFLOW GMG solver.

#### Bug fixes

* Multiplier in array control record was not being applied to arrays
* vani parameter was not supported

### Version 3.2.1

* FloPy can now be used with **Python 3.x**
* Revised setters for package class variables stored using the `util_2d` or `util_3d` classes.
* Added option to load a subset of MODFLOW packages in a MODFLOW model name file using `load_only=` keyword.

### Version 3.1

* FloPy now supports some simple mapping and cross-section capabilities through the `flopy.plot` submodule. See the notebook [flopy3_MapExample](https://github.com/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_MapExample.ipynb).
* Full support for all Output Control (OC) options including DDREFERENCE, SAVE IBOUND, and layer lists. All Output Control Input is specified using words. Output Control Input using numeric codes is still available in the ModflowOc88 class. The ModflowOc88 class is currently deprecated and no longer actively maintained.
* Added support for standard MULT package FUNCTION and EXPRESSION functionality are supported. MODFLOW parameters are not supported in `write()` methods.

### Version 3.0

FloPy is significantly different from earlier versions of FloPy (previously hosted on [googlecode](https://code.google.com/p/flopy/)). The main changes are:

* FloPy is fully zero-based. This means that layers, rows and columns start counting at *zero*. The reason for this is consistency. Arrays are zero-based by default in Python, so it was confusing to have a mix.
* Input for packages that take *layer, row, column, data* input (like the wel or ghb package) has changed and is much more flexible now. See the notebook [flopy3boundaries](https://github.com/modflowpy/flopy/blob/master/examples/Notebooks/flopy3boundaries.ipynb)
* Input for the MT3DMS Source/Sink Mixing (SSM) Package has been modified to be consistent with the new MODFLOW boundary package input and is more flexible than previous versions of FloPy. See the notebook [flopy3ssm](https://github.com/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_multi-component_SSM.ipynb)
* Support for use of EXTERNAL and OPEN/CLOSE array specifiers has been improved.
* *load()* methods have been developed for all of the standard MODFLOW packages and a few less used packages (*e.g.* SWI2).
* MODFLOW parameter support has been added to the `load()` methods. MULT, PVAL, and ZONE packages are now supported and parameter data are converted to arrays in the `load()` methods. MODFLOW parameters are not supported in `write()` methods.
