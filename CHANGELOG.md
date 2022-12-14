### Version 3.3.6

#### New features

* [feat(lak6)](https://github.com/modflowpy/flopy/commit/5974d8044f404a5221960e837dcce73d0db140d2): Support none lake bedleak values (#1189). Committed by jdhughes-usgs on 2021-08-16.
* [feat(mf6)](https://github.com/modflowpy/flopy/commit/6c0df005bd1f61d7ce54f160fea82cc34916165b): Allow multi-package for stress package concentrations (SPC) (#1242). Committed by langevin-usgs on 2021-09-16.
* [feat(CellBudget)](https://github.com/modflowpy/flopy/commit/45df13abfd0611cc423b01bff341b450064e3b01): Add support for full3D keyword (#1254). Committed by jdhughes-usgs on 2021-10-05.
* [feat(get_ts)](https://github.com/modflowpy/flopy/commit/376b3234a7931785518e478b422c4987b2d8fd52): Added support to get_ts for HeadUFile (#1260). Committed by Ross Kushnereit on 2021-10-09.
* [feat(Gridintersect)](https://github.com/modflowpy/flopy/commit/806d60149c4b3d821fa409d71dd0592a94070c51): Add shapetype kwarg (#1301). Committed by Davíd Brakenhoff on 2021-12-03.
* [feat(multiple package instances)](https://github.com/modflowpy/flopy/commit/2eeaf1db72150806bf0cee0aa30c813e02256741): FloPy support for multiple instances of the same package stored in dfn files (#1239) (#1321). Committed by spaulins-usgs on 2022-01-07.
* [feat(inspect cells)](https://github.com/modflowpy/flopy/commit/ee009fd2826c15bd91b81b84d1b7daaba0b8fde3): New feature that returns model data associated with specified model cells (#1140) (#1325). Committed by spaulins-usgs on 2022-01-11.
* [feat(gwtgwt-mvt)](https://github.com/modflowpy/flopy/commit/8663a3474dd42211be2b8124316f16ea2ae8de18): Add support for gwt-gwt with mover transport (#1356). Committed by langevin-usgs on 2022-02-18.
* [feat(mvt)](https://github.com/modflowpy/flopy/commit/aa0baac73db0ef65ac975511f6182148708e75be): Add simulation-level support for mover transport (#1357). Committed by langevin-usgs on 2022-02-18.
* [feat(time step length)](https://github.com/modflowpy/flopy/commit/ea6a0a190070f065d74824b421de70d4a66ebcc2): Added feature that returns time step lengths from listing file (#1435) (#1437). Committed by scottrp on 2022-06-30.
* [feat](https://github.com/modflowpy/flopy/commit/3e2b5fb0ec0fc4b58854fce6155eaa40c25603c6): Get modflow utility (#1465). Committed by Mike Taves on 2022-07-27.
* [feat(Gridintersect)](https://github.com/modflowpy/flopy/commit/4f86fcf70fcb0d4bb76af136d45fd6857730811b): New grid intersection options (#1468). Committed by Davíd Brakenhoff on 2022-07-27.
* [feat](https://github.com/modflowpy/flopy/commit/af42e7827fe053af911efef0f37dcb76dad7e9c0): Unstructured grid from specification file (#1524). Committed by w-bonelli on 2022-09-08.
* [feat(aux variable checking)](https://github.com/modflowpy/flopy/commit/c3cdc1323ed416eda7027a8d839fcc3b29d5cfaa): Check now performs aux variable checking (#1399) (#1536). Committed by spaulins-usgs on 2022-09-12.
* [feat(get/set data record)](https://github.com/modflowpy/flopy/commit/984227d8f5762a4687cbac0a8175a6b07f751aee): Updated get_data/set_data functionality and new get_record/set_record methods (#1568). Committed by spaulins-usgs on 2022-10-06.
* [feat(get_modflow)](https://github.com/modflowpy/flopy/commit/b8d471cd27b66bbb601b909043843bb7c59e8b40): Support modflow6 repo releases (#1573). Committed by w-bonelli on 2022-10-11.
* [feat(contours)](https://github.com/modflowpy/flopy/commit/00757a4dc7ea03ba3582242ba4d2590f407c0d39): Use standard matplotlib contours for StructuredGrid map view plots (#1615). Committed by w-bonelli on 2022-11-10.

#### Bug fixes

* [fix(grid, plotting)](https://github.com/modflowpy/flopy/commit/fdc4608498884526d11079ecbd3c0c3548b11a61): Bugfixes for all grid instances and plotting code (#1174). Committed by Joshua Larsen on 2021-08-09.
* [fix(raster)](https://github.com/modflowpy/flopy/commit/b893017995e01dd5c6dda5eb4347ebb707a16ece): Rework raster threads to work on osx (#1180). Committed by jdhughes-usgs on 2021-08-10.
* [fix(pakbase)](https://github.com/modflowpy/flopy/commit/74289dc397a3cd41861ddbf2001726d8489ba6af): Specify dtype=bool for 0-d active array used for USG (#1188). Committed by Mike Taves on 2021-08-16.
* [fix(MF6Output)](https://github.com/modflowpy/flopy/commit/0d9faa34974c1e9cf32b9897e9a2a68f695b0936): Fix add new obs package issue (#1193). Committed by Joshua Larsen on 2021-08-17.
* [fix(Grid.saturated_thick)](https://github.com/modflowpy/flopy/commit/e9dff6136c19631532de4a2f714092037fb81c98): Update saturated_thick to filter confining bed layers (#1197). Committed by Joshua Larsen on 2021-08-18.
* [fix(numpy)](https://github.com/modflowpy/flopy/commit/8ec8d096afb63080d754a11d7d477ac4133c0cf6): Handle deprecation warnings from numpy 1.21 (#1211). Committed by Mike Taves on 2021-08-23.
* [fix(rename and comments)](https://github.com/modflowpy/flopy/commit/c2a01df3d49b0e06a21aa6aec532125b03992048): Package rename code and comment propagation (#1226). Committed by spaulins-usgs on 2021-09-03.
* [fix(numpy elementwise comparison)](https://github.com/modflowpy/flopy/commit/779aa50afd9fb8105d0684b0ab50c64e0bc347de): Fixed numpy FutureWarning. No elementwise compare is needed if user passes in a numpy recarray, this is only necessary for dictionaries. (#1235). Committed by spaulins-usgs on 2021-09-13.
* [fix(writing tas)](https://github.com/modflowpy/flopy/commit/047c9e6fdb345c373b48b07fe0b81bc88105a543): Fixed writing non-constant tas (#1244) (#1245). Committed by spaulins-usgs on 2021-09-22.
* [fix(voronoi)](https://github.com/modflowpy/flopy/commit/ccdb36a8dd00347889e6048f7e6ddb7538e6308f): VoronoiGrid class upgraded to better support irregular domains (#1253). Committed by langevin-usgs on 2021-10-01.
* [fix(MFPackage)](https://github.com/modflowpy/flopy/commit/658bf2ccbc9c6baeec4184e7c08aeb452f83c4ea): Fix mfsim.nam relative paths (#1252). Committed by Joshua Larsen on 2021-10-01.
* [fix(_mg_resync)](https://github.com/modflowpy/flopy/commit/fe913fe6e364501832194c87deb962d913a22a35): Added checks to reset the modelgrid resync (#1258). Committed by Joshua Larsen on 2021-10-06.
* [fix(MFFileMgmt.string_to_file_path)](https://github.com/modflowpy/flopy/commit/459aacbc0ead623c79bc4701b4c23666a280fbe8): Updated to support UNC paths (#1256). Committed by Joshua Larsen on 2021-10-06.
* [fix(ModflowFhb)](https://github.com/modflowpy/flopy/commit/fc094353a375b26f5afd6befb1bcdfef407c3a64): Update datasets 4, 5, 6, 7, 8 loading routine for multiline records (#1264). Committed by Joshua Larsen on 2021-10-15.
* [fix(keystring records)](https://github.com/modflowpy/flopy/commit/0d21b925ebd4e2f31413ea67d28591d77a02d8c1): Fixed problem with keystring records containing multiple keywords (#1266). Committed by spaulins-usgs on 2021-10-16.
* [fix(1247)](https://github.com/modflowpy/flopy/commit/71f6cc6b6391a460dd01e58b739372b35899f09c): Make flopy buildable with pyinstaller (#1248). Committed by Tim Mitchell on 2021-10-20.
* [fix(io)](https://github.com/modflowpy/flopy/commit/1fa24026d890abc4508a39eddf9049399c1e4d3f): Read comma separated list (#1285). Committed by Michael Ou on 2021-11-02.
* [fix(path)](https://github.com/modflowpy/flopy/commit/220974ae98c8fbc9a278ce02eec71378fd84dc1a): Fix subdirectory path issues (#1298). Committed by Brioch Hemmings on 2021-11-18.
* [fix(geospatial_utils.py)](https://github.com/modflowpy/flopy/commit/976ad8130614374392345ce8ee3adb766d378797): Added pyshp and shapely imports check to geospatial_utils.py (#1305). Committed by Joshua Larsen on 2021-12-03.
* [fix(autotests)](https://github.com/modflowpy/flopy/commit/bbdf0082749d04805af470bc1b0f197d73202e21): Added pytest.ini to declare test naming convention (#1307). Committed by Joshua Larsen on 2021-12-03.
* [fix(plot_pathline)](https://github.com/modflowpy/flopy/commit/849f68e96c151e64b63a351b50955d0d5e01f1f7): Sort projected pathline points by travel time instead of cell order (#1304). Committed by Joshua Larsen on 2021-12-03.
* [fix(array)](https://github.com/modflowpy/flopy/commit/fad091540e3f75efa38668e74a304e6d7f6b715f): Getting array data (#1028) (#1290). Committed by spaulins-usgs on 2021-12-03.
* [fix(UnstructuredGrid)](https://github.com/modflowpy/flopy/commit/8ab2e8d1a869de14ee76b11d0c53f439560c7a72): Load vertices for unstructured grids (#1312). Committed by Chris Nicol on 2021-12-09.
* [fix(paths)](https://github.com/modflowpy/flopy/commit/0432ce96a0a5eec4d20adb4d384505632a2db3dc): Path code made more robust so that non-standard model folder structures are supported (#1311) (#1316). Committed by scottrp on 2021-12-09.
* [fix(filenames)](https://github.com/modflowpy/flopy/commit/fb0955c174cd47e41cd15eac6b2fc5e4cc587935): Fixed how spaces in filenames are handled (#1236) (#1318). Committed by spaulins-usgs on 2021-12-16.
* [fix(voronoi)](https://github.com/modflowpy/flopy/commit/24f142ab4e02ba96e04d9c5040c87cea35ad8fd9): Clean up voronoi examples and add error check (#1323). Committed by langevin-usgs on 2022-01-03.
* [fix(Raster)](https://github.com/modflowpy/flopy/commit/dced106e5f9bd9a5234a2cbcd74c67b959ca950c): Resample_to_grid failure no data masking failure with int dtype (#1328). Committed by Joshua Larsen on 2022-01-21.
* [fix(postprocessing)](https://github.com/modflowpy/flopy/commit/0616db5f158a97d1b38544d993d866c19a672a80): Get_structured_faceflows fix to support 3d models (#1333). Committed by langevin-usgs on 2022-01-21.
* [fix(cellid)](https://github.com/modflowpy/flopy/commit/33e61b8ccd2556b65d33d25c5a3fb27114f5ff25): Fixes some issues with flopy properly identifying cell ids (#1335) (#1336). Committed by spaulins-usgs on 2022-01-25.
* [fix(ModflowUtllaktab)](https://github.com/modflowpy/flopy/commit/0ee7c8849a14bc1a0736d5680c761b887b9c04db): Utl-lak-tab.dfn is redundant to utl-laktab.dfn (#1339). Committed by Mike Taves on 2022-01-27.
* [fix(tab files)](https://github.com/modflowpy/flopy/commit/b59f1fef153a235b0e11d40107c1e30d5eae4f84): Fixed searching for tab file packages (#1337) (#1344). Committed by scottrp on 2022-02-16.
* [fix(exchange obs)](https://github.com/modflowpy/flopy/commit/14d461588e1dbdb7bbe396e89e2a9cfc9366c8f3): Fixed building of obs package for exchange packages (through MFSimulation) (#1363). Committed by spaulins-usgs on 2022-02-28.
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
* [fix](https://github.com/modflowpy/flopy/commit/e9292af129cc3fadbeb557f72d9983b945d58a1b): Example script and notebook tests (#1546). Committed by w-bonelli on 2022-09-19.
* [fix(parsenamefile)](https://github.com/modflowpy/flopy/commit/d6e7f3fd4fbce44a0ca2a8f6c8f611922697d93a): Only do lowercase comparison when parent dir exists (#1554). Committed by Mike Taves on 2022-09-22.
* [fix(obs)](https://github.com/modflowpy/flopy/commit/06f08beb440e01209523efb2c144f8cf6e97bba4): Modify observations to load single time step files (#1559). Committed by jdhughes-usgs on 2022-09-29.
* [fix(PlotMapView notebook)](https://github.com/modflowpy/flopy/commit/4d4c68e471cd4b7bc2092d57cf593bf0bf21d65f): Remove exact contour count assertion (#1564). Committed by w-bonelli on 2022-10-03.
* [fix(test markers)](https://github.com/modflowpy/flopy/commit/aff35857e13dfc514b62db3dbc44e9a3b7abb5ab): Add missing markers for exes required (#1577). Committed by w-bonelli on 2022-10-07.
* [fix(csvfile)](https://github.com/modflowpy/flopy/commit/2eeefd4a0b00473ef13480ba7bb8741d3410bdfe): Default csvfile to retain all characters in column names (#1587). Committed by langevin-usgs on 2022-10-14.
* [fix(csvfile)](https://github.com/modflowpy/flopy/commit/a0a92ed07d87e0def97dfc4a87d291aff91cd0f8): Correction to read_csv args, close file handle (#1590). Committed by Mike Taves on 2022-10-16.
* [fix(data shape)](https://github.com/modflowpy/flopy/commit/09e3bc5552af5e40e207ca67808397df644a20c8): Fixed incorrect data shape for sfacrecord (#1584).  Fixed a case where time array series data did not load correctly from a file when the data shape could not be determined (#1594). (#1598). Committed by spaulins-usgs on 2022-10-21.
* [fix(write_shapefile)](https://github.com/modflowpy/flopy/commit/8a4e204eff0162021a895ee194feb2ccef4fe50d): Fix transform call for exporting (#1608). Committed by Joshua Larsen on 2022-10-27.
* [fix(multiple)](https://github.com/modflowpy/flopy/commit/e20ea054fdc1ab54bb0f1118a243b943c390ea4f): Miscellanous fixes/enhancements (#1614). Committed by w-bonelli on 2022-11-03.
* [fix(ci)](https://github.com/modflowpy/flopy/commit/09354561e204d42466dec8fe07c85a6edc9c4f27): Don't run benchmarks, examples and regression tests on push (#1618). Committed by w-bonelli on 2022-11-03.
* [fix](https://github.com/modflowpy/flopy/commit/9d328534aa272d8b4f8cfe77d30117f980794a8a): Not reading comma separated data correctly (#1634). Committed by Michael Ou on 2022-11-23.
* [fix(get-modflow)](https://github.com/modflowpy/flopy/commit/57d08623ea4c91a0ba3d1e6cfead44d203359ac4): Fix code.json handling (#1641). Committed by w-bonelli on 2022-12-08.
* [fix(quotes+exe_path+nam_file)](https://github.com/modflowpy/flopy/commit/3131e97fd85f5cfa57b8cbb5646b4a1884469989): Fixes for quoted strings, exe path, and nam file (#1645). Committed by spaulins-usgs on 2022-12-08.
* [fix(ci)](https://github.com/modflowpy/flopy/commit/c2a86b51cd7f41062af813fb69ca45b2fd6c2321): Fix release.yml version update step (#1646). Committed by w-bonelli on 2022-12-14.

#### Refactoring

* [refactor](https://github.com/modflowpy/flopy/commit/b5115945a2bdc509b59466ce87287e23a6093ca2): Clean-up python 2 'next' and redundant '__future__' imports (#1181). Committed by Mike Taves on 2021-08-10.
* [refactor(MF6Output)](https://github.com/modflowpy/flopy/commit/0288f969959f04f47aa15256a5650d2e0dafe7fb): Access list file object via <model>.output.list() (#1173). Committed by Joshua Larsen on 2021-08-16.
* [refactor(python)](https://github.com/modflowpy/flopy/commit/d8eb3567c8f9c4e194b5e01f09b22245ab457e8c): Set python 3.7 as minimum supported version (#1190). Committed by jdhughes-usgs on 2021-08-16.
* [refactor(numpy)](https://github.com/modflowpy/flopy/commit/feea04eaa594cf8c77f32bd17c4bd08d2e7cbd13): Remove support for numpy<1.15 (#1191). Committed by Mike Taves on 2021-08-17.
* [refactor(plot)](https://github.com/modflowpy/flopy/commit/771f1d12e0a0fd07eacd9c0154790446134a0194): Remove deprecated plot methods (#1192). Committed by jdhughes-usgs on 2021-08-17.
* [refactor(thickness)](https://github.com/modflowpy/flopy/commit/8402dca68b41e2ca11ba6ab199cdbb56690d1f9a): Remove deprecated dis and disu thickness property (#1195). Committed by jdhughes-usgs on 2021-08-17.
* [refactor(SR)](https://github.com/modflowpy/flopy/commit/5fcf9709ec8a655106c57d55657c47b1d4987812): Remove deprecated SpatialReference class (#1200). Committed by jdhughes-usgs on 2021-08-19.
* [refactor(pakbase)](https://github.com/modflowpy/flopy/commit/63ce93d23eb028017cccf6a83a9180560ba36f77): Generate heading from base class (#1196). Committed by Mike Taves on 2021-08-19.
* [refactor](https://github.com/modflowpy/flopy/commit/be74751ecd37dbc2e09da3e1e9398a22f2311b93): Replace OrderedDict with dict (#1201). Committed by Mike Taves on 2021-08-20.
* [refactor(reference)](https://github.com/modflowpy/flopy/commit/806ee8ffa50923f701eba15c4a24cb51b0393936): Remove deprecated crs and epsgRef classes (#1202). Committed by jdhughes-usgs on 2021-08-20.
* [refactor(deprecated)](https://github.com/modflowpy/flopy/commit/bf4371e7ba9a5f81129753a2c9d795355ca0bb51): Remove additional deprecated classes and functions (#1204). Committed by jdhughes-usgs on 2021-08-20.
* [refactor(zonbud, output_util, mfwel, OptionBlock)](https://github.com/modflowpy/flopy/commit/86206ff91d92df433f3d8f1e90978123430e5a71): Updates and bug fixes (#1209). Committed by Joshua Larsen on 2021-08-21.
* [refactor(zonebud)](https://github.com/modflowpy/flopy/commit/a267d153e43008bbcc65f539b732982a5e8d0d84): Reapply OrderedDict to dict changes from PR #1201 (#1210). Committed by jdhughes-usgs on 2021-08-21.
* [refactor](https://github.com/modflowpy/flopy/commit/12a062aa51fd3e2b98198bd1245cebe95e2175ad): Use print() instead of sys.stdout.write() (#1223). Committed by Mike Taves on 2021-09-04.
* [refactor(exceptions)](https://github.com/modflowpy/flopy/commit/349ba260e5c471bf162621a0efb5854b64bedae4): Reduce redundancy, simplify message generation (#1218). Committed by Mike Taves on 2021-09-04.
* [refactor(exceptions)](https://github.com/modflowpy/flopy/commit/81b17fa93df67f938c2d1b1bea34e8292359208d): Reclassify IOError as OSError or other type (#1227). Committed by Mike Taves on 2021-09-04.
* [refactor(Vtk)](https://github.com/modflowpy/flopy/commit/706f6dbe3888babcdee69fc62369702cc8a49543): Updates to Vtk class for MF6, HFB, and MODPATH (#1249). Committed by Joshua Larsen on 2021-09-30.
* [refactor(Package)](https://github.com/modflowpy/flopy/commit/bb21686c61a8c442724ad96e10d5baba3e373681): Simplify __init__ in subclasses, remove "extra" (#1238). Committed by Mike Taves on 2021-10-05.
* [refactor(netcdf)](https://github.com/modflowpy/flopy/commit/8f35966bbea31510f1e5a6aa93deff45891f6004): Sync data, revise logger and exceptions (#1230). Committed by Mike Taves on 2021-10-05.
* [refactor(imports)](https://github.com/modflowpy/flopy/commit/8ab73a9a9ac6435a047ba7128a3980b09dda6a06): Add function to import optional packages (#1262). Committed by jdhughes-usgs on 2021-10-13.
* [refactor(setup)](https://github.com/modflowpy/flopy/commit/c252a3413ea658d8bbec54fbd723df5c809ecbad): Prefer setup.cfg for package metadata, and other changes (#1267). Committed by Mike Taves on 2021-10-21.
* [refactor(MF6Output)](https://github.com/modflowpy/flopy/commit/999efc5b90af524fe37d65d979f5055e0e66d51b): Added budgetcsv method to .output attribute (#1275). Committed by Joshua Larsen on 2021-11-01.
* [refactor(CellBudgetFile, MF6Output)](https://github.com/modflowpy/flopy/commit/3e770ec23bf76a84e6145b4d25d5fe00328e2613): Update to pass modelgrid to CellBudgetFile and reshape arrays using full3D option (#1282). Committed by Joshua Larsen on 2021-11-09.
* [refactor(Raster)](https://github.com/modflowpy/flopy/commit/76bd34e731496f711f31edda930e61c3578c35f8): Added min and max resampling for hydrologic resampling and streamflow representation (#1294). Committed by Joshua Larsen on 2021-11-12.
* [refactor(Gridgen)](https://github.com/modflowpy/flopy/commit/e4efe8e4cdc5ce56bb2370c3dbc41edd3f56184a): Added keyword arguments for smoothing_level_vertical and smoothing_level_horizontal (#1322). Committed by Joshua Larsen on 2021-12-29.
* [refactor(Grid.intersect)](https://github.com/modflowpy/flopy/commit/eecd1ad193c5972093c9712e5c4b7a83284f0688): Added optional z-coordinate to intersect() method (#1326). Committed by Joshua Larsen on 2022-01-14.
* [refactor(distutils)](https://github.com/modflowpy/flopy/commit/7ec884f954e808b85b7d81a03bafe60cbd2e0fc4): Version classes are deprecated, use internal class (#1331). Committed by Mike Taves on 2022-01-21.
* [refactor(MFBlockHeader)](https://github.com/modflowpy/flopy/commit/8ac31ddff4790067208c422451b4e9835dac45cf): Fix for extremely slow loading of package OBS files when many blocks exist in OBS file. (#1351). Committed by Joshua Larsen on 2022-02-09.
* [refactor(read_binary_data_from_file)](https://github.com/modflowpy/flopy/commit/141baee1f0bb91f84907527c6da6fc401311673b): Update for binary layer files with multiple layers (#1352). Committed by Joshua Larsen on 2022-02-14.
* [refactor(geometry.py)](https://github.com/modflowpy/flopy/commit/89af81fbe50c0f086d085067a4323947d0b6ff8c): Added __reversed__ to geometry objects (#1347). Committed by Joshua Larsen on 2022-02-16.
* [refactor(contouring routines)](https://github.com/modflowpy/flopy/commit/8e3edd924dc885ff4f4d609b32fa6604335123aa): Updates for contour_array and export_contourf routines (#1353). Committed by Joshua Larsen on 2022-02-16.
* [refactor(flake8)](https://github.com/modflowpy/flopy/commit/e3d15ee7d41672812972009bc09ec843de4f2621): F821 for 'undefined name' errors (#1350). Committed by Mike Taves on 2022-02-18.
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
* [refactor(get_modflow)](https://github.com/modflowpy/flopy/commit/d1a7155ec2e60ed4f55e1d5b02d6df3fe69575c3): Misc (#1580). Committed by w-bonelli on 2022-10-20.
* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/f9068756311cf76f5e8f435f7ef9291af9c88eec): Updates for errant interpolation in point_scalar routines (#1601). Committed by Joshua Larsen on 2022-10-21.
* [refactor(docs/examples)](https://github.com/modflowpy/flopy/commit/798cf0d8b53ef61248a8b6d8da570eaca5483540): Correction to authors, use Path.cwd() (#1602). Committed by Mike Taves on 2022-10-26.
* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/2f71612197efedb08b0874d494b4658eb3f1d5c2): Use pathlib, corrections to docstrings (#1603). Committed by Mike Taves on 2022-10-26.
* [refactor(docs)](https://github.com/modflowpy/flopy/commit/66a28c50043d8dcaec4d6b6ee4906e4519bfacd6): Add ORCID icon and link to authors (#1606). Committed by Mike Taves on 2022-10-27.
* [refactor(plotting)](https://github.com/modflowpy/flopy/commit/b5d64e034504d962ebff1ea08dafef1abe00396d): Added default masked values  (#1610). Committed by Joshua Larsen on 2022-10-31.

