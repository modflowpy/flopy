# Changelog
### Version 3.9.3

#### New features

* [feat(modeltime)](https://github.com/modflowpy/flopy/commit/932bdcc8c07a10148cf733c02ba9190167239066): Add from_headers and reverse methods (#2481). Committed by wpbonelli on 2025-03-29.
* [feat](https://github.com/modflowpy/flopy/commit/d8048a868b51d75039f0c3c097429144c6d44714): Add the angldegx to the kw dictionary produced by the get_disu_kwargs method (#2484). Committed by Sunny Titus on 2025-04-14.

#### Bug fixes

* [fix(resolve_exe)](https://github.com/modflowpy/flopy/commit/ca3adb34031285bc795b58fa42c4965d67278adc): Typecast exe name to string before passing to _resolve (#2457). Committed by martclanor on 2025-02-20.
* [fix(HeadFile,CellBudgetFile)](https://github.com/modflowpy/flopy/commit/39d8d32625151979d78b82e709fa46ca4f5f7d9f): Fix tdis reversal (#2460). Committed by wpbonelli on 2025-02-25.
* [fix(output_util.py)](https://github.com/modflowpy/flopy/commit/61e52f03ff69ec50b03a17c863738ad4a3c1dc4b): Trap grb in MF6Output (#2468). Committed by wpbonelli on 2025-03-07.
* [fix(binaryfile)](https://github.com/modflowpy/flopy/commit/02ab7f1ff18a5c03f06d98fe1b35d6d6f7ad12df): Tdis in head/budget reversal methods (#2475). Committed by wpbonelli on 2025-03-18.
* [fix(binaryfile)](https://github.com/modflowpy/flopy/commit/0eba4babad994a7cd73a3853714b9ff3bb326c90): Fix head/budget file reversal (#2483). Committed by wpbonelli on 2025-04-01.
* [fix(flopy/utils/sfroutputfile.py::SfrFile.get_results)](https://github.com/modflowpy/flopy/commit/bf02be00ded034d2765948c7d7b075bf9d6ed4f1): Refactor deprecated DataFrame.append() call to pandas.concat() (#2491). Committed by aleaf on 2025-04-21.
* [fix(model_splitter.py)](https://github.com/modflowpy/flopy/commit/838b37cc999c713aa7e71c4215f7535c5c9df27b): Add trap for obs packages in bc packages (#2493). Committed by Joshua Larsen on 2025-04-22.
* [fix(evt)](https://github.com/modflowpy/flopy/commit/dcba9cf5ca1d8b946b568748f713a6dcf21e2bc3): Optional field mishandling (#2490). Committed by mjreno on 2025-04-23.
* [fix](https://github.com/modflowpy/flopy/commit/0d9c91489a6a9df2c2c51c962e59cb5e9dd4ff2b): Update numpy array comparisons to use isin (#2504). Committed by Emmanuel Ferdman on 2025-05-06.
* [fix(column lengths)](https://github.com/modflowpy/flopy/commit/351c5b72401b5de52d230ef9f0897f58e4edf475): Autoscale array write to ncol for structured multi-model simulations (#2507). Committed by Joshua Larsen on 2025-05-12.

#### Refactoring

* [refactor(resolve_exe)](https://github.com/modflowpy/flopy/commit/c61643fd8d61e6e4866527104324a6c468eb0e41): Also fix tests (#2464). Committed by Mike Taves on 2025-03-03.
* [refactor(createpackages)](https://github.com/modflowpy/flopy/commit/c5b5a41f626ad24d8f0e564a83ab8dc673cc9b7b): Use jinja for mf6 module code generation (#2333). Committed by wpbonelli on 2025-03-07.
* [refactor(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/20829b704b20be5175119fcd685fe808c55d51b6): Change how node mapping is stored and loaded (#2465). Committed by Joshua Larsen on 2025-03-14.
* [refactor(gridutil)](https://github.com/modflowpy/flopy/commit/48f46fb87a1a23bc13a41be881f3852fdab7d682): Improve arg handling in get_disu_kwargs (#2480). Committed by wpbonelli on 2025-04-01.
* [refactor(model_splitter.py)](https://github.com/modflowpy/flopy/commit/14640530dd46598d8d76417b6b640db0850cbf80): Support for SSM and ATS (#2505). Committed by Joshua Larsen on 2025-05-08.
* [refactor(codegen)](https://github.com/modflowpy/flopy/commit/69f6e09a8bea004454e9a83b76e5b37f8c0a7810): Move dfn utils from devtools (#2508). Committed by wpbonelli on 2025-05-13.

### Version 3.9.2

#### New features

* [feat(mf2005,mf2k)](https://github.com/modflowpy/flopy/commit/f058122a51404aadb5d28d23548ec96eff82ec8a): Allow loading custom flopy packages (#2404). Committed by Davíd Brakenhoff on 2025-01-10.
* [feat(generate_classes.py)](https://github.com/modflowpy/flopy/commit/48ec3b2f2cfa0287c94fb6d62a596196294cfc2d): Allow excluding components (#2447). Committed by wpbonelli on 2025-02-11.

#### Bug fixes

* [fix(mp7particledata)](https://github.com/modflowpy/flopy/commit/942a4cd0eadec1201b430867f84412a03751747e): Add global_xy option for to_coords/to_prp (#2405). Committed by Davíd Brakenhoff on 2024-12-23.
* [fix(resolve_exe)](https://github.com/modflowpy/flopy/commit/4deabe1d34b3476d9134bf0b64614ef0779634bb): Allow shutil.which() to find exe without suffix in Windows (#2408). Committed by martclanor on 2025-01-10.
* [fix(cvfdutil)](https://github.com/modflowpy/flopy/commit/f9da244347f6e076e8e5d65562efc06640b07677): Fix skip_hanging_node_check (#2423). Committed by Oscar  Sanchez on 2025-01-21.
* [fix(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/25ec5cd4ce32e8862760ef672ef709fbb870d19f): Multiple bug fixes and added support (#2418). Committed by Joshua Larsen on 2025-01-30.
* [fix(mp7particledata.py)](https://github.com/modflowpy/flopy/commit/9385daac51d064b34ca72117a27d5ec741a640c7): Avoid attribute error (#2441). Committed by wpbonelli on 2025-02-08.

#### Perf

* [perf(tutorials)](https://github.com/modflowpy/flopy/commit/8fd63758606091b323b01b1b77d39be964650cdd): Skip some writes in export_vtk_tutorial.py (#2432). Committed by wpbonelli on 2025-02-03.

#### Refactoring

* [refactor(model_splitter)](https://github.com/modflowpy/flopy/commit/bb587c3e46d934799e1e1060681dd10073ad935e): Add timeseries support (#2403). Committed by Joshua Larsen on 2025-01-03.
* [refactor(FlopyBinaryData)](https://github.com/modflowpy/flopy/commit/ef5ca6a6361bab020f9e198563c18d7242f12e28): Make properties readonly, deprecate set_float() (#2421). Committed by wpbonelli on 2025-01-20.
* [refactor(get-modflow)](https://github.com/modflowpy/flopy/commit/6a8f313df441c73673f8780a203a7233ae4f3ecc): Don't hard-code available os tags (#2426). Committed by wpbonelli on 2025-01-26.
* [refactor(mf6)](https://github.com/modflowpy/flopy/commit/816ff8108f6985e0b2ead549a13f6446639d8f61): Allow using existing dfn files in generate_classes.py (#2431). Committed by wpbonelli on 2025-02-03.
* [refactor(ModelTime)](https://github.com/modflowpy/flopy/commit/069d5430760fd0a9eb5eff6e8f36ce39974e25d3): Refactor ModelTime and add new features (#2367). Committed by Joshua Larsen on 2025-02-05.

### Version 3.9.1

#### Mf6

* [mf6](https://github.com/modflowpy/flopy/commit/4952f81f3b4935c51513593011166fbd88dfd85c): Update dfns and mf6 module (#2399). Committed by wpbonelli on 2024-12-20.

### Version 3.9.0

#### New features

* [feat(plot_centers)](https://github.com/modflowpy/flopy/commit/c6a41abb496039b65fbe52fd6c3f2df011e492be): Add plot_centers support to PlotMapView and PlotCrossSection (#2318). Committed by Joshua Larsen on 2024-10-07.
* [feat(get-modflow)](https://github.com/modflowpy/flopy/commit/d800ce5638e7a0983985881f2fa5d37207e3560b): Support windows extended build (#2356). Committed by mjreno on 2024-11-06.
* [feat(binaryfile)](https://github.com/modflowpy/flopy/commit/4eac63176f1328a22a55e1cf131db55b8ca929a8): Add head/budget file reversal script (#2383). Committed by wpbonelli on 2024-11-27.

#### Bug fixes

* [fix(ZoneFile6.load)](https://github.com/modflowpy/flopy/commit/99d57e6fd0d0d41c364f9c76f7800ec3be0d179a): Add split statement to input read (#2330). Committed by Joshua Larsen on 2024-10-09.
* [fix(resample_to_grid)](https://github.com/modflowpy/flopy/commit/15f1b94a45487e42aa36afdd2abb2b111c2197a6): Fix unintended extrapolation  (#2331). Committed by Joshua Larsen on 2024-10-09.
* [fix(utils)](https://github.com/modflowpy/flopy/commit/558f4a8c9d17241e21e7d41c3a75a9a6380a9fb1): Exclude ncf from mf6 output utils (#2336). Committed by mjreno on 2024-10-16.
* [fix(masked_4D_arrays)](https://github.com/modflowpy/flopy/commit/332a310ef0b43130fd2f37cf7cd4abe954484524): Allow re-use of preceding spd data if empty (#2314). Committed by martclanor on 2024-10-20.
* [fix(gridintersect)](https://github.com/modflowpy/flopy/commit/34043ab58eb254feee0c2f47cb63e057905b49d7): Fix multiple issues (#2343). Committed by Davíd Brakenhoff on 2024-10-25.

#### Refactoring

* [refactor(PackageContainer)](https://github.com/modflowpy/flopy/commit/f378f84a5677b99191ce178bb1c5b67ac1d1bd66): Compose not inherit, deprecate methods (#2324). Committed by Marnix on 2024-10-14.
* [refactor(Modpath7.create_mp7)](https://github.com/modflowpy/flopy/commit/3a2c4946a047e20e35341d7e4266f8dae3ac5316): Expose porosity parameter of Modpath7Bas (#2340). Committed by martclanor on 2024-10-20.
* [refactor(gridintersect)](https://github.com/modflowpy/flopy/commit/f8810c20097004a940e96ee0f2bc0229366a3899): Clean up gridintersect (#2346). Committed by Davíd Brakenhoff on 2024-10-24.
* [refactor(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/acc5a5b6580bdd6e93a24970db7d0b62d54e2485): Added split_multi_model method (#2352). Committed by Joshua Larsen on 2024-11-06.
* [refactor(mf6)](https://github.com/modflowpy/flopy/commit/4e0906426ef70235a7546c31d263904fa3249a9d): Deprecate mf6 checks (#2357). Committed by wpbonelli on 2024-11-06.
* [refactor](https://github.com/modflowpy/flopy/commit/a5545c6fc23c2ea1476f5b5650ce294ae9d2509f): Apply suggestions from pyupgrade (#2361). Committed by Mike Taves on 2024-11-11.
* [refactor](https://github.com/modflowpy/flopy/commit/bb9824e8aaea04a43d911724445cf0610301d236): Fix long lines to resolve check E501 (#2368). Committed by Mike Taves on 2024-11-14.
* [refactor](https://github.com/modflowpy/flopy/commit/373b82d3b864fe54bf5675e2a1aabbe4b6eee58e): Resolve ruff check F821 for undefined names (#2374). Committed by Mike Taves on 2024-11-18.
* [refactor](https://github.com/modflowpy/flopy/commit/22b5992ccd0e8280ee831611e8823bdb0d22ae3b): Apply fixes for flake8 comprehensions (C4) (#2376). Committed by Mike Taves on 2024-11-18.
* [refactor(deprecations)](https://github.com/modflowpy/flopy/commit/1993af155f9db97f5e4fb1a0090926e4cb65cf19): Deprecate flopy.mf6.utils.reference module (#2375). Committed by wpbonelli on 2024-11-19.
* [refactor](https://github.com/modflowpy/flopy/commit/4c1bf6cb486f8bd5bd9d25c5c7cf159fc65ecd4b): Apply Ruff-specific rule checks (#2377). Committed by Mike Taves on 2024-11-22.

### Version 3.8.2

#### Bug fixes

* [fix(mp7particledata)](https://github.com/modflowpy/flopy/commit/a0e9219407b7be208d38f4f902cd2b5b96da1351): Fix get_extent() vertical extent calculation (#2307). Committed by wpbonelli on 2024-09-12.
* [fix(array3d_export)](https://github.com/modflowpy/flopy/commit/693f01a0ce41f08d05395832b540fcc4f7dcff43): Fix exporting of array3d to shp (#2310). Committed by martclanor on 2024-09-16.
* [fix(binaryfile)](https://github.com/modflowpy/flopy/commit/181e101a605bdb9b628c6781abff7b3276aca635): Accommodate windows drives for in-place reversal  (#2312). Committed by wpbonelli on 2024-09-16.
* [fix(get_modflow)](https://github.com/modflowpy/flopy/commit/38180335445fa09f5463cf8b9239e6ed0c10bf5b): Accommodate missing ratelimit info on api response (#2320). Committed by wpbonelli on 2024-10-01.

### Version 3.8.1

#### New features

* [feat(cell1d)](https://github.com/modflowpy/flopy/commit/4ea71927f251c3675acf1b578bca623d023ccd2c): Add support for 1D vertex grids (#2296). Committed by langevin-usgs on 2024-08-23.

#### Bug fixes

* [fix(ParticleTrackFile.write_shapefile)](https://github.com/modflowpy/flopy/commit/f86881d6354071bf7c675384c2684ea184e281eb): Check for "k" even if "i", "j are not present   (#2294). Committed by Joshua Larsen on 2024-08-17.
* [fix(modelgrid)](https://github.com/modflowpy/flopy/commit/c42d8787bbcd4131b2f493ebc5a5a384b2e8b861): Add more support for mf6 surface water models  (#2295). Committed by langevin-usgs on 2024-08-22.

#### Refactoring

* [refactor(model_splitter.py)](https://github.com/modflowpy/flopy/commit/d02967db167b98e32cfaa26ef6636475ea2441a8): Update UnstructuredGrid support (#2292). Committed by Joshua Larsen on 2024-08-16.

### Version 3.8.0

#### New features

* [feat(datafile)](https://github.com/modflowpy/flopy/commit/d36bb78c3b7a12ab6f77bbe31e3572915753c86b): Add .headers property with data frame (#2221). Committed by Mike Taves on 2024-06-11.
* [feat(lgr-disv)](https://github.com/modflowpy/flopy/commit/7dec7c52db7c7bf3f8bca61de4d4a953ac1317d2): Add to_disv_gridprops() method to lgr object (#2271). Committed by langevin-usgs on 2024-07-26.

#### Bug fixes

* [fix(docs)](https://github.com/modflowpy/flopy/commit/4a26cab4e0af4f49775fd0dc327c8f5ff51843f6): Section underline matches section title (#2208). Committed by Mike Taves on 2024-06-06.
* [fix(vtk)](https://github.com/modflowpy/flopy/commit/d81d7c089f0688173f25c1f6d1e860e08c3a17ba): Fix __transient_vector access (#2209). Committed by mickey-tsai on 2024-06-06.
* [fix(swt)](https://github.com/modflowpy/flopy/commit/667774231a3c3e40fb68067331ead4b8a576cbee): Pass load_only down to Mt3dms.load() (#2222). Committed by wpbonelli on 2024-06-11.
* [fix(ParticleTrackFile)](https://github.com/modflowpy/flopy/commit/f15caaa0554f306eb5839588e4c75f9e14ef9641): Fix particle filtering in get_alldata (#2223). Committed by martclanor on 2024-06-11.
* [fix(regression)](https://github.com/modflowpy/flopy/commit/c69990ac37ce5d6828472af1eadab4dc6687c1e8): Corrections to test_create_tests_transport (#2228). Committed by Mike Taves on 2024-06-13.
* [fix(binaryread)](https://github.com/modflowpy/flopy/commit/e2a85a38640656d5795f8859defb0de14cf668e6): Raise/handle EOFError, deprecate vartype=str (#2226). Committed by Mike Taves on 2024-06-13.
* [fix(pandas warnings)](https://github.com/modflowpy/flopy/commit/5cdd609748cc70d93859192519d87d34194aec40): Catch pandas warnings and display them in a more useful way (#2229). Committed by scottrp on 2024-06-14.
* [fix](https://github.com/modflowpy/flopy/commit/d9ebd81903bb6aa03864e156a0488128867286ef): Test_uzf_negative_iuzfopt (#2236). Committed by Mike Taves on 2024-06-17.
* [fix(PlotMapView)](https://github.com/modflowpy/flopy/commit/678bb61346bc226831ae5b66615bc9a00c355cc5): Default to all layers in plot_pathline() (#2242). Committed by wpbonelli on 2024-06-19.
* [fix(Raster)](https://github.com/modflowpy/flopy/commit/a2a159f1758781fc633710f68af5441eb1e4dafb): Reclassify np.float64 correctly (#2235). Committed by martclanor on 2024-06-24.
* [fix(HeadFile)](https://github.com/modflowpy/flopy/commit/9db562a3b1d18af3801036b1d79d74668c0f71c6): Fix dis reversal, expand tests (#2247). Committed by wpbonelli on 2024-06-25.
* [fix(mfmodel)](https://github.com/modflowpy/flopy/commit/576cefe5e9826a53a5085d7e3aee9ce7765be22f): Fix get_ims_package (#2272). Committed by martclanor on 2024-08-06.
* [fix(modelgrid)](https://github.com/modflowpy/flopy/commit/b64f2bdae803830936da89cf1c8e97ab4f660981): Fix missing coord info if disv (#2284). Committed by martclanor on 2024-08-07.
* [fix(examples)](https://github.com/modflowpy/flopy/commit/2eace7843409b78497bc941d49eab68394833bfb): Restore example notebooks skipped after #2264 (#2286). Committed by wpbonelli on 2024-08-08.

#### Refactoring

* [refactor(expired deprecation)](https://github.com/modflowpy/flopy/commit/31955a7536b1f53d2a572580e05ff282a933716e): Raise AttributeError with to_shapefile (#2200). Committed by Mike Taves on 2024-05-30.
* [refactor](https://github.com/modflowpy/flopy/commit/bbabf86c0292ed2b237f89371afba01140050592): Deprecate unused flopy.utils.binaryfile.binaryread_struct (#2201). Committed by Mike Taves on 2024-05-31.
* [refactor(exceptions)](https://github.com/modflowpy/flopy/commit/0d9947eb8301561569676d4e3bdbc28a869e5bad): Raise NotImplementedError where appropriate (#2213). Committed by Mike Taves on 2024-06-07.
* [refactor(datafile)](https://github.com/modflowpy/flopy/commit/e2d16df5cc1a27a43e274a5b16eee7d91d5decfa): Use len(obj) rather than obj.get_nrecords() (#2215). Committed by Mike Taves on 2024-06-11.
* [refactor(binarygrid_util)](https://github.com/modflowpy/flopy/commit/ae388ef5a2f40abc950c05ca5b156f7e42337983): Refactor get_iverts to be general and not dependent on grid type (#2230). Committed by langevin-usgs on 2024-06-14.
* [refactor(datafile)](https://github.com/modflowpy/flopy/commit/cfdedbcb35c2f812e2b7efd78706d4eaa8cdc8f5): Deprecate list_records() and other list_ methods (#2232). Committed by Mike Taves on 2024-06-14.
* [refactor](https://github.com/modflowpy/flopy/commit/1e44b3fd57bfad1602a06247e44878a7237e0e3a): Fixes for numpy-2.0 deprecation warnings, require numpy>=1.20.3 (#2237). Committed by Mike Taves on 2024-06-17.
* [refactor](https://github.com/modflowpy/flopy/commit/59040d0948337245d6527671960b56446d39d4d3): Np.where(cond) -> np.asarray(cond).nonzero() (#2238). Committed by wpbonelli on 2024-06-17.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/e48198c661d8b10d1c1120a88a6cd0c7987d7b22): Support numpy 2 (#2241). Committed by wpbonelli on 2024-06-19.
* [refactor(get-modflow)](https://github.com/modflowpy/flopy/commit/baf8dff95ae3cc55adee54ec3e141437ae153b9c): Support ARM macs by default (previously opt-in) (#2225). Committed by wpbonelli on 2024-06-21.
* [refactor(Raster)](https://github.com/modflowpy/flopy/commit/bad483b3910218dc828c993863d540793111090d): Add new methods and checks (#2267). Committed by Joshua Larsen on 2024-07-17.
* [refactor(resample_to_grid)](https://github.com/modflowpy/flopy/commit/bd7f0a578b9093697948255eb9ecc164d5574f6e): Filter raster nan values from scipy resampling routines (#2285). Committed by Joshua Larsen on 2024-08-08.

### Version 3.7.0

#### New features

* [feat(get-modflow)](https://github.com/modflowpy/flopy/commit/4c5e2ee3ec6699f61acb04a5e7bd35407b16f9ff): Support ARM mac distributions (#2110). Committed by wpbonelli on 2024-02-19.
* [feat(get-modflow)](https://github.com/modflowpy/flopy/commit/53a94a9c6d57f8311318836b4291ac73b4b7f728): Support ARM mac nightly build (#2115). Committed by wpbonelli on 2024-02-23.
* [feat(binaryfile)](https://github.com/modflowpy/flopy/commit/5ec612a1a5b21c094bd910c9ed39f3ea5fef5084): Get budget by second package name `paknam2` (#2050). Committed by Michael Ou@SSPA on 2024-03-14.
* [feat(get-modflow)](https://github.com/modflowpy/flopy/commit/43e5178db2b52f136cab3d53cb290c2576af21ee): Support windows parallel nightly build (#2128). Committed by wpbonelli on 2024-03-22.
* [feat](https://github.com/modflowpy/flopy/commit/f75853f9fe9921c5ed1aa4e6ab4cec594338905f): Add optional custom print callable (#2121). Committed by Mike Müller on 2024-03-25.
* [feat(dis2d)](https://github.com/modflowpy/flopy/commit/18014af25dff00d659ebe30f9b5dc82da0143be3): Introduce limited support for a 2D structured grid (for overland flow) (#2131). Committed by langevin-usgs on 2024-04-01.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/3028863dc48b6eb8622b0dcd1a4918cd31547429): Improve vtk particle track export (#2132). Committed by wpbonelli on 2024-04-02.
* [feat(disv1d)](https://github.com/modflowpy/flopy/commit/4c44cb0d9d6ef02c092938dad943cdb8285f82a7): Rename DISL to DISV1D (#2133). Committed by langevin-usgs on 2024-04-02.
* [feat(disv2d)](https://github.com/modflowpy/flopy/commit/e023235ad40cd373d428ce5c1533bd0047109a96): Introduce support for a 2D vertex grid (for overland flow) (#2151). Committed by langevin-usgs on 2024-04-15.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/43cbe4762f6f339a554052a1d0d5836d863da13f): Include all arrays on pathline input (#2161). Committed by wpbonelli on 2024-04-19.
* [feat(mp7particledata)](https://github.com/modflowpy/flopy/commit/e50ab9ae4fce67db947a022da994b26150df13aa): Add localz option for PRT PRP conversions (#2166). Committed by wpbonelli on 2024-04-24.
* [feat(sim options block packages)](https://github.com/modflowpy/flopy/commit/4e1d53ac3685d9027b1ced45749c9e20d8b701d5): Support for packages declared in simulation name file's options block (#2164). Committed by spaulins-usgs on 2024-05-01.
* [feat(sim options block packages)](https://github.com/modflowpy/flopy/commit/6237cecd5f393a15857b653e030130ab4861e3a5): Support for packages declared in simulation name file's options block (#2174). Committed by spaulins-usgs on 2024-05-02.
* [feat(MfList)](https://github.com/modflowpy/flopy/commit/712221918b1f32b881244a19781d01f92d150196): Support kper field in stress period data (#2179). Committed by wpbonelli on 2024-05-04.

#### Bug fixes

* [fix(PRT)](https://github.com/modflowpy/flopy/commit/dda482b18a6e857ca66a44bbbd69bc3145a68060): Allow empty recarray or dataframe for output conversion fns (#2103). Committed by wpbonelli on 2024-02-19.
* [fix(gridintersect)](https://github.com/modflowpy/flopy/commit/40c03913bdf4329e8540d365ccfffe1c2bf4fccf): Gridintersect does not work for rotated vertex grids (#2107). Committed by Davíd Brakenhoff on 2024-02-20.
* [fix(str and repr)](https://github.com/modflowpy/flopy/commit/1fe51578309a693fae45d843e50aa6f9c86ad70c): Better repr and str output for transient data with multiple blocks (#2058) (#2102). Committed by scottrp on 2024-03-13.
* [fix(get_package and model_time)](https://github.com/modflowpy/flopy/commit/11f573b0341b461a584faae4fd82ea4b2bbffc69): #2117, #2118 (#2123). Committed by spaulins-usgs on 2024-03-20.
* [fix(modflow)](https://github.com/modflowpy/flopy/commit/aa9f410a4cfd2eecb9f5fa824fbf20103bf45bc0): Dataframe support was patchy in a few packages (#2136). Committed by wpbonelli on 2024-04-04.
* [fix(dependencies)](https://github.com/modflowpy/flopy/commit/0fe415058118ca67a1e02cd5e583ce589a65c9f1): Pin pyzmq >= 25.1.2 for arm macs (#2138). Committed by wpbonelli on 2024-04-05.
* [fix(empty transient data)](https://github.com/modflowpy/flopy/commit/0f14f1fdc02a34338795396549f15758e3ab2ba0): Empty first stress period block (#1091) (#2139). Committed by spaulins-usgs on 2024-04-05.
* [fix(comma delimited, scientific notation)](https://github.com/modflowpy/flopy/commit/5aaa5fff0fd9004d16f1699261ed8ef72577a67c): #2053 (#2144). Committed by spaulins-usgs on 2024-04-11.
* [fix(empty transient arrays)](https://github.com/modflowpy/flopy/commit/f4a4274ce676713614f47fc4968315ed30297b35): #2145 (#2146). Committed by spaulins-usgs on 2024-04-12.
* [fix(#2152)](https://github.com/modflowpy/flopy/commit/57cf82ebfcfabfaaf65bde88619070a20a7744ac): Improve gridintersect geometry creation for vertex grids (#2154). Committed by Davíd Brakenhoff on 2024-04-15.
* [fix(grb)](https://github.com/modflowpy/flopy/commit/30e03490e4afadda6ed155cff1c319ebd3c6d5b0): Update binary grid file reader for new grid types (#2157). Committed by langevin-usgs on 2024-04-17.
* [fix(OptionBlock)](https://github.com/modflowpy/flopy/commit/187885b4a3b9f54dbe27d568639931b8094b488e): Deprecate attribute typo 'auxillary' -> 'auxiliary' (#2159). Committed by Mike Taves on 2024-04-19.
* [fix(typos)](https://github.com/modflowpy/flopy/commit/ff82488c0105db52931730fac801fa49a504d1cf): Fixed a variety of typos throughout project (#2160). Committed by Mike Taves on 2024-04-19.
* [fix(cvfdutil)](https://github.com/modflowpy/flopy/commit/bb5461ba32a2047d7451ca53d96ab1643f47309b): Polygon area and centroid calculations now use shapely (#2165). Committed by langevin-usgs on 2024-04-23.
* [fix(gridgen)](https://github.com/modflowpy/flopy/commit/50bbd01a5bae8e80d8345c3f314cabd03df5bb22): Remove duplicate disv grid vertices #1492 (#2119). Committed by wpbonelli on 2024-05-02.
* [fix(mfmodel)](https://github.com/modflowpy/flopy/commit/8e16aab76b6e4f892fcf7031488324c3d490b75b): Fix budgetkey for transport models (#2176). Committed by wpbonelli on 2024-05-03.
* [fix(gridintersect)](https://github.com/modflowpy/flopy/commit/15d1d7f5eaf5f50455ff99a876e05c93a947aa5b): Relax cell boundary checks with np.isclose (#2173). Committed by wpbonelli on 2024-05-06.
* [fix(MFFileAccessArray)](https://github.com/modflowpy/flopy/commit/344579b9e23484cd7947948a9ca24a01e161b20e): Read_text_data_from_file modified for non-layered (#2183). Committed by langevin-usgs on 2024-05-06.
* [fix(styles)](https://github.com/modflowpy/flopy/commit/acfd0d37ec0dcb186635af8e70f8a5afbd1eadb6): Remove need for platform evaluation (#2188). Committed by jdhughes-usgs on 2024-05-09.
* [fix(get_structured_faceflows)](https://github.com/modflowpy/flopy/commit/29e247dbe464a861be8ccb4a7ad84f3c4020dd59): Fix lower face flows when idomain is -1 (#2192). Committed by vincentpost on 2024-05-17.
* [fix(tutorial, verbosity setter)](https://github.com/modflowpy/flopy/commit/7879c2f0e41299f70ec0b3b03d756feaf4f20f37): Fixed tutorial model name and verbosity setter (#2182) (#2193). Committed by scottrp on 2024-05-21.

#### Refactoring

* [refactor(datautil)](https://github.com/modflowpy/flopy/commit/b58a70379030d55735858201b16b2599b266ec50): In is_int/float use .item() for np arrays (#2068). Committed by wpbonelli on 2024-02-19.
* [refactor(plotting)](https://github.com/modflowpy/flopy/commit/4e6f4c1b788ec5e64559975afffbcea539def8ac): Check for user set axes limits (#2108). Committed by Joshua Larsen on 2024-02-21.
* [refactor(get_cell_vertices)](https://github.com/modflowpy/flopy/commit/00b3d1c75bcb98bd0ab99c4a9fb45cd8f829e8d3): Raise helpful messages, improve docs, add tests (#2125). Committed by Mike Taves on 2024-03-18.
* [refactor(modpathfile)](https://github.com/modflowpy/flopy/commit/77e5e1dfbadad3fac3be9bf9cf6f0de64f826f66): Toward unified particle tracking api (#2127). Committed by wpbonelli on 2024-03-28.
* [refactor(MFSimulationBase)](https://github.com/modflowpy/flopy/commit/9e87acddcf0740aac491826ef86fd45ae2304dac): Allow simulations to have no attached models (#2140). Committed by Joshua Larsen on 2024-04-06.
* [refactor(lgrutil)](https://github.com/modflowpy/flopy/commit/029a4e165caed6af760517ed7bc1f2e62e218858): Convert numpy types to builtins for np2 compat (#2158). Committed by wpbonelli on 2024-04-19.
* [refactor(mp7particledata)](https://github.com/modflowpy/flopy/commit/c7af787110eb1d984fa660c49014c999d87b0774): Match mp7 order in to_coords()/to_prp() (#2172). Committed by wpbonelli on 2024-05-01.

### Version 3.6.0

#### New features

* [feat(set all data external options)](https://github.com/modflowpy/flopy/commit/02a2f91e802185fd6b31929f99e31fecbb41c499): Additional parameters added (#2041). Committed by scottrp on 2023-12-18.
* [feat(PRT)](https://github.com/modflowpy/flopy/commit/a53cda7ff8f724dfc362b432ac9be8c98cc04165): Add conversion/plotting utils for MF6 particle tracking models (#1753). Committed by wpbonelli on 2023-12-22.
* [feat](https://github.com/modflowpy/flopy/commit/6899553828725a302e23b5ec787cea481ce85d59): Add static methods to read gridgen quadtreegrid files (#2061). Committed by Martin Vonk on 2024-01-17.
* [feat(GeoSpatialCollection)](https://github.com/modflowpy/flopy/commit/f8eac0feafd981d84367128a0fdf8e7e1789fa9b): Add support for GeoDataFrame objects (#2063). Committed by Joshua Larsen on 2024-01-26.
* [feat(GeoSpatialCollection)](https://github.com/modflowpy/flopy/commit/86eb092bf5f4d44cc449a458b77114dcefcbdda7): Add support for geopandas GeoSeries and GeoArray (#2085). Committed by Joshua Larsen on 2024-02-02.

#### Bug fixes

* [fix(gridgen)](https://github.com/modflowpy/flopy/commit/b3510e99b062ba9e950c68db4df95e87375a1ab8): Fix add_refinement_feature() shapefile support (#2022). Committed by wpbonelli on 2023-11-30.
* [fix(gridgen)](https://github.com/modflowpy/flopy/commit/51109751cf0f6c4ebfbc70ccf0ba0ced7fabd4b8): Support arbitrary path-like for shapefiles (#2026). Committed by wpbonelli on 2023-12-04.
* [fix(subpackages)](https://github.com/modflowpy/flopy/commit/788a8df62b1519a621c2721bec12f75cfb11a8d7): Fixed detection issue of subpackages in some filein records (#2025). Committed by scottrp on 2023-12-04.
* [fix(recarrays with cellid)](https://github.com/modflowpy/flopy/commit/a8800396b987e676bf0710a9d52a8cdce7b4d13f): Fixes bug when setting data as recarrays with cellids (#2029). Committed by scottrp on 2023-12-05.
* [fix(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/44abb51b4fb2034d94d82cf686f44c3ddf238a1f): Preserve MFSimulation version & exe_name (#2033). Committed by wpbonelli on 2023-12-07.
* [fix(data storage)](https://github.com/modflowpy/flopy/commit/97da3961f531b26001e8cf55db0de45fc41befad): Added numpy type check for consistent integer and float sizes (32-bit vs 64-bit) (#2062). Committed by scottrp on 2024-01-17.
* [fix(obs package loading)](https://github.com/modflowpy/flopy/commit/a017b77493482124c2f9a635bad19a0c0dfed8c9): Fixed problem with loading multiple continuous blocks (#2058) (#2064). Committed by scottrp on 2024-01-22.
* [fix(particledata)](https://github.com/modflowpy/flopy/commit/ea73e0d2f52cf22511c211ea4f5dd5b0a96be69b): Support 1D numpy array for partlocs (#2074). Committed by wpbonelli on 2024-01-25.
* [fix(tri2vor)](https://github.com/modflowpy/flopy/commit/1ab25fe1cd9b55a5128eaade60bd09122f82d096): Remove invalid geometries from voronoi nodes (#2076). Committed by Joshua Larsen on 2024-01-26.
* [fix(MFSimulationList)](https://github.com/modflowpy/flopy/commit/f37610d1f775dbfcc935b129d76c13c407b30e3b): Fix comma spacing in error message (#2090). Committed by wpbonelli on 2024-02-04.
* [fix(numpy 2.0 deprecation)](https://github.com/modflowpy/flopy/commit/ad35b8dda86959f8cdf4560934ab5f309dc28d56): Replace np.alltrue with np.all (#2088). Committed by mnfienen on 2024-02-04.
* [fix(usgcln)](https://github.com/modflowpy/flopy/commit/4ffa04e0757aabf000e6d87a33901c7265b1fa9d): add explicit second dimension to util2d.load calls (#2097). Committed by cnicol-gwlogic on 2024-02-07.

#### Refactoring

* [refactor(.gitattributes)](https://github.com/modflowpy/flopy/commit/0f33a22f1a01a46949cbb56b61bf41a0c73d78f2): Configure github-linguist exclusions (#2023). Committed by Mike Taves on 2023-12-01.
* [refactor(remap_array)](https://github.com/modflowpy/flopy/commit/5a4533bf115dc4876a811b11d87d353ae01da8ea): Trap for None type idomain (#2034). Committed by Joshua Larsen on 2023-12-07.
* [refactor(mbase)](https://github.com/modflowpy/flopy/commit/66b18624f78f36acd587c5a5bdcd90503397c3d1): Append not prepend flopy bindir to PATH (#2037). Committed by wpbonelli on 2023-12-08.
* [refactor(pyproject.toml)](https://github.com/modflowpy/flopy/commit/b94745dc4f72a4568ab084a762f4d6a1502afbfd): Add dev dependency group (#2075). Committed by wpbonelli on 2024-01-25.
* [refactor(contour_array)](https://github.com/modflowpy/flopy/commit/92853a95f6d38b73e561e3e155d92b75b93afb5d): Add tri_mask kwarg to parameters (#2078). Committed by Joshua Larsen on 2024-02-01.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/da8a3bdd6ecdbced4a22054b57c45dca470ec709): Remove python-dateutil (#2080). Committed by wpbonelli on 2024-02-01.
* [refactor(_plot_package_helper)](https://github.com/modflowpy/flopy/commit/cff4f2351d92ef8f894633481248b645c07f5c83): Pass kwargs to datatype helpers (#2081). Committed by Joshua Larsen on 2024-02-02.
* [refactor(convert_grid)](https://github.com/modflowpy/flopy/commit/b9ca77160cab2bf0e1af1d1d17ac82f3fa6e9d84): Added offset and angrot info to conversion (#2083). Committed by Joshua Larsen on 2024-02-02.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/391a3690c57ccf2951f7d618c1afd4168dfa0f9b): Pin numpy<2 until other reqs support it (#2092). Committed by wpbonelli on 2024-02-07.
* [refactor(mf6)](https://github.com/modflowpy/flopy/commit/d3e8a2a4342766288166e2251e96721007f5217f): Update DFNS for mf6.4.3, regen/reformat .py files (#2095). Committed by wpbonelli on 2024-02-07.

### Version 3.5.0

#### New features

* [feat(simulation+model options)](https://github.com/modflowpy/flopy/commit/a16a379ef61cc16594ac6ac9eadb5ce4c6a4cee1): Dynamically generate simulation options from simulation namefile dfn (#1842). Committed by spaulins-usgs on 2023-07-10.
* [feat(binaryfile)](https://github.com/modflowpy/flopy/commit/79311120d0daeb3ba1281c70faf97fcd92a1fde5): Add reverse() method to HeadFile, CellBudgetFile (#1829). Committed by w-bonelli on 2023-07-29.
* [feat(get-modflow)](https://github.com/modflowpy/flopy/commit/1cb7594d12a2aa385a567613e9f840de63ba7157): Allow specifying repo owner (#1910). Committed by w-bonelli on 2023-08-08.
* [feat(generate_classes)](https://github.com/modflowpy/flopy/commit/e02030876c20d6bf588134bc0ab0cc1c75c0c46e): Create a command-line interface (#1912). Committed by Mike Taves on 2023-08-16.
* [feat(gridutil)](https://github.com/modflowpy/flopy/commit/e20a29814afd97650d6ee2b10f0939893be64551): Add function to help create DISV grid (#1952). Committed by langevin-usgs on 2023-09-18.
* [feat(pandas list)](https://github.com/modflowpy/flopy/commit/6a46a9b75286d7e268d4addc77946246ccb5db56): Fix for handling special case where boundname set but not used (#1982). Committed by scottrp on 2023-10-06.
* [feat(MfSimulationList)](https://github.com/modflowpy/flopy/commit/b34f154362320059d92050b54014ec1fc9c9f09f): Add functionality to parse the mfsim.lst file (#2005). Committed by jdhughes-usgs on 2023-11-14.
* [feat(modflow)](https://github.com/modflowpy/flopy/commit/be104c4a38d2b1f4acfbdd5a49a5b3ddb325f852): Support dataframe for pkg data (#2010). Committed by wpbonelli on 2023-11-22.
* [feat(mfsimlist)](https://github.com/modflowpy/flopy/commit/121ca78f6073813cec082cfd01f483dad6838781): Add functionality to parse memory_print_options (#2009). Committed by jdhughes-usgs on 2023-11-22.

#### Bug fixes

* [fix(exchange and gnc package cellids)](https://github.com/modflowpy/flopy/commit/a84d88596f8b8e8f9f8fa074ad8fce626a84ebd0): #1866 (#1871). Committed by spaulins-usgs on 2023-07-11.
* [fix(modelgrid)](https://github.com/modflowpy/flopy/commit/99f680feb39cb9450e1ce052024bb3da45e264d8): Retain crs data from classic nam files (#1904). Committed by Mike Taves on 2023-08-10.
* [fix(generate_classes)](https://github.com/modflowpy/flopy/commit/4f6cd47411b96460495b1f7d6582860a4d655060): Use branch arg if provided (#1938). Committed by w-bonelli on 2023-08-31.
* [fix(remove_model)](https://github.com/modflowpy/flopy/commit/71855bdfcd37e95194b6e5928ae909a6f7ef15fa): Remove_model method fix and tests (#1945). Committed by scottrp on 2023-09-14.
* [fix(model_splitter.py)](https://github.com/modflowpy/flopy/commit/5b5eb4ec20c833df4117415d433513bf3a1b4aa5): Standardizing naming of iuzno, rno, lakeno, & wellno to ifno (#1963). Committed by Eric Morway on 2023-09-25.
* [fix(pandas list)](https://github.com/modflowpy/flopy/commit/f77989d33955baa76032d9341226385215e45821): Deal with cellids with inconsistent types (#1980). Committed by scottrp on 2023-10-06.
* [fix(model_splitter)](https://github.com/modflowpy/flopy/commit/5ebc216822a86f32c7a2ab9a3735f021475dc0f4): Check keys in mftransient array (#1998). Committed by jdhughes-usgs on 2023-11-13.
* [fix(benchmarks)](https://github.com/modflowpy/flopy/commit/16183c3059f4a9bb10d74423500b566eb41d3d6e): Fix benchmark post-processing (#2004). Committed by wpbonelli on 2023-11-14.
* [fix(MfSimulationList)](https://github.com/modflowpy/flopy/commit/7cc5e6f0b4861e2f9d85eb1e25592721d0427227): Add missing seek to get_runtime method (#2006). Committed by mjr-deltares on 2023-11-15.
* [fix(get_disu_kwargs)](https://github.com/modflowpy/flopy/commit/419ae0e0c15b4f95f6ea334fe49f79e29917130b): Incorrect indexing of delr and delc (#2011). Committed by langevin-usgs on 2023-11-21.
* [fix(PlotCrossSection)](https://github.com/modflowpy/flopy/commit/6e3c7911c1e94a9f8c36563662471c68a7aeeefc): Boundary conditions not plotting for DISU (#2012). Committed by langevin-usgs on 2023-11-21.
* [fix(release.yml)](https://github.com/modflowpy/flopy/commit/11f518cd68be596c3f07b052f9d3320880c38170): Don't regenerate pkgs from mf6 main on release (#2014). Committed by wpbonelli on 2023-11-24.
* [fix(release.yml)](https://github.com/modflowpy/flopy/commit/0da6b8ecc9087cc9c68a7d90999f61eb78f0f9ac): Fix update changelog step (#2015). Committed by wpbonelli on 2023-11-25.

#### Refactoring

* [refactor(_set_neighbors)](https://github.com/modflowpy/flopy/commit/9168b3a538a5f604010ad5df17ea7e868b8f913f): Check for closed iverts and remove closing ivert (#1876). Committed by Joshua Larsen on 2023-07-14.
* [refactor(crs)](https://github.com/modflowpy/flopy/commit/02fae7d8084f52e852722338038b6ef7a2691e61): Provide support without pyproj, other deprecations (#1850). Committed by Mike Taves on 2023-07-19.
* [refactor(Notebooks)](https://github.com/modflowpy/flopy/commit/c261ee8145e2791be2c2cc944e5a55cf19ef1560): Apply pyformat and black QA tools (#1879). Committed by Mike Taves on 2023-07-24.
* [refactor](https://github.com/modflowpy/flopy/commit/8c3d7dbbaa753893d746965f9ecc48dc68b274cd): Require pandas>=2.0.0 as core dependency (#1887). Committed by w-bonelli on 2023-08-01.
* [refactor(expired deprecation)](https://github.com/modflowpy/flopy/commit/47e5e359037518f29d14469531db68999d308a0f): Raise AttributeError with Grid.thick and Grid.saturated_thick (#1884). Committed by Mike Taves on 2023-08-01.
* [refactor(pathline/endpoint plots)](https://github.com/modflowpy/flopy/commit/54d6099eb71a7789121bcfde8816b23fa06c258e): Support recarray or dataframe (#1888). Committed by w-bonelli on 2023-08-01.
* [refactor(expired deprecation)](https://github.com/modflowpy/flopy/commit/70b9a37153bf51be7cc06db513b24950ce122fd2): Remove warning for third parameter of Grid.intersect (#1883). Committed by Mike Taves on 2023-08-01.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/dd77e72b9676be8834dcf0a08e6f55fd35e2bc29): Constrain sphinx >=4 (#1898). Committed by w-bonelli on 2023-08-02.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/03fa01fe09db45e30fe2faf4c5160622b78f1b24): Constrain sphinx-rtd-theme >=1 (#1900). Committed by w-bonelli on 2023-08-03.
* [refactor(mf6)](https://github.com/modflowpy/flopy/commit/809e624fe77c4d112e482698434e2199345e68c2): Remove deprecated features (#1894). Committed by w-bonelli on 2023-08-03.
* [refactor(plotutil)](https://github.com/modflowpy/flopy/commit/cf675c6b892a0d484f91465a4ad6600ffe4d518e): Remove deprecated utilities (#1891). Committed by w-bonelli on 2023-08-03.
* [refactor(shapefile_utils)](https://github.com/modflowpy/flopy/commit/6f2da88f9b2d53f94f9c003ddfde3daf6952e3ba): Remove deprecated SpatialReference usages (#1892). Committed by w-bonelli on 2023-08-03.
* [refactor(vtk)](https://github.com/modflowpy/flopy/commit/ca838b17e89114be8952d07e85d88b128d2ec2c3): Remove deprecated export_* functions (#1890). Committed by w-bonelli on 2023-08-03.
* [refactor(generate_classes)](https://github.com/modflowpy/flopy/commit/72370269e2d6933245c65b3c93beb532016593f3): Deprecate branch for ref, introduce repo, test commit hashes (#1907). Committed by w-bonelli on 2023-08-09.
* [refactor(expired deprecation)](https://github.com/modflowpy/flopy/commit/ed3a0cd68ad1acb68fba750f415fa095a19b741a): Remaining references to SpatialReference (#1914). Committed by Mike Taves on 2023-08-11.
* [refactor(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/3a1ae0bba3171561f0b4b78a2bb892fdd0c55e90): Control record and additional splitting checks (#1919). Committed by Joshua Larsen on 2023-08-21.
* [refactor(triangle)](https://github.com/modflowpy/flopy/commit/9ca601259604b1805d8ec1a30b0a883a7b5f019a): Raise if output files not found (#1954). Committed by wpbonelli on 2023-09-22.
* [refactor(recarray_utils)](https://github.com/modflowpy/flopy/commit/941a5f105b10f50fc71b14ea05abf499c4f65fa5): Deprecate functions, use numpy builtins (#1960). Committed by wpbonelli on 2023-09-27.
* [refactor(contour_array)](https://github.com/modflowpy/flopy/commit/4ed68a2be1e6b4f607e27c38ce5eab9f66492e3f): Add layer param, update docstrings, expand tests (#1975). Committed by wpbonelli on 2023-10-18.
* [refactor(model_splitter.py)](https://github.com/modflowpy/flopy/commit/4ef699e10b1fbd80055b85ca00a17b96d1c627ce): (#1994). Committed by Joshua Larsen on 2023-11-01.
* [refactor(modflow)](https://github.com/modflowpy/flopy/commit/696a209416f91bc4a9e61b4392f4218e9bd83408): Remove deprecated features (#1893). Committed by wpbonelli on 2023-11-03.
* [refactor](https://github.com/modflowpy/flopy/commit/af8954b45903680a2931573453228b03ea7beb6b): Support python3.12, simplify tests and dependencies (#1999). Committed by wpbonelli on 2023-11-13.
* [refactor(msfsr2)](https://github.com/modflowpy/flopy/commit/4eea1870cb27aa2976cbf5366a6c2cb942a2f0c4): Write sfr_botm_conflicts.chk to model workspace (#2002). Committed by wpbonelli on 2023-11-14.
* [refactor(shapefile_utils)](https://github.com/modflowpy/flopy/commit/0f8b521407694442c3bb2e50ce3d4f6207f86e48): Warn if fieldname truncated per 10 char limit (#2003). Committed by wpbonelli on 2023-11-14.
* [refactor(pakbase)](https://github.com/modflowpy/flopy/commit/09b59ce8c0e33b32727e0f07e44efadf166ffba2): Standardize ipakcb docstrings/defaults (#2001). Committed by wpbonelli on 2023-11-22.
* [refactor(.gitattributes)](https://github.com/modflowpy/flopy/commit/338f6d073d6ba7edecc12405f3da83ea51e6cc26): Exclude examples/data from linguist (#2017). Committed by wpbonelli on 2023-11-25.

### Version 3.4.3

#### Bug fixes

* [fix(export_contours/f)](https://github.com/modflowpy/flopy/commit/30209f2ca2e69289227203e4afd2f33bfceed097): Support matplotlib 3.8+ (#1951). Committed by wpbonelli on 2023-09-19.
* [fix(usg bcf)](https://github.com/modflowpy/flopy/commit/d2dbacb65d2579e42bbd7965c1321a89ccce7d56): ksat util3d call --> util2d call (#1959). Committed by @cnicol-gwlogic on 2023-09-22.
* [fix(resolve_exe)](https://github.com/modflowpy/flopy/commit/3522dced8a49dc93fb0140d9ac360a88f31b11bb): Support extensionless abs/rel paths on windows (#1957). Committed by wpbonelli on 2023-09-24.
* [fix(mbase)](https://github.com/modflowpy/flopy/commit/b848f968af4179d8618b811cd4fe6f8de66d09cb): Warn if duplicate pkgs or units (#1964). Committed by wpbonelli on 2023-09-26.
* [fix(get_structured_faceflows)](https://github.com/modflowpy/flopy/commit/92632d26be2ecb21b6d9d56717faadaa13e08369): Cover edge cases, expand tests (#1968). Committed by wpbonelli on 2023-09-29.
* [fix(CellBudgetFile)](https://github.com/modflowpy/flopy/commit/015d6399baa48819f9f0f78bf2f34f60bdd8ef18): Detect compact fmt by negative nlay (#1966). Committed by wpbonelli on 2023-09-30.

### Version 3.4.2

#### Bug fixes

* [fix(binaryfile/gridutil)](https://github.com/modflowpy/flopy/commit/b1e6b77af34448fee388efed5bbfa8d902fe93dc): Avoid numpy deprecation warnings (#1868). Committed by w-bonelli on 2023-07-12.
* [fix(binary)](https://github.com/modflowpy/flopy/commit/aa74356708137223ffed501f01d759832c336457): Fix binary header information (#1877). Committed by jdhughes-usgs on 2023-07-16.
* [fix(time series)](https://github.com/modflowpy/flopy/commit/021159bed614e80b0676a0e4c1dd61ac68c531de): Fix for multiple time series attached to single package (#1867) (#1873). Committed by spaulins-usgs on 2023-07-20.
* [fix(check)](https://github.com/modflowpy/flopy/commit/7e8a0cba122707ab1a87b0d1f13e05afbce29e94): Check now works properly with confined conditions (#1880) (#1882). Committed by spaulins-usgs on 2023-07-27.
* [fix(mtlistfile)](https://github.com/modflowpy/flopy/commit/2222245a1fbc4c9955f99a0f98c319843f4eb18a): Fix reading MT3D budget (#1899). Committed by Ralf Junghanns on 2023-08-03.
* [fix(check)](https://github.com/modflowpy/flopy/commit/5d21410cbe0c36d998fb3e6bcc4f8e443fcd7448): Updated flopy's check to work with cellid -1 values (#1885). Committed by spaulins-usgs on 2023-08-06.
* [fix(BaseModel)](https://github.com/modflowpy/flopy/commit/82bc3f1100eec3167f546e71a8c4d4520b7c0a3d): Don't suppress error if exe not found (#1901). Committed by w-bonelli on 2023-08-07.
* [fix(keyword data)](https://github.com/modflowpy/flopy/commit/4e00489f8d2a7626786338942210b5764995dd8a): Optional keywords (#1920). Committed by spaulins-usgs on 2023-08-16.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/22205f446bdd1e72ee204e5802af30dd501eece9): Combine list of geometries using unary_union (#1923). Committed by Mike Taves on 2023-08-21.
* [fix(gridintersect)](https://github.com/modflowpy/flopy/commit/672d6be6b07e1bf3b8cb6b0c3b6ce54be669399f): Add multilinestring tests (#1924). Committed by Davíd Brakenhoff on 2023-08-21.
* [fix(binary file)](https://github.com/modflowpy/flopy/commit/d1c60717b1d04922a718bea134cd28e6e050c660): Was writing binary file information twice to external files (#1925) (#1928). Committed by scottrp on 2023-08-25.
* [fix(ParticleData)](https://github.com/modflowpy/flopy/commit/00e99c1e07a07225829c5f2bd8e8992eeb50aeb9): Fix docstring, structured default is False (#1935). Committed by w-bonelli on 2023-08-25.

#### Refactoring

* [refactor(_set_neighbors)](https://github.com/modflowpy/flopy/commit/89fa273e8fd3ab7788ba4a65e62455d7d65c504d): Check for closed iverts and remove closing ivert (#1876). Committed by Joshua Larsen on 2023-07-14.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/364b4d17da421b15e69918a67b4d1c0b159fbf77): Constrain sphinx >=4 (#1898). Committed by w-bonelli on 2023-08-02.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/ee92091b096ad16a14cd8dd1f142abbb944bf91f): Constrain sphinx-rtd-theme >=1 (#1900). Committed by w-bonelli on 2023-08-03.

### Version 3.4.1

#### Bug fixes

* [fix(get-modflow)](https://github.com/modflowpy/flopy/commit/b8dffbc6bc7bee70d18d7ec24c96e33517e4f0e8): Accommodate mf6 release asset name change (#1855). Committed by w-bonelli on 2023-06-29.

### Version 3.4.0

#### New features

* [feat(Simulation)](https://github.com/modflowpy/flopy/commit/e2cbb25ce6356fa44d9ed4ba4460674f3b9e6760): Support pathlike (#1712). Committed by aleaf on 2023-02-13.
* [feat(solvers)](https://github.com/modflowpy/flopy/commit/ea04e83ed6199bd93fd6926e58a16a2d0d6686e1): Support for multiple solver types (#1706) (#1709). Committed by spaulins-usgs on 2023-02-15.
* [feat(pathlike)](https://github.com/modflowpy/flopy/commit/656751a5d34c74e9a16ba39fce7cb85505888cc4): Support pathlike in user-facing APIs (#1730). Committed by w-bonelli on 2023-03-03.
* [feat(export)](https://github.com/modflowpy/flopy/commit/17031e58a4ebf68e497771d2c1f1ffe6f71d1625): Include particle track polylines in VTK exports (#1750). Committed by w-bonelli on 2023-04-19.
* [feat(crs)](https://github.com/modflowpy/flopy/commit/604183550e4a2a6b2d19cc38528e4de0686e58be): Pyproj crs (#1737). Committed by aleaf on 2023-04-26.
* [feat(vtk)](https://github.com/modflowpy/flopy/commit/5352425694f2236e88fb2d77880b91f922d6f86e): Add to_pyvista() method (#1771). Committed by w-bonelli on 2023-04-30.
* [feat(run_simulation)](https://github.com/modflowpy/flopy/commit/183117342d07e6a80fb5cc0aefad3453f61d1a51): Add support for running parallel simulations with flopy (#1807). Committed by jdhughes-usgs on 2023-06-05.
* [feat(model_splitter.py)](https://github.com/modflowpy/flopy/commit/df091e4c73ffce3eefbc57d71e9372a44aeda830): Integrate model_splitter.py into FloPy (#1799). Committed by Joshua Larsen on 2023-06-05.
* [feat(model_splitter)](https://github.com/modflowpy/flopy/commit/9e397631b0c371ab41807e8e6f04f1722968fba5): Add optional pymetis dependency (#1812). Committed by jdhughes-usgs on 2023-06-06.
* [feat(model_splitter)](https://github.com/modflowpy/flopy/commit/cdd25125df0454404644d37af1064ec29ce5aa46): Add support for models that do not use IDOMAIN (#1834). Committed by jdhughes-usgs on 2023-06-21.
* [feat(generate_classes)](https://github.com/modflowpy/flopy/commit/43a39c2383d7aa72b303e490d12177e93df35c7a): Add optional owner param (#1833). Committed by w-bonelli on 2023-06-21.

#### Bug fixes

* [fix(MFPackage kwargs check)](https://github.com/modflowpy/flopy/commit/19b3daa3a63abda992bfc75938f9cf79b92f333d): Now verifying that only valid kwargs are passed to MFPackage (#1667). Committed by spaulins-usgs on 2022-12-22.
* [fix(factor)](https://github.com/modflowpy/flopy/commit/d99e0cc8be014c1ae796f917a83db2cb359e6a24): Fixed factor bug where converting data from internal to external can cause the factor to be applied to the data (#1673). Committed by scottrp on 2023-01-09.
* [fix(package dictionary)](https://github.com/modflowpy/flopy/commit/ebaef18341fa8966fe3eed539821552d701635f4): Removed package_key_dict since it is redundant and can cause errors (#1690). Committed by spaulins-usgs on 2023-01-26.
* [fix(intersect)](https://github.com/modflowpy/flopy/commit/ea1f5f06e935d7e3dc54d665e3ccb494cc0371e7): Multiple (#1696). Committed by w-bonelli on 2023-01-31.
* [fix(datautil)](https://github.com/modflowpy/flopy/commit/3b2800e720a786250c00d822e01d4fd2669140a0): Fix SFR connection file parsing (#1694). Committed by Wes Kitlasten on 2023-02-03.
* [fix(CellBudgetFile)](https://github.com/modflowpy/flopy/commit/c2c44fe2dc38653c84ee337eb3b2f91d4d8b6c42): Strip auxname for imeth 5 (#1716). Committed by Mike Taves on 2023-02-13.
* [fix(mfdatastorage)](https://github.com/modflowpy/flopy/commit/940e69038d397379dbc7d9799420c10eb34f5904): Use appropriate fill_value for data type (#1689). Committed by Mike Taves on 2023-02-13.
* [fix(test_sfr)](https://github.com/modflowpy/flopy/commit/915c268675c391e61fe0a3ef2d7f3fa26a9db2b8): Update test to be more robust with Matplotlib versions (#1717). Committed by Mike Taves on 2023-02-14.
* [fix(flopy performance)](https://github.com/modflowpy/flopy/commit/82a1c0ccfc9727fc6c591565dd3ede91198b5a67): FloPy performance modifications + best practices documented (#1674). Committed by spaulins-usgs on 2023-02-15.
* [fix(exe path)](https://github.com/modflowpy/flopy/commit/cf3a5177d16137455ce79fc9848c21b73fe7bb22): FloPy now correctly resolves relative paths to mf6 executable (#1633) (#1727). Committed by spaulins-usgs on 2023-03-02.
* [fix(ParticleData)](https://github.com/modflowpy/flopy/commit/8d52ecef28bdf5624d78ce0b809620353014e42c): Support partlocs as ndarray or list of lists (#1752). Committed by w-bonelli on 2023-03-23.
* [fix(mp6sim)](https://github.com/modflowpy/flopy/commit/ae14dd5caa4c2d580a6c5e7819b2d68e9effaa23): Use keyword args for pandas DataFrame.drop (#1757). Committed by w-bonelli on 2023-04-06.
* [fix(MFFileMgmt)](https://github.com/modflowpy/flopy/commit/6a8161aba0a8d42d770208201e54ec1d8d027c06): Remove string_to_file_path (#1759). Committed by w-bonelli on 2023-04-06.
* [fix(contours)](https://github.com/modflowpy/flopy/commit/baa322a8344b7baf5be196d44f1684f5fee99c41): Use nan for mpl contour masks on structured grids (#1766). Committed by w-bonelli on 2023-04-14.
* [fix(MFFileMgmt)](https://github.com/modflowpy/flopy/commit/7db9263aa9fa200f0a60b4d47e33187b8de32130): Avoid IndexError in strip_model_relative_path (#1748). Committed by w-bonelli on 2023-04-27.
* [fix(mtdsp)](https://github.com/modflowpy/flopy/commit/f9188eb1f963c442fef360fe459969209ce01ceb): Add support for keyword 'nocross' in MT3D dsp package (#1778). Committed by Eric Morway on 2023-05-09.
* [fix(shapefile_utils)](https://github.com/modflowpy/flopy/commit/6569568709e2615bab4b06a19d4e0127b6fac371): Tolerate missing arrays in model_attributes_to_shapefile (#1785). Committed by w-bonelli on 2023-05-17.
* [fix(Modpath6Sim)](https://github.com/modflowpy/flopy/commit/caadb03e0f54e45fe9a4c3f4366eee2131c35862): Move import_optional_dependency("pandas") to method (#1783). Committed by Mike Taves on 2023-05-17.
* [fix(float32, empty stress period)](https://github.com/modflowpy/flopy/commit/6ad17acc25d3667c2da631423ba5281bf90123c6): #1779 and #1793 (#1806). Committed by spaulins-usgs on 2023-06-02.
* [fix(load_node_mapping)](https://github.com/modflowpy/flopy/commit/22ef330bcfb9259fc23735d6b174d27804b624a0): Add sim parameter to populate Mf6Splitter._model_dict (#1828). Committed by Joshua Larsen on 2023-06-13.
* [fix(keystring)](https://github.com/modflowpy/flopy/commit/7ca806d420a675c890a9966ef736210eed52ebcf): Flopy now does not rely on keystring name being a substring of the keystring record name (#1616) (#1830). Committed by spaulins-usgs on 2023-06-15.
* [fix(get-modflow)](https://github.com/modflowpy/flopy/commit/a72a8c9fdaf6963fc5c9a89de00568665dfcfd9c): Manage internal "bin" dir structures (#1837). Committed by Mike Taves on 2023-06-23.
* [fix(GridIntersect)](https://github.com/modflowpy/flopy/commit/736c5b89035b079bb2956a9e2241e5155c046a76): Fix indexing error for empty intersection comparison (#1838). Committed by Joshua Larsen on 2023-06-23.
* [fix(mf6)](https://github.com/modflowpy/flopy/commit/8113a8856a61fdbb7a743d4687ddb3fb68d99e62): Fix external binary files for vertex grids (#1839). Committed by jdhughes-usgs on 2023-06-26.
* [fix(binary)](https://github.com/modflowpy/flopy/commit/edb70b48695de9483a58d449e648f829818de197): Revert a few changes in PR #1839 (#1846). Committed by jdhughes-usgs on 2023-06-28.

#### Perf

* [perf(Gridintersect)](https://github.com/modflowpy/flopy/commit/1114c93d1c52de862e00a594d41f5d724d91f358): Optimize intersection methods for shapely 2.0 (#1666). Committed by Davíd Brakenhoff on 2022-12-23.

#### Refactoring

* [refactor(tests)](https://github.com/modflowpy/flopy/commit/96fc3ad8f78470811c04effa2ce2d4bf3318eb6d): Use modflow-devtools fixtures and utilities (#1665). Committed by w-bonelli on 2022-12-20.
* [refactor(utils)](https://github.com/modflowpy/flopy/commit/94cd19d73a392738d29fbc742a41f1e1dc81d662): Move utils from modflow-devtools (#1621). Committed by w-bonelli on 2022-12-20.
* [refactor](https://github.com/modflowpy/flopy/commit/1e6991d7c05f97b6b9464edecfd49101dd7969a2): Drop Python 3.7, add Python 3.11 (#1662). Committed by Mike Taves on 2022-12-21.
* [refactor(dependencies)](https://github.com/modflowpy/flopy/commit/72890e175a9f91bc1021b89e7cc595b3dbce9778): Use devtools & relocated utils, drop pymake (#1670). Committed by w-bonelli on 2022-12-24.
* [refactor](https://github.com/modflowpy/flopy/commit/2e6a7e1b7fb905157d7ff8b29b57485b4a3bec20): Move project metadata to pyproject.toml (#1678). Committed by Mike Taves on 2023-01-18.
* [refactor(Modpath7)](https://github.com/modflowpy/flopy/commit/3c2a93e8e8f38fd3a168ea68daddf777a47cf35f): Update path construction for modpath nam file (#1679). Committed by Joshua Larsen on 2023-01-20.
* [refactor(styles)](https://github.com/modflowpy/flopy/commit/2552de521a7adc550133bfd3427a66d406851087): Add graph_legend fontsize parameters (#1702). Committed by Joshua Larsen on 2023-02-07.
* [refactor](https://github.com/modflowpy/flopy/commit/649596d0eeabc422a63e649111e3e12b411411ee): Run pyupgrade and adjust a few f-strings (#1710). Committed by Mike Taves on 2023-02-14.
* [refactor(MfGrdFile)](https://github.com/modflowpy/flopy/commit/9b780aa84fcd95ba940eb5543e11637d1ed8f5d0): Update docstrings in MfGrdFile (#1685). Committed by Joshua Larsen on 2023-02-16.
* [refactor(notebooks, mfhob.py)](https://github.com/modflowpy/flopy/commit/17dabb5494c91d931166f31809269a2704ba40e1): Clean paths in jupyter notebooks (#1711). Committed by Joshua Larsen on 2023-02-16.
* [refactor(grid.py)](https://github.com/modflowpy/flopy/commit/7aa9ecf1cebcff6898163b26b721fd728092dc42): Add size property for model splitting support (#1720). Committed by Joshua Larsen on 2023-02-19.
* [refactor(flopy_io, notebooks)](https://github.com/modflowpy/flopy/commit/fb3504bad39969dc75d88cf031b145cdcedfc810): Update path cleaning mechanism (#1728). Committed by w-bonelli on 2023-02-24.
* [refactor(PlotMapView)](https://github.com/modflowpy/flopy/commit/c307420640d039f7b3f5811088c9d54321f3f9fd): Support color kwarg in plot_endpoint() (#1745). Committed by w-bonelli on 2023-03-21.
* [refactor(Grid)](https://github.com/modflowpy/flopy/commit/945be7ce1cfc770b77651e6a9d0b21bf622d8af6): Refactor thick and saturated_thick: (#1768). Committed by Joshua Larsen on 2023-04-15.
* [refactor(neighbors)](https://github.com/modflowpy/flopy/commit/0c6e2f797e374d64c60b25cdd73fdfd05d711094): Overhaul of neighbors calculation (#1787). Committed by Joshua Larsen on 2023-05-19.
* [refactor(triangle-gridgen)](https://github.com/modflowpy/flopy/commit/9aa81e167741e062704edc789a12813fb0d00a44): Refactor resolving triangle and gridgen exe (#1819). Committed by jdhughes-usgs on 2023-06-08.
* [refactor(get-modflow)](https://github.com/modflowpy/flopy/commit/0deb8f6e695ad278cfeb7590675df59183485827): Use Path.replace instead of Path.rename (#1822). Committed by w-bonelli on 2023-06-08.
* [refactor(Mf6Splitter)](https://github.com/modflowpy/flopy/commit/2fa3d65af244ac5d31b523e1c8a310cd4be7b401): Feature updates and bugfixes (#1821). Committed by Joshua Larsen on 2023-06-10.

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
* Added link to Binder on [README](README.md) and notebooks_examples markdown documents. Binder provides an environment that runs and interactively serves the FloPy Jupyter notebooks.

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
