"""
Tests to prevent performance regressions
"""
import os
import shutil
import time
import numpy as np
import flopy.modflow as fm


class TestModflowPerformance():
    """Test flopy.modflow performance with realistic model/package sizes,
    in a reasonable timeframe.
    """
    @classmethod
    def setup_class(cls):
        """Make a modflow model."""
        print('setting up model...')
        t0 = time.time()
        size = 100
        nlay = 10
        nper = 10
        nsfr = int((size ** 2)/5)

        cls.modelname = 'junk'
        cls.model_ws = 'temp/t064'
        external_path = 'external/'

        if not os.path.isdir(cls.model_ws):
            os.makedirs(cls.model_ws)
        if not os.path.isdir(os.path.join(cls.model_ws, external_path)):
            os.makedirs(os.path.join(cls.model_ws, external_path))

        m = fm.Modflow(cls.modelname, model_ws=cls.model_ws, external_path=external_path)

        dis = fm.ModflowDis(m, nper=nper, nlay=nlay, nrow=size, ncol=size,
                            top=nlay, botm=list(range(nlay)))

        rch = fm.ModflowRch(m, rech={k: .001 - np.cos(k) * .001 for k in range(nper)})

        ra = fm.ModflowWel.get_empty(size ** 2)
        well_spd = {}
        for kper in range(nper):
            ra_per = ra.copy()
            ra_per['k'] = 1
            ra_per['i'] = (np.ones((size, size)) * np.arange(size)).transpose().ravel().astype(int)
            ra_per['j'] = list(range(size)) * size
            well_spd[kper] = ra
        wel = fm.ModflowWel(m, stress_period_data=well_spd)

        # SFR package
        rd = fm.ModflowSfr2.get_empty_reach_data(nsfr)
        rd['iseg'] = range(len(rd))
        rd['ireach'] = 1
        sd = fm.ModflowSfr2.get_empty_segment_data(nsfr)
        sd['nseg'] = range(len(sd))
        sfr = fm.ModflowSfr2(reach_data=rd, segment_data=sd, model=m)
        cls.init_time = time.time() - t0
        cls.m = m

    def test_init_time(self):
        """test model and package init time(s)."""
        mfp = TestModflowPerformance()
        target = 0.3 # seconds
        assert mfp.init_time < target, "model init took {:.2fs}, should take {:.1f}s".format(mfp.init_time, target)
        print('setting up model took {:.2f}s'.format(mfp.init_time))

    def test_0_write_time(self):
        """test write time"""
        print('writing files...')
        mfp = TestModflowPerformance()
        target = 5
        t0 = time.time()
        mfp.m.write_input()
        t1 = time.time() - t0
        assert t1 < target, "model write took {:.2fs}, should take {:.1f}s".format(t1, target)
        print('writing input took {:.2f}s'.format(t1))

    def test_9_load_time(self):
        """test model load time"""
        print('loading model...')
        mfp = TestModflowPerformance()
        target = 3
        t0 = time.time()
        m = fm.Modflow.load('{}.nam'.format(mfp.modelname),
                            model_ws=mfp.model_ws, check=False)
        t1 = time.time() - t0
        assert t1 < target, "model load took {:.2fs}, should take {:.1f}s".format(t1, target)
        print('loading the model took {:.2f}s'.format(t1))

    @classmethod
    def teardown_class(cls):
        # cleanup
        shutil.rmtree(cls.model_ws)
