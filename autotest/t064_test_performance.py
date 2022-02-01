"""
Tests to prevent performance regressions
"""
import os
import random
import shutil
import string
import sys
import time

import numpy as np
from ci_framework import base_test_dir

import flopy.modflow as fm

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)


class TestModflowPerformance:
    """Test flopy.modflow performance with realistic model/package sizes,
    in a reasonable timeframe.
    """

    @classmethod
    def setup_class(cls):
        """Make a modflow model."""
        print("setting up model...")
        t0 = time.time()
        size = 100
        nlay = 10
        nper = 10
        nsfr = int((size ** 2) / 5)

        letters = string.ascii_lowercase
        prepend = "".join(random.choice(letters) for i in range(10))

        cls.modelname = "junk"
        cls.model_ws = f"{base_dir}_{prepend}"
        external_path = "external/"

        if not os.path.isdir(cls.model_ws):
            os.makedirs(cls.model_ws, exist_ok=True)
        if not os.path.isdir(os.path.join(cls.model_ws, external_path)):
            os.makedirs(
                os.path.join(cls.model_ws, external_path), exist_ok=True
            )

        m = fm.Modflow(
            cls.modelname, model_ws=cls.model_ws, external_path=external_path
        )

        dis = fm.ModflowDis(
            m,
            nper=nper,
            nlay=nlay,
            nrow=size,
            ncol=size,
            top=nlay,
            botm=list(range(nlay)),
        )

        rch = fm.ModflowRch(
            m, rech={k: 0.001 - np.cos(k) * 0.001 for k in range(nper)}
        )

        ra = fm.ModflowWel.get_empty(size ** 2)
        well_spd = {}
        for kper in range(nper):
            ra_per = ra.copy()
            ra_per["k"] = 1
            ra_per["i"] = (
                (np.ones((size, size)) * np.arange(size))
                .transpose()
                .ravel()
                .astype(int)
            )
            ra_per["j"] = list(range(size)) * size
            well_spd[kper] = ra
        wel = fm.ModflowWel(m, stress_period_data=well_spd)

        # SFR package
        rd = fm.ModflowSfr2.get_empty_reach_data(nsfr)
        rd["iseg"] = range(len(rd))
        rd["ireach"] = 1
        sd = fm.ModflowSfr2.get_empty_segment_data(nsfr)
        sd["nseg"] = range(len(sd))
        sfr = fm.ModflowSfr2(reach_data=rd, segment_data=sd, model=m)
        cls.init_time = time.time() - t0
        cls.m = m

    def test_init_time(self):
        """test model and package init time(s)."""
        mfp = TestModflowPerformance()
        target = 0.3  # seconds
        assert (
            mfp.init_time < target
        ), f"model init took {mfp.init_time:.2f}s, should take {target:.1f}s"
        print(f"setting up model took {mfp.init_time:.2f}s")

    def test_0_write_time(self):
        """test write time"""
        print("writing files...")
        if "CI" in os.environ and sys.platform.lower() == "darwin":
            assert_time = False
        else:
            assert_time = True
        mfp = TestModflowPerformance()
        target = 6.0
        t0 = time.time()
        mfp.m.write_input()
        t1 = time.time() - t0
        if assert_time:
            assert (
                t1 < target
            ), f"model write took {t1:.2f}s, should take {target:.1f}s"
        print(f"writing input took {t1:.2f}s")

    def test_9_load_time(self):
        """test model load time"""
        print("loading model...")
        mfp = TestModflowPerformance()
        mfp.m.write_input()
        target = 3
        t0 = time.time()
        m = fm.Modflow.load(
            f"{mfp.modelname}.nam", model_ws=mfp.model_ws, check=False
        )
        t1 = time.time() - t0
        assert (
            t1 < target
        ), f"model load took {t1:.2f}s, should take {target:.1f}s"
        print(f"loading the model took {t1:.2f}s")

    @classmethod
    def teardown_class(cls):

        shutil.rmtree(cls.model_ws)
