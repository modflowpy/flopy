import numpy as np

import flopy


class Mp6Cases1:
    nrow = 3
    ncol = 4
    nlay = 2
    nper = 1
    l1_ibound = np.array(
        [[[-1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]]]
    )
    l2_ibound = np.ones((1, nrow, ncol))
    l2_ibound_alt = np.ones((1, nrow, ncol))
    l2_ibound_alt[0, 0, 0] = 0
    ibound = {
        "mf1": np.concatenate(
            (l1_ibound, l2_ibound), axis=0
        ),  # constant heads around model on top row
        "mf2": np.concatenate(
            (l1_ibound, l2_ibound_alt), axis=0
        ),  # constant heads around model on top row
    }
    laytype = {"mf1": [0, 1], "mf2": [0, 0]}
    hnoflow = -888
    hdry = -777
    top = np.zeros((1, nrow, ncol)) + 10
    bt1 = np.ones((1, nrow, ncol)) + 5
    bt2 = np.ones((1, nrow, ncol)) + 3
    botm = np.concatenate((bt1, bt2), axis=0)
    ipakcb = 740

    def case_1(self, function_tmpdir):
        m = flopy.modflow.Modflow(
            modelname="mf1",
            namefile_ext="nam",
            version="mf2005",
            exe_name="mf2005",
            model_ws=function_tmpdir,
        )

        # dis
        dis = flopy.modflow.ModflowDis(
            model=m,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            nper=self.nper,
            delr=1.0,
            delc=1.0,
            laycbd=0,
            top=self.top,
            botm=self.botm,
            perlen=1,
            nstp=1,
            tsmult=1,
            steady=True,
        )

        # bas
        bas = flopy.modflow.ModflowBas(
            model=m,
            ibound=self.ibound[m.name],
            strt=10,
            ifrefm=True,
            ixsec=False,
            ichflg=False,
            stoper=None,
            hnoflo=self.hnoflow,
            extension="bas",
            unitnumber=None,
            filenames=None,
        )
        # lpf
        lpf = flopy.modflow.ModflowLpf(
            model=m,
            ipakcb=self.ipakcb,
            laytyp=self.laytype[m.name],
            hk=10,
            vka=10,
            hdry=self.hdry,
        )

        # well
        wel = flopy.modflow.ModflowWel(
            model=m,
            ipakcb=self.ipakcb,
            stress_period_data={0: [[1, 1, 1, -5.0]]},
        )

        flopy.modflow.ModflowPcg(
            m,
            hclose=0.001,
            rclose=0.001,
            mxiter=150,
            iter1=30,
        )

        ocspd = {}
        for p in range(self.nper):
            ocspd[(p, 0)] = ["save head", "save budget"]
        ocspd[(0, 0)] = [
            "save head",
            "save budget",
        ]  # pretty sure it just uses the last for everything
        flopy.modflow.ModflowOc(m, stress_period_data=ocspd)

        m.write_input()
        success, buff = m.run_model()
        assert success
        return m

    def case_2(self, function_tmpdir):
        m = flopy.modflow.Modflow(
            modelname="mf2",
            namefile_ext="nam",
            version="mf2005",
            exe_name="mf2005",
            model_ws=function_tmpdir,
        )

        # dis
        dis = flopy.modflow.ModflowDis(
            model=m,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            nper=self.nper,
            delr=1.0,
            delc=1.0,
            laycbd=0,
            top=self.top,
            botm=self.botm,
            perlen=1,
            nstp=1,
            tsmult=1,
            steady=True,
        )

        # bas
        bas = flopy.modflow.ModflowBas(
            model=m,
            ibound=self.ibound[m.name],
            strt=10,
            ifrefm=True,
            ixsec=False,
            ichflg=False,
            stoper=None,
            hnoflo=self.hnoflow,
            extension="bas",
            unitnumber=None,
            filenames=None,
        )
        # lpf
        lpf = flopy.modflow.ModflowLpf(
            model=m,
            ipakcb=self.ipakcb,
            laytyp=self.laytype[m.name],
            hk=10,
            vka=10,
            hdry=self.hdry,
        )

        # well
        wel = flopy.modflow.ModflowWel(
            model=m,
            ipakcb=self.ipakcb,
            stress_period_data={0: [[1, 1, 1, -5.0]]},
        )

        flopy.modflow.ModflowPcg(
            m,
            hclose=0.001,
            rclose=0.001,
            mxiter=150,
            iter1=30,
        )

        ocspd = {}
        for p in range(self.nper):
            ocspd[(p, 0)] = ["save head", "save budget"]
        ocspd[(0, 0)] = [
            "save head",
            "save budget",
        ]  # pretty sure it just uses the last for everything
        flopy.modflow.ModflowOc(m, stress_period_data=ocspd)

        m.write_input()
        success, buff = m.run_model()
        assert success
        return m


class Mp6Cases2:
    nrow = 3
    ncol = 4
    nlay = 2
    nper = 1
    l1_ibound = np.array(
        [[[-1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]]]
    )
    l2_ibound = np.ones((1, nrow, ncol))
    l2_ibound_alt = np.ones((1, nrow, ncol))
    l2_ibound_alt[0, 0, 0] = 0
    ibound = {
        "mf1": np.concatenate(
            (l1_ibound, l2_ibound), axis=0
        ),  # constant heads around model on top row
    }
    laytype = {
        "mf1": [0, 1],
    }
    hnoflow = -888
    hdry = -777
    top = np.zeros((1, nrow, ncol)) + 10
    bt1 = np.ones((1, nrow, ncol)) + 5
    bt2 = np.ones((1, nrow, ncol)) + 3
    botm = np.concatenate((bt1, bt2), axis=0)
    ipakcb = 740

    def case_1(self, function_tmpdir):
        m = flopy.modflow.Modflow(
            modelname=f"mf1",
            namefile_ext="nam",
            version="mf2005",
            exe_name="mf2005",
            model_ws=function_tmpdir,
        )

        # dis
        dis = flopy.modflow.ModflowDis(
            model=m,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            nper=self.nper,
            delr=1.0,
            delc=1.0,
            laycbd=0,
            top=self.top,
            botm=self.botm,
            perlen=1,
            nstp=1,
            tsmult=1,
            steady=True,
        )

        # bas
        bas = flopy.modflow.ModflowBas(
            model=m,
            ibound=self.ibound[m.name],
            strt=10,
            ifrefm=True,
            ixsec=False,
            ichflg=False,
            stoper=None,
            hnoflo=self.hnoflow,
            extension="bas",
            unitnumber=None,
            filenames=None,
        )
        # lpf
        lpf = flopy.modflow.ModflowLpf(
            model=m,
            ipakcb=self.ipakcb,
            laytyp=self.laytype[m.name],
            hk=10,
            vka=10,
            hdry=self.hdry,
        )

        # well
        wel = flopy.modflow.ModflowWel(
            model=m,
            ipakcb=self.ipakcb,
            stress_period_data={0: [[1, 1, 1, -5.0]]},
        )

        flopy.modflow.ModflowPcg(
            m, hclose=0.001, rclose=0.001, mxiter=150, iter1=30
        )

        ocspd = {}
        for p in range(self.nper):
            ocspd[(p, 0)] = ["save head", "save budget"]
        ocspd[(0, 0)] = [
            "save head",
            "save budget",
        ]  # pretty sure it just uses the last for everything
        flopy.modflow.ModflowOc(m, stress_period_data=ocspd)

        m.write_input()
        success, buff = m.run_model()
        assert success

        return m
