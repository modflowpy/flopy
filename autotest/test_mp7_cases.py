import numpy as np

from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfdis,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfwel,
    ModflowIms,
    ModflowTdis,
)
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowRch,
    ModflowRiv,
    ModflowWel,
)
from flopy.modpath import (
    Modpath7,
    Modpath7Bas,
    Modpath7Sim,
    ParticleData,
    ParticleGroup,
)


class Mp7Cases:
    nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
    nlay, nrow, ncol = 3, 21, 20
    delr = delc = 500.0
    top = 400.0
    botm = [220.0, 200.0, 0.0]
    laytyp = [1, 0, 0]
    kh = [50.0, 0.01, 200.0]
    kv = [10.0, 0.01, 20.0]
    wel_loc = (2, 10, 9)
    wel_q = -150000.0
    rch = 0.005
    riv_h = 320.0
    riv_z = 317.0
    riv_c = 1.0e5

    zone3 = np.ones((nrow, ncol), dtype=np.int32)
    zone3[wel_loc[1:]] = 2
    zones = [1, 1, zone3]

    # create particles
    partlocs = []
    partids = []
    for i in range(nrow):
        partlocs.append((0, i, 2))
        partids.append(i)
    part0 = ParticleData(partlocs, structured=True, particleids=partids)
    pg0 = ParticleGroup(
        particlegroupname="PG1", particledata=part0, filename="ex01a.sloc"
    )

    v = [(0,), (400,)]
    pids = [1, 2]  # [1000, 1001]
    part1 = ParticleData(v, structured=False, drape=1, particleids=pids)
    pg1 = ParticleGroup(
        particlegroupname="PG2", particledata=part1, filename="ex01a.pg2.sloc"
    )

    particlegroups = [pg0, pg1]

    @staticmethod
    def mf6(function_tmpdir):
        """
        MODPATH 7 example 1 for MODFLOW 6
        """

        ws = function_tmpdir / "mf6"
        nm = "ex01_mf6"

        # Create the Flopy simulation object
        sim = MFSimulation(
            sim_name=nm, exe_name="mf6", version="mf6", sim_ws=ws
        )

        # Create the Flopy temporal discretization object
        pd = (Mp7Cases.perlen, Mp7Cases.nstp, Mp7Cases.tsmult)
        tdis = ModflowTdis(
            sim,
            pname="tdis",
            time_units="DAYS",
            nper=Mp7Cases.nper,
            perioddata=[pd],
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = f"{nm}.nam"
        gwf = ModflowGwf(
            sim, modelname=nm, model_nam_file=model_nam_file, save_flows=True
        )

        # Create the Flopy iterative model solver (ims) Package object
        ims = ModflowIms(
            sim,
            pname="ims",
            complexity="SIMPLE",
            rcloserecord=1e-3,
            inner_dvclose=1e-6,
            outer_dvclose=1e-6,
            outer_maximum=50,
            inner_maximum=100,
        )

        # create gwf file
        dis = ModflowGwfdis(
            gwf,
            pname="dis",
            nlay=Mp7Cases.nlay,
            nrow=Mp7Cases.nrow,
            ncol=Mp7Cases.ncol,
            length_units="FEET",
            delr=Mp7Cases.delr,
            delc=Mp7Cases.delc,
            top=Mp7Cases.top,
            botm=Mp7Cases.botm,
        )
        # Create the initial conditions package
        ic = ModflowGwfic(gwf, pname="ic", strt=Mp7Cases.top)

        # Create the node property flow package
        npf = ModflowGwfnpf(
            gwf,
            pname="npf",
            icelltype=Mp7Cases.laytyp,
            k=Mp7Cases.kh,
            k33=Mp7Cases.kv,
        )

        # recharge
        ModflowGwfrcha(gwf, recharge=Mp7Cases.rch)
        # wel
        wd = [(Mp7Cases.wel_loc, Mp7Cases.wel_q)]
        ModflowGwfwel(gwf, maxbound=1, stress_period_data={0: wd})
        # river
        rd = []
        for i in range(Mp7Cases.nrow):
            rd.append(
                [
                    (0, i, Mp7Cases.ncol - 1),
                    Mp7Cases.riv_h,
                    Mp7Cases.riv_c,
                    Mp7Cases.riv_z,
                ]
            )
        ModflowGwfriv(gwf, stress_period_data={0: rd})
        # Create the output control package
        headfile = f"{nm}.hds"
        head_record = [headfile]
        budgetfile = f"{nm}.cbb"
        budget_record = [budgetfile]
        saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
        oc = ModflowGwfoc(
            gwf,
            pname="oc",
            saverecord=saverecord,
            head_filerecord=head_record,
            budget_filerecord=budget_record,
        )

        return sim

    @staticmethod
    def mp7_mf6(function_tmpdir):
        sim = Mp7Cases.mf6(function_tmpdir)
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success, "mf6 model did not run"

        # create modpath files
        mp = Modpath7(
            modelname=f"{sim.name}_mp",
            flowmodel=sim.get_model(sim.name),
            exe_name="mp7",
            model_ws=sim.sim_path,
        )
        defaultiface6 = {"RCH": 6, "EVT": 6}
        mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface6)
        mpsim = Modpath7Sim(
            mp,
            simulationtype="combined",
            trackingdirection="forward",
            weaksinkoption="pass_through",
            weaksourceoption="pass_through",
            budgetoutputoption="summary",
            budgetcellnumbers=[1049, 1259],
            traceparticledata=[1, 1000],
            referencetime=[0, 0, 0.0],
            stoptimeoption="extend",
            timepointdata=[500, 1000.0],
            zonedataoption="on",
            zones=Mp7Cases.zones,
            particlegroups=Mp7Cases.particlegroups,
        )

        return mp

    @staticmethod
    def mf2005(function_tmpdir):
        """
        MODPATH 7 example 1 for MODFLOW-2005
        """

        ws = function_tmpdir / "mf2005"
        nm = "ex01_mf2005"
        iu_cbc = 130
        m = Modflow(nm, model_ws=ws, exe_name="mf2005")
        ModflowDis(
            m,
            nlay=Mp7Cases.nlay,
            nrow=Mp7Cases.nrow,
            ncol=Mp7Cases.ncol,
            nper=Mp7Cases.nper,
            itmuni=4,
            lenuni=1,
            perlen=Mp7Cases.perlen,
            nstp=Mp7Cases.nstp,
            tsmult=Mp7Cases.tsmult,
            steady=True,
            delr=Mp7Cases.delr,
            delc=Mp7Cases.delc,
            top=Mp7Cases.top,
            botm=Mp7Cases.botm,
        )
        ModflowLpf(
            m,
            ipakcb=iu_cbc,
            laytyp=Mp7Cases.laytyp,
            hk=Mp7Cases.kh,
            vka=Mp7Cases.kv,
        )
        ModflowBas(m, ibound=1, strt=Mp7Cases.top)
        # recharge
        ModflowRch(m, ipakcb=iu_cbc, rech=Mp7Cases.rch, nrchop=1)
        # wel
        wd = [i for i in Mp7Cases.wel_loc] + [Mp7Cases.wel_q]
        ModflowWel(m, ipakcb=iu_cbc, stress_period_data={0: wd})
        # river
        rd = []
        for i in range(Mp7Cases.nrow):
            rd.append(
                [
                    0,
                    i,
                    Mp7Cases.ncol - 1,
                    Mp7Cases.riv_h,
                    Mp7Cases.riv_c,
                    Mp7Cases.riv_z,
                ]
            )
        ModflowRiv(m, ipakcb=iu_cbc, stress_period_data={0: rd})
        # output control
        ModflowOc(
            m,
            stress_period_data={
                (0, 0): ["save head", "save budget", "print head"]
            },
        )
        ModflowPcg(m, hclose=1e-6, rclose=1e-3, iter1=100, mxiter=50)

        return m

    @staticmethod
    def mp7_mf2005(function_tmpdir):
        m = Mp7Cases.mf2005(function_tmpdir)
        m.write_input()
        success, buff = m.run_model()
        assert success, "mf2005 model did not run"

        # create modpath files
        mp = Modpath7(
            modelname=f"{m.name}_mp",
            flowmodel=m,
            exe_name="mp7",
            model_ws=m.model_ws,
        )
        defaultiface = {"RECHARGE": 6, "ET": 6}
        mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface)
        mpsim = Modpath7Sim(
            mp,
            simulationtype="combined",
            trackingdirection="forward",
            weaksinkoption="pass_through",
            weaksourceoption="pass_through",
            budgetoutputoption="summary",
            budgetcellnumbers=[1049, 1259],
            traceparticledata=[1, 1000],
            referencetime=[0, 0, 0.0],
            stoptimeoption="extend",
            timepointdata=[500, 1000.0],
            zonedataoption="on",
            zones=Mp7Cases.zones,
            particlegroups=Mp7Cases.particlegroups,
        )

        return mp

    def case_mp7_mf6(self, function_tmpdir):
        return Mp7Cases.mp7_mf6(function_tmpdir)

    def case_mp7_mf2005(self, function_tmpdir):
        return Mp7Cases.mp7_mf2005(function_tmpdir)
