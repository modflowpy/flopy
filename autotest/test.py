import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg
from test_mp7_cases import Mp7Cases

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
from flopy.modpath import (
    CellDataType,
    FaceDataType,
    LRCParticleData,
    Modpath7,
    Modpath7Bas,
    Modpath7Sim,
    NodeParticleData,
    ParticleData,
    ParticleGroup,
    ParticleGroupLRCTemplate,
    ParticleGroupNodeTemplate,
)
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile

function_tmpdir = Path("temp/test")

mf6sim = Mp7Cases.mf6(function_tmpdir)
mf6sim.write_simulation()
mf6sim.run_simulation()

# create mp7 model
mp = Modpath7(
    modelname=f"{mf6sim.name}_mp",
    flowmodel=mf6sim.get_model(mf6sim.name),
    exe_name="mp7",
    model_ws=mf6sim.sim_path,
    verbose=True,
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
# add a duplicate mp7sim package
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
