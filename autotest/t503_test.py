import pytest
import sys
import os
import shutil
import flopy
import pymake
from ci_framework import (
    baseTestDir,
    flopyTest,
    download_mf6_examples,
)

exe_name = "mf6"
v = flopy.which(exe_name)
run = True
if v is None:
    run = False

mf6path = download_mf6_examples()

exclude_models = ("lnf",)
exclude_examples = (
    "sagehen",
    "ex-gwt-keating",
    # "ex-gwt-moc3d-p02",
    # "ex-gwt-mt3dms-p01",
    # "ex-gwt-mt3dsupp632",
    # "ex-gwt-prudic2004t2",
)

exdirs = sorted(
    [os.path.join(mf6path, exdir) for exdir in list(os.listdir(mf6path))]
)

print("example simulations")
for exdir in exdirs:
    print(f"  {exdir}")


def copy_folder(baseDir, src):
    subDir = src.partition("{0}mf6examples{0}".format(os.path.sep))[2]
    if os.path.basename(subDir) in os.path.basename(baseDir):
        dst = baseDir
    else:
        dst = os.path.join(baseDir, subDir)

    # clean the destination directory if it exists
    if os.path.isdir(dst):
        shutil.rmtree(dst)

    # copy the files
    print(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)

    # remove the src directory
    shutil.rmtree(src)

    return dst


def simulation_subdirs(baseDir):
    exsubdirs = []
    for dirName, subdirList, fileList in os.walk(baseDir):
        for file_name in fileList:
            if file_name.lower() == "mfsim.nam":
                print(f"Found directory: {dirName}")
                exsubdirs.append(dirName)
    return sorted(exsubdirs)


def runmodel(exdir):
    baseDir = (
        baseTestDir(__file__, relPath="temp", verbose=True)
        + "_"
        + os.path.basename(exdir)
    )
    fpTest = flopyTest(verbose=True)

    simulations = simulation_subdirs(exdir)
    for src in simulations:
        ws = copy_folder(baseDir, src)
        fpTest.addTestDir(ws)
        f = os.path.basename(os.path.normpath(ws))
        print("\n\n")
        print(f"**** RUNNING TEST: {f} ****")
        print("\n")

        # load the model into a flopy simulation
        print(f"loading {f}")
        sim = flopy.mf6.MFSimulation.load(f, "mf6", exe_name, ws)
        assert isinstance(sim, flopy.mf6.MFSimulation)

        # run the simulation in folder if executable is available
        if run:
            success, buff = sim.run_simulation()
            assert success

            headfiles = [
                f for f in os.listdir(ws) if f.lower().endswith(".hds")
            ]

            # set the comparison directory
            ws2 = f"{ws}-RERUN"
            fpTest.addTestDir(ws2)
            sim.simulation_data.mfpath.set_sim_path(ws2)

            # remove the comparison directory if it exists
            if os.path.isdir(ws2):
                shutil.rmtree(ws2)

            sim.write_simulation()
            success, buff = sim.run_simulation()
            assert success

            headfiles1 = []
            headfiles2 = []
            for hf in headfiles:
                headfiles1.append(os.path.join(ws, hf))
                headfiles2.append(os.path.join(ws2, hf))

            success = pymake.compare_heads(
                None,
                None,
                precision="double",
                text="head",
                files1=headfiles1,
                files2=headfiles2,
                outfile=os.path.join(ws, "head_compare.dat"),
            )
            assert success, f"comparision for {ws} failed"

    fpTest.addTestDir(baseDir)

    fpTest.teardown()


# for running tests with pytest
@pytest.mark.parametrize(
    "exdir",
    exdirs,
)
def test_load_mf6_distribution_models(exdir):
    runmodel(exdir)
    return


# for running outside of pytest
def runmodels():
    for exdir in exdirs:
        runmodel(exdir)
    return


if __name__ == "__main__":
    runmodels()
