import pytest
import os
import shutil
import numpy as np
import flopy
import pymake


def download_mf6_examples(delete_existing=False):
    """
    Download mf6 examples and return location of folder

    """
    # save current directory
    cpth = os.getcwd()

    # create folder for mf6 distribution download
    dirname = "mf6examples"
    dstpth = os.path.join("temp", dirname)

    # delete the existing examples
    if delete_existing:
        if os.path.isdir(dstpth):
            shutil.rmtree(dstpth)

    # download the MODFLOW 6 distribution does not exist
    if not os.path.isdir(dstpth):
        print(f"create...{dstpth}")
        if not os.path.exists(dstpth):
            os.makedirs(dstpth)
        os.chdir(dstpth)

        # Download the distribution
        url = (
            "https://github.com/MODFLOW-USGS/modflow6-examples/releases/"
            "download/current/modflow6-examples.zip"
        )
        pymake.download_and_unzip(url, verify=True)

        # change back to original path
        os.chdir(cpth)

    # return the absolute path to the distribution
    return os.path.abspath(dstpth)


exe_name = "mf6"
v = flopy.which(exe_name)
run = True
if v is None:
    run = False

out_dir = os.path.join("temp", "t503")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

mf6path = download_mf6_examples()
distpth = os.path.join(mf6path)

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


def copy_folder(src):
    dirBase = src.partition("{0}mf6examples{0}".format(os.path.sep))[2]
    dst = os.path.join(out_dir, dirBase)

    # clean the destination directory if it exists
    if os.path.isdir(dst):
        shutil.rmtree(dst)

    # copy the files
    print(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)

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
    simulations = simulation_subdirs(exdir)
    for src in simulations:
        ws = copy_folder(src)
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
    # to run them all with python
    runmodels()
