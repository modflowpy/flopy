import os
import shutil
import flopy
import pymake


def download_mf6_examples():
    """
    Download mf6 examples and return location of folder

    """

    # set url
    dirname = "mf6examples"
    url = "https://github.com/MODFLOW-USGS/modflow6-examples/releases/" + \
          "download/current/modflow6-examples.zip"

    # create folder for mf6 distribution download
    cpth = os.getcwd()
    dstpth = os.path.join('temp', dirname)
    print('create...{}'.format(dstpth))
    if not os.path.exists(dstpth):
        os.makedirs(dstpth)
    os.chdir(dstpth)

    # Download the distribution
    pymake.download_and_unzip(url, verify=True)

    # change back to original path
    os.chdir(cpth)

    # return the absolute path to the distribution
    mf6path = os.path.abspath(dstpth)

    return mf6path


out_dir = os.path.join("temp", "t503")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

mf6path = download_mf6_examples()
distpth = os.path.join(mf6path)

exclude_models = (
    "lnf",
)
exclude_examples = (
    "sagehen",
    "ex-gwt-keating",
    # "ex-gwt-moc3d-p02",
    # "ex-gwt-mt3dms-p01",
    # "ex-gwt-mt3dsupp632",
    # "ex-gwt-prudic2004t2",
)
src_folders = []

for dirName, subdirList, fileList in os.walk(mf6path):
    dirBase = os.path.basename(os.path.normpath(dirName))
    useModel = True
    for exclude in exclude_models:
        if exclude in dirName:
            useModel = False
            break
    if useModel:
        for exclude in exclude_examples:
            if exclude in dirName:
                useModel = False
                break
    if useModel:
        for file_name in fileList:
            if file_name.lower() == "mfsim.nam":
                print('Found directory: {}'.format(dirName))
                src_folders.append(dirName)
src_folders = sorted(src_folders)

folders = []
for src in src_folders:
    dirBase = src.partition("{0}mf6examples{0}".format(os.path.sep))[2]
    dst = os.path.join(out_dir, dirBase)

    print('copying {} -> {}'.format(src, dst))
    folders.append(dst)
    shutil.copytree(src, dst)
folders = sorted(folders)

exe_name = 'mf6'
v = flopy.which(exe_name)
run = True
if v is None:
    run = False


def runmodel(folder):
    f = os.path.basename(os.path.normpath(folder))
    print('\n\n')
    print('**** RUNNING TEST: {} ****'.format(f))
    print('\n')

    # load the model into a flopy simulation
    print('loading {}'.format(f))
    sim = flopy.mf6.MFSimulation.load(f, 'mf6', exe_name, folder)
    assert isinstance(sim, flopy.mf6.MFSimulation)

    # run the simulation in folder if executable is available
    if run:
        success, buff = sim.run_simulation()
        assert success

        headfiles = [f for f in os.listdir(folder)
                     if f.lower().endswith('.hds')]

        folder2 = folder + '-RERUN'
        sim.simulation_data.mfpath.set_sim_path(folder2)
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success

        headfiles1 = []
        headfiles2 = []
        for hf in headfiles:
            headfiles1.append(os.path.join(folder, hf))
            headfiles2.append(os.path.join(folder2, hf))

        outfile = os.path.join(folder, 'head_compare.dat')
        success = pymake.compare_heads(None, None, precision='double',
                                       text='head', files1=headfiles1,
                                       files2=headfiles2, outfile=outfile)
        assert success


# for running tests with nosetests
def test_load_mf6_distribution_models():
    for f in folders:
        yield runmodel, f
    return


# for running outside of nosetests
def runmodels():
    for f in folders:
        runmodel(f)
    return


if __name__ == '__main__':
    # to run them all with python
    runmodels()

    # or to run just test, pass the example name into runmodel
    # runmodel('ex30-vilhelmsen-gf')
