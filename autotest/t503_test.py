import os
import shutil
import flopy
import pymake
import platform


def download_mf6_distribution():
    """
    Download mf6 distribution and return location of folder

    """

    # set url
    dirname = 'mf6.0.1'
    url = 'https://water.usgs.gov/ogw/modflow/{0}.zip'.format(dirname)

    # create folder for mf6 distribution download
    cpth = os.getcwd()
    dstpth = os.path.join('temp', 'mf6dist')
    print('create...{}'.format(dstpth))
    if not os.path.exists(dstpth):
        os.makedirs(dstpth)
    os.chdir(dstpth)

    # Download the distribution
    pymake.download_and_unzip(url, verify=True)

    # change back to original path
    os.chdir(cpth)

    # return the absolute path to the distribution
    mf6path = os.path.abspath(os.path.join(dstpth, dirname))

    return mf6path


out_dir = os.path.join('temp', 't503')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

mf6path = download_mf6_distribution()
distpth = os.path.join(mf6path, 'examples')
folders = [f for f in os.listdir(distpth)
           if os.path.isdir(os.path.join(distpth, f))]

for f in folders:
    src = os.path.join(distpth, f)
    dst = os.path.join(out_dir, f)
    print('copying {}'.format(f))
    shutil.copytree(src, dst)

exe_name = 'mf6'
if platform.system() == 'Windows':
    exe_name += '.exe'
v = flopy.which(exe_name)
run = True
if v is None:
    run = False


def run_test(f):

    print('\n\n')
    print('**** RUNNING TEST: {} ****'.format(f))
    print('\n')

    # load the model into a flopy simulation
    folder = os.path.join(out_dir, f)
    print('loading {}'.format(folder))
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
        yield run_test, f
    return


# for running outside of nosetests
def run_tests():
    for f in folders:
        run_test(f)
    return


if __name__ == '__main__':
    run_tests()
