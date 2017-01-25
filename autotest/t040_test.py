import os
import shutil
import flopy

model_ws = os.path.join('temp', 't040')
# delete the directory if it exists
if os.path.isdir(model_ws):
    shutil.rmtree(model_ws)

ibound_path = os.path.join('..', 'examples', 'data', 'subwt_example',
                           'ibound.ref')

exe_name = 'mf2005'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False

def build_model():
    import matplotlib.pyplot as plt

    ml = test_subwt()
    ml.write_input()
    if run:
        ml.run_model()

        hds_geo = flopy.utils.HeadFile(os.path.join(
            model_ws, ml.name + '.geostatic_stress.hds'),
            text='stress').get_alldata()
        hds_eff = flopy.utils.HeadFile(os.path.join(
            model_ws, ml.name + '.eff_stress.hds'),
            text='effective stress').get_alldata()

        hds_sub = flopy.utils.HeadFile(os.path.join(
            model_ws, ml.name + '.subsidence.hds'),
            text='subsidence').get_alldata()

        hds_comp = flopy.utils.HeadFile(os.path.join(
            model_ws, ml.name + '.total_comp.hds'),
            text='layer compaction').get_alldata()

        hds_precon = flopy.utils.HeadFile(os.path.join(
            model_ws, ml.name + '.precon_stress.hds'),
            text='preconsol stress').get_alldata()

        # make 6 from subwt manual
        i, j = 8, 9
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(hds_precon[:, 0, i, j], color='0.5', dashes=(1, 1))
        ax1.plot(hds_eff[:, 0, i, j], color='r')

        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(hds_geo[:, 0, i, j], color='b')

        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(hds_precon[:, 1, i, j], color='0.5', dashes=(1, 1))
        ax3.plot(hds_eff[:, 1, i, j], color='r')

        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(hds_geo[:, 1, i, j], color='b')
        plt.savefig(os.path.join(model_ws, 'fig6.pdf'))

        fig2 = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        i1, j1 = 8, 9
        i2, j2 = 11, 6
        for k in range(hds_comp.shape[1]):
            hds_comp_sum = hds_comp[:, k:, :, :].sum(axis=1)
            print(hds_comp_sum.shape)
            ax1.plot(hds_comp_sum[:, i1, j1])
            ax2.plot(hds_comp_sum[:, i2, j2])
        plt.show()


def test_subwt():
    '''
    test040 subwt example problem test
    '''
    import copy
    import numpy as np

    ml = flopy.modflow.Modflow('subwt_mf2005', model_ws=model_ws,
                               exe_name=exe_name)
    perlen = [1.0, 60. * 365.25, 60 * 365.25]
    nstp = [1, 60, 60]
    flopy.modflow.ModflowDis(ml, nlay=4, nrow=20, ncol=15, delr=2000.0,
                             delc=2000.0,
                             nper=3, steady=[True, False, False]
                             , perlen=perlen, nstp=nstp,
                             top=150.0, botm=[50, -100, -150.0, -350.0])

    flopy.modflow.ModflowLpf(ml, laytyp=[1, 0, 0, 0], hk=[4, 4, 0.01, 4],
                             vka=[0.4, 0.4, 0.01, 0.4],
                             sy=0.3, ss=1.0e-6)

    # temp_ib = np.ones((ml.nrow,ml.ncol),dtype=np.int)
    # np.savetxt('temp_ib.dat',temp_ib,fmt='%1d')
    ibound = np.loadtxt(ibound_path)
    ibound[ibound == 5] = -1
    flopy.modflow.ModflowBas(ml, ibound=ibound, strt=100.0)

    # sp1_wells = pd.DataFrame(data=np.argwhere(ibound == 2), columns=['i', 'j'])
    # sp1_wells.loc[:, 'k'] = 0
    # sp1_wells.loc[:, 'flux'] = 2200.0
    # sp1_wells = sp1_wells.loc[:, ['k', 'i', 'j', 'flux']].values.tolist()
    idxs = np.argwhere(ibound == 2)
    sp1_wells = []
    for idx in idxs:
        sp1_wells.append([0, idx[0], idx[1], 2200.0])

    sp2_wells = copy.copy(sp1_wells)
    sp2_wells.append([1, 8, 9, -72000.0])
    sp2_wells.append([3, 11, 6, -72000.0])

    flopy.modflow.ModflowWel(ml, stress_period_data=
    {0: sp1_wells, 1: sp2_wells, 2: sp1_wells})

    flopy.modflow.ModflowSwt(ml, iswtoc=1, nsystm=4, sgs=2.0, sgm=1.7,
                             lnwt=[0, 1, 2, 3], thick=[45, 70, 50, 90],
                             icrcc=0,
                             cr=0.01, cc=0.25, istpcs=1, pcsoff=15.0,
                             void=0.82, ithk=1, ivoid=1)
    flopy.modflow.ModflowOc(ml, stress_period_data={
        (0, 0): ['save head', 'save budget']})
    flopy.modflow.ModflowPcg(ml, hclose=0.01, rclose=1.0)

    ml.write_input()

    ml1 = flopy.modflow.Modflow.load(ml.namefile, model_ws=model_ws,
                                     verbose=True, exe_name='mf2005',
                                     load_only=['SWT'], forgive=False)

    return ml


if __name__ == '__main__':
    build_model()
