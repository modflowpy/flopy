def build_model():
    import os
    import numpy as np
    import pandas as pd
    import flopy

    model_ws = os.path.join("temp","t040_test_subwt")
    ibound_path = os.path.join("..", "examples", "data", "subwt_example"
                               , "ibound.ref")

    ml = flopy.modflow.Modflow("subwt_mf2005", model_ws=model_ws,exe_name="mf2005")
    perlen = [1.0, 60. * 365.25, 60 * 365.25]
    nstp = [1, 60, 60]
    flopy.modflow.ModflowDis(ml, nlay=4, nrow=20, ncol=15, delr=2000.0,
                             delc=2000.0,
                             nper=3, steady=[True, False, False]
                             , perlen=perlen, nstp=nstp,
                             top=150.0, botm=[50, -100, -150.0, -350.0])

    flopy.modflow.ModflowLpf(ml, laytyp=[1, 0, 0, 0], hk=[4, 4, 0.01, 4]
                             , vka=[0.4, 0.4, 0.01, 0.4],
                             sy=0.3, ss=1.0e-6)

    # temp_ib = np.ones((ml.nrow,ml.ncol),dtype=np.int)
    # np.savetxt("temp_ib.dat",temp_ib,fmt="%1d")
    ibound = np.loadtxt(ibound_path)
    ibound[ibound == 5] = -1
    flopy.modflow.ModflowBas(ml, ibound=ibound, strt=100.0)

    sp1_wells = pd.DataFrame(data=np.argwhere(ibound == 2), columns=['i', 'j'])
    sp1_wells.loc[:, "k"] = 0
    sp1_wells.loc[:, "flux"] = 2200.0
    sp1_wells = sp1_wells.loc[:, ["k", "i", "j", "flux"]].values.tolist()

    sp2_wells = sp1_wells.copy()
    sp2_wells.append([1, 8, 9, -72000.0])
    sp2_wells.append([3, 11, 6, -72000.0])

    flopy.modflow.ModflowWel(ml, stress_period_data=
                            {0: sp1_wells, 1: sp2_wells, 2: sp1_wells})

    flopy.modflow.ModflowSubwt(ml,iswtoc=1,nsystm=4,sgs=2.0,sgm=1.7,
                               lnwt=[0,1,2,3],thick=[45,70,50,90],icrcc=0,
                               cr=0.01,cc=0.25,istpcs=1,pcsoff=15.0,
                               void=0.82,ithk=1,ivoid=1)
    flopy.modflow.ModflowOc(ml,stress_period_data={(0,0):["save head","save budget"]})
    flopy.modflow.ModflowPcg(ml,hclose=0.01,rclose=1.0)
    ml.write_input()
    ml.run_model()
    hds_stress = flopy.utils.HeadFile(os.path.join(
            model_ws,ml.name+".eff_stress.hds"),text="effective stress")
    print(hds_stress.recordarray)
    d = hds_stress.get_alldata()
    d1 = d[:,1,8,9]
    print(d1)


if __name__ == "__main__":
    build_model()
