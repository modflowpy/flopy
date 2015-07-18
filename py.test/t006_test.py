# Test reference class

def test_reference():
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import flopy
    mf = flopy.modflow.Modflow(model_ws='data/')
    nrow,ncol = 10,10
    nlay = 3
    botm = [20,10,0]
    top = 30
    perlen  = np.zeros(10) + 10.0
    perlen[1] = 45.2
    start = datetime(1979,9,29)
    ibound = np.ones((nrow,ncol))
    ibound[:,0] = 2
    ibound[:,9] = -1
    k = np.random.random((nrow,ncol))
    dis = flopy.modflow.ModflowDis(mf,nrow=nrow,ncol=ncol,nlay=nlay,nstp=3,tsmult=1.2,nper=perlen.shape[0],botm=botm,perlen=perlen,
                                   start_datetime=start,xul=2000.0,yul=5000.0,rotation=10.0)
    bas = flopy.modflow.ModflowBas(mf,ibound=ibound)
    lpf = flopy.modflow.ModflowLpf(mf, hk=k)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data={0:[[0,0,0, -100]]})
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data={0:[[1,1,1,5.9,1000.]]})
    oc = flopy.modflow.ModflowOc(mf)
    sms = flopy.modflow.ModflowPcg(mf)
    wel.write_file()
    try:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_title("1")
        mm = flopy.plot.ModelMap(ax,model=mf)
        mm.plot_grid(ax=ax)
        mm.plot_bc("WEL")
        mm.plot_ibound(ax=ax)
        #plt.show()
        plt.close(fig)
    except Exception as e:
        raise Exception("error in modelmap: "+str(e))

    #print(dis.tr.stressperiod_deltas)
    assert len(dis.tr.stressperiod_deltas) == len(perlen)
    assert len(dis.tr.stressperiod_start) == len(perlen)
    assert len(dis.tr.stressperiod_end) == len(perlen)
    print(dis.tr.stressperiod_start)
    #print(dis.tr.stressperiod_end)
    print(dis.tr.timestep_start)
    s = dis.tr.timestep_start[dis.tr.kperkstp_loc[(0,0)]]
    e = dis.tr.timestep_end[dis.tr.kperkstp_loc[(9,2)]]

    print((e-s).days,np.cumsum(perlen)[-1])
    fig, axes = plt.subplots(nlay)
    lpf.hk.plot(axes=axes)
    plt.show()
    return

test_reference()