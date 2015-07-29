# Test reference class

def test_reference():
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import shapefile
    import flopy
    loadpth = os.path.join('data', 'freyberg')
    ml = flopy.modflow.Modflow.load('freyberg.nam', model_ws=loadpth)


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
    rech = {}
    for kper in range(perlen.shape[0]):
        rech[kper] = np.random.random((nrow,ncol))

    dis = flopy.modflow.ModflowDis(mf,nrow=nrow,ncol=ncol,nlay=nlay,nstp=3,tsmult=1.2,nper=perlen.shape[0],botm=botm,perlen=perlen,
                                   xul=2000.0,yul=5000.0,rotation=10.0)
    bas = flopy.modflow.ModflowBas(mf,ibound=ibound)
    lpf = flopy.modflow.ModflowLpf(mf, hk=k)
    rch = flopy.modflow.ModflowRch(mf,rech=rech)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data={0:[[0,0,0, -100]]})
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data={0:[[1,1,1,5.9,1000.]]})
    oc = flopy.modflow.ModflowOc(mf)
    sms = flopy.modflow.ModflowPcg(mf)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_title("1")
    mm = flopy.plot.ModelMap(ax=ax,sr=mf.dis.sr,model=mf)
    mm.plot_grid(ax=ax)
    mm.plot_bc("WEL")
    mm.plot_ibound(ax=ax)
    plt.close(fig)

    shapename = os.path.join('data', 'test1.shp')
    lpf.hk.to_shapefile(shapename)
    shp = shapefile.Reader(shapename)
    assert shp.numRecords == mf.nrow * mf.ncol
    return

def test_binaryfile_reference():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import shapefile
    import flopy
    import flopy.modflow as fmf

    #--make the model
    ml = fmf.Modflow(exe_name="mf2005",model_ws="reference_testing")
    perlen = np.arange(1,20,1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow,ncol = 50,40
    botm = np.arange(0,-100,-10)
    dis = fmf.ModflowDis(ml,delr=100.0,delc=100.0,nrow=nrow,ncol=ncol,nlay=nlay,nper=perlen.shape[0],perlen=perlen,
        nstp=nstp,tsmult=tsmult,top=10,botm=botm,steady=False,rotation=45)
    lpf = fmf.ModflowLpf(ml,hk=10.0,vka=10.0,laytyp=1)
    pcg = fmf.ModflowPcg(ml)
    oc =  fmf.ModflowOc(ml)
    ibound = np.ones((nrow,ncol))
    ibound[:,0] = -1
    ibound[25:30,30:39] = 0
    bas = fmf.ModflowBas(ml,strt=5.0,ibound=ibound)
    rch = fmf.ModflowRch(ml,rech={0:0.00001,5:0.0001,6:0.0})
    wel_data = [9,25,20,-200]
    wel = fmf.ModflowWel(ml,stress_period_data={0:wel_data})
    ml.write_input()
    ml.run_model()
    #instance without any knowledge of sr tr - builds defaults from info in hds file
    hds = os.path.join("reference_testing", 'modflowtest.hds')
    if not os.path.exists(hds):
        print("could not find hds file " + hds)
        return
    bf = flopy.utils.HeadFile(hds)
    name = os.path.join('data', 'test2.shp')
    bf.to_shapefile(name)
    shp = shapefile.Reader(name)
    assert shp.numRecords == ml.nrow * ml.ncol

def test_mflist_reference():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import shapefile
    import flopy
    import flopy.modflow as fmf

    #model_ws = os.path.join('..', 'data', 'freyberg')
    #ml = fmf.Modflow.load('freyberg.nam', model_ws=model_ws)
    #--make the model
    ml = fmf.Modflow()
    perlen = np.arange(1,20,1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow,ncol = 50,40
    botm = np.arange(0,-100,-10)
    hk = np.random.random((nrow,ncol))
    dis = fmf.ModflowDis(ml,delr=100.0,delc=100.0,nrow=nrow,ncol=ncol,nlay=nlay,nper=perlen.shape[0],perlen=perlen,
        nstp=nstp,tsmult=tsmult,top=10,botm=botm,steady=False,rotation=45)
    lpf = fmf.ModflowLpf(ml,hk=hk,vka=10.0,laytyp=1)
    pcg = fmf.ModflowPcg(ml)
    oc =  fmf.ModflowOc(ml)
    ibound = np.ones((nrow,ncol))
    ibound[:,0] = -1
    ibound[25:30,30:39] = 0
    bas = fmf.ModflowBas(ml,strt=5.0,ibound=ibound)
    rch = fmf.ModflowRch(ml,rech={0:0.00001,5:0.0001,6:0.0})
    wel_dict = {}
    wel_data = [[9,25,20,-200],[0,0,0,-400],[5,20,32,500]]
    wel_dict[0] = wel_data
    wel_data2 = [[45,20,200],[9,49,39,400],[5,20,32,500]]
    wel_dict[10] = wel_data2
    wel = fmf.ModflowWel(ml,stress_period_data={0:wel_data})
    ghb_dict = {0:[1,10,10,400,300]}
    ghb = fmf.ModflowGhb(ml,stress_period_data=ghb_dict)

    test = os.path.join('data', 'test3.shp')
    ml.wel.stress_period_data.to_shapefile(test, kper=0)
    shp = shapefile.Reader(test)
    assert len(shp.fields) == nlay + 3
    assert shp.numRecords == nrow * ncol


#test_mflist_reference()
#test_reference()
test_binaryfile_reference()