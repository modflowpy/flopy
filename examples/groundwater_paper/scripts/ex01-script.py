import flopy
model = flopy.modflow.Modflow()
dis = flopy.modflow.ModflowDis()
bas = flopy.modflow.ModflowBas()
lpf = flopy.modflow.ModflowLpf()
pcg = flopy.modflow.ModflowPcg()
model.write_input()
success, stdout = model.run_model()
