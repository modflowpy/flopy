import os
import shutil
import numpy as np
import flopy
from flopy.utils.util_array import Util2d, Util3d, Transient2d

out_dir = "temp"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)


def test_transient2d():
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=10, nrow=10, ncol=10, nper=3)
    t2d = Transient2d(ml, (10, 10), np.float32, 10., "fake")
    a1 = t2d.array
    assert a1.shape == (3, 1, 10, 10), a1.shape
    t2d.cnstnt = 2.0
    assert np.array_equal(t2d.array, np.zeros((3, 1, 10, 10)) + 20.0)

    t2d[0] = 1.0
    t2d[2] = 999
    assert np.array_equal(t2d[0].array,np.ones((ml.nrow,ml.ncol)))
    assert np.array_equal(t2d[2].array,np.ones((ml.nrow,ml.ncol))*999)

    m4d = t2d.array
    t2d2 = Transient2d.from_4d(ml,"rch",{"rech":m4d})
    m4d2 = t2d2.array
    assert np.array_equal(m4d,m4d2)

def test_util2d():
    ml = flopy.modflow.Modflow()
    u2d = Util2d(ml, (10, 10), np.float32, 10., "test")
    a1 = u2d.array
    a2 = np.ones((10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)

    # test external filenames - ascii and binary
    fname_ascii = os.path.join(out_dir, 'test_a.dat')
    fname_bin = os.path.join(out_dir, 'test_b.dat')
    np.savetxt(fname_ascii,a1,fmt="%15.6E")
    u2d.write_bin(a1.shape,fname_bin,a1,bintype="head")
    dis = flopy.modflow.ModflowDis(ml,2,10,10)
    lpf = flopy.modflow.ModflowLpf(ml,hk=[fname_ascii,fname_bin])
    ml.lpf.hk[1].fmtin = "(BINARY)"
    assert np.array_equal(lpf.hk[0].array,a1)
    assert np.array_equal(lpf.hk[1].array,a1)


    # test external filenames - ascii and binary with model_ws and external_path
    ml = flopy.modflow.Modflow(model_ws="temp",external_path="ref")
    u2d = Util2d(ml, (10, 10), np.float32, 10., "test")
    fname_ascii = os.path.join(out_dir, 'test_a.dat')
    fname_bin = os.path.join(out_dir, 'test_b.dat')
    np.savetxt(fname_ascii,a1,fmt="%15.6E")
    u2d.write_bin(a1.shape,fname_bin,a1,bintype="head")
    dis = flopy.modflow.ModflowDis(ml,2,10,10)
    lpf = flopy.modflow.ModflowLpf(ml,hk=[fname_ascii,fname_bin])
    ml.lpf.hk[1].fmtin = "(BINARY)"
    assert np.array_equal(lpf.hk[0].array,a1)
    assert np.array_equal(lpf.hk[1].array,a1)

    # bin read write test
    fname = os.path.join(out_dir, 'test.bin')
    u2d.write_bin((10, 10), fname, u2d.array)
    a3 = u2d.load_bin((10, 10), fname, u2d.dtype)[1]
    assert np.array_equal(a3, a1)
    # ascii read write test
    fname = os.path.join(out_dir, 'text.dat')
    u2d.write_txt((10, 10), fname, u2d.array)
    a4 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(a1, a4)

    # fixed format read/write with touching numbers - yuck!
    data = np.arange(100).reshape(10, 10)
    u2d_arange = Util2d(ml, (10, 10), np.float32, data, "test")
    u2d_arange.write_txt((10, 10), fname, u2d_arange.array, python_format=[7, "{0:10.4E}"])
    a4a = u2d.load_txt((10, 10), fname, np.float32, "(7E10.6)")
    assert np.array_equal(u2d_arange.array, a4a)

    # test view vs copy with .array
    a5 = u2d.array
    a5 += 1
    assert not np.array_equal(a5, u2d.array)

    # Util2d.__mul__() overload
    new_2d = u2d * 2
    assert np.array_equal(new_2d.array, u2d.array * 2)

    # test the cnstnt application
    u2d.cnstnt = 2.0
    a6 = u2d.array
    assert not np.array_equal(a1, a6)
    u2d.write_txt((10, 10), fname, u2d.array)
    a7 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(u2d.array, a7)

    return


def stress_util2d(ml, nlay, nrow, ncol):
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol)
    hk = np.ones((nlay, nrow, ncol))
    vk = np.ones((nlay, nrow, ncol)) + 1.0
    # save hk up one dir from model_ws
    fnames = []
    for i, h in enumerate(hk):
        fname = os.path.join(out_dir, "test_{0}.ref".format(i))
        fnames.append(fname)
        np.savetxt(fname, h, fmt="%15.6e", delimiter='')
        vk[i] = i + 1.

    lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vk)
    # util2d binary check
    ml.lpf.vka[0].format.binary = True

    # util3d cnstnt propogation test
    ml.lpf.vka.cnstnt = 2.0
    ml.write_input()

    # check that binary is being respect - it can't get no respect!
    vka_1 = ml.lpf.vka[0]
    a = vka_1.array
    vka_1_2 = vka_1 * 2.0
    assert np.array_equal(a * 2.0,vka_1_2.array)

    if ml.external_path is not None:
        files = os.listdir(os.path.join(ml.model_ws, ml.external_path))
    else:
        files = os.listdir(ml.model_ws)

    print("\n\nexternal files: " + ','.join(files) + '\n\n')
    ml1 = flopy.modflow.Modflow.load(ml.namefile,
                                     model_ws=ml.model_ws,
                                     verbose=True, forgive=False)
    print("testing load")
    assert ml1.load_fail == False
    # check that both binary and cnstnt are being respected through
    # out the write and load process.
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.vka.array, ml.lpf.vka.array)
    assert np.array_equal(ml1.lpf.hk.array, hk)
    assert np.array_equal(ml1.lpf.hk.array, ml.lpf.hk.array)

    print("change model_ws")
    ml.model_ws = out_dir
    ml.write_input()
    if ml.external_path is not None:
        files = os.listdir(os.path.join(ml.model_ws, ml.external_path))
    else:
        files = os.listdir(ml.model_ws)
    print("\n\nexternal files: " + ','.join(files) + '\n\n')
    ml1 = flopy.modflow.Modflow.load(ml.namefile,
                                     model_ws=ml.model_ws,
                                     verbose=True, forgive=False)
    print("testing load")
    assert ml1.load_fail == False
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.hk.array, hk)

    # more binary testing
    ml.lpf.vka[0]._array[0,0] *= 3.0
    ml.write_input()
    ml1 = flopy.modflow.Modflow.load(ml.namefile,
                                     model_ws=ml.model_ws,
                                     verbose=True, forgive=False)
    assert np.array_equal(ml.lpf.vka.array,ml1.lpf.vka.array)
    assert np.array_equal(ml.lpf.hk.array,ml1.lpf.hk.array)

def stress_util2d_for_joe_the_file_king(ml, nlay, nrow, ncol):
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol)
    hk = np.ones((nlay, nrow, ncol))
    vk = np.ones((nlay, nrow, ncol)) + 1.0
    # save hk up one dir from model_ws
    fnames = []
    for i, h in enumerate(hk):
        fname = os.path.join("test_{0}.ref".format(i))
        fnames.append(fname)
        np.savetxt(fname, h, fmt="%15.6e", delimiter='')
        vk[i] = i + 1.

    lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vk)
    ml.lpf.vka[0].format.binary = True
    ml.lpf.vka.cnstnt = 2.0
    ml.write_input()

    assert np.array_equal(ml.lpf.hk.array,hk)
    assert np.array_equal(ml.lpf.vka.array,vk * 2.0)

    ml1 = flopy.modflow.Modflow.load(ml.namefile,
                                     model_ws=ml.model_ws,
                                     verbose=True, forgive=False)
    print("testing load")
    assert ml1.load_fail == False
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.hk.array, hk)
    assert np.array_equal(ml1.lpf.vka.array, ml.lpf.vka.array)
    assert np.array_equal(ml1.lpf.hk.array, ml.lpf.hk.array)



    # more binary testing
    ml.lpf.vka[0]._array[0,0] *= 3.0
    ml.write_input()
    ml1 = flopy.modflow.Modflow.load(ml.namefile,
                                     model_ws=ml.model_ws,
                                     verbose=True, forgive=False)
    assert np.array_equal(ml.lpf.vka.array,ml1.lpf.vka.array)
    assert np.array_equal(ml.lpf.hk.array,ml1.lpf.hk.array)


def test_util2d_external_free():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ml = flopy.modflow.Modflow(model_ws=model_ws)
    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_free_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow()
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_free_path():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(model_ws=model_ws,
                               external_path=ext_path)
    stress_util2d(ml, 1, 1, 1)

    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_free_path_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    base_dir = os.getcwd()
    os.chdir(out_dir)
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(external_path=ext_path)

    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_fixed():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ml = flopy.modflow.Modflow(model_ws=model_ws)
    ml.array_free_format = False

    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_fixed_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)

    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow()
    ml.array_free_format = False
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_fixed_path():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(model_ws=model_ws,
                               external_path=ext_path)
    ml.array_free_format = False

    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_fixed_path_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)

    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow(external_path=ext_path)
    ml.array_free_format = False
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util3d():
    ml = flopy.modflow.Modflow()
    u3d = Util3d(ml, (10, 10, 10), np.float32, 10., 'test')
    a1 = u3d.array
    a2 = np.ones((10, 10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)

    new_3d = u3d * 2.0
    assert np.array_equal(new_3d.array, u3d.array * 2)

    # test the mult list-based overload for Util3d
    mult = [2.0] * 10
    mult_array = (u3d * mult).array
    assert np.array_equal(mult_array, np.zeros((10, 10, 10)) + 20.0)
    u3d.cnstnt = 2.0
    assert not np.array_equal(a1, u3d.array)

    return


def test_arrayformat():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    u2d = Util2d(ml, (15, 2), np.float32, np.ones((15, 2)), 'test')

    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.npl = 1
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.npl = 2
    u2d.format.width = 8
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.free = True
    u2d.format.width = 8
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.free = False
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.fmtin = "(10G15.6)"
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.binary = True
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()


def test_new_get_file_entry():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    u2d = Util2d(ml, (5, 2), np.float32, np.ones((5, 2)), 'test', locat=99)
    print(u2d.get_file_entry(how="internal"))
    print(u2d.get_file_entry(how="constant"))
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = False
    print(u2d.get_file_entry(how="openclose"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="openclose"))

    ml.array_free_format = False
    u2d = Util2d(ml, (5, 2), np.float32, np.ones((5, 2)), 'test', locat=99)
    print(u2d.get_file_entry(how="internal"))
    print(u2d.get_file_entry(how="constant"))
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="external"))


def test_mflist():

    ml = flopy.modflow.Modflow(model_ws="temp")
    dis = flopy.modflow.ModflowDis(ml,10,10,10,10)
    sp_data = {0: [[1, 1, 1, 1.0], [1, 1, 2, 2.0], [1, 1, 3, 3.0]],1:[1,2,4,4.0]}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data)
    m4ds = ml.wel.stress_period_data.masked_4D_arrays

    sp_data = flopy.utils.MfList.masked4D_arrays_to_stress_period_data\
        (flopy.modflow.ModflowWel.get_default_dtype(),m4ds)
    assert np.array_equal(sp_data[0],ml.wel.stress_period_data[0])
    assert np.array_equal(sp_data[1],ml.wel.stress_period_data[1])
    # the last entry in sp_data (kper==9) should equal the last entry
    # with actual data in the well file (kper===1)
    assert np.array_equal(sp_data[9],ml.wel.stress_period_data[1])

    pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    ml = flopy.modflow.Modflow.load(os.path.join(pth,"swi2ex4sww.nam"),
                                    verbose=True)
    m4ds = ml.wel.stress_period_data.masked_4D_arrays

    sp_data = flopy.utils.MfList.masked4D_arrays_to_stress_period_data\
        (flopy.modflow.ModflowWel.get_default_dtype(),m4ds)

    # make a new wel file
    wel = flopy.modflow.ModflowWel(ml,stress_period_data=sp_data)
    flx1 = m4ds["flux"]
    flx2 = wel.stress_period_data.masked_4D_arrays["flux"]

    flx1 = np.nan_to_num(flx1)
    flx2 = np.nan_to_num(flx2)

    assert flx1.sum() == flx2.sum()


def test_how():
    import numpy as np
    import flopy
    ml = flopy.modflow.Modflow(model_ws="temp")
    ml.array_free_format = False
    dis = flopy.modflow.ModflowDis(ml,nlay=2,nrow=10,ncol=10)

    arr = np.ones((ml.nrow,ml.ncol))
    u2d = flopy.utils.Util2d(ml,arr.shape,np.float32,arr,"test",locat=1)
    print(u2d.get_file_entry())
    u2d.how = "constant"
    print(u2d.get_file_entry())
    u2d.fmtin = "(binary)"
    print(u2d.get_file_entry())


def test_util3d_reset():
    import numpy as np
    import flopy
    ml = flopy.modflow.Modflow(model_ws="temp")
    ml.array_free_format = False
    dis = flopy.modflow.ModflowDis(ml,nlay=2,nrow=10,ncol=10)
    bas = flopy.modflow.ModflowBas(ml,strt=999)
    arr = np.ones((ml.nlay,ml.nrow,ml.ncol))
    ml.bas6.strt = arr

if __name__ == '__main__':
    test_util3d_reset()
    # test_mflist()
    # test_new_get_file_entry()
    # test_arrayformat()
    # test_util2d_external_free_nomodelws()
    # test_util2d_external_free_path_nomodelws()
    # test_util2d_external_free()
    # test_util2d_external_free_path()
    # test_util2d_external_fixed()
    # test_util2d_external_fixed_path()
    # test_util2d_external_fixed_nomodelws()
    # test_util2d_external_fixed_path_nomodelws()
    #test_transient2d()
    # test_util2d()
    # test_util3d()
    # test_how()
