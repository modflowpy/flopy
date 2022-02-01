"""
Try to load all of the MODFLOW examples in ../examples/data/mf2005_test.
These are the examples that are distributed with MODFLOW-2005.
"""

import os

import pytest
from ci_framework import FlopyTestSetup, base_test_dir

import flopy

pth = os.path.join("..", "examples", "data", "mf2005_test")
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith(".nam")]

ppth = os.path.join("..", "examples", "data", "parameters")
pnamfiles = [
    "Oahu_02.nam",
]

test_nwt_pth = os.path.join("..", "examples", "data", "nwt_test")
nwt_pak_files = [
    os.path.join(test_nwt_pth, f)
    for f in os.listdir(test_nwt_pth)
    if f.endswith(".nwt")
]

nwt_nam = [
    os.path.join(test_nwt_pth, f)
    for f in os.listdir(test_nwt_pth)
    if f.endswith(".nam")
]


def load_model(namfile):
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=pth, version="mf2005", verbose=True
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False


def load_parameter_model(namfile):
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=ppth, version="mf2005", verbose=True
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False


def load_only_bas6_model(namfile):
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=pth,
        version="mf2005",
        verbose=True,
        load_only=["bas6"],
        check=False,
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False


@pytest.mark.parametrize(
    "namfile",
    list(namfiles),
)
def test_modflow_load(namfile):
    load_model(namfile)
    return


@pytest.mark.parametrize(
    "namfile",
    pnamfiles,
)
def test_parameter_load(namfile):
    load_parameter_model(namfile)
    return


@pytest.mark.parametrize(
    "namfile",
    namfiles,
)
def test_modflow_loadonly(namfile):
    load_only_bas6_model(namfile)
    return


@pytest.mark.parametrize(
    "fname",
    nwt_pak_files,
)
def test_nwt_pack_load(fname):
    load_nwt_pack(fname)


@pytest.mark.parametrize(
    "namfile",
    nwt_nam,
)
def test_nwt_model_load(namfile):
    load_nwt_model(namfile)


def load_nwt_pack(nwtfile):
    new_ws = (
        base_test_dir(__file__, rel_path="temp", verbose=True)
        + "_"
        + os.path.basename(nwtfile).replace(".nam", "_nwtpack")
    )
    test_setup = FlopyTestSetup(test_dirs=new_ws, verbose=True)

    ws = os.path.dirname(nwtfile)
    ml = flopy.modflow.Modflow(model_ws=ws, version="mfnwt")
    if "fmt." in nwtfile.lower():
        ml.array_free_format = False
    else:
        ml.array_free_format = True

    nwt = flopy.modflow.ModflowNwt.load(nwtfile, ml)
    msg = f"{os.path.basename(nwtfile)} load unsuccessful"
    assert isinstance(nwt, flopy.modflow.ModflowNwt), msg

    # write the new file in the working directory
    ml.change_model_ws(new_ws)
    nwt.write_file()

    fn = os.path.join(new_ws, ml.name + ".nwt")
    msg = f"{os.path.basename(nwtfile)} write unsuccessful"
    assert os.path.isfile(fn), msg

    ml2 = flopy.modflow.Modflow(model_ws=new_ws, version="mfnwt")
    nwt2 = flopy.modflow.ModflowNwt.load(fn, ml2)
    lst = [
        a
        for a in dir(nwt)
        if not a.startswith("__") and not callable(getattr(nwt, a))
    ]
    for l in lst:
        msg = (
            "{} data instantiated from {} load  is not the same as written to "
            "{}".format(l, nwtfile, os.path.basename(fn))
        )
        assert nwt2[l] == nwt[l], msg


def load_nwt_model(nfile):
    new_ws = (
        base_test_dir(__file__, rel_path="temp", verbose=True)
        + "_"
        + os.path.basename(nfile).replace(".nam", "")
    )
    test_setup = FlopyTestSetup(test_dirs=new_ws, verbose=True)

    f = os.path.basename(nfile)
    model_ws = os.path.dirname(nfile)
    ml = flopy.modflow.Modflow.load(f, model_ws=model_ws)
    msg = "Error: flopy model instance was not created"
    assert isinstance(ml, flopy.modflow.Modflow), msg

    # change the model work space and rewrite the files
    ml.change_model_ws(new_ws)
    ml.write_input()

    # reload the model that was just written
    ml2 = flopy.modflow.Modflow.load(f, model_ws=new_ws)

    # check that the data are the same
    for pn in ml.get_package_list():
        p = ml.get_package(pn)
        p2 = ml2.get_package(pn)
        lst = [
            a
            for a in dir(p)
            if not a.startswith("__") and not callable(getattr(p, a))
        ]
        for l in lst:
            msg = (
                "{}.{} data instantiated from {} load  is not the same as "
                "written to {}".format(pn, l, model_ws, new_ws)
            )
            assert p[l] == p2[l], msg


if __name__ == "__main__":
    for namfile in namfiles:
        load_model(namfile)
    for namfile in namfiles:
        load_only_bas6_model(namfile)
    for fnwt in nwt_nam:
        load_nwt_model(fnwt)
    for fnwt in nwt_pak_files:
        load_nwt_pack(fnwt)
