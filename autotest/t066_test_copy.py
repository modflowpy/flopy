"""Test copying of flopy objects.
"""
import sys
import os
import copy
import inspect
import numpy as np
import flopy
fm = flopy.modflow
mf6 = flopy.mf6
from flopy.datbase import DataType, DataInterface
from flopy.mbase import ModelInterface
from flopy.utils import TemporalReference


def get_package_list(model):
    if model.version == 'mf6':
        packages = [p.name[0].upper() for p in model.packagelist]
    else:
        packages = model.get_package_list()
    return packages


def model_is_copy(m1, m2):
    """Test that m2 is a copy of m1 by
    checking for different identities in their attributes
    and equality between their attributes.
    """
    if m1 is m2:
        return False
    m1packages = get_package_list(m1)
    m2packages = get_package_list(m2)
    if m1packages is m2packages:
        return False
    if m2.modelgrid != m1.modelgrid:
        if not package_is_copy(m1.modelgrid, m2.modelgrid):
            return False
    for k, v in m1.__dict__.items():
        v2 = m2.__dict__[k]
        if v2 is v and type(v) not in [bool, str, type(None), float, int]:
            # some mf6 objects aren't copied with deepcopy
            if isinstance(v, mf6.mfsimulation.MFSimulationData):
                continue
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    item2 = v2[i]
                    if item is item2:
                        return False
            else:
                return False
        if k in [
                 '_packagelist',
                 '_package_paths',
                 'package_key_dict',
                 'package_type_dict',
                 'package_name_dict',
                 '_ftype_num_dict']:
            continue
        elif k not in m2.__dict__:
            return False
        elif type(v) == bool:
            if not v == v2:
                return False
        elif type(v) in [str, int, float, dict, list]:
            if v != v2:
                return False
            continue
    for pk in m1packages:
        if not package_is_copy(getattr(m1, pk), getattr(m2, pk)):
            return False
    return True


def package_is_copy(pk1, pk2):
    """Test that pk2 is a copy of pk1 by
    checking for different identities in their attributes
    and equality between their attributes.
    """
    for k, v in pk1.__dict__.items():
        v2 = pk2.__dict__[k]
        if v2 is v and type(v) not in [bool, str, type(None),
                                       float, int, tuple
                                       ]:
            # Deep copy doesn't work for ModflowUtltas
            if not inspect.isclass(v):
                return False
        if k in ['_child_package_groups',
                 '_data_list',
                 '_packagelist',
                 '_simulation_data',
                 'blocks',
                 'dimensions',
                 'package_key_dict',
                 'package_name_dict',
                 'package_type_dict',
                 'post_block_comments',
                 'simulation_data',
                 'structure'
                 ]:
            continue
        elif isinstance(v, mf6.mfpackage.MFPackage):
            continue
        elif isinstance(v, mf6.mfpackage.MFChildPackages):
            if not package_is_copy(v, v2):
                return False
        elif k not in pk2.__dict__:
            return False
        elif type(v) == bool:
            if not v == v2:
                return False
        elif type(v) in [str, int, float, dict, list]:
            if v != v2:
                return False
        elif isinstance(v, ModelInterface):
            # weak, but calling model_eq would result in recursion
            if v.__repr__() != v2.__repr__():
                return False
        elif isinstance(v, DataInterface):
            if v != v2:
                if v.data_type == DataType.transientlist or \
                        v.data_type == DataType.list:
                    if not list_is_copy(v, v2):
                        return False
                else:
                    a1, a2 = v.array, v2.array
                    if a2 is a1 and type(a1) not in [bool, str, type(None), float, int, tuple]:
                        return False
                    if a1 is None and a2 is None:
                        continue
                    if not isinstance(a1, np.ndarray):
                        if a1 != a2:
                            return False
                    # TODO: this may return False if there are nans
                    elif not np.allclose(v.array, v2.array):
                        return False
        elif isinstance(v, TemporalReference):
            pass
        elif isinstance(v, np.ndarray):
            if not np.allclose(v, v2):
                return False
        elif v != v2:
            return False
    return True


def list_is_copy(mflist1, mflist2):
    """Test that mflist2 is a copy of mflist1 by
    checking that their arrays have different identities
    but are equal or close.
    """
    if mflist2 is mflist1:
        return False
    if isinstance(mflist1, mf6.data.mfdatalist.MFTransientList):
        data1 = {per: ra for per, ra in enumerate(mflist1.array)}
        data2 = {per: ra for per, ra in enumerate(mflist2.array)}
    elif isinstance(mflist1, mf6.data.mfdatalist.MFList):
        data1 = {0: mflist1.array}
        data2 = {0: mflist2.array}
    elif hasattr(mflist1, 'data'):
        data1 = mflist1.data
        data2 = mflist2.data
    for k, v in data1.items():
        if k not in data2:
            return False
        v2 = data2[k]
        if v2 is v and type(v) not in [bool, str, type(None), float, int, tuple]:
            return False
        if v is None and v2 is None:
            continue
        elif not isinstance(v, np.recarray):
            if v != v2:
                return False
        else:
            # compare the two np.recarrays
            # not sure if this will work for all relevant cases
            for c, dtype in v.dtype.fields.items():
                c1 = v[c].copy()
                c2 = v2[c].copy()
                if np.issubdtype(dtype[0].type, np.floating):
                    c1[np.isnan(c1)] = 0
                    c2[np.isnan(c2)] = 0
                if not np.array_equal(c1, c2):
                    return False
    return True


def test_mf2005_copy():
    if sys.version_info[0] < 3:
        return
    path = '../examples/data/freyberg_multilayer_transient/freyberg.nam'
    model_ws, namefile = os.path.split(path)
    m = fm.Modflow.load(namefile, model_ws=model_ws)
    m_c = copy.copy(m)
    m_dc = copy.deepcopy(m)
    assert model_is_copy(m, m_dc)
    assert not model_is_copy(m, m_c)


def test_mf6_copy():
    if sys.version_info[0] < 3:
        return
    sim_ws = '../examples/data/mf6/test045_lake2tr'
    sim = mf6.MFSimulation.load('mfsim.nam', 'mf6', sim_ws=sim_ws)
    m = sim.get_model('lakeex2a')
    m_c = copy.copy(m)
    m_dc = copy.deepcopy(m)
    assert model_is_copy(m, m_dc)
    assert not model_is_copy(m, m_c)
