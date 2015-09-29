from __future__ import print_function
import numpy as np
import params
import tplarray


def Refarray2Params(mfpackage, partype, parname, parzone,
                 startvalue, lbound, ubound, transform, refarr):
    """

    """

    idx = np.where(refarr == parzone)

    curr_params = params.Params(mfpackage, partype, parname,
                 startvalue, lbound, ubound, idx, transform)

    return curr_params


class Parameters(object):
    def __init__(self, model, plist):
        self.model = model
        self.plist = plist
        return

    def write_template1(self):
        import copy
        lpftpl = copy.copy(self.model.get_package('LPF'))
        tpla = tplarray.Util3dTpl(lpftpl.hk.array)
        lpftpl.hk = tpla
        for p in self.plist:
            lpftpl.hk.chararray[p.idx] = '~{0:^13s}~'.format(p.name)
        lpftpl.heading = 'ptf ~\n' + lpftpl.heading
        lpftpl.fn_path += '.tpl'
        lpftpl.write_file()
        return

    def write_template(self):
        import copy

        pakdict = {}

        # Store the package with the template copy of the package
        for p in self.plist:
            pakdictkey = p.mfpackage.upper()
            if pakdictkey not in pakdict:
                paktpl = copy.copy(self.model.get_package(pakdictkey))
                pakdict[pakdictkey] = paktpl

        # Store parameter type with a pointer to the template array object
        for p in self.plist:
            paktpl = pakdict[p.mfpackage.upper()]
            if not hasattr(paktpl, p.type.lower()):
                msg = 'Parameter type {} not found in package.'.format(p.type.lower())
                raise Exception(msg)
            pakarray = getattr(paktpl, p.type.lower())
            if not isinstance(pakarray, tplarray.Util3dTpl):
                tpla = tplarray.Util3dTpl(pakarray.array)
                setattr(paktpl, p.type.lower(), tpla)

            # Fill the template array with the string name
            tpla = getattr(paktpl, p.type.lower())
            tpla.chararray[p.idx] = '~{0:^13s}~'.format(p.name)

        # Go through each package template and write the template file
        for pakdictkey, paktpl in pakdict.items():
            paktpl.heading = 'ptf ~\n' + paktpl.heading
            paktpl.fn_path += '.tpl'
            paktpl.write_file()

        return


if __name__ == '__main__':
    import flopy
    mfpackage = 'lpf'
    partype = 'hk'
    parname = 'hk2'
    parzone = 2
    startvalue = 120
    lbound = 5
    ubound = 500
    transform = 'log'
    refarr = np.ones((2, 20, 14), dtype=int)
    refarr[0,1:3,3:5] = 2
    refarr[1,2:4,3:5] = 2

    print (refarr)


    par1 = Refarray2Params(mfpackage, partype, parname, parzone, startvalue, lbound, ubound, transform, refarr)

    m = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(m, nlay=2, nrow=20, ncol=14)
    lpf = flopy.modflow.ModflowLpf(m, hk=10.)

    p = Parameters(m, [par1])
    p.write_template()
