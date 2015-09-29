from __future__ import print_function
import numpy as np
import params
import tplarray


def zonearray2params(mfpackage, partype, parzones, lbound, ubound,
                     parvals, transform, zonearray):
    """
    Helper function to create a list of flopy parameters from a zone array
    and list of parameter zone numbers.

    The parameter name is set equal to the parameter type and the parameter
    zone value, separated by an underscore.
    """
    plist = []
    for i, iz in enumerate(parzones):
        idx = np.where(zonearray == iz)
        parname = partype + '_' + str(iz)
        startvalue = parvals[i]
        p = params.Params(mfpackage, partype, parname, startvalue, lbound,
                          ubound, idx, transform)
        plist.append(p)
    return plist


class TemplateWriter(object):
    """
    Class for writing PEST template files.
    """
    def __init__(self, model, plist):
        self.model = model
        self.plist = plist
        return

    def write_template(self):
        """
        Write the template files for all model files that have arrays that
        have been parameterized.
        """

        # Import and initialize
        import copy
        pakdict = {}

        # Create a copy of any flopy model packages that have arrays that have
        # been parameterized.
        for p in self.plist:
            pakdictkey = p.mfpackage.upper()
            if pakdictkey not in pakdict:
                paktpl = copy.copy(self.model.get_package(pakdictkey))
                pakdict[pakdictkey] = paktpl

        # Go through each copy of parameterized packages, replace parameterized
        # arrays with template arrays (Util3dTpl) and then replace the string
        # array values with the parameter name.
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
    parzones = [2, 3, 4]
    parvals = [56.777, 78.999, 99.]
    lbound = 5
    ubound = 500
    transform = 'log'
    zonearray = np.ones((3, 20, 14), dtype=int)
    zonearray[0, 10:, 7:] = 2
    zonearray[0, 15:, 9:] = 3
    zonearray[1] = 4
    print (zonearray)
    plisthk = zonearray2params(mfpackage, 'hk', parzones, lbound, ubound,
                                 parvals, transform, zonearray)


    parzones = [1, 2]
    parvals = [0.001, 0.0005]
    zonearray = np.ones((3, 20, 14), dtype=int)
    zonearray[1] = 2
    plistvk = zonearray2params(mfpackage, 'vka', parzones, lbound, ubound,
                                 parvals, transform, zonearray)

    paramlist = plisthk + plistvk

    m = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(m, nlay=3, nrow=20, ncol=14)
    lpf = flopy.modflow.ModflowLpf(m, hk=10.)

    tw = TemplateWriter(m, paramlist).write_template()
