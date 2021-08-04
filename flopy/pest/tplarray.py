from __future__ import print_function
import numpy as np
from ..utils.util_array import Util3d as Util3d
from ..utils.util_array import Transient2d as Transient2d


def get_template_array(pakarray):
    """
    Convert the package array into the appropriate template array

    """
    tpla = pakarray
    if isinstance(pakarray, Util3d):
        tpla = Util3dTpl(pakarray)
    elif isinstance(pakarray, Transient2d):
        tpla = Transient2dTpl(pakarray)
    return tpla


class Transient2dTpl:
    def __init__(self, transient2d):
        self.transient2d = transient2d
        self.params = {}
        self.multipliers = {}
        return

    def add_parameter(self, p):
        """
        Store the parameters in a list for later substitution

        """
        # Verify parameter span contents
        if "kpers" not in p.span:
            raise Exception(
                "Parameter {} span does not contain kper.".format(p.name)
            )
        if "idx" not in p.span:
            raise Exception(
                "Parameter {} span does not contain idx.".format(p.name)
            )

        if p.span["idx"] is None:
            # Multiplier parameter is when p.span['idx'] is None
            for kper in p.span["kpers"]:
                self.multipliers[kper] = "~ {0:^13s} ~".format(p.name)
        else:
            # Index parameter otherwise
            for kper in p.span["kpers"]:
                if kper not in self.params:
                    self.params[kper] = []
                self.params[kper].append(p)
        return

    def get_kper_entry(self, kper):

        # Set defaults
        parameterized = False
        multiplier = None
        indexed_param = False

        # Check to see if a multiplier applies
        if kper in self.multipliers:
            multiplier = self.multipliers[kper]
            parameterized = True

        # Check to see if there are any array index parameters.
        if kper in self.params:
            parameterized = True
            indexed_param = True

        # If parameterized return the parameter array, otherwise return
        # regular transient2d array
        if parameterized:
            u2d = self.transient2d[kper]
            chararray = np.array(u2d.array, dtype="str")
            if kper in self.params:
                for p in self.params[kper]:
                    idx = p.span["idx"]
                    chararray[idx] = "~{0:^13s}~".format(p.name)
            u2dtpl = Util2dTpl(chararray, u2d.name, multiplier, indexed_param)
            return (1, u2dtpl.get_file_entry())
        else:
            return self.transient2d.get_kper_entry(kper)


class Util3dTpl:
    """
    Class to define a three-dimensional template array for use with parameter
    estimation.

    Parameters
    ----------
    u3d : Util3d object

    """

    def __init__(self, u3d):
        self.u3d = u3d
        self.chararray = np.array(u3d.array, dtype="str")
        self.multipliers = {}
        self.indexed_params = False
        if self.chararray.ndim == 3:
            # Then multi layer array, so set all multipliers to None
            for k in range(self.chararray.shape[0]):
                self.multipliers[k] = None
        return

    def __getitem__(self, k):
        return Util2dTpl(
            self.chararray[k],
            self.u3d.name_base[k] + str(k + 1),
            self.multipliers[k],
            self.indexed_params,
        )

    def add_parameter(self, p):
        """
        Fill the chararray with the parameter name.

        Parameters
        ----------
        p : flopy.pest.params.Params
            Parameter.  Must have .idx and .name attributes

        """

        if "layers" in p.span and "idx" in p.span:
            if p.span["idx"] is not None:
                raise Exception(
                    "For a Util3d object, cannot have layers and "
                    "idx in parameter.span"
                )

        if "layers" in p.span:
            for l in p.span["layers"]:
                self.multipliers[l] = "~ {0:^13s} ~".format(p.name)

        if "idx" in p.span and p.span["idx"] is not None:
            idx = p.span["idx"]
            self.chararray[idx] = "~{0:^13s}~".format(p.name)
            self.indexed_params = True

        return


class Util2dTpl:
    """
    Class to define a two-dimensional template array for use with parameter
    estimation.

    Parameters
    ----------
    chararray : A Numpy ndarray of dtype 'str'.
    name : The parameter type.  This will be written to the control record
        as a comment.
    indexed_param : bool
        A flag to indicated whether or not the array contains parameter names
        within the array itself.

    """

    def __init__(self, chararray, name, multiplier, indexed_param):
        self.chararray = chararray
        self.name = name
        self.multiplier = multiplier
        self.indexed_param = indexed_param
        return

    def get_file_entry(self):
        """
        Convert the array into a string.

        Returns
        -------
        file_entry : str

        """
        ncol = self.chararray.shape[-1]
        au = np.unique(self.chararray)
        if au.shape[0] == 1 and self.multiplier is None:
            file_entry = "CONSTANT {0}    #{1}\n".format(au[0], self.name)
        else:
            mult = 1.0
            if self.multiplier is not None:
                mult = self.multiplier
            cr = "INTERNAL {0} (FREE) -1      #{1}\n".format(mult, self.name)
            astring = ""
            icount = 0
            for i in range(self.chararray.shape[0]):
                for j in range(self.chararray.shape[1]):
                    icount += 1
                    astring += " {0:>15s}".format(self.chararray[i, j])
                    if icount == 10 or j == ncol - 1:
                        astring += "\n"
                        icount = 0
            file_entry = cr + astring
        return file_entry
