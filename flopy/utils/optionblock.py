from collections import OrderedDict
import numpy as np
from ..utils import flopy_io


class OptionBlock:
    """
    Parent class to for option blocks within
    Modflow-nwt models. This class contains base
    information and routines that can be shared throughout
    all option block classes.

    Parameters
    ----------
    options_line : str
        single line based options string
    package : flopy.pakbase.Package instance
        valid packages include ModflowWel, ModflowSfr2, ModflowUzf1
    block : bool
        flag to write as single line or block type

    """

    nested = "nested"
    dtype = "dtype"
    n_nested = "nvars"
    vars = "vars"
    optional = "optional"

    simple_flag = OrderedDict(
        [(dtype, np.bool_), (nested, False), (optional, False)]
    )
    simple_str = OrderedDict(
        [(dtype, str), (nested, False), (optional, False)]
    )
    simple_float = OrderedDict(
        [(dtype, float), (nested, False), (optional, False)]
    )
    simple_int = OrderedDict(
        [(dtype, int), (nested, False), (optional, False)]
    )

    simple_tabfile = OrderedDict(
        [
            (dtype, np.bool_),
            (nested, True),
            (n_nested, 2),
            (
                vars,
                OrderedDict([("numtab", simple_int), ("maxval", simple_int)]),
            ),
        ]
    )

    def __init__(self, options_line, package, block=True):
        self._context = package._options
        self._attr_types = {}
        self.options_line = options_line
        self.package = package
        self.auxillary = []
        self.noprint = False
        self.block = block

        self.__build_attr_types()
        self._set_attributes()

    @property
    def single_line_options(self):
        """
        Method to get the single line representation of the
        Options Block

        Returns
        -------
        t : (str) single line representation of Options
        """
        t = repr(self).split("\n")
        t = t[1:-2]
        return " ".join(t)

    def update_from_package(self, pak):
        """
        Updater method to check the package and update
        OptionBlock attribute values based on package
        values.

        Parameters
        ----------
        pak : flopy.package
            valid packages include ModflowWel, ModflowSfr2,
            and ModflowUzf1 instances
        """
        for key, ctx in self._context.items():
            if key in pak.__dict__:
                val = pak.__dict__[key]
                self.__setattr__(key, val)
                if ctx[OptionBlock.nested]:
                    for k2, ctx2 in ctx[OptionBlock.vars].items():
                        if k2 in pak.__dict__:
                            v2 = pak.__dict__[k2]
                            self.__setattr__(k2, v2)

    def __repr__(self):
        """
        Syntactic sugar that creates a dynamic representation
        of the OptionsBlock. Makes it very easy to write to file
        """
        s = "OPTIONS\n"
        for key, ctx in self._context.items():
            try:
                val = []
                if ctx[OptionBlock.dtype] == np.bool_:
                    if not object.__getattribute__(self, key):
                        continue
                    else:
                        val.append(str(key))
                else:
                    val.append(str(object.__getattribute__(self, key)))

                if ctx[OptionBlock.nested]:
                    for k, d in ctx[OptionBlock.vars].items():
                        if d[OptionBlock.dtype] == np.bool_:
                            if not object.__getattribute__(self, k):
                                pass
                            else:
                                val.append(str(k))
                        else:
                            v = str(object.__getattribute__(self, k))
                            if v == "None" and d[OptionBlock.optional]:
                                pass
                            else:
                                val.append(
                                    str((object.__getattribute__(self, k)))
                                )

                if "None" in val:
                    pass
                else:
                    s += " ".join(val)
                    s += "\n"
            except:
                pass

        s += "END\n"
        return s.upper()

    def __setattr__(self, key, value):
        """
        Syntactic sugar to allow for dynamic recarray/attribute
        interactions and data type enforcement on dynamic attributes

        Parameters
        ----------
            key : str
                string refering to an attribute
            value : object
                a python object (int, str, float, bool) that
                is consistant with the attribute data type

        """
        err_msg = "Data type must be compatible with {}"
        if key in ("_context", "_attr_types", "options_line"):
            self.__dict__[key] = value

        elif value is None:
            super().__setattr__(key, value)

        elif isinstance(value, np.recarray):
            for name in value.dtype.names:
                if self._attr_types[name] == np.bool_:
                    if not isinstance(value, (bool, np.bool_)):
                        raise TypeError(err_msg.format(self._attr_types[name]))
                else:
                    try:
                        value = self._attr_types[name](value)
                    except ValueError:
                        raise TypeError(err_msg.format(self._attr_types[name]))

                self.__dict__[name] = value[name][0]

        elif key in self._attr_types:
            if self._attr_types[key] == np.bool_:
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(err_msg.format(self._attr_types[key]))
            else:
                try:
                    value = self._attr_types[key](value)
                except ValueError:
                    raise TypeError(err_msg.format(self._attr_types[key]))

            self.__dict__[key] = value

        else:
            super().__setattr__(key, value)

    def __getattribute__(self, item):
        """
        Syntactic sugar that creates recarrays of nested/related items.
        Ex. Tabfiles, numtab, and maxval will be returned in a
        recarray if the user calls <.tabfiles>

        """
        if item in ("__dict__", "_context", "package"):
            value = object.__getattribute__(self, item)

        elif item in object.__getattribute__(self, "_context"):
            ctx = object.__getattribute__(self, "_context")[item]
            if ctx[OptionBlock.nested]:
                vals = [object.__getattribute__(self, item)]
                dtypes = [(item, ctx[OptionBlock.dtype])]
                for key, d in ctx[OptionBlock.vars].items():
                    vals.append(object.__getattribute__(self, key))
                    dtypes.append((key, d[OptionBlock.dtype]))

                if not vals[0]:
                    value = False
                elif None in vals:
                    value = vals[0]
                else:
                    value = np.recarray((1,), dtype=dtypes)
                    value[0] = tuple(vals)

            else:
                value = object.__getattribute__(self, item)
        else:
            value = object.__getattribute__(self, item)

        return value

    def __build_attr_types(self):
        """
        Method to build a type dictionary for type
        enforcements in __setattr__. This uses the package's
        contex tree to build and enforce attribute
        types for the class

        """
        for key, value in self._context.items():
            self._attr_types[key] = value[OptionBlock.dtype]
            if OptionBlock.vars in value:
                for k, d in value[OptionBlock.vars].items():
                    self._attr_types[k] = d[OptionBlock.dtype]

    def _set_attributes(self):
        """
        Dynamic attribute creation method. This uses the
        package's context tree to build attributes for the class

        """
        # set up all attributes for the class!
        for key, ctx in self._context.items():
            if ctx[OptionBlock.dtype] in (np.bool_, bool):
                self.__setattr__(key, False)
            else:
                self.__setattr__(key, None)

            if ctx[OptionBlock.nested]:
                for k, d in ctx[OptionBlock.vars].items():
                    if d[OptionBlock.dtype] in (np.bool_, bool):
                        self.__setattr__(k, False)
                    else:
                        self.__setattr__(k, None)

        t = self.options_line.split()
        nested = False
        ix = 0
        while ix < len(t):
            if not nested:
                if t[ix] in self._context:
                    key = t[ix]
                    ctx = self._context[key]
                    dtype = ctx[OptionBlock.dtype]
                    nested = ctx[OptionBlock.nested]

                    OptionUtil.isvalid(dtype, t[ix])

                    if dtype == np.bool_:
                        self.__setattr__(key, True)
                    else:
                        self.__setattr__(key, dtype(t[ix]))

                    ix += 1

                else:
                    err_msg = "Option: {} not a valid option".format(t[ix])
                    raise KeyError(err_msg)

            else:
                ctx = self._context[t[ix - 1]]
                for key, d in ctx[OptionBlock.vars].items():
                    dtype = d[OptionBlock.dtype]

                    if d[OptionBlock.optional]:
                        if ix >= len(t):
                            continue
                        else:
                            try:
                                OptionUtil.isvalid(dtype, t[ix])
                            except TypeError:
                                continue

                    OptionUtil.isvalid(dtype, t[ix])

                    if dtype == np.bool_:
                        self.__setattr__(key, True)
                    else:
                        self.__setattr__(key, dtype(t[ix]))

                    ix += 1

                nested = False

    def write_options(self, f):
        """
        Method to write the options block or options line to
        an open file object.

        Parameters
            f : file, str
                open file object, or path to file
        """
        if isinstance(f, str):
            with open(f, "w") as optfile:
                if self.block:
                    optfile.write(repr(self))
                else:
                    optfile.write(self.single_line_options)
                    optfile.write("\n")
        else:
            if self.block:
                f.write(repr(self))
            else:
                f.write(self.single_line_options)
                f.write("\n")

    @classmethod
    def load_options(cls, options, package):
        """
        Loader for the options class. Reads in an options
        block and uses context from option util dictionaries
        to check the validity of the data

        Parameters
        ----------
            options: str or file
                string path to a file or file object
            package : flopy.package type
                valid packages include flopy.modflow.ModflowWel,
                flopy.modflow.ModflowUzf1, flopy.modflow.ModflowSfr2

        Returns
        -------
            OptionBlock object

        """
        context = package._options

        openfile = not hasattr(options, "read")
        if openfile:
            try:
                options = open(options, "r")
            except IOError:
                err_msg = (
                    "Unrecognized type for options"
                    " variable: {}".format(type(options))
                )
                raise TypeError(err_msg)

        option_line = ""
        while True:
            line = flopy_io.line_strip(options.readline())
            line = line.lower()
            if not line:
                continue

            if line.split()[0] == "options":
                pass

            elif line.split()[0] != "end":
                t = line.split()
                if t[0] in context:
                    key = t[0]
                    option_line += key + " "
                    ctx = context[key]

                    if ctx[OptionBlock.nested]:
                        ix = 1

                        for k, d in ctx[OptionBlock.vars].items():
                            if ix >= len(t) and d[OptionBlock.optional]:
                                continue

                            if d[OptionBlock.dtype] == float:
                                valid = OptionUtil.isfloat(t[ix])
                            elif d[OptionBlock.dtype] == int:
                                valid = OptionUtil.isint(t[ix])
                            else:
                                valid = True

                            if not valid:
                                err_msg = (
                                    "Invalid type set to variable "
                                    "{} in option block".format(k)
                                )
                                raise TypeError(err_msg)

                            option_line += t[ix] + " "
                            ix += 1

            else:
                if openfile:
                    options.close()
                return cls(options_line=option_line, package=package)


class OptionUtil:
    @staticmethod
    def isfloat(s):
        """
        Simple method to check that a string is a valid
        floating point variable

        Parameters
        ----------
        s : str

        Returns
        -------
            bool

        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def isint(s):
        """
        Simple data check method to check that a string
        is a valid integer

        Parameters
        ----------
        s : str

        Returns
        -------
            bool

        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def isvalid(dtype, val):
        """
        Check to see if a dtype is valid before setting
        as an attribute

        Parameters
        ----------
        dtype : type
            int, float, str, bool, etc...
        val : string

        Returns
        -------
            bool

        """
        valid = False
        if dtype == np.bool_:
            valid = True
        elif dtype == str:
            valid = True
        else:
            # check if valid
            if dtype == int:
                valid = OptionUtil.isint(val)
            elif dtype == float:
                valid = OptionUtil.isfloat(val)
            else:
                pass

        if not valid:
            err_msg = (
                "Invalid type set to variable "
                "{} in option block".format(val)
            )
            raise TypeError(err_msg)

        return valid
