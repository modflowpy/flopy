import os

from ...mbase import ModelInterface
from ...pakbase import PackageInterface
from ...utils import (
    CellBudgetFile,
    HeadFile,
    Mf6ListBudget,
    Mf6Obs,
    ZoneBudget6,
    ZoneFile6,
)
from ...utils.observationfile import CsvFile


class MF6Output:

    """
    A class that uses meta programming to get output

    Parameters
    ----------
    obj : PackageInterface object

    """

    def __init__(self, obj):
        from ..modflow import ModflowGwfoc, ModflowGwtoc, ModflowUtlobs

        # set initial observation definitions
        methods = {
            "budget": self.__budget,
            "budgetcsv": self.__budgetcsv,
            "zonebudget": self.__zonebudget,
            "obs": self.__obs,
            "csv": self.__csv,
            "package_convergence": self.__csv,
        }
        delist = ("ts", "wc")
        self._obj = obj
        self._methods = []
        self._sim_ws = obj.simulation_data.mfpath.get_sim_path()
        self.__budgetcsv = False

        if not isinstance(obj, (PackageInterface, ModelInterface)):
            raise TypeError("Only mf6 PackageInterface types can be used")

        # capture the list file for Models and for OC packages
        if isinstance(obj, (ModelInterface, ModflowGwfoc, ModflowGwtoc)):
            if isinstance(obj, ModelInterface):
                self._model = obj
            else:
                self._model = obj.model_or_sim
            self._mtype = self._model.model_type
            nam_file = self._model.model_nam_file[:-4]
            self._lst = (
                self._model.name_file.blocks["options"].datasets["list"].array
            )
            if self._lst is None:
                self._lst = f"{nam_file}.lst"
            setattr(self, "list", self.__list)
            self._methods.append("list()")
            if isinstance(obj, ModelInterface):
                return

        else:
            if obj.model_or_sim.type == "Model":
                self._model = obj.model_or_sim
            else:
                self._model = None

        obspkg = False
        if isinstance(obj, ModflowUtlobs):
            # this is a package
            obspkg = True
            rectype = "obs"

        layerfiles = {}
        if not obspkg:
            # skim through the dfn file
            try:
                datasets = obj.blocks["options"].datasets
            except KeyError:
                return

            for key, value in datasets.items():
                if "_filerecord" in key:
                    tmp = key.split("_")
                    if tmp[0] in methods:
                        rectype = tmp[0]
                    else:
                        rectype = "_".join(tmp[:-1])
                    data = value.array
                    if rectype in delist:
                        continue
                    else:
                        if rectype not in methods:
                            layerfiles[rectype] = data
                        else:
                            setattr(self, rectype, methods[rectype])
                            if rectype == "budget":
                                setattr(
                                    self, "zonebudget", methods["zonebudget"]
                                )
                                self._methods.append("zonebudget()")
                            elif rectype == "budgetcsv":
                                self.__budgetcsv = True
                            self._methods.append(f"{rectype}()")
                            if rectype == "obs":
                                data = None
                                for ky in obj._simulation_data.mfdata:
                                    if obj.path == (ky[0:2]):
                                        if str(ky[-2]).lower() == "fileout":
                                            data = [[ky[-1]]]
                                            break
                                        elif (
                                            str(ky[-3]) == "continuous"
                                            and str(ky[-1]) == "output"
                                        ):
                                            if (
                                                obj._simulation_data.mfdata[
                                                    ky
                                                ].array[0][0]
                                                == "fileout"
                                            ):
                                                data = [
                                                    [
                                                        obj._simulation_data.mfdata[
                                                            ky
                                                        ].array[
                                                            0
                                                        ][
                                                            -2
                                                        ]
                                                    ]
                                                ]
                                                break

                            if rectype == "package_convergence":
                                rectype = "csv"
                            attr_name = f"_{rectype}"
                            # need a check for obs....
                            if data is not None:
                                if not hasattr(self, attr_name):
                                    setattr(self, attr_name, [data[0][0]])
                                else:
                                    attr = getattr(self, attr_name)
                                    if attr is None:
                                        attr = [data[0][0]]
                                    else:
                                        attr.append(data[0][0])
                                    setattr(self, attr_name, attr)
                            else:
                                setattr(self, attr_name, data)

        else:
            setattr(self, rectype, methods[rectype])
            self._methods.append(f"{rectype}()")
            data = obj.data_list[2].data
            for f in data.keys():
                attr_name = f"_{rectype}"
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, [f])
                else:
                    attr = getattr(self, attr_name)
                    if attr is None:
                        attr = [f]
                    else:
                        attr.append(f)
                    setattr(self, attr_name, attr)

        if layerfiles:
            for rectype, data in layerfiles.items():
                if data is not None:
                    data = data[0][0]

                def get_layerfile_data(self, f=data, text=rectype):
                    """
                    Method to get data from a binary layer file

                    Parameters
                    ----------
                    self : MetaMF6Output
                        placeholder for the self attr after setting
                        as an attribute of the base class.
                    f : str
                        model head or other layer file
                    text : str
                        layer file header search string

                    Returns
                    -------
                        HeadFile object
                    """
                    if f is not None:
                        try:
                            f = os.path.join(self._sim_ws, f)
                            return HeadFile(f, text=text)
                        except OSError:
                            return

                setattr(self.__class__, rectype, get_layerfile_data)
                self._methods.append(f"{rectype}()")

    def methods(self):
        """
        Method that returns a list of available method calls

        Returns
        -------
            list
        """
        if self._methods:
            return self._methods

    @property
    def obs_names(self):
        """
        Method to get obs file names

        Returns
        -------
            list
        """
        try:
            return self._obs
        except AttributeError:
            return

    @property
    def csv_names(self):
        """
        Method to get csv file names

        Returns
        -------
            list
        """
        try:
            return self._csv
        except AttributeError:
            return

    def __zonebudget(self, izone):
        """

        Returns
        -------

        """
        budget = self.__budget()
        grb = None
        if budget is not None:
            zonbud = ZoneBudget6(model_ws=self._sim_ws)
            ZoneFile6(zonbud, izone)
            zonbud.bud = budget
            try:
                if (
                    "gwf" in self._obj.model_or_sim.model_type
                    or "gwt" in self._obj.model_or_sim.model_type
                ):
                    if self._obj.package_type == "oc":
                        dis = self._obj.model_or_sim.dis
                        if (
                            dis.blocks["options"].datasets["nogrb"].array
                            is None
                        ):
                            grb = os.path.join(
                                self._sim_ws, f"{dis.filename}.grb"
                            )
            except AttributeError:
                pass

            zonbud.grb = grb
            return zonbud

    def __budgetcsv(self):
        """
        Convience method to open and return a budget csv object

        Returns
        -------
            flopy.utils.CsvFile object
        """
        return self.__csv(budget=True)

    def __budget(self, precision="double"):
        """
        Convenience method to open and return a budget object

        Returns
        -------
            flopy.utils.CellBudgetFile object
        """
        if self._budget is not None:
            try:
                budget_file = os.path.join(self._sim_ws, self._budget[0])
                return CellBudgetFile(
                    budget_file,
                    precision=precision,
                    modelgrid=self._model.modelgrid,
                )
            except OSError:
                return None

    def __obs(self, f=None):
        """
        Method to read and return obs files

        Parameters
        ----------
        f : str, None
            observation file name, if None the first observation file
            will be returned

        Returns
        -------
        flopy.utils.Mf6Obs file object

        """
        if self._obs is not None:
            obs_file = self.__mulitfile_handler(f, self._obs)

            try:
                obs_file = os.path.join(self._sim_ws, obs_file)
                return Mf6Obs(obs_file)
            except OSError:
                return None

    def __csv(self, f=None, budget=False):
        """
        Method to get csv file outputs

        Parameters
        ----------
        f : str
            csv file name path
        budget : bool
            boolean flag to indicate budgetcsv file

        Returns
        -------
        flopy.utils.CsvFile object

        """
        if budget and self._budgetcsv is not None:
            csv_file = self.__mulitfile_handler(f, self._budgetcsv)
        elif self._csv is not None:
            csv_file = self.__mulitfile_handler(f, self._csv)
        else:
            return

        try:
            csv_file = os.path.join(self._sim_ws, csv_file)
            return CsvFile(csv_file)
        except OSError:
            return None

    def __list(self):
        """
        Method to read list files

        Returns
        -------
            Mf6ListBudget object
        """
        if self._lst is not None:
            try:
                list_file = os.path.join(self._sim_ws, self._lst)
                return Mf6ListBudget(list_file)
            except (AssertionError, OSError):
                return None

    def __mulitfile_handler(self, f, flist):
        """
        Method to parse multiple output files of the same type

        Parameters
        ----------
        f : str
            file name
        flist : list
            list of output file names

        Returns
        -------
            file name string of valid file or first file is f is None
        """
        if len(flist) > 1 and f is None:
            print("Multiple csv files exist, selecting first")
            filename = flist[0]
        else:
            if f is None:
                filename = flist[0]
            else:
                idx = flist.index(f)
                if idx is None:
                    err = f"File name not found, available files are {', '.join(flist)}"
                    raise FileNotFoundError(err)
                else:
                    filename = flist[idx]

        return filename
