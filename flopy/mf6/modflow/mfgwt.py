# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfmodel
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwt(mfmodel.MFModel):
    """
    Modflowgwt defines a gwt model

    Parameters
    ----------
    modelname : string
        name of the model
    model_nam_file : string
        relative path to the model name file from model working folder
    version : string
        version of modflow
    exe_name : string
        model executable name
    model_ws : string
        model working folder path
    sim : MFSimulation
        Simulation that this model is a part of.  Model is automatically
        added to simulation when it is initialized.
    list : string
        * list (string) is name of the listing file to create for this GWT
          model. If not specified, then the name of the list file will be the
          basename of the GWT model name file and the '.lst' extension. For
          example, if the GWT name file is called "my.model.nam" then the list
          file will be called "my.model.lst".
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of all model
          stress package information will be written to the listing file
          immediately after it is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of all model
          package flow rates will be printed to the listing file for every
          stress period time step in which "BUDGET PRINT" is specified in
          Output Control. If there is no Output Control option and
          "PRINT_FLOWS" is specified, then flow rates are printed for the last
          time step of each stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that all model package flow
          terms will be written to the file specified with "BUDGET FILEOUT" in
          Output Control.
    packages : [ftype, fname, pname]
        * ftype (string) is the file type, which must be one of the following
          character values shown in table in mf6io.pdf. Ftype may be entered in
          any combination of uppercase and lowercase.
        * fname (string) is the name of the file containing the package input.
          The path to the file should be included if the file is not located in
          the folder where the program was run.
        * pname (string) is the user-defined name for the package. PNAME is
          restricted to 16 characters. No spaces are allowed in PNAME. PNAME
          character values are read and stored by the program for stress
          packages only. These names may be useful for labeling purposes when
          multiple stress packages of the same type are located within a single
          GWT Model. If PNAME is specified for a stress package, then PNAME
          will be used in the flow budget table in the listing file; it will
          also be used for the text entry in the cell-by-cell budget file.
          PNAME is case insensitive and is stored in all upper case letters.

    Methods
    -------
    load : (simulation : MFSimulationData, model_name : string,
        namfile : string, version : string, exe_name : string,
        model_ws : string, strict : boolean) : MFSimulation
        a class method that loads a model from files
    """

    model_type = "gwt"

    def __init__(
        self,
        simulation,
        modelname="model",
        model_nam_file=None,
        version="mf6",
        exe_name="mf6.exe",
        model_rel_path=".",
        list=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        packages=None,
        **kwargs
    ):
        super(ModflowGwt, self).__init__(
            simulation,
            model_type="gwt6",
            modelname=modelname,
            model_nam_file=model_nam_file,
            version=version,
            exe_name=exe_name,
            model_rel_path=model_rel_path,
            **kwargs
        )

        self.name_file.list.set_data(list)
        self.name_file.print_input.set_data(print_input)
        self.name_file.print_flows.set_data(print_flows)
        self.name_file.save_flows.set_data(save_flows)
        self.name_file.packages.set_data(packages)

    @classmethod
    def load(
        cls,
        simulation,
        structure,
        modelname="NewModel",
        model_nam_file="modflowtest.nam",
        version="mf6",
        exe_name="mf6.exe",
        strict=True,
        model_rel_path=".",
        load_only=None,
    ):
        return mfmodel.MFModel.load_base(
            simulation,
            structure,
            modelname,
            model_nam_file,
            "gwt",
            version,
            exe_name,
            strict,
            model_rel_path,
            load_only,
        )
