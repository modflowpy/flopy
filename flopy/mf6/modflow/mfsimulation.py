# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on April 19, 2024 19:08:53 UTC
import os
from typing import Union

from .. import mfsimbase


class MFSimulation(mfsimbase.MFSimulationBase):
    """
    MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation.
    A MFSimulation object must be created before creating any of the MODFLOW 6
    model objects.

    Parameters
    ----------
    sim_name : str
       Name of the simulation
    continue_ : boolean
        * continue (boolean) keyword flag to indicate that the simulation
          should continue even if one or more solutions do not converge.
    nocheck : boolean
        * nocheck (boolean) keyword flag to indicate that the model input check
          routines should not be called prior to each time step. Checks are
          performed by default.
    memory_print_option : string
        * memory_print_option (string) is a flag that controls printing of
          detailed memory manager usage to the end of the simulation list file.
          NONE means do not print detailed information. SUMMARY means print
          only the total memory for each simulation component. ALL means print
          information for each variable stored in the memory manager. NONE is
          default if MEMORY_PRINT_OPTION is not specified.
    maxerrors : integer
        * maxerrors (integer) maximum number of errors that will be stored and
          printed.
    print_input : boolean
        * print_input (boolean) keyword to activate printing of simulation
          input summaries to the simulation list file (mfsim.lst). With this
          keyword, input summaries will be written for those packages that
          support newer input data model routines. Not all packages are
          supported yet by the newer input data model routines.
    hpc : {varname:data} or hpc_data data
        * Contains data for the hpc package. Data can be stored in a dictionary
          containing data for the hpc package with variable names as keys and
          package data as values. Data just for the hpc variable is also
          acceptable. See hpc package documentation for more information.
    tdis6 : string
        * tdis6 (string) is the name of the Temporal Discretization (TDIS)
          Input File.
    models : [mtype, mfname, mname]
        * mtype (string) is the type of model to add to simulation.
        * mfname (string) is the file name of the model name file.
        * mname (string) is the user-assigned name of the model. The model name
          cannot exceed 16 characters and must not have blanks within the name.
          The model name is case insensitive; any lowercase letters are
          converted and stored as upper case letters.
    exchanges : [exgtype, exgfile, exgmnamea, exgmnameb]
        * exgtype (string) is the exchange type.
        * exgfile (string) is the input file for the exchange.
        * exgmnamea (string) is the name of the first model that is part of
          this exchange.
        * exgmnameb (string) is the name of the second model that is part of
          this exchange.
    mxiter : integer
        * mxiter (integer) is the maximum number of outer iterations for this
          solution group. The default value is 1. If there is only one solution
          in the solution group, then MXITER must be 1.
    solutiongroup : [slntype, slnfname, slnmnames]
        * slntype (string) is the type of solution. The Integrated Model
          Solution (IMS6) is the only supported option in this version.
        * slnfname (string) name of file containing solution input.
        * slnmnames (string) is the array of model names to add to this
          solution. The number of model names is determined by the number of
          model names the user provides on this line.

    Methods
    -------
    load : (sim_name : str, version : string,
        exe_name : str or PathLike, sim_ws : str or PathLike, strict : bool,
        verbosity_level : int, load_only : list, verify_data : bool,
        write_headers : bool, lazy_io : bool, use_pandas : bool,
        ) : MFSimulation
        a class method that loads a simulation from files
    """

    def __init__(
        self,
        sim_name="sim",
        version="mf6",
        exe_name: Union[str, os.PathLike] = "mf6",
        sim_ws: Union[str, os.PathLike] = os.curdir,
        verbosity_level=1,
        write_headers=True,
        use_pandas=True,
        lazy_io=False,
        continue_=None,
        nocheck=None,
        memory_print_option=None,
        maxerrors=None,
        print_input=None,
        hpc_data=None,
    ):
        super().__init__(
            sim_name=sim_name,
            version=version,
            exe_name=exe_name,
            sim_ws=sim_ws,
            verbosity_level=verbosity_level,
            write_headers=write_headers,
            lazy_io=lazy_io,
            use_pandas=use_pandas,
        )

        self.name_file.continue_.set_data(continue_)
        self.name_file.nocheck.set_data(nocheck)
        self.name_file.memory_print_option.set_data(memory_print_option)
        self.name_file.maxerrors.set_data(maxerrors)
        self.name_file.print_input.set_data(print_input)

        self.continue_ = self.name_file.continue_
        self.nocheck = self.name_file.nocheck
        self.memory_print_option = self.name_file.memory_print_option
        self.maxerrors = self.name_file.maxerrors
        self.print_input = self.name_file.print_input
        self.hpc_data = self._create_package("hpc", hpc_data)

    @classmethod
    def load(
        cls,
        sim_name="modflowsim",
        version="mf6",
        exe_name: Union[str, os.PathLike] = "mf6",
        sim_ws: Union[str, os.PathLike] = os.curdir,
        strict=True,
        verbosity_level=1,
        load_only=None,
        verify_data=False,
        write_headers=True,
        lazy_io=False,
        use_pandas=True,
    ):
        return mfsimbase.MFSimulationBase.load(
            cls,
            sim_name,
            version,
            exe_name,
            sim_ws,
            strict,
            verbosity_level,
            load_only,
            verify_data,
            write_headers,
            lazy_io,
            use_pandas,
        )
