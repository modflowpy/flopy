from ..coordinates import modeldimensions
from ..data import mfstructure


def get_grid_type(simulation_data, model_name):
    """
    Return the type of grid used by model 'model_name' in simulation containing
    simulation data 'simulation_data'.

    Parameters
    ----------
    simulation_data : MFSimulationData
        object containing simulation data for a simulation
    model_name : string
        name of a model in the simulation
    Returns
    -------
    grid type : DiscritizationType
    """
    package_recarray = simulation_data.mfdata[(model_name, 'nam', 'packages', 'packagerecarray')]
    structure = mfstructure.MFStructure()
    if package_recarray.search_data('dis{}'.format(structure.get_version_string()),0) is not None:
        return modeldimensions.DiscritizationType.DIS
    elif package_recarray.search_data('disv{}'.format(structure.get_version_string()),0) is not None:
        return modeldimensions.DiscritizationType.DISV
    elif package_recarray.search_data('disu{}'.format(structure.get_version_string()),0) is not None:
        return modeldimensions.DiscritizationType.DISU

    return modeldimensions.DiscritizationType.UNDEFINED


def convert_to_unstruct_jagged_array(unstruct_con_array, iac):
    """
    Converts and unstructured connection array (ja, cl12, ...) that is currently
    stored as a 1-d array into a jagged array

    Example:
        cell1_connection cell1_connection cell1_connection cell1_connection
        cell2_connection cell2_connection cell2_connection
        cell3_connection cell2_connection cell2_connection cell2_connection cell2_connection

    Parameters
    ----------
    unstruct_con_array : list
        unstructured connection array
    iac : list
        contents of the MODFLOW unstructured iac array
    Returns
    -------
    jagged_unstruct_con_array : list
    """
    jagged_list = []
    current_index = 0
    for num_con in iac:
        cell_con = []
        for con in range(0, num_con):
            cell_con.append(unstruct_con_array[current_index])
            current_index += 1
        jagged_list.append(cell_con)
    return jagged_list