class CoordUtil(object):
    @staticmethod
    def convert_to_unstruct_jagged_array(unstruct_con_array, iac):
        """
        Converts and unstructured connection array (ja, cl12, ...) that is
        currently stored as a 1-d array into a jagged array

        Example:
            cell1_connection cell1_connection cell1_connection cell1_connection
            cell2_connection cell2_connection cell2_connection
            cell3_connection cell2_connection cell2_connection cell2_connection
            cell2_connection

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
