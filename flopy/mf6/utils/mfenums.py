from enum import Enum


class ModflowDataAxis(Enum):
    """
    Enumeration of model data axis
    """
    time = 1
    row = 2
    column = 3
    layer = 4
    x_coord = 5
    y_coord = 6
    elv = 7


class DiscretizationType(Enum):
    """
    Enumeration of discretization types
    """
    UNDEFINED = 0
    DIS = 1
    DISV = 2
    DISU = 3
