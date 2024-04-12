from enum import Enum


class DiscretizationType(Enum):
    """
    Enumeration of discretization types
    """

    UNDEFINED = 0
    DIS = 1
    DISV = 2
    DISU = 3
    DISV1D = 4
    DIS2D = 5
    DISV2D = 6
