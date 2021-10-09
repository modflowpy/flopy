"""Mfusg CLN dtype class."""
import numpy as np


class MfUsgClnDtypes:
    """Defines MfUsgCln dtypes for various CLN geometries."""

    @staticmethod
    def get_clnnode_dtype():
        """Define dtype of CLN node properties.

        Returns
        -------
        dtype

        """
        dtype = np.dtype(
            [
                ("ifno", int),  # node number
                ("iftyp", int),  # type-index
                ("ifdir", int),  # directional index
                ("fleng", np.float32),  # length
                ("felev", np.float32),  # elevation of the bottom
                ("fangle", np.float32),  # angle
                ("iflin", int),  # flag of flow conditions
                ("iccwadi", int),  # flag of vertical flow correction
                ("x1", np.float32),  # coordinates
                ("y1", np.float32),  # coordinates
                ("z1", np.float32),  # coordinates
                ("x2", np.float32),  # coordinates
                ("y2", np.float32),  # coordinates
                ("z2", np.float32),  # coordinates
            ]
        )
        return dtype

    @staticmethod
    def get_gwconn_dtype(structured=True):
        """Dtype of CLN node - GW node connection properties.

        Parameters
        ----------
        structured : True = structured grid

        Returns
        -------
        dtype

        """
        if structured:
            dtype = np.dtype(
                [
                    ("ifnod", int),  # CLN node number
                    ("igwlay", int),  # layer number of connecting gw node
                    ("igwrow", int),  # row number of connecting gw node
                    ("igwfcol", int),  # col number of connecting gw node
                    ("ifcon", int),  # index of connectivity equation
                    ("fskin", np.float32),  # leakance across a skin
                    ("flengw", np.float32),  # length of connection
                    (
                        "faniso",
                        np.float32,
                    ),  # anisotropy or thickness of sediments
                    ("icgwadi", int),  # flag of vertical flow correction
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("ifnod", int),  # CLN node number
                    ("igwnod", int),  # node number of connecting gw node
                    ("ifcon", int),  # index of connectivity equation
                    ("fskin", np.float32),  # leakance across a skin
                    ("flengw", np.float32),  # length of connection
                    (
                        "faniso",
                        np.float32,
                    ),  # anisotropy or thickness of sediments
                    ("icgwadi", int),  # flag of vertical flow correction
                ]
            )
        return dtype

    @staticmethod
    def get_clncirc_dtype(bhe=False):
        """Dtype of CLN node circular conduit type properties.

        Parameters
        ----------
        bhe : borehole heat exchanger (bhe)

        Returns
        -------
        dtype

        """
        if bhe:
            dtype = np.dtype(
                [
                    ("iconduityp", int),  # index of circular conduit type
                    ("frad", np.float32),  # radius
                    (
                        "conduitk",
                        np.float32,
                    ),  # conductivity or resistance factor
                    ("tcond", np.float32),  # thermal conductivity of bhe tube
                    ("tthk", np.float32),  # thickness
                    (
                        "tcfluid",
                        np.float32,
                    ),  # thermal conductivity of the fluid
                    ("tconv", np.float32),  # thermal convective coefficient
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("iconduityp", int),  # index of circular conduit type
                    ("frad", np.float32),  # radius
                    (
                        "conduitk",
                        np.float32,
                    ),  # conductivity or resistance factor
                ]
            )
        return dtype

    @staticmethod
    def get_clnrect_dtype(bhe=False):
        """Returns the dtype of CLN node rectangular conduit type properties.

        Parameters
        ----------
        bhe : borehole heat exchanger (bhe)

        Returns
        -------
        dtype

        """
        if bhe:
            dtype = np.dtype(
                [
                    ("irectyp", int),  # index of rectangular conduit type
                    ("flength", np.float32),  # width
                    ("fheight", np.float32),  # height
                    (
                        "conduitk",
                        np.float32,
                    ),  # conductivity or resistance factor
                    ("tcond", np.float32),  # thermal conductivity of bhe tube
                    ("tthk", np.float32),  # thickness of bhe tube
                    ("tcfluid", np.float32),  # thermal conductivity of fluid
                    ("tconv", np.float32),  # thermal convective
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("irectyp", int),  # index of rectangular conduit type
                    ("flength", np.float32),  # width
                    ("fheight", np.float32),  # height
                    (
                        "conduitk",
                        np.float32,
                    ),  # conductivity or resistance factor
                ]
            )
        return dtype
