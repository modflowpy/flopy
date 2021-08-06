import abc
from enum import Enum


class DataType(Enum):
    array2d = 1
    array3d = 2
    transient2d = 3
    transient3d = 4
    list = 5
    transientlist = 6
    scalar = 7
    transientscalar = 8


class DataInterface:
    @property
    @abc.abstractmethod
    def data_type(self):
        raise NotImplementedError(
            "must define dat_type in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def dtype(self):
        def dtype(self):
            raise NotImplementedError(
                "must define dtype in child class to use this base class"
            )

    @property
    @abc.abstractmethod
    def array(self):
        raise NotImplementedError(
            "must define array in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError(
            "must define name in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def model(self):
        raise NotImplementedError(
            "must define name in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def plottable(self):
        raise NotImplementedError(
            "must define plottable in child class to use this base class"
        )


class DataListInterface:
    @property
    @abc.abstractmethod
    def package(self):
        raise NotImplementedError(
            "must define package in child class to use this base class"
        )

    @property
    @abc.abstractmethod
    def to_array(self, kper=0, mask=False):
        def to_array(self):
            raise NotImplementedError(
                "must define to_array in child class to use this base class"
            )

    @abc.abstractmethod
    def masked_4D_arrays_itr(self):
        def masked_4D_arrays_itr(self):
            raise NotImplementedError(
                "must define masked_4D_arrays_itr in child "
                "class to use this base class"
            )
