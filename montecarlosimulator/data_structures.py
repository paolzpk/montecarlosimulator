import numpy as np

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Literal, Iterator


InitialConditions = namedtuple('InitialConditions', 't0, y0')


# TODO when updating to python 3.10.x, remove all the defaults from the fields but for the 'kind' field.
# They are confusing and not necessary for python >3.10.0.
@dataclass(frozen=True, eq=False)
class DispersionType:
    """
    This class should be used as a base class for specifying the different possible specific distributions a dispersion
    can have. The members:
    - nominal: should contain the nominal value to be assigned;
    - kind: should never be changed by the user and should be defaulted to a string containing the name of the
    dispersion.
    """
    nominal: object
    kind: str = field(init=False, default='')


@dataclass(frozen=True, eq=False)
class NormalDispersions(DispersionType):
    loc: float = 0  # mean
    scale: float = 0  # standard deviation
    kind: Literal['normal', 'gaussian'] = 'normal'


@dataclass(frozen=True, eq=False)
class UniformDispersions(DispersionType):
    low: float = 0
    high: float = 0
    kind: Literal['uniform'] = 'uniform'


@dataclass(frozen=True, eq=False)
class ExperimentalDistributionDispersions(DispersionType):
    population: np.array = np.array(())
    kind: Literal['resampling'] = 'resampling'


@dataclass(frozen=True, eq=False)
class Constant(DispersionType):
    value: object = 0
    kind: Literal['constant'] = 'constant'


@dataclass(frozen=True, eq=False)
class IteratorDispersions(DispersionType):
    value: Iterator = ()
    kind: Literal['iterator'] = 'iterator'


@dataclass
class SimulationEntryData:
    args: list
    kwargs: dict

