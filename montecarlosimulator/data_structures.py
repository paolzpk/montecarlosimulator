import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, Iterator, Any, Tuple, Dict

InitialConditions = namedtuple('InitialConditions', 't0, y0')


class DispersionType:
    """
    This class should be used as a base class for specifying the different possible specific distributions a dispersion
    can have. The members:
    - nominal: should contain the nominal value to be assigned;
    - kind: should never be changed by the user and should be defaulted to a string containing the name of the
    dispersion.
    """

    def __init_subclass__(cls, parallel=False, **kwargs):
        # Make sure DispersionType subclasses are dataclasses
        dataclass(DispersionType)
        super().__init_subclass__(**kwargs)

    nominal: object
    kind: str


@dataclass
class NormalDispersions(DispersionType):
    loc: float = 0  # mean
    scale: float = 0  # standard deviation
    kind: Literal['normal', 'gaussian'] = 'normal'


@dataclass
class UniformDispersions(DispersionType):
    low: float = 0
    high: float = 0
    kind: Literal['uniform'] = 'uniform'


@dataclass
class ExperimentalDistributionDispersions(DispersionType):
    population: np.array = np.array(())
    kind: Literal['resampling'] = 'resampling'


@dataclass
class Constant(DispersionType):
    value: object = 0
    kind: Literal['constant'] = 'constant'


@dataclass
class IteratorDispersions(DispersionType):
    value: Iterator = ()
    kind: Literal['iterator'] = 'iterator'


@dataclass
class SimulationEntryData:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
