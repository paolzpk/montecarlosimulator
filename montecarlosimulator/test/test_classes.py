from montecarlosimulator.dispersions import SimulationDataclass
import montecarlosimulator as mcs

from typing import Union


class SimulationData(SimulationDataclass):
    parameter1: float
    parameter2: Union[float, mcs.DispersionType]
    parameter3: float


class PhysicalData(SimulationDataclass):
    mass1: float
    mass2: float
    elastic_constant1: float
    elastic_constant2: float
    damper_constant1: float
    damper_constant2: float
