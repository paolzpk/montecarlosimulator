from montecarlosimulator.Exceptions import SimulationFailureError
from montecarlosimulator.data_structures import NormalDispersions, UniformDispersions, \
    ExperimentalDistributionDispersions, DispersionType, Constant, IteratorDispersions

import numpy as np

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from inspect import isclass
from typing import Literal, Union, Any, Type, Iterable


class Extractor:
    def __init__(self, data, n_dispersions=20):
        """
        :param data: generic data structure containing the data for the specific distribution (check children for details)
        :param n_dispersions: (opt) number of dispersions to be generated
        """
        self.data = data
        self.n_dispersions = n_dispersions

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    @abstractmethod
    def get(self, *args, **kwargs):
        return


# TODO change all Extract* to lower_case
def ExtractUniform(data: UniformDispersions, n_dispersions=20):
    """
    :param data: must contain float like data.low and data.high
    :param n_dispersions: (opt) number of dispersions to be generated
    """
    return np.random.uniform(low=data.low, high=data.high, size=n_dispersions)


def ExtractGaussian(data: NormalDispersions, n_dispersions=20):
    """
    :param data: must contain :
        data.loc float or array_like of floats: Mean  ("centre") of the distribution
        data.scale float or array_like of floats: Standard deviation (spread or "width") of the distribution.
        Must be non-negative
    :param n_dispersions: (opt) number of dispersions to be generated
    """
    return np.random.normal(data.loc, data.scale, n_dispersions)


def ExtractResampling(data: ExperimentalDistributionDispersions, n_dispersions=20):
    """
    :param data: must contain data.population array-like: sampled population for an experimental distribution
    :param n_dispersions: (opt) number of dispersions to be generated
    """
    idx = np.random.randint(low=0, high=len(data.population), size=n_dispersions)
    return data.population[idx]


def ExtractConstant(data, n_dispersions=20):
    """
    A constant (float, np.array or other) cannot be dispersed, but it is seamlessly returned without errors.
    """
    return [data.value for _ in range(n_dispersions)]


def ExtractIterator(data, n_dispersions=20):
    """
    Create dispersed extractions for each of the values of the data (an iterator)
    """
    generators = [ExtractValue(value, n_dispersions) for value in data.value]
    return [tuple(extracted) for _, *extracted in zip(range(n_dispersions), *generators)]


def ExtractClass(data, n_dispersions=20):
    """
    Continuously yield dispersed values for a dataclass. All fields are dispersed recursively if they can be dispersed
    else they will be treated as constants.
    """
    generators = {param_name: ExtractValue(getattr(data.value, param_name), n_dispersions)
                  for param_name in data.value.__dataclass_fields__.keys()}

    extracted_classes = []
    for _ in range(n_dispersions):
        extracted_arguments = {param_name: next(generator) for param_name, generator in generators.items()}
        extracted_classes.append(data.target_class(**extracted_arguments))
    return extracted_classes


def ExtractValue(data: DispersionType, n_dispersions=20) -> np.array:
    """
    Continuously yield dispersed values for the data. The dispersed values are generated in bunches of n_dispersions for
    computational efficiency.
    :param data: DispersionType containing the data to be dispersed along with the type of dispersion to be applied
    (see the DispersionType inheritance structure).
    :param n_dispersions: For efficiency reason the dispersions are generated in bunches of n_dispersions, but
    regardless of n_dispersions, ExtractValue always yields one dispersed data point per call.
    :return: generator infinitely yielding dispersed values according to the specifies dispersion type.
    """
    kind = data.kind.lower()
    dispersors = {
        'constant': ExtractConstant,
        'uniform': ExtractUniform,
        'normal': ExtractGaussian,
        'gaussian': ExtractGaussian,
        'resampling': ExtractResampling,
        'iterator': ExtractIterator,
        'class': ExtractClass,
    }
    while True:
        for dispersed in dispersors[kind](data, n_dispersions):
            yield dispersed


def dispersion_type_wrap(value):
    if isinstance(value, DispersionType):
        return value
    elif isinstance(value, np.ndarray):
        for x in np.nditer(value):
            if isinstance(x, DispersionType):
                raise SimulationFailureError('Numpy arrays cannot contain values to disperse!')
        return Constant(nominal=value, value=value)
    elif isinstance(value, Iterable):
        wrapped_values = [dispersion_type_wrap(element) for element in value]
        return IteratorDispersions(value=wrapped_values,
                                   nominal=[element.nominal for element in wrapped_values])
    else:
        return Constant(nominal=value, value=value)


def get_dispersions_generator(*args, **kwargs):
    args = [dispersion_type_wrap(arg) for arg in args]
    dispersed_args_generator = [ExtractValue(arg) for arg in args]

    kwargs = {arg_key: dispersion_type_wrap(arg_value) for arg_key, arg_value in kwargs.items()}

    dispersed_kwargs_generator = {arg_key: ExtractValue(arg_value) for arg_key, arg_value in kwargs.items()}

    # Yield nominal parameters
    yield [arg.nominal for arg in args], {arg_key: arg_value.nominal for arg_key, arg_value in kwargs.items()}

    # Yield dispersed parameters
    while True:
        dispersed_args = [next(arg_generator) for arg_generator in dispersed_args_generator]
        dispersed_kwargs = {arg_key: next(value_generator)
                            for arg_key, value_generator in dispersed_kwargs_generator.items()}
        yield dispersed_args, dispersed_kwargs


def simulation_dataclass(fcn=None, no_parallel=False):
    def simulation_dataclass_internal(cls):
        if not isclass(cls):
            raise SimulationFailureError('The simulation_class decorator can be applied only to classes')
        if not is_dataclass(cls):
            # reassignment is useless as dataclass() modifies the argument, but it is clearer
            cls = dataclass(cls, frozen=True, eq=False)
        if not no_parallel and (not cls.__dataclass_params__.frozen or cls.__dataclass_params__.eq):
            raise SimulationFailureError('Dataclasses must be used with frozen=True and eq=False')
        for parameter_name, dataclass_field in cls.__dataclass_fields__.items():
            if hasattr(dataclass_field.type, '__name__') and dataclass_field.type.__name__ == 'field':
                dataclass_field.type = Union[Any, DispersionType]
            else:
                dataclass_field.type = Union[dataclass_field.type, DispersionType]

        class DataClassDispersions(DispersionType):
            target_class: Type = cls
            nominal: Any = None
            kind: Literal['class'] = 'class'

            __slots__ = ('value',)

            def __init__(self, *args, **kwargs):
                w_args = [dispersion_type_wrap(arg) for arg in args]
                w_kwargs = {key: dispersion_type_wrap(arg) for key, arg in kwargs.items()}
                self.value = self.target_class(*w_args, **w_kwargs)

                nominal_instance = self.target_class(
                    *[arg.nominal for arg in w_args],
                    **{key: arg.nominal for key, arg in w_kwargs.items()}
                )
                super().__init__(nominal=nominal_instance)

        return DataClassDispersions

    if fcn:
        return simulation_dataclass_internal(fcn)
    else:
        return simulation_dataclass_internal
