from montecarlosimulator.Exceptions import SimulationFailureError
from montecarlosimulator.data_structures import NormalDispersions, UniformDispersions, \
    ExperimentalDistributionDispersions, DispersionType, Constant, IteratorDispersions

import numpy as np

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from inspect import isclass
from typing import Literal, Union, Any, Type, Iterable
from functools import partial
from itertools import chain


class Extractor:
    def __init__(self, data, n_dispersions=20):
        """
        :param data: generic data structure containing the data for the specific distribution (check children for
         details)
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
    value = data.value
    generators = {param_name: ExtractValue(value.get(param_name, None), n_dispersions)
                  for param_name in value.keys()}

    extracted_classes = []
    for _ in range(n_dispersions):
        extracted_arguments = {param_name: next(generator) for param_name, generator in generators.items()}
        extracted = type(data)(**extracted_arguments)  # TODO the class here must not have the nominal parameter
        extracted_classes.append(extracted)
    return extracted_classes


# TODO Substitute this with a class implementing __init_subclass__ and replace dispersors by _registry
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
    elif isinstance(value, str):
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


# TODO this decorator is deprecated and should be deleted
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


dataclass_handled_fields = ('__dataclass_fields__', '__dataclass_params__', '__init__', '__repr__', '__doc__', '__eq__',
                            '__setattr__', '__delattr__')


class SimulationDataclass(DispersionType):
    target_class: Type
    nominal: Any
    kind: Literal['class']

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure cls is a dataclass (if the decorator is applied afterwards, nothing changes (though for the moment no
        # parameters are supported, i.e. @dataclass(frozen=True) will raise an exception).
        dataclass(cls)

        # Store the original __init__: a new one will be constructed during __new__ execution in order to correctly
        # initialize the internal data (this is a trick to overcome the fact that @dataclass does not provide a way
        # to add code to the auto-generated __init__ (via a __pre_init__, the __post_init__ does not solve the issue).
        cls.__orig_init__ = cls.__init__

        # Dynamically create an internal class to contain the nominal data. This class
        # contains the same data as the child class of SimulationDataclass but does
        # not inherit from SimulationDataclass.
        # The __init__ method is constructed by @dataclass, hence we need to make sure
        # we are creating it without any of the methods that are filled by @dataclass.
        # Another solution to achieve the same would be to have
        internal_class_name = f'__montecarlosimulator_{cls.__name__}'
        cls_dict = dict(cls.__dict__)
        for f in dataclass_handled_fields:
            cls_dict.pop(f, None)

        if {'nominal', 'kind', 'target_class'} & set(chain(cls_dict['__annotations__'], cls_dict)):
            msg = f'{cls.__name__} cannot contain members named "nominal", "kind" or "target_class"'
            raise SimulationFailureError(msg)

        # Note this _internal_dataclass will belong to the class inheriting from SimulationDataclass and NOT to
        # SimulationDataclass itself.
        cls._internal_dataclass = dataclass(type(internal_class_name, (), cls_dict), frozen=True, eq=False)

    def __new__(cls, *args, **kwargs):
        # This method is responsible for constructing an instance of cls AND providing the necessary missing inputs to
        # the __init__ method generated by the dataclass decorator ('nominal' and 'kind' members). As dataclass does not
        # provide a __pre_init__ function this must be done here (mangling with the cls.__init__)
        # The missing parts in the __init__ are the initialization of the value and nominal members.
        if cls is SimulationDataclass:
            msg = 'SimulationDataclass should not be instantiated on its own but used via inheritance'
            raise SimulationFailureError(msg)

        obj = super().__new__(cls)
        w_args = [dispersion_type_wrap(arg) for arg in args]
        w_kwargs = {key: dispersion_type_wrap(arg) for key, arg in kwargs.items()}

        # Do not fill the 'nominal' field if not strictly necessary. Note because of this it is possible to pickle cls
        # without implementing __reduce__, __get_state__ or __set_state__.
        is_nominal_needed = not all(isinstance(val, Constant) for val in chain(w_args, w_kwargs.values()))
        nominal_instance = cls._internal_dataclass(
            *[arg.nominal for arg in w_args],
            **{key: arg.nominal for key, arg in w_kwargs.items()}
        ) if is_nominal_needed else None

        # Restore the original __init__ if necessary
        if cls.__init__ is not cls.__orig_init__:
            cls.__init__ = cls.__orig_init__

        # Modify the cls.__init__ to add the initialization of 'nominal' and 'kind'. For some reason I do not understand
        # the incorrect __init__ is called if I modify obj.__init__, hence the need to use the cls.__init__ as a proxy.
        # If ever dataclass will provide a __pre_init__ this hack should be substituted with a __pre_init__.
        partial_kwargs = dict(nominal=nominal_instance, kind='class')
        cls.__init__ = partial(obj.__init__, **partial_kwargs)
        return obj

    @property
    def value(self):
        # TODO this is a needless copy and should be optimized: if type(self) contains a lot of fields is a bottleneck
        return {k: dispersion_type_wrap(v) for k, v in self.__dict__.items() if k not in ['nominal', 'kind']}

    @classmethod
    def target_class(cls, *args, **kwargs):
        return cls._internal_dataclass(*args, **kwargs)
