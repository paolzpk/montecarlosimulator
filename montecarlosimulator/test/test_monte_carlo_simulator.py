import montecarlosimulator as mcs
from montecarlosimulator import SimulationFailureError
from montecarlosimulator.test import test_classes
from montecarlosimulator.dispersions import SimulationDataclass

from scipy.integrate import solve_ivp

import pandas as pd
import numpy as np

from dataclasses import dataclass, is_dataclass

from typing import Iterable, Union
import pytest
import re
import pickle


@pytest.fixture
def low_len():
    return 1.3


@pytest.fixture
def high_len():
    return 1.4


@pytest.fixture
def nominal_len():
    return 1.32


@pytest.fixture
def length_pendulum(nominal_len, low_len, high_len):
    return mcs.UniformDispersions(nominal=nominal_len, low=low_len, high=high_len)


class TestDispersions:
    def test_generate_dispersions_constant_value_arg(self):
        # Disperse a float
        a_constant = 42
        dispersion_generator = mcs.get_dispersions_generator(a_constant)
        dispersed_args, dispersed_kwargs = next(dispersion_generator)
        assert dispersed_args == [a_constant], 'A constant must not be dispersed'
        assert dispersed_kwargs == {}, 'A constant must not be dispersed'

        # Disperse an object
        a_constant = np.ones((10, 15))
        dispersion_generator = mcs.get_dispersions_generator(a_constant)
        dispersed_args, dispersed_kwargs = next(dispersion_generator)
        assert dispersed_args == [a_constant], 'A constant must not be dispersed'
        assert dispersed_kwargs == {}, 'A constant must not be dispersed'

        # Infinitely disperse
        for _, (dispersed_args, dispersed_kwargs) in zip(range(500), dispersion_generator):
            assert dispersed_args == [a_constant], 'A constant must not be dispersed'
            assert dispersed_kwargs == {}, 'A constant must not be dispersed'

    def test_generate_dispersions_constant_value_kwarg(self):
        # Disperse a float
        a_constant = 42
        dispersion_generator = mcs.get_dispersions_generator(parameter=a_constant)
        dispersed_args, dispersed_kwargs = next(dispersion_generator)
        assert dispersed_args == [], 'A constant must not be dispersed'
        assert dispersed_kwargs == dict(parameter=a_constant), 'A constant must not be dispersed'

        # Disperse an object
        a_constant = np.ones((10, 15))
        dispersion_generator = mcs.get_dispersions_generator(parameter=a_constant)
        dispersed_args, dispersed_kwargs = next(dispersion_generator)
        assert dispersed_args == [], 'A constant must not be dispersed'
        assert dispersed_kwargs == dict(parameter=a_constant), 'A constant must not be dispersed'

        # Infinitely disperse
        for _, (dispersed_args, dispersed_kwargs) in zip(range(500), dispersion_generator):
            assert dispersed_args == [], 'A constant must not be dispersed'
            assert dispersed_kwargs == dict(parameter=a_constant), 'A constant must not be dispersed'

    def test_generate_uniform_dispersions(self):
        a_parameter = mcs.UniformDispersions(nominal=42.3, low=42, high=43)
        dispersion_generator = mcs.get_dispersions_generator(a_parameter)
        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert 42 < dispersed_args[0] < 43, 'The dispersed value is not coherent with the expected range, ' \
                                                'if this fails it is bad as it is aleatory!'

    def test_generate_dispersions_from_iterator(self):
        a_parameter = mcs.UniformDispersions(nominal=42.3, low=42, high=43)
        b_parameter = mcs.NormalDispersions(nominal=1, loc=1, scale=0.1)
        dispersion_generator = mcs.get_dispersions_generator(parameter=a_parameter,
                                                             iterator_dispersions=(a_parameter, b_parameter))
        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert dispersed_args == [], 'not supposed to have dispersed arguments'
            assert {'parameter', 'iterator_dispersions'} == set(
                dispersed_kwargs.keys()), 'Unexpected dispersed elements'
            assert isinstance(dispersed_kwargs['iterator_dispersions'], Iterable)
            assert len(dispersed_kwargs['iterator_dispersions']) == 2
            assert dispersed_kwargs['iterator_dispersions'][0] > 40
            assert dispersed_kwargs['iterator_dispersions'][1] < 40, \
                'This can technically fail because of statistics, ' \
                'but if it does fail twice in a row chances are something is wrong'

    def test_generate_dispersions_from_iterator_with_constants(self):
        a_parameter = mcs.UniformDispersions(nominal=2.3, low=2, high=3)
        dispersion_generator = mcs.get_dispersions_generator(parameter=a_parameter,
                                                             iterator_dispersions=(a_parameter, 42))
        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert dispersed_args == [], 'not supposed to have dispersed arguments'
            assert {'parameter', 'iterator_dispersions'} == set(
                dispersed_kwargs.keys()), 'Unexpected dispersed elements'
            assert isinstance(dispersed_kwargs['iterator_dispersions'], Iterable)
            assert len(dispersed_kwargs['iterator_dispersions']) == 2
            assert dispersed_kwargs['iterator_dispersions'][0] < 3
            assert dispersed_kwargs['iterator_dispersions'][1] == 42

    def test_generate_dispersions_from_nested_iterators(self):
        a_parameter = mcs.UniformDispersions(nominal=2.3, low=2, high=3)
        it_dispersions = ((a_parameter, (a_parameter, a_parameter, 42)), 42)
        dispersion_generator = mcs.get_dispersions_generator(parameter=a_parameter, iterator_dispersions=it_dispersions)
        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert dispersed_args == [], 'not supposed to have dispersed arguments'
            assert {'parameter', 'iterator_dispersions'} == set(
                dispersed_kwargs.keys()), 'Unexpected dispersed elements'
            assert isinstance(dispersed_kwargs['iterator_dispersions'], Iterable), \
                'Input was iterable, output should as well'
            assert isinstance(dispersed_kwargs['iterator_dispersions'][0], Iterable), \
                'Input was iterable, output should as well'
            assert isinstance(dispersed_kwargs['iterator_dispersions'][0][1], Iterable), \
                'Input was iterable, output should as well'
            assert len(dispersed_kwargs['iterator_dispersions']) == 2
            assert dispersed_kwargs['iterator_dispersions'][0][0] < 3
            assert all(np.array(dispersed_kwargs['iterator_dispersions'][0][1][0:2]) < 3)
            assert dispersed_kwargs['iterator_dispersions'][0][1][2] == 42
            assert dispersed_kwargs['iterator_dispersions'][1] == 42

    def test_arugments_dataclass_dispersions(self):
        # Test inheritance and instantiation does not raise exceptions
        class SimulationData(SimulationDataclass):
            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: object
            parameter4: str = ''

        SimulationData(parameter1=np.zeros((3, 1)), parameter2=3, parameter3=None, parameter4='p4')
        assert is_dataclass(SimulationData)
        assert not SimulationData.__dataclass_params__.frozen
        assert SimulationData.__dataclass_params__.eq

        # Test addition of @dataclass decorator has no effect
        @dataclass
        class SimulationData(SimulationDataclass):
            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: object

        # Test multiple instantiations
        s_data = SimulationData(parameter1=np.zeros((3, 1)), parameter2=3, parameter3=None)
        assert is_dataclass(SimulationData)
        assert not SimulationData.__dataclass_params__.frozen
        assert SimulationData.__dataclass_params__.eq
        assert all(s_data.parameter1 == np.zeros((3, 1)))
        assert s_data.parameter2 == 3
        assert s_data.parameter3 is None
        s_data1 = SimulationData(parameter1=np.ones((4, 1)), parameter2=4, parameter3=None)
        assert all(s_data1.parameter1 == np.ones((4, 1)))
        assert s_data1.parameter2 == 4
        assert s_data1.parameter3 is None
        assert all(s_data.parameter1 == np.zeros((3, 1)))
        assert s_data.parameter2 == 3
        assert s_data.parameter3 is None

        msg = r'SimulationData cannot contain members named "nominal", "kind" or "target_class"'
        with pytest.raises(SimulationFailureError, match=msg):
            @dataclass
            class SimulationData(SimulationDataclass):
                parameter1: np.array
                parameter2: Union[float, mcs.DispersionType]
                parameter3: object
                nominal: float

        with pytest.raises(SimulationFailureError, match=msg):
            @dataclass
            class SimulationData(SimulationDataclass):
                parameter1: np.array
                parameter2: Union[float, mcs.DispersionType]
                parameter3: object
                kind: float

        with pytest.raises(SimulationFailureError, match=msg):
            @dataclass
            class SimulationData(SimulationDataclass):
                parameter1: np.array
                parameter2: Union[float, mcs.DispersionType]
                parameter3: object
                target_class: float

    def test_can_pickle_dataclass(self, tmp_path):
        """
        A SimulationDataClass must be a non-local variable (i.e. module-scoped variable).
        NOTE: the current implementation uses nested classes to store all the necessary data for the dispersions:
        this is probably not ideal as SimulationDataclass does handle two very different things, that is adapting the
        entry data to be able to disperse as well as storing the dispersed data.
        A possible solution is to split the data storage and the dispersion generation in
        two different classes:
        """

        class SimulationDataNonLocal(SimulationDataclass):
            parameter1: float
            parameter2: Union[float, mcs.DispersionType]
            parameter3: float

        data = test_classes.SimulationData(parameter1=1, parameter2=2, parameter3=3)
        p_data = pickle.loads(pickle.dumps(data))
        assert p_data == data

        data_non_local = SimulationDataNonLocal(parameter1=1, parameter2=1, parameter3=1)
        with pytest.raises(AttributeError,
                           match="Can't pickle local object "
                                 "'TestDispersions.test_can_pickle_dataclass.<locals>.SimulationDataNonLocal'"):
            pickle.dumps(data_non_local)

    def test_generate_dispersions_from_dataclass(self):
        class SimulationData(SimulationDataclass):
            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: object

        class ObjData:
            pass

        params = SimulationData(parameter1=np.array([1]),
                                parameter2=mcs.UniformDispersions(nominal=1.0, low=0.9, high=1.1),
                                parameter3=ObjData())
        dispersion_generator = mcs.get_dispersions_generator(params, dataclass=params)

        nominal_args, nominal_kwargs = next(dispersion_generator)
        assert len(nominal_args) == 1
        assert len(nominal_kwargs) == 1
        assert all(nominal_args[0].parameter1 == np.array([1]))
        assert nominal_args[0].parameter2 == 1.0
        assert isinstance(nominal_args[0].parameter3, ObjData)
        assert all(nominal_kwargs['dataclass'].parameter1 == np.array([1]))
        assert nominal_kwargs['dataclass'].parameter2 == 1.0
        assert isinstance(nominal_kwargs['dataclass'].parameter3, ObjData)

        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert len(dispersed_args) == 1
            assert len(dispersed_kwargs) == 1
            assert all(dispersed_args[0].parameter1 == np.array([1]))
            assert dispersed_args[0].parameter2 < 1.1
            assert isinstance(dispersed_args[0].parameter3, ObjData)
            assert all(dispersed_kwargs['dataclass'].parameter1 == np.array([1]))
            assert dispersed_kwargs['dataclass'].parameter2 < 1.1
            assert isinstance(dispersed_kwargs['dataclass'].parameter3, ObjData)

    def test_generate_dispersion_from_generic_class(self):
        class ObjData:
            pass

        # Missing inheritance: no exception is thrown, the data is not dispersed
        class SimulationData:
            def __init__(self, parameter1, parameter2, parameter3):
                self.parameter1 = parameter1
                self.parameter2 = parameter2
                self.parameter3 = parameter3

            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: object

        params = SimulationData(parameter1=np.array([1]),
                                parameter2=mcs.UniformDispersions(nominal=1.0, low=0.9, high=1.1),
                                parameter3=ObjData())

        dispersion_generator = mcs.get_dispersions_generator(params, dataclass=params)

        nominal_args, nominal_kwargs = next(dispersion_generator)
        assert len(nominal_args) == 1
        assert len(nominal_kwargs) == 1
        assert all(nominal_args[0].parameter1 == np.array([1]))
        assert isinstance(nominal_args[0].parameter2, mcs.UniformDispersions)
        assert isinstance(nominal_args[0].parameter3, ObjData)
        assert all(nominal_kwargs['dataclass'].parameter1 == np.array([1]))
        assert isinstance(nominal_kwargs['dataclass'].parameter2, mcs.UniformDispersions)
        assert isinstance(nominal_kwargs['dataclass'].parameter3, ObjData)

        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert len(dispersed_args) == 1
            assert len(dispersed_kwargs) == 1
            assert all(dispersed_args[0].parameter1 == np.array([1]))
            assert isinstance(dispersed_args[0].parameter2, mcs.UniformDispersions)
            assert isinstance(dispersed_args[0].parameter3, ObjData)
            assert all(dispersed_kwargs['dataclass'].parameter1 == np.array([1]))
            assert isinstance(dispersed_kwargs['dataclass'].parameter2, mcs.UniformDispersions)
            assert isinstance(dispersed_kwargs['dataclass'].parameter3, ObjData)

        # @mcs.simulation_dataclass
        class SimulationData(SimulationDataclass):
            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: object

        params = SimulationData(parameter1=np.array([1]),
                                parameter2=mcs.UniformDispersions(nominal=1.0, low=0.9, high=1.1),
                                parameter3=ObjData())
        dispersion_generator = mcs.get_dispersions_generator(params, dataclass=params)

        nominal_args, nominal_kwargs = next(dispersion_generator)
        assert len(nominal_args) == 1
        assert len(nominal_kwargs) == 1
        assert all(nominal_args[0].parameter1 == np.array([1]))
        assert nominal_args[0].parameter2 == 1.0
        assert isinstance(nominal_args[0].parameter3, ObjData)
        assert all(nominal_kwargs['dataclass'].parameter1 == np.array([1]))
        assert nominal_kwargs['dataclass'].parameter2 == 1.0
        assert isinstance(nominal_kwargs['dataclass'].parameter3, ObjData)

        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert len(dispersed_args) == 1
            assert len(dispersed_kwargs) == 1
            assert all(dispersed_args[0].parameter1 == np.array([1]))
            assert dispersed_args[0].parameter2 < 1.1
            assert isinstance(dispersed_args[0].parameter3, ObjData)
            assert all(dispersed_kwargs['dataclass'].parameter1 == np.array([1]))
            assert dispersed_kwargs['dataclass'].parameter2 < 1.1
            assert isinstance(dispersed_kwargs['dataclass'].parameter3, ObjData)

    def test_generate_dispersions_from_nested_dataclass(self):
        # @mcs.simulation_dataclass
        class NestedSimulationData(SimulationDataclass):
            nested_parameter1: int
            nested_parameter2: Union[float, mcs.DispersionType]

        # @mcs.simulation_dataclass
        class SimulationData(SimulationDataclass):
            parameter1: np.array
            parameter2: Union[float, mcs.DispersionType]
            parameter3: NestedSimulationData

        nested_params = NestedSimulationData(nested_parameter1=42,
                                             nested_parameter2=mcs.UniformDispersions(nominal=20, low=21, high=22))
        params = SimulationData(parameter1=np.array([1]),
                                parameter2=mcs.UniformDispersions(nominal=1.0, low=0.9, high=1.1),
                                parameter3=nested_params)
        dispersion_generator = mcs.get_dispersions_generator(params, dataclass=params)

        nominal_args, nominal_kwargs = next(dispersion_generator)
        assert len(nominal_args) == 1
        assert len(nominal_kwargs) == 1
        assert all(nominal_args[0].parameter1 == np.array([1]))
        assert nominal_args[0].parameter2 == 1.0
        # assert isinstance(nominal_args[0].parameter3, NestedSimulationData), \
        #     'This fails because of the wrapping NestedSimulationData has changed type, but otherwise it would be correct'
        assert hasattr(nominal_args[0].parameter3, 'nested_parameter1') and \
               hasattr(nominal_args[0].parameter3, 'nested_parameter2'), 'Probably it is not a NestedSimulationData'
        assert nominal_args[0].parameter3.nested_parameter1 == 42
        assert nominal_args[0].parameter3.nested_parameter2 == 20
        assert all(nominal_kwargs['dataclass'].parameter1 == np.array([1]))
        assert nominal_kwargs['dataclass'].parameter2 == 1.0
        # assert isinstance(nominal_kwargs['dataclass'].parameter3, NestedSimulationData), \
        #     'This fails because of the wrapping NestedSimulationData has changed type, but otherwise it would be correct'
        assert hasattr(nominal_kwargs['dataclass'].parameter3, 'nested_parameter1') and \
               hasattr(nominal_kwargs['dataclass'].parameter3, 'nested_parameter2'), \
            'Probably it is not a NestedSimulationData'
        assert nominal_kwargs['dataclass'].parameter3.nested_parameter1 == 42
        assert nominal_kwargs['dataclass'].parameter3.nested_parameter2 == 20

        for _, (dispersed_args, dispersed_kwargs) in zip(range(50), dispersion_generator):
            assert len(dispersed_args) == 1
            assert len(dispersed_kwargs) == 1
            assert all(dispersed_args[0].parameter1 == np.array([1]))
            assert dispersed_args[0].parameter2 < 1.1
            # assert isinstance(dispersed_args[0].parameter3, NestedSimulationData), \
            # 'This fails because of the wrapping NestedSimulationData has changed type, but otherwise it would be correct'
            assert hasattr(dispersed_args[0].parameter3, 'nested_parameter1') and \
                   hasattr(dispersed_args[0].parameter3,
                           'nested_parameter2'), 'Probably it is not a NestedSimulationData'
            assert all(dispersed_kwargs['dataclass'].parameter1 == np.array([1]))
            assert dispersed_kwargs['dataclass'].parameter2 < 1.1
            # assert isinstance(dispersed_kwargs['dataclass'].parameter3, NestedSimulationData), \
            # 'This fails because of the wrapping NestedSimulationData has changed type, but otherwise it would be correct'
            assert hasattr(dispersed_kwargs['dataclass'].parameter3, 'nested_parameter1') and \
                   hasattr(dispersed_kwargs['dataclass'].parameter3, 'nested_parameter2'), \
                'Probably it is not a NestedSimulationData'
            assert dispersed_kwargs['dataclass'].parameter3.nested_parameter1 == 42
            assert 21 < dispersed_kwargs['dataclass'].parameter3.nested_parameter2 < 22


@pytest.fixture
def n_sim():
    return 150


def period(length, g=9.81):
    T = 2 * np.pi * np.sqrt(length / g)
    return pd.DataFrame(dict(T=[T]))


def coupled_mass_spring_damper(t_span, t_eval, initial_conditions, physical_data, forces):
    r"""
    The system is composed of 2 masses m1 and m2 attached one another by a spring k2 and a damper b2. The first mass is
    attached to a spring k1 and a damper b1 as well. k1 and b1 are attached to a wall. To each of the masses a force is
    applied as well (F1 to m1 and F2 to m2).
    //|    k1    |------|    k2    |------|
    //|--/\/\/\--|  m1  |--/\/\/\--|  m2  |
    //|--[==-----|      |--[==-----|      |
    //|    b1    |______|    b2    |______|
    //|             |--> x1           |--> x2

    We have:
    T = kinetic_energy = 0.5*m1*dx1^2 + 0.5*m2*dx2^2
    V = potential_energy = 0.5*k1*x1^2 + 0.5*k1*(x2-x1)^2
    deltaWe = delta_xi*Qi = external_virtual_work = delta_x1*(-b1*dx1 - b2*dx1 + F1) + delta_x2*(-b2(dx2-dx1) + F2)

    Lagrange equation:
    d/dt(dT/d(dxi)) - dT/dxi + dV/dxi = Qi

    Hence:
    m1*ddx1 + (b1+b2)*dx1 +        + (k1+k2)*x1 - k2*x2 = F1
    m2*ddx2 -      b2*dx1 + b2*dx2 -      k2*x1 + k2*x2 = F2

    M = [[m1 0],    C = [[(b1+b2),  0]       K = [[(k1+k2), -k2]
         [0 m2]]               b2, b2]]           [    -k2,  k2]]
    """
    f1, f2 = forces
    M = np.diag((physical_data.mass1, physical_data.mass2))
    C = np.diag((physical_data.damper_constant1, physical_data.damper_constant2))
    K = np.diag((physical_data.elastic_constant1, physical_data.elastic_constant2))
    zero_m = np.zeros((2, 2))
    id_m = np.eye(2)
    M_inv = np.linalg.inv(M)
    A = np.block([[zero_m, id_m], [-M_inv * K, -M_inv * C]])
    B = np.block([[zero_m], [M_inv]])

    def equation(t, y, A, B, forces):
        forces = forces(t) if callable(forces) else forces
        dx = A @ np.array(y).reshape((4, 1)) + B @ np.array(forces).reshape((2, 1))
        return dx.flatten().tolist()

    solution = solve_ivp(equation, t_span, initial_conditions, method='RK45', args=(A, B, forces), t_eval=t_eval)

    if not solution.success:
        raise mcs.SimulationFailureError
    df = pd.DataFrame(data=solution.y.T, index=solution.t,
                      columns=('position1', 'position2', 'velocity1', 'velocity2'))
    df.index.name = 'time'
    df.reset_index(inplace=True)

    return df


def mass_spring_damper(t_span, t_eval, initial_conditions, mass, elastic_constant, damper_constant, force):
    """
    The equation to solve is a simple mass spring damper:
    m*ddx + b*dx + kx = F

    Let x1 = dx and x2 = x:
    [[dx1]  = [[-b/m -k/m]  [[x1]   [[1]  F
     [dx2]]    [   1    0]]  [x2]]   [0]]
    """

    def equation(t, y, m, b, k, force):
        f = force(t) if callable(force) else force
        x1, x2 = y
        dx1 = -b / m * x1 - k / m * x2 + f
        dx2 = x1
        return dx1, dx2

    args = (mass, damper_constant, elastic_constant, force)
    solution = solve_ivp(equation, t_span, initial_conditions, method='RK45', args=args, t_eval=t_eval)

    if not solution.success:
        raise mcs.SimulationFailureError
    df = pd.DataFrame(data=solution.y.T, index=solution.t, columns=('velocity', 'position'))
    df.index.name = 'time'
    df.reset_index(inplace=True)

    return df


@pytest.fixture
def mcs_pendulum(n_sim):
    return mcs.MonteCarloSimulator(model=period, n_simulations=n_sim)


@pytest.fixture
def mcs_ivp_simple_model():
    return mcs.MonteCarloSimulator(model=mass_spring_damper, n_simulations=20, stop_on_failure=False)


@pytest.fixture
def mcs_complex():
    return mcs.MonteCarloSimulator(coupled_mass_spring_damper, n_simulations=20, stop_on_failure=False)


class TestSimulations:
    def test_empty_simulation(self, capsys):
        mcsim = mcs.MonteCarloSimulator(stop_on_failure=True)

        with pytest.raises(TypeError):
            mcsim.compute()

        mcsim.stop_on_failure = False
        mcsim.n_simulations = 1
        mcsim.compute()
        captured = capsys.readouterr()
        re_pattern = re.compile(
            r".*Exception thrown \(skipped\) during execution of a simulation: Traceback \(most recent call last\):\n.*"
        )
        assert re_pattern.match(captured.out), 'Expecting error message on stdout'

    def test_identity_model(self):
        mcsim = mcs.MonteCarloSimulator(n_simulations=1, stop_on_failure=True)

        def wrong_model(x):
            return x

        mcsim.model = wrong_model
        with pytest.raises(mcs.SimulationFailureError,
                           match=re.escape("Simulation failed using data='The model must return a pandas.DataFrame "
                                           "with columns containing the different output parameters and rows containing "
                                           "the dependent variable (e.g. time).'")):
            mcsim.compute(42)

        def correct_model(x):
            return pd.DataFrame(dict(x=[x]))

        mcsim.model = correct_model
        result = mcsim.compute(42)

        expected_result = pd.DataFrame(
            dict(x=[42], sim_name=0, entry_data=mcs.SimulationEntryData(args=[42], kwargs={})))

        assert isinstance(result, pd.DataFrame), 'result must be a list of pandas DataFrame'
        assert len(result) == 2, 'there must be 2 results, the nominal and another'
        assert all(result.iloc[1] == expected_result), \
            'The Answer to the Great Question... Of Life, the Universe and Everything... Is... Forty-two'

    def test_identity_model_multiple_sims(self):
        def identity(x):
            return pd.DataFrame(dict(x=[x]))

        mcsim = mcs.MonteCarloSimulator(n_simulations=100, stop_on_failure=True, model=identity)

        results = mcsim.compute(42)
        assert isinstance(results, pd.DataFrame), 'result must be a pandas DataFrame'
        assert len(results) == 101, 'there must be only one result'
        assert isinstance(results.iloc[0], pd.Series), 'result must be a pandas DataFrame'

    def test_simple_simulation(self, mcs_pendulum, n_sim, length_pendulum, low_len, high_len):
        """ Period of a pendulum: T = 2*pi*sqrt(l/g) """
        results = mcs_pendulum.compute(length=length_pendulum, g=9.81)
        assert len(results) == n_sim + 1, 'Not all the simulations have been run'
        assert np.all(period(low_len).values < results['T'].values), 'At least one of the results is wrong'
        assert np.all(results['T'].values < period(high_len).values), 'At least one of the results is wrong'

    def test_results_incorporate_entry_data(self, mcs_pendulum, n_sim, length_pendulum):
        results = mcs_pendulum.compute(length=length_pendulum)

        assert 'sim_name' in results.columns, 'Must contain a column named sim_name'
        assert 'entry_data' in results.columns, 'Must contain a column named entry_data'
        assert results['entry_data'].dtype == np.dtype('O')
        assert isinstance(results['entry_data'].iloc[0], mcs.SimulationEntryData)
        assert results['entry_data'].iloc[0].args == [], 'Only kwargs passed'
        assert 'length' in results['entry_data'].iloc[0].kwargs

    def test_nominal_simulation(self, mcs_pendulum, n_sim, length_pendulum):
        results = mcs_pendulum.compute(length=length_pendulum)

        assert mcs.MonteCarloSimulator.NOMINAL == -1
        assert mcs.MonteCarloSimulator.NOMINAL in results['sim_name'].unique(), 'Nominal simulation missing'
        assert results[results['sim_name'] == mcs.MonteCarloSimulator.NOMINAL].shape == (1, 3), \
            'Expecting only one df entry for the results of the ' \
            'nominal simulation and 3 columns (sim_name, entry_data, length)'
        assert all(
            results[results['sim_name'] == mcs.MonteCarloSimulator.NOMINAL]['T'].values == period(
                length_pendulum.nominal))

    def test_results_of_failed_sim(self):
        """ Test the df of 2 pendulum simulations successful (nominal + dispersed) and a failed simulation """
        length_pendulum = mcs.ExperimentalDistributionDispersions(nominal=3, population=np.array([2, 4]))

        def period_with_error(length):
            if length < 2.5:
                raise ValueError()
            return period(length)

        # This test has technically aleatory results, but with n_simulations = 100, the probability of an erroneous result
        # is 1/2^100.
        mcs_pendulum = mcs.MonteCarloSimulator(model=period_with_error, n_simulations=100, stop_on_failure=False)
        results = mcs_pendulum.compute(length=length_pendulum)

        assert 'T' in results.columns
        assert 'sim_name' in results.columns
        assert 'entry_data' in results.columns
        assert any(results['T'].isna()), 'The results should contain at least a NaN'

    def test_ivp_model(self, mcs_ivp_simple_model):
        mass = mcs.UniformDispersions(low=0.8, high=1.2, nominal=1)
        elastic_constant = mcs.NormalDispersions(nominal=3, loc=3, scale=0.2)
        damper_constant = mcs.NormalDispersions(nominal=0.5, loc=0.5, scale=0.05)
        force = 0
        t_span = [0, 10]
        initial_conditions = (0, 0.5)  # (velocity, position)
        results = mcs_ivp_simple_model.compute(
            mass=mass,
            elastic_constant=elastic_constant,
            damper_constant=damper_constant,
            force=force,
            t_span=t_span,
            t_eval=np.linspace(min(t_span), max(t_span), 50),
            initial_conditions=initial_conditions,
        )

        assert set(results.columns) == {'sim_name', 'entry_data', 'velocity', 'position', 'time'}, 'Wrong columns'
        assert len(results['sim_name'].unique()) == mcs_ivp_simple_model.n_simulations + 1, \
            'Unexpected number of simulations'
        assert results[
                   results['sim_name'] == 19].index.size == 50, 'One simulation should have 50 instants as a results'
        assert len(results) == (mcs_ivp_simple_model.n_simulations + 1) * 50, \
            'Each simulation should contain 50 instants as a result'

        # Uncomment below to debug visually
        # import matplotlib.pyplot as plt
        # result = results[results['sim_name'] == -1]
        # plt.plot(result['time'], result['position'])
        # plt.show()

    def test_ivp_model_dispersing_initial_conditions(self, mcs_ivp_simple_model):
        mass = 1
        elastic_constant = 3
        damper_constant = 0.5
        force = 0
        t_span = [0, 10]
        initial_conditions = (mcs.NormalDispersions(nominal=0, loc=0, scale=0.2),  # velocity
                              mcs.NormalDispersions(nominal=0.5, loc=0.5, scale=0.05))  # position
        results = mcs_ivp_simple_model.compute(
            mass=mass,
            elastic_constant=elastic_constant,
            damper_constant=damper_constant,
            force=force,
            t_span=t_span,
            # t_eval=None,
            t_eval=np.linspace(min(t_span), max(t_span), 50),
            initial_conditions=initial_conditions,
        )

        assert set(results.columns) == {'sim_name', 'entry_data', 'velocity', 'position', 'time'}, 'Wrong columns'
        assert len(results['sim_name'].unique()) == mcs_ivp_simple_model.n_simulations + 1, \
            'Unexpected number of simulations'
        assert results[
                   results['sim_name'] == 19].index.size == 50, 'One simulation should have 50 instants as a results'
        assert len(results) == (mcs_ivp_simple_model.n_simulations + 1) * 50, \
            'Each simulation should contain 50 instants as a result'

        # Uncomment below to debug visually
        # import matplotlib.pyplot as plt
        # result = results[results['sim_name'] == -1]
        # plt.plot(result['time'], result['position'])
        # result = results[results['sim_name'] == 0]
        # plt.plot(result['time'], result['position'])
        # result = results[results['sim_name'] == 2]
        # plt.plot(result['time'], result['position'])
        # plt.show()

    def test_complex_simulation_parallel(self, mcs_complex):
        """
        To fix this test, fix first test_can_pickle_dataclass which is a simpler version of the problem here.
        """

        physical_data = test_classes.PhysicalData(
            mass1=mcs.UniformDispersions(nominal=1, low=0.9, high=1.1),
            mass2=mcs.UniformDispersions(nominal=3, low=2.9, high=3.1),
            elastic_constant1=mcs.UniformDispersions(nominal=3, low=2.9, high=3.1),
            elastic_constant2=mcs.UniformDispersions(nominal=2, low=1.9, high=2.1),
            damper_constant1=mcs.UniformDispersions(nominal=0.5, low=0.4, high=0.6),
            damper_constant2=mcs.UniformDispersions(nominal=0.5, low=0.4, high=0.6),
        )
        forces = (0, 0)
        t_span = [0, 10]
        initial_conditions = (
            mcs.NormalDispersions(nominal=0.5, loc=0.5, scale=0.05),  # position1
            mcs.NormalDispersions(nominal=0.5, loc=0.5, scale=0.05),  # position2
            mcs.NormalDispersions(nominal=0, loc=0, scale=0.2),  # velocity1
            mcs.NormalDispersions(nominal=0, loc=0, scale=0.2),  # velocity2
        )
        mcs_complex.parallel = True
        results = mcs_complex.compute(
            t_span=t_span,
            t_eval=np.linspace(min(t_span), max(t_span), 50),
            initial_conditions=initial_conditions,
            physical_data=physical_data,
            forces=forces,
        )

        assert set(results.columns) == {'sim_name', 'entry_data', 'velocity1', 'position1', 'velocity2', 'position2',
                                        'time'}, 'Wrong columns'
        assert len(results['sim_name'].unique()) == mcs_complex.n_simulations + 1, \
            'Unexpected number of simulations'
        assert results[
                   results['sim_name'] == 19].index.size == 50, 'One simulation should have 50 instants as a results'
        assert len(results) == (mcs_complex.n_simulations + 1) * 50, \
            'Each simulation should contain 50 instants as a result'

    def test_simple_simulation_parallel(self, mcs_pendulum, n_sim, length_pendulum, low_len, high_len):
        mcs_pendulum.parallel = True
        results = mcs_pendulum.compute(length=length_pendulum, g=9.81)
        assert len(results) == n_sim + 1, 'Not all the simulations have been run'
        assert np.all(period(low_len).values < results['T'].values), 'At least one of the results is wrong'
        assert np.all(results['T'].values < period(high_len).values), 'At least one of the results is wrong'

    def test_ivp_model_parallel(self, mcs_ivp_simple_model):
        mass = mcs.UniformDispersions(low=0.8, high=1.2, nominal=1)
        elastic_constant = mcs.NormalDispersions(nominal=3, loc=3, scale=0.2)
        damper_constant = mcs.NormalDispersions(nominal=0.5, loc=0.5, scale=0.05)
        force = 0
        t_span = [0, 10]
        initial_conditions = (0, 0.5)  # (velocity, position)
        mcs_ivp_simple_model.parallel = True
        results = mcs_ivp_simple_model.compute(
            mass=mass,
            elastic_constant=elastic_constant,
            damper_constant=damper_constant,
            force=force,
            t_span=t_span,
            t_eval=np.linspace(min(t_span), max(t_span), 50),
            initial_conditions=initial_conditions,
        )

        assert set(results.columns) == {'sim_name', 'entry_data', 'velocity', 'position', 'time'}, 'Wrong columns'
        assert len(results['sim_name'].unique()) == mcs_ivp_simple_model.n_simulations + 1, \
            'Unexpected number of simulations'
        assert results[
                   results['sim_name'] == 19].index.size == 50, 'One simulation should have 50 instants as a results'
        assert len(results) == (mcs_ivp_simple_model.n_simulations + 1) * 50, \
            'Each simulation should contain 50 instants as a result'

        # Uncomment below to debug visually
        # import matplotlib.pyplot as plt
        # result = results[results['sim_name'] == -1]
        # plt.plot(result['time'], result['position'])
        # plt.show()
