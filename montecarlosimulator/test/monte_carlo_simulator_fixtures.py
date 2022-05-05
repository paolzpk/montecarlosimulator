import montecarlosimulator as mcs

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import pytest

from test import test_classes


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


@pytest.fixture
def mcs_complex_datadict():
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
    return dict(
        t_span=t_span,
        t_eval=np.linspace(min(t_span), max(t_span), 50),
        initial_conditions=initial_conditions,
        physical_data=physical_data,
        forces=forces,
    )
