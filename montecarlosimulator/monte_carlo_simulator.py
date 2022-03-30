from montecarlosimulator.data_structures import SimulationEntryData
from montecarlosimulator.dispersions import get_dispersions_generator
from montecarlosimulator.Exceptions import SimulationFailureError

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain, islice
import math
import multiprocessing
import traceback
from typing import Union, Literal


def make_chunks(iterable, chunk_size):
    """
    Creates a generator whose elements are iterable of size chunk_size of the original iterator.
    >>> from types import GeneratorType
    >>> result = make_chunks(range(100), 49)
    >>> assert isinstance(result, GeneratorType)
    >>> assert next(result) == tuple(range(49)), 'The first chunk is of size 49'
    >>> assert next(result) == tuple(range(49, 98)), 'The second chunk is of size 49'
    >>> assert next(result) == tuple(range(98, 100)), 'The third chunk is of size 2'
    """
    iterator = iter(iterable)
    for first in iterator:
        yield tuple(chain([first], islice(iterator, chunk_size - 1)))


class MonteCarloSimulator:
    NOMINAL = -1
    DEFAULT_N_BATCHES = max(multiprocessing.cpu_count() - 1, 2)

    def __init__(self, model=None, n_simulations=None, stop_on_failure=False, parallel=False, n_batches='auto'):
        """
        :param model: function representing the model to be simulated. The function must return a pd.DataFrame having
        as columns the different studied variables and as rows values for the independent variable (usually time).
        :param n_simulations: number of dispersed simulations (NOTE: a nominal simulation will also be run, hence the
        actual number of simulations is n_simulations+1)
        :param stop_on_failure: stop if any of the simulations fails (an Exception is raised).
        """
        self.n_simulations = n_simulations
        self.model = model
        self.stop_on_failure = stop_on_failure
        self.parallel = parallel
        self.n_batches = n_batches

    @property
    def stop_on_failure(self):
        return self._stop_on_failure

    @stop_on_failure.setter
    def stop_on_failure(self, stop):
        self._stop_on_failure = stop

    @property
    def n_simulations(self):
        return int(self._n_simulations)

    @n_simulations.setter
    def n_simulations(self, n):
        if n is not None and n < 0:
            raise ValueError('Number of simulations must be positive')
        self._n_simulations = n

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, parallel):
        self._parallel = parallel

    @property
    def n_batches(self):
        return self._n_batches

    @n_batches.setter
    def n_batches(self, n: Union[int, Literal['auto']]):
        if n is None or n == 'auto':
            n = MonteCarloSimulator.DEFAULT_N_BATCHES
        self._n_batches = n

    def compute(self, *args, **kwargs):
        simulation_it = zip(range(MonteCarloSimulator.NOMINAL, self.n_simulations),
                            get_dispersions_generator(*args, **kwargs))
        sim_name, (dispersed_args, dispersed_kwargs) = next(simulation_it)
        nominal_results = self._run_single_simulation(dispersed_args, dispersed_kwargs)
        nominal_results = self._prepare_single_result(nominal_results, sim_name, dispersed_args, dispersed_kwargs)
        if self.parallel:
            dispersed_results = self._compute_parallel(simulation_it)
        else:
            dispersed_results = self._compute_sequential(simulation_it)

        return pd.concat([dispersed_results, nominal_results])

    def _compute_parallel(self, simulation_it):
        raise NotImplementedError('The tests for the parallel execution are failing!')
        batches_it = make_chunks(simulation_it, math.ceil(self.n_simulations / self.n_batches))

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._compute_sequential, list(simulation_it)) for simulation_it in batches_it]

        results = []
        for i_batch, future in enumerate(as_completed(futures)):
            print(f'Completed {i_batch} batch.')
            try:
                results.append(future.result())
            except SimulationFailureError:
                print('Something went horribly wrong: ' + traceback.format_exc())
                raise
        return pd.concat(results)

    def _compute_sequential(self, simulation_it):
        results = []
        for sim_name, (dispersed_args, dispersed_kwargs) in simulation_it:
            result = self._run_single_simulation(dispersed_args, dispersed_kwargs)
            result = self._prepare_single_result(result, sim_name, dispersed_args, dispersed_kwargs)
            results.append(result)

        return pd.concat(results)

    def _prepare_single_result(self, result, sim_name, dispersed_args, dispersed_kwargs):
        data = dict(sim_name=sim_name, entry_data=SimulationEntryData(args=dispersed_args, kwargs=dispersed_kwargs))
        simulation_data = pd.DataFrame(data, index=result.index if not result.index.empty else [0])
        result = pd.concat((result, simulation_data), ignore_index=False, axis=1)
        return result

    def _run_single_simulation(self, dispersed_args, dispersed_kwargs):
        try:
            if not dispersed_args and not dispersed_kwargs:
                raise SimulationFailureError(data=None, message='Must provide at least one argument to disperse')
            result = self.model(*dispersed_args, **dispersed_kwargs)

            if not isinstance(result, pd.DataFrame):
                error_msg = 'The model must return a pandas.DataFrame with columns containing the different ' \
                            'output parameters and rows containing the dependent variable (e.g. time).'
                raise SimulationFailureError(error_msg)
        except Exception:
            print('Exception thrown (skipped) during execution of a simulation: ' + traceback.format_exc())
            result = pd.DataFrame()
            if self.stop_on_failure:
                raise
        return result
