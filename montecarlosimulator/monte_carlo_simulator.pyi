from typing import Iterable, Callable, Literal, Union, Any, Iterator, Tuple, Dict

import pandas as pd


def make_chunks(iterable: Iterable, chunk_size: int) -> Iterable: ...

class MonteCarloSimulator:
    """
    Utility to run Monte Carlo simulations.
    """
    NOMINAL: int
    DEFAULT_N_BATCHES: int

    def __init__(self,
                 model: Callable[[...], pd.DataFrame],
                 n_simulations: int,
                 stop_on_failure: bool,
                 parallel: bool,
                 n_batches: Union[int, Literal['auto']]) -> None: ...

    @stop_on_failure
    def stop_on_failure(self) -> bool: ...

    @stop_on_failure.setter
    def stop_on_failure(self, stop: bool) -> None: ...

    @n_simulations
    def n_simulations(self) -> int: ...

    @n_simulations.setter
    def n_simulations(self, n: int) -> None: ...

    @property
    def model(self) -> Callable[[...], pd.DataFrame]: ...

    @model.setter
    def model(self, model: Callable[[...], pd.DataFrame]) -> None: ...

    @property
    def parallel(self) -> bool: ...

    @parallel.setter
    def parallel(self, parallel: bool) -> None: ...

    @property
    def n_batches(self) -> int: ...

    @n_batches.setter
    def n_batches(self, n: Union[int, Literal['auto']]) -> None: ...

    def compute(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> pd.DataFrame: ...

    def _compute_parallel(self, simulation_it: Iterator[int, (Tuple[Any, ...], Dict[str, Any])]) -> pd.DataFrame: ...

    def _compute_sequential(self, simulation_it: Iterator[int, (Tuple[Any, ...], Dict[str, Any])]) -> pd.DataFrame: ...

    def _prepare_single_result(self,
                               result: pd.DataFrame,
                               sim_name: int,
                               dispersed_args: Tuple[Any, ...],
                               dispersed_kwargs: Dict[str, Any]) -> pd.DataFrame: ...

    def _run_single_simulation(self,
                               dispersed_args: Tuple[Any, ...],
                               dispersed_kwargs: Dict[str, Any]) -> pd.DataFrame(): ...
