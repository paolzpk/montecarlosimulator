import pytest

from montecarlosimulator import MonteCarloVisualizer, MonteCarloSimulator

from matplotlib import pyplot as plt  # add plt.show() to visually test the graphs


class TestMonteCarloVisualizer:
    def test_montecarlo_visualizer_no_data_provided(self, tmpdir, mcs_complex_sim_results):
        mcviz = MonteCarloVisualizer(tmpdir)
        with pytest.raises(ValueError, match='No simulations to visualize'):
            mcviz.plotsims('position1')

        with pytest.raises(ValueError, match='No simulations to visualize'):
            mcviz.plot3D('position1', 'position2', 'velocity1')

        with pytest.raises(ValueError, match='No statistics to visualize'):
            mcviz.distplot('position1')

        with pytest.raises(ValueError, match='No statistics to visualize'):
            mcviz.scatter('position1', 'position2')

    def test_montecarlo_visualizer_only_sims_provided(self, tmpdir, mcs_complex_sim_results):
        mcviz = MonteCarloVisualizer(tmpdir, mcs_complex_sim_results)

        mcviz.plotsims('position1')
        mcviz.plot3D('position1', 'position2', 'velocity1')

        with pytest.raises(ValueError, match='No statistics to visualize'):
            mcviz.distplot('position1')

        with pytest.raises(ValueError, match='No statistics to visualize'):
            mcviz.scatter('position1', 'position2')

    def test_montecarlo_visualizer_only_statistics_provided(self, tmpdir, mcs_complex_sim_stats_results):
        mcviz = MonteCarloVisualizer(tmpdir, sim_statistics=mcs_complex_sim_stats_results)

        with pytest.raises(ValueError, match='No simulations to visualize'):
            mcviz.plotsims('position1')

        with pytest.raises(ValueError, match='No simulations to visualize'):
            mcviz.plot3D('position1', 'position2', 'velocity1')

        mcviz.distplot('position1')
        mcviz.scatter('position1', 'position2')

    def test_scatter_reference_pt_no_statsref(self, tmpdir, mcs_complex_sim_stats_results):
        mcviz = MonteCarloVisualizer(tmpdir, sim_statistics=mcs_complex_sim_stats_results)
        mcviz.scatter('position1', 'position2')
        mcviz.scatter('position1', 'position2', ref_pt=(0.5, 0.5))
        mcviz.scatter('position1', 'position2', ref_pt=(0.5, 0.6))

    def test_scatter_reference_pt_with_statsref(self, tmpdir, mcs_complex_sim_stats_results):
        stats = mcs_complex_sim_stats_results[mcs_complex_sim_stats_results['sim_name'] == MonteCarloSimulator.NOMINAL]
        del stats['sim_name']
        mcviz = MonteCarloVisualizer(
            tmpdir,
            sim_statistics=mcs_complex_sim_stats_results,
            stats_ref=stats
        )
        mcviz.scatter('position1', 'position2')
        mcviz.scatter('position1', 'position2', ref_pt=(0.5, 0.5))
        mcviz.scatter('position1', 'position2', ref_pt=(0.5, 0.6))

    def test_distplot_reference_pt_no_dataref(self, tmpdir, mcs_complex_sim_stats_results):
        mcviz = MonteCarloVisualizer(tmpdir, sim_statistics=mcs_complex_sim_stats_results)
        mcviz.distplot('position1')
        mcviz.distplot('position1', ref_pt=0.5)
        mcviz.distplot('position1', ref_pt=0.6)

    def test_distplot_reference_pt_with_statsref(self, tmpdir, mcs_complex_sim_stats_results):
        stats = mcs_complex_sim_stats_results[mcs_complex_sim_stats_results['sim_name'] == MonteCarloSimulator.NOMINAL]
        del stats['sim_name']
        mcviz = MonteCarloVisualizer(
            tmpdir,
            sim_statistics=mcs_complex_sim_stats_results,
            stats_ref=stats
        )
        mcviz.distplot('position1')
        mcviz.distplot('position1', ref_pt=0.5)
        mcviz.distplot('position1', ref_pt=0.6)
