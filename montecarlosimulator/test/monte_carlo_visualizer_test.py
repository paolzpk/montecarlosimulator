from montecarlosimulator import MonteCarloVisualizer


class TestMonteCarloVisualizer:
    def test_montecarlo_visualizer_only_sims_provided(self, tmpdir, mcs_complex_sim_results):
        mcviz = MonteCarloVisualizer(tmpdir, mcs_complex_sim_results)
        mcviz.plotsims('position1')
