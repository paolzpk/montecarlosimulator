"""
List of things to do
TODO implement
    - extract groups of plots in functions or files
    - add a way to run from command line
TODO read the units from the input data instead of trying to guess
TODO find a better way to change units of measure (is there a lib to do so? probably yes)
TODO add at least one test (using pytest ideally)
"""
from collections.abc import Sequence
from typing import NamedTuple, Callable, Iterable

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

from monte_carlo_simulator import MonteCarloSimulator

if mpl.__version__ == '3.0.2':
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from scipy import stats

import pandas as pd
import numpy as np

from functools import partial, reduce
from functools import update_wrapper
from collections import OrderedDict
from pathlib import Path
from numbers import Number

import re


class FigureCreator:
    def __init__(self, func, figsize=(15, 8)):
        # Setting default settings for matplotlib and seaborn
        # Latex-style font
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'

        # Seaborn style - dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        sns.set_style("whitegrid")

        self.figsize = figsize
        self.func = func
        update_wrapper(self, func, "save_dir")

    def __call__(self, instance, *args, **kwargs):
        plt.figure(figsize=self.figsize)
        axes = self.func(instance, *args, **kwargs)
        if not isinstance(axes, list):
            axes = [axes]
        self.__format_figure(axes)

        title = axes[0].get_title()

        if not title:
            sub_str1 = re.sub(r'\[.+?\]', '',
                              axes[0].get_ylabel().replace('$', '').replace(r'\var', '').replace('\\', '').replace(',',
                                                                                                                   ''))
            sub_str2 = re.sub(r'\[.+?\]', '',
                              axes[0].get_xlabel().replace('$', '').replace(r'\var', '').replace('\\', '').replace(',',
                                                                                                                   ''))
            title = sub_str1 + ' vs ' + sub_str2
            axes[0].set_title(title, fontsize=18)
            print(f'Using generated title {title}')

        if not title:
            UserWarning("Title not set! cannot save the picture")
            return axes[0]

        save_dir = Path(instance.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / title
        print(f'Saving in {save_path}')
        try:
            plt.savefig(save_path, transparent=True)
        except Exception as ex:
            print()
            raise type(ex)('An error occurred, try specifying the labels of the axes. ' + str(ex))

        if len(axes) == 1:
            return axes[0]
        elif len(axes) == 2:
            return axes[0], axes[1]

    def __get__(self, instance, owner):
        return partial(self, instance)

    def __format_figure(self, axes):
        pass
        # plt.gcf().tight_layout()


class MonteCarloVisualizer:
    BUNDLE_LINE_NUMBER_LIMIT = 1000
    NOMINAL = -1

    def __init__(self, save_dir, /, sim_results=None, sim_statistics=None, *, data_ref=None, stats_ref=None, ci=0.95,
                 bundle_size=30, experimental=False):
        self.sims_stats = sim_statistics
        self.sims = sim_results
        self.data_ref = data_ref
        self.stats_ref = stats_ref

        if self.data_ref is not None:
            self.data_ref = self.data_ref.loc[:, ~self.data_ref.columns.duplicated()]

        self.save_dir = save_dir

        if experimental:
            if ci < 0 or ci > 1:
                ValueError('Confidence interval must be between 0 and 1')
            self.ci = ci
            self.bundle_size = bundle_size
            self.random_sims = self.__randomly(range(0, self.sim_number))[0:min(self.sim_number, self.bundle_size)]

    @staticmethod
    def __regression_line(x, intercept, slope):
        return x, intercept + slope * x

    def __confidence_interval(self, x, y, intercept, slope, ci=0.05):
        p = 1 - ci / 2
        _, y1 = MonteCarloVisualizer.__regression_line(x, intercept, slope)
        N = x.size
        x2 = np.linspace(np.min(x), np.max(x), N)
        _, y2 = MonteCarloVisualizer.__regression_line(x2, intercept, slope)
        df = N - 2
        q_t = stats.t.ppf(p, df)
        s_err = y.std(ddof=2)
        c = q_t * s_err * np.sqrt(1 / N + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        return x2, y2 + c, y2 - c, y

    def __randomly(self, seq):
        from random import shuffle
        shuffled = list(seq)
        shuffle(shuffled)
        return shuffled

    @FigureCreator
    def probplot_ci(self, x=None, data=None, dist=stats.norm, ci=0.05):
        """
        This function is not tested and the results are potentially wrong
        """

        if not self.experimental:
            ValueError(
                'Experimental flag must be turned on to use this function. '
                'Try defining ev = MonteCarloVisualizer(..., experimental=True)')

        if data is None:
            raise ValueError('No data specified')

        if isinstance(x, str):
            if data is None:
                raise ValueError('You must specify the data type!')
            if isinstance(data, str):
                if not isinstance(x, str):
                    raise ValueError('You must specify the data to use!')
                else:
                    raise NotImplementedError('Implement it!')
            data = data[x]

        probplot_data, probplot_fit = stats.probplot(data, dist=dist)  # plot=sns.mpl.pyplot
        qqplot_data_x, qqplot_data_y = probplot_data
        qqplot_fit_slope, qqplot_fit_intercept, qqplot_fit_r = probplot_fit

        ci_data = self.__confidence_interval(qqplot_data_x, qqplot_data_y, qqplot_fit_intercept, qqplot_fit_slope,
                                             ci=self.ci)

        axes = plt.axes()
        axes.plot(ci_data[0], ci_data[3])
        axes.scatter(qqplot_data_x, qqplot_data_y)
        axes.fill_between(ci_data[0], ci_data[1], ci_data[2], interpolate=True, alpha=0.4)

        # TODO add label with the confidence interval, and qqplot_fit_r
        x_text = min(axes.get_xlim()) + np.diff(axes.get_xlim()) * 0.1
        y_text = min(axes.get_ylim()) + np.diff(axes.get_ylim()) * 0.75
        #     axes.title('QQ Plot')
        axes.text(x_text, y_text, f"Confidence Interval {(1 - ci) * 100}%\nR={qqplot_fit_r}",
                  fontsize=18,
                  bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2, 'boxstyle': 'round'})
        return axes

    replace_pairs = [('theta', r'\vartheta'), ('phi', r'\varphi'), ('psi', r'\psi'), ('delta', r'\Delta'),
                     ('time', r't'), ('altitude', r'h'),
                     ('vx_r', r'v_{x_r}'), ('vy_r', r'v_{y_r}'), ('vz_r', r'v_{z_r}'),
                     ('vx_b', r'v_{x_b}'), ('vy_b', r'v_{y_b}'), ('vz_b', r'v_{z_b}'), ]

    def __infer_label(self, src, factor, delta=False):
        par = src.replace('$', '').lower()

        for key, val in self.replace_pairs:
            par = par.replace(key, val)

        if any(angle in par for angle in ['phi', 'theta', 'psi', 'beta', 'alpha']):
            unit = r'^{\circ}'
        elif 'omega' in par:
            unit = r'^{\circ}/s'
        elif any(speed in src.upper() for speed in ['VX', 'VY', 'VZ', 'V_X', 'V_Y', 'V_Z']):
            unit = 'm/s'
        elif any(length in par for length in ['x', 'y', 'z']) or 'altitude' in src.lower():
            unit = r'm'
        elif 'time' in src.lower() or 't' == src.lower():
            unit = 's'
        else:
            unit = ''

        if factor == 1e-3:
            unit = 'k' + unit

        if delta:
            par = r'\Delta ' + par

        out = '$' + par + '\,[' + unit + ']$'
        # print(f'Generated label: {out}')
        return out

    @FigureCreator
    def plot3D(self, x, y, z, data=None, data_ref=None, ax=None, title=None, xfac=1, yfac=1, zfac=1, xlabel='infer',
               ylabel='infer', zlabel='infer', sim_filter=None, view=None):
        data, data_ref, sim_filter = self.__prepare_data(data, data_ref, sim_filter, x, y)
        filtered_data = self.filter_data(data, sim_filter, [x, y, z])

        if ax is None:
            ax = plt.axes(projection='3d')

        for i_sim in np.append(data['sim_name'].unique(), MonteCarloVisualizer.NOMINAL):
            label, palette_n = self.__plot_details_handler(i_sim)
            _data = filtered_data[(filtered_data['sim_name'] == i_sim)]
            ax.plot3D(_data[x].values * xfac, _data[y].values * yfac, _data[z].values * zfac,
                      color=sns.color_palette()[palette_n],
                      label=label, alpha=0.8)
            if view is not None:
                ax.view_init(**view)

        self.__handle_label_and_title(ax, title, x, xfac, xlabel, y, yfac, ylabel, z, zfac, zlabel)
        self.__remove_duplicate_legend_entries(ax)

        return ax

    @FigureCreator
    def plotsims(self, y, x='time', data=None, data_ref=None, ax=None, title=None, xfac=1, yfac=1, yyfac=1,
                 xlabel='infer',
                 ylabel='infer', yylabel='infer', sim_filter=None, fill_between=False, plot_deltas=True):
        """
        :param y: string with the name of the column to plot on y axis or data vector (numpy.array, pandas.Series etc.)
            with values to plot on y axis
        :param x: string with the name of the column to plot on x axis or data vector (numpy.array, pandas.Series etc.)
            with values to plot on x axis
        :param data: optional, DataFrame with columns to use for main data, defaults to the sims Dataframe of
            MonteCarloVisualizer
        :param data_ref: optional, DataFrame with columns to use for reference data, defaults to the data_ref
        Dataframe of MonteCarloVisualizer
        :param ax: optional, axis on which to plot
        :param title: optional, title of the plot, if not provided it is inferred
        :param xfac: scale factor for the x axis
        :param yfac: scale factor for the y axis
        :param yyfac: scale factor for the yy axis
        :param xlabel: optional, x label of the plot, if not provided it is inferred
        :param ylabel: optional, y label of the plot, if not provided it is inferred
        :param yylabel: optional, yy label of the plot, if not provided it is inferred
        :param sim_filter: optional, list of tuples used to filter data to plot of the form
                           [('name_of_parameter_to_filter', 'value_filtered')],
                           it defaults to plot simulated data and not navigation data
        :param fill_between:
        :param plot_deltas:
        :return: list of axes used to plot
        """
        data, data_ref, sim_filter = self.__prepare_data(data, data_ref, sim_filter, x, y)
        filtered_data = self.filter_data(data, sim_filter, [x, y])

        if ax is None:
            ax = plt.axes()

        if not fill_between:
            # lines = []
            # for i_sim in data['sim_name'].unique():
            #     _data = data[(data['sim_name'] == i_sim) & _filter]
            #     lines.append(_data[x] * xfac)
            #     lines.append(_data[y] * yfac)
            #
            # ax.plot(*lines)

            # _x_data = []
            # _y_data = []
            # for i_sim in data['sim_name'].unique():
            #     _data = data[(data['sim_name'] == i_sim) & _filter]
            #     _x_data.extend(_data[x].values * xfac)
            #     _x_data.append(None)
            #     _y_data.extend(_data[y].values * yfac)
            #     _y_data.append(None)
            #
            # ax.plot(_x_data, _y_data, color=sns.color_palette()[1], label='Monte Carlo', alpha=0.8)
            #
            # _data = data[(data['sim_name'] == 'nominal') & _filter]
            # ax.plot(_data[x], _data[y], color=sns.color_palette()[3], label='Nominal', alpha=0.8)

            # Append nominal simulation as last to plot to make sure it is plotted on top of everything else
            for _, i_sim in zip(range(0, MonteCarloVisualizer.BUNDLE_LINE_NUMBER_LIMIT), data['sim_name'].unique()):
                label, palette_n = self.__plot_details_handler(i_sim)
                _data = filtered_data[(filtered_data['sim_name'] == i_sim)]
                ax.plot(_data[x].values * xfac, _data[y].values * yfac, color=sns.color_palette()[palette_n],
                        label=label, alpha=0.8)
            else:
                label, palette_n = self.__plot_details_handler(MonteCarloVisualizer.NOMINAL)
                _data = filtered_data[(filtered_data['sim_name'] == MonteCarloVisualizer.NOMINAL)]
                if len(_data[x].values) > 0:
                    ax.plot(_data[x].values * xfac, _data[y].values * yfac, color=sns.color_palette()[palette_n],
                            label=label, alpha=0.8)
        else:
            NotImplementedError('Code below should be updated: it does not distinguish nav and real data')
            # ax.fill_between(x=data[x].unique() * xfac,
            #                 y1=[data[data[x] == x_i][y].max() * yfac for x_i in data[x].unique()],
            #                 y2=[data[data[x] == x_i][y].min() * yfac for x_i in data[x].unique()],
            #                 color=sns.color_palette()[1], label='Monte Carlo', alpha=0.8)

        self.__handle_label_and_title(ax, title, x, xfac, xlabel, y, yfac, ylabel)

        if data_ref is None or x not in data_ref.columns or y not in data_ref.columns:
            self.__remove_duplicate_legend_entries(ax)
            return [ax]

        ax.plot(data_ref[x] * xfac, data_ref[y] * yfac,
                linestyle='dashdot', label='Ref', alpha=0.8, color=sns.color_palette()[9])

        if not plot_deltas:
            self.__remove_duplicate_legend_entries(ax)
            return [ax]

        yyax = ax.twinx()

        if not fill_between:
            data_ref_len = self.data_ref['Time'].size
            # _x_data = []
            # _y_data = []
            # for i_sim in data['sim_name'].unique():
            #     y_data = data[(data['sim_name'] == i_sim) & _filter][y]
            #     data_size = min(data_ref_len, y_data.size)
            #     _x_data.extend(data_ref[x][:data_size])
            #     _x_data.append(None)
            #     _y_data.extend((data_ref[y][:data_size] - y_data[:data_size].values) * yyfac)
            #     _y_data.append(None)
            #
            # yyax.plot(_x_data, _y_data, color=sns.color_palette('pastel')[1], label='Monte Carlo', alpha=0.8)

            # Append nominal simulation as last to plot to make sure it is plotted on top of everything else
            for i_sim in np.append(data['sim_name'].unique(), MonteCarloVisualizer.NOMINAL):
                y_data = filtered_data[filtered_data['sim_name'] == i_sim][y]
                data_size = min(data_ref_len, y_data.size)
                label, palette_n = self.__plot_details_handler(i_sim)

                yyax.plot(data_ref[x][:data_size].values,
                          (data_ref[y][:data_size].values - y_data[:data_size].values) * yyfac,
                          color=sns.color_palette("pastel")[palette_n],
                          label=label,
                          alpha=0.8)
        else:
            NotImplementedError('This is not implemented yet! if you want it, code it! '
                                'Below find something that I don\'t know why it does not work ;)')

        yyax.grid(False)

        # Fixing legends
        self.__remove_duplicate_legend_entries(ax, title='Axe gauche', loc='upper left')
        self.__remove_duplicate_legend_entries(yyax, title='Axe droite (difference ref/sim)', loc='upper right')

        if yylabel == 'infer':
            yyax.set_ylabel(self.__infer_label(y, yyfac, delta=True), fontsize=18)
        yyax.yaxis.label.set_color(sns.color_palette()[1])
        yyax.tick_params(axis='y', colors=sns.color_palette()[1])

        return [ax, yyax]

    def __prepare_data(self, data, data_ref, sim_filter, x, y):
        if sim_filter is None:
            sim_filter = []
        if data is None:
            if self.sims is None:
                raise ValueError('No simulations to visualize')
            data = self.sims
        if data_ref is None:
            data_ref = self.data_ref
        else:
            data_ref.columns = [x, y]
        return data, data_ref, sim_filter

    class MCSFilterData(NamedTuple):
        what: str
        on_value: object
        comparison: Callable = None

    @staticmethod
    def filter_data(data, sim_filter: Iterable[MCSFilterData], cols=None):
        if cols is None or not isinstance(cols, list):
            ValueError(
                "Argument 'cols' is mandatory and should be a list of strings with the names of the columns to select")
        cols.append('sim_name')

        if not sim_filter:
            return data[cols]

        # TODO implement here filtering not based only on equality
        _filter = reduce(lambda x, y: x & y, [data[column] == value for column, value in sim_filter])
        filtered_data = data[_filter][cols]
        filtered_data.reindex()
        return filtered_data

    @staticmethod
    def __remove_duplicate_legend_entries(ax, **kwarg):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), **kwarg)

    @staticmethod
    def __plot_details_handler(i_sim):
        palette_n = 1
        if i_sim != MonteCarloVisualizer.NOMINAL:
            label = 'Monte Carlo'
        elif i_sim == MonteCarloVisualizer.NOMINAL:
            label = 'Nominal'
            palette_n = 3
        else:
            label = ''
        return label, palette_n

    @FigureCreator
    def distplot(self, parameter, data=None, factor=1, title=None, xlabel='infer', legend_labels_override=None,
                 hue='Type', kde=True, data_ref=None, ax=None, ref_pt=None, **kwargs):
        """
        Plots the histogram and kde estimation of the given parameter.
        :param hue: string, list of strings or list of tuples containing the different datasets to be plotted.
                    Example:
                    - hue='Type' plots different distributions for all data[hue].unique()
                    - hue=[('Type', ['Type1', 'Type2'])] plots different distribution for
                    data[data[hue[0][0] == data[hue[0][1]], i.e. data[data['Type'] == ['Type1', 'Type2'].
                    -

        """
        data, hue = self.__prepare_stats_data(data, hue)

        # Copying only the necessary columns. Copying because the original df must not be modified
        necessary_columns = [parameter, 'sim_name'] if hue is None else [parameter, hue, 'sim_name']
        _data = data[necessary_columns].dropna().reset_index(drop=True).copy()
        _data.loc[:, parameter] = _data.loc[:, parameter] * factor

        ax = plt.axes() if ax is None else ax
        if sns.__version__ == '0.9.0':
            for typ in _data[hue].unique():
                # seaborn is bugged and does not accept the pd.DataFrame nor the underlying np.array
                a = np.array(_data[_data[hue] == typ][parameter].reset_index(drop=True).to_numpy().tolist())
                ax = sns.distplot(a=a, kde=kde, hist=True, rug=True, label=typ,
                                  rug_kws=dict(alpha=0.3), **kwargs)
                ax.legend(loc='best')
        else:  # sns.__version__ == '0.11.0'
            ax = sns.histplot(data=_data, x=parameter, kde=kde, hue=hue, ax=ax, **kwargs)
            ax = sns.rugplot(data=_data, x=parameter, hue=hue, ax=ax, alpha=0.3, **kwargs)

        averages, stddevs = self.__get_stats(_data, hue, parameter)

        stats_strings = [f"$\mu$ = ${avg:.2f}$ ; $\sigma$ = ${stddev:.2f}${' (' + dye + ')' if hue is not None else ''}"
                         for (dye, avg), stddev in zip(averages.items(), stddevs.values())]

        self.__add_anchored_text(ax, stats_strings)

        ref_pt = self.__get_reference_point(ref_pt, self.stats_ref, necessary_columns)
        nominal_pt = self.__get_nominal_point(_data)
        self.__plot_reference_and_nominal_pt(ax, (nominal_pt, 0), (ref_pt, 0), marker='v', xfac=factor)

        if title is not None:
            ax.set_title(title, fontsize=18)
        else:
            ax.set_title(f'Distribution de {parameter}', fontsize=18)

        if xlabel == 'infer':
            xlabel = self.__infer_label(parameter, factor)
        ax.set_xlabel(xlabel, fontsize=18)

        # TODO adding this removes the legend for the hue, must find a way to add the legend for the ref pt
        ax.legend(loc='best')
        if legend_labels_override is not None:
            ax.legend(labels=legend_labels_override)

        return ax

    @staticmethod
    def __add_anchored_text(ax, stats_strings, loc='upper left'):
        stats_string = '\n'.join(stats_strings)
        at = AnchoredText(stats_string, frameon=True, prop=dict(size=14), loc=loc)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    @FigureCreator
    def scatter(self, x, y, ax=None, xlabel='infer', ylabel='infer', title=None, xfac=1, yfac=1, ref_pt=None,
                kdeplot=True, hue='Type', legend_labels_override=None, data=None, loc_annotation='lower left',
                **kwargs):
        """
        :param x: array-like x data or name of column of data to use on x axis
        :param y: array-like y data or name of column of data to use on y axis
        :param ax: (optional) axis on which to plot
        :param xlabel: (optional) label to use on x axis
        :param ylabel: (optional) label to use on y axis
        :param title: (optional) title of the plot
        :param xfac: (optional) scale factor to be applied to the x axis data
        :param yfac: (optional) scale factor to be applied to the y axis data
        :param ref_pt: (optional) reference point to plot
        :param kdeplot: (optional) if True, overlap a kde plot (for newer versions of seaborn not necessary...)
        :param legend_labels_override:
        :param data: (optional) pd.DataFrame with the data to plot
        :param kwargs: (optional) keyword args to be forwarded to Seaborn
        :param loc_annotation: position of the annotation with the statistics (matplotlib style)
        :return: axis used to plot
        """
        data, hue = self.__prepare_stats_data(data, hue)

        necessary_columns = [x, y, 'sim_name'] if hue is None else [x, y, hue, 'sim_name']
        _data = data[necessary_columns].dropna().reset_index(drop=True).copy()
        _data[x] = _data[x] * xfac
        _data[y] = _data[y] * yfac

        ax = plt.axes() if ax is None else ax
        ax = sns.scatterplot(data=_data, x=x, y=y, ax=ax, alpha=0.3, hue=hue, **kwargs, label='_nolegend_')
        try:
            if sns.__version__ == '0.9.0':
                for typ in _data[hue].unique():
                    __data = _data[_data[hue] == typ]
                    ax = sns.kdeplot(data=__data[x].dropna(), data2=__data[y].dropna(), label='_nolegend_')
            else:  # sns.__version__ >= '0.9.0'
                ax = sns.kdeplot(data=_data, x=x, y=y, ax=ax, hue=hue, **kwargs, label='_nolegend_') if kdeplot else ax
        except np.linalg.LinAlgError as ex:
            print(str(ex) + f'\nCannot plot kde for {x} and {y}. Something went wrong in the calculations! Skipping!')

        ref_pt = self.__get_reference_point(ref_pt, self.stats_ref, necessary_columns)
        nominal_pt = self.__get_nominal_point(_data)
        self.__plot_reference_and_nominal_pt(ax, nominal_pt, ref_pt, marker='P', xfac=xfac, yfac=yfac)

        averages, stddevs = self.__get_stats(_data, hue, [x, y])

        stats_strings = [f"$\mu_x$ = ${avg[x]:.2f}$ ; $\sigma_x$ = ${stddev[x]:.2f}$ ; "
                         f"$\mu_y$ = ${avg[y]:.2f}$ ; $\sigma_y$ = ${stddev[y]:.2f}${' (' + dye + ')' if hue is not None else ''}"
                         for (dye, avg), stddev in zip(averages.items(), stddevs.values())]

        self.__add_anchored_text(ax, stats_strings, loc=loc_annotation)

        self.__handle_label_and_title(ax, title, x, xfac, xlabel, y, yfac, ylabel)

        ax.legend(loc='best')
        if legend_labels_override is not None:
            ax.legend(labels=legend_labels_override)

        return ax

    @staticmethod
    def __plot_reference_and_nominal_pt(ax, nominal_pt, ref_pt, /, *, marker, xfac, yfac=1):
        if ref_pt == nominal_pt:
            ax.plot(ref_pt[0] * xfac, ref_pt[1] * yfac, ms=10, label='Reference/Nominal', marker=marker, color='r')
            ref_pt = None
            nominal_pt = None
        if ref_pt is not None and all(val is not None for val in ref_pt):
            ax.plot(ref_pt[0] * xfac, ref_pt[1] * yfac, ms=10, label='Reference', marker=marker, color='r')
        if nominal_pt is not None and all(val is not None for val in nominal_pt):
            ax.plot(nominal_pt[0] * xfac, nominal_pt[1] * yfac, ms=10, label='Nominal', marker=marker, color='b')

    @staticmethod
    def __get_reference_point(ref_pt, data, cols):
        # Trying to find the reference point:
        # - If ref_pt is provided, just return it
        # - else look in self.stats_ref
        # - If no suitable reference point could be found, return None
        # TODO add the possibility to calculate statistics on the fly (e.g. if self.statistics_calculator is not None..)
        if ref_pt is not None:
            return ref_pt

        try:
            if data is None:
                return None
            if 'sim_name' in cols:
                cols.remove('sim_name')
            assert len(data[cols].index) == 1, 'Reference data must have just one possible value (per parameter)'
            ref_pt = tuple(data[cols].iloc[0])
            return ref_pt[0] if len(ref_pt) == 1 else ref_pt
        except IndexError:
            return None
        return None

    @staticmethod
    def __get_nominal_point(data):
        try:
            nominal_data = data[data['sim_name'] == MonteCarloSimulator.NOMINAL].iloc[0]
            del nominal_data['sim_name']
            nominal_pt = tuple(nominal_data)
            return nominal_pt[0] if len(nominal_pt) == 1 else nominal_pt
        except IndexError:
            return None

    def __prepare_stats_data(self, data, hue):
        if data is None:
            if self.sims_stats is None:
                raise ValueError('No statistics to visualize')
            data = self.sims_stats
        if hue is None:
            hue = 'Type'
        if hue not in data.columns:
            hue = None
        return data, hue

    @staticmethod
    def __get_stats(_data, hue, parameters):
        if hue is None:
            averages = {'__NO_DYE__': _data[parameters].mean()}
            stddevs = {'__NO_DYE__': _data[parameters].std()}
        else:
            averages = {dye: _data[_data[hue] == dye][parameters].mean() for dye in _data[hue].unique()}
            stddevs = {dye: _data[_data[hue] == dye][parameters].std() for dye in _data[hue].unique()}
        return averages, stddevs

    def __handle_label_and_title(self, ax, title, x, xfac, xlabel, y, yfac, ylabel, z=None, zfac=None, zlabel=None):
        if not (z is not None and zfac is not None and zlabel is not None) and z is not None:
            ValueError('If you provide z, you should provide as well zfac and zlabel')

        if xlabel == 'infer':
            xlabel = self.__infer_label(x, xfac)
        ax.set_xlabel(xlabel, fontsize=18)
        if ylabel == 'infer':
            ylabel = self.__infer_label(y, yfac)
        ax.set_ylabel(ylabel, fontsize=18)
        if z is not None:
            if zlabel == 'infer':
                zlabel = self.__infer_label(z, zfac)
            ax.set_zlabel(zlabel, fontsize=18)
        if title is not None:
            ax.set_title(title, fontsize=18)
        else:
            ax.set_title(f'{y} vs {x}', fontsize=18)
