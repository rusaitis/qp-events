import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# from collections import OrderedDict
from matplotlib import dates
from matplotlib import rc
from scipy.integrate import simps
from numpy import trapz
from cassinilib import Plot
import copy
# for ax in np.ravel(ax_array):
        # _profile(ax, np.arange(50), np.random.rand(50))
#=============================================================
#                       PLOT MAG DATA
#=============================================================

clist = ['#12d5ae', '#f29539', '#f26b59', '#fef0b3', '#38ff81', '#cf9bff', '#fdf33c', '#ff396a']

def hourRounder(t, even=True):
    h = t.hour
    if t.second > 0:
        h += 1
    if h%2 != 0:
        h += 1
    if h >= 24:
        h = h - 24
    return t.replace(second=0, microsecond=0, minute=0, hour=h)

def magnetospherePosition(time, MLoc_time, MLoc, returnIndex=False):
    ind = np.argwhere(MLoc_time >= time)[0][0]
    loc = MLoc[ind]
    if loc == 1:
        cl = clist[0]
        label = 'MS'
    elif loc == 2:
        cl = clist[1]
        label = 'SH'
    elif loc == 3:
        cl = clist[2]
        label = 'SW'
    else:
        cl = 'black'
        label = '--'
    returnValues = [label, cl]
    if returnIndex:
        returnValues.append(ind)
    return returnValues

# yrang = 0.003
# ymin, ymax = ax.get_ylim()
# ymid = np.mean([ymin,ymax])
# ax.set_ylim([ymid - yrang/2 , ymid + yrang/2])
# ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.005))
def plotInfoAxis(ax,
                 series,
                 coords=None,
                 lt=None,
                 major_locator_hours=2,
                 minor_locator_hours=1,
                 xaxis_format = "%H:%M",
                 fontsize = 9,
                 MLoc = None,
                 MLoc_time = None,
                 ):

    # Inerpolation?
    # time = [DatetimeFunctions.UnixTime(t) for t in self.datetime]
    # data = data.tolist()
    # f = interp1d(time, data, kind='cubic')
    # time = [time[0] + self.dt*n for n in range(0, N)]
    # timeDatetime = [self.datetime[0]
                       # + datetime.timedelta(seconds=self.dt*n)
                       # for n in range(0, N)]
    # data = f(time)


    series = np.atleast_1d(series)
    # plotVars = np.atleast_1d(plotVars)
    # time = np.atleast_1d(time)
    # s = series[0]
    timeaxis = series[0].datetime
    # ax.plot([1, 2, 5], [4, 7, 3])

    coords = [s for s in series if s.kind == 'Coord']
    # LTS =    [s for s in timeseries if s.kind == 'lt']
    # FIELDS = [s for s in timeseries if s.kind == 'Field']

    # coord_names = []
    # for coord in coords:
        # coord_names.append(coord.name)

    majorHourLocator = dates.HourLocator(interval=major_locator_hours)
    minorHourLocator = dates.HourLocator(interval=minor_locator_hours)
    dfmt = dates.DateFormatter(xaxis_format)
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_major_locator(majorHourLocator)
    ax.xaxis.set_minor_locator(minorHourLocator)

    # Set the tick positions
    ax.set_yticks([0,1,2,3])
    # Set the tick labels
    ax.set_yticklabels(['Location', 'X', 'Y', 'Z'])
    ax.set_xticklabels([])
    # ax.get_yticklabels().set_fontsize(20)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    dateFrom = timeaxis[0]
    timeaxis = np.asarray(timeaxis)   #BUG TODO why is it no longer array?
    bbox_props = dict(boxstyle="round, pad=0.15", ec=None, lw=0, fc='orange', alpha=0.4)
    time_delta = datetime.timedelta(hours=major_locator_hours)
    current_time = hourRounder(dateFrom)
    while current_time < timeaxis[-1]:
        ind = np.argwhere(timeaxis >= current_time)[0][0]
        for i, coord in enumerate(coords):
            ax.text(current_time, i+1, '{:.2f}'.format(coord.y[ind]),
                    ha='center', va='center', color='grey', fontsize=fontsize)
        loc, cl = magnetospherePosition(current_time, MLoc_time, MLoc)
        bbox_props['fc'] = cl
        ax.text(current_time, 0, loc, ha='center', va='center', color='white', bbox=bbox_props, fontsize=fontsize)
        current_time += time_delta

    ax.tick_params(which='both', width=0)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set(frame_on=False)
    ax.set_xlim(timeaxis[0], timeaxis[-1])
    ax.set_ylim(0,4)

def findSeriesRange(series):
    yranges = []
    series = np.atleast_1d(series)
    for i, s in enumerate(series):
        data = s.y
        if data is not None:
            ymax = np.amax(data)
            ymin = np.amin(data)
            yranges.append(ymax-ymin)
    return max(yranges)

def findSeriesLim(series, selection=None):
    ymax = []
    ymin = []
    series = np.atleast_1d(series)
    for i, s in enumerate(series):
        data = s.y
        if selection:
            data = data[selection[0] : selection[1]]
        if data is not None:
            ymax.append(np.amax(data))
            ymin.append(np.amin(data))
    return [min(ymin), max(ymax)]

def findSeriesAverage(series):
    ymids = []
    for n, s in enumerate(series):
        ymids.append(np.average(s.y))
    return ymids

def plotEachTimeSeries(ax,
                       series,
                       plotVars=None,
                       time=None,
                       **plot_kwds,
                       ):
    ''' Plot Each Time Series to the provided axis '''
    changeColor = False if 'color' in plot_kwds else True
    if plotVars is None:
        plotVars = [['y'] for s in series]
    for n, s in enumerate(series):
        # If provided, plot a different variable for each time series
        var = plotVars[n] if plotVars.size > 1 else plotVars[0]

        tAxis = s.datetime
        yAxis = s.y
        label = s.name

        # If alternative Time Axis is given, overwrite tAxis
        if time is not None:
            tAxis = time[n] if time.size > 1 else time[0]

        # Overwrite color with Time Series color property
        changeColor = True
        if changeColor:
            plot_kwds['color'] = s.color

        # Plot
        timeplot = ax.plot(tAxis,
                           yAxis,
                           label=label,
                           **plot_kwds,
                           )
    return timeplot


def PlotTimeseries(ax,
                   series,
                   plotVars=['y'], # E.g., 'run_avg', 'peaks', 'hl'
                   time=None,
                   ax2series=None,
                   ax2plotVars=None,
                   ax2time=None,
                   highlight=None,
                   highlight_color=None,
                   ylim=None,
                   xlim=None,
                   yrange=None,
                   ymid=None,
                   ylabel=None,
                   xlabel=None,
                   plot_kwds={'ls' : '-',
                              'alpha' : 1},
                   plot2_kwds={'ls' : '--',
                               'alpha' : 0.8},
                   major_locator_hours=2,
                   minor_locator_hours=1,
                   fs=12,
                   plot_peaks=False,
                   plotZeroAxis=False,
                   minimal=True,
                   show_xticks=True,
                   xaxis_format = "%H:%M",
                   ):
    series = np.atleast_1d(series)
    plotVars = np.atleast_1d(plotVars)
    if time is not None:
        time = np.atleast_1d(time)

    # Plot Every Timeseries Inputted
    plotEachTimeSeries(ax,
                       series,
                       plotVars,
                       time=time,
                       **plot_kwds,
                       )

    # (Optional) Second Axis on the Right
    if ax2series is not None:
        axr=ax.twinx()
        plotEachTimeSeries(axr,
                           ax2series,
                           ax2plotVars,
                           time=ax2time,
                           **plot2_kwds,
                           )

    if plotZeroAxis:
        ax.axhline(y=0, ls='--', lw=2, c='black', alpha=0.2)

    if plot_peaks:
        s = series[0]
        timeaxis = s.datetime[s.Nmargin : s.N - s.Nmargin]
        properties = s.properties
        peaks = s.peaks
        yaxis = s.y
        ax.plot(timeaxis[peaks], yaxis[peaks], "x", color=s.color)
        ax.vlines(x=timeaxis[peaks], ymin=yaxis[peaks] - properties["prominences"],
               ymax = yaxis[peaks], color = "orange")
        # ax.hlines(y=properties["width_heights"], xmin=timeaxis[np.round(properties["left_ips"]).astype(int)],
               # xmax=timeaxis[np.round(properties["right_ips"]).astype(int)], color = "orange")
        ax.hlines(y=properties["width_heights"], xmin=timeaxis[np.round(properties["left_bases"]).astype(int)],
               xmax=timeaxis[np.round(properties["right_bases"]).astype(int)], color = "orange")

        for i, pk in enumerate(peaks):
            peak_height = properties["prominences"][i]
            peak_width = (timeaxis[np.round(properties["right_bases"][i]).astype(int)] -
                         timeaxis[np.round(properties["left_bases"][i]).astype(int)])
            peak_width = peak_width.total_seconds() / 60
            ax.text(timeaxis[pk], yaxis[pk],
                    '%.2f nT \n %.0f min' %(peak_height, peak_width),
                    alpha=1, color='#ff6a00')

        peaks = s.peaksCWT
        ax.plot(timeaxis[peaks], yaxis[peaks], "o", color='black', ms=10, alpha=0.3)
        ax.axvline(x=timeaxis[peaks[0]], color='black', ls='--', alpha=0.4)
        for i in range(1, len(peaks)):
            ax.axvline(x=timeaxis[peaks[i]], color='black', ls='--', alpha=0.4)
            interval = timeaxis[peaks[i]] - timeaxis[peaks[i-1]]
            interval_min = interval.total_seconds() / 60
            ax.text(timeaxis[peaks[i-1]] + interval/2, ax.get_ylim()[0],
                    '%.0f min' % interval_min, color='black', alpha=0.4)

    # Draw Highlights (given as a times array) as vertical lines
    if highlight is not None:
        highlight = np.atleast_1d(highlight)
        hlcolor = 'black' if highlight_color is None else highlight_color
        highlight_color = np.atleast_1d(highlight_color)
        # hlcolor = np.asarray(hlcolor)
        for i, hl in enumerate(highlight):
            cl = hlcolor[i] if len(hlcolor) > 1 else hlcolor[0]
            ax.axvline(x=hl, ls='--', lw=2, c=cl, alpha=0.7)

    # Formatting and Styling
    # GRID
    majorHourLocator = dates.HourLocator(interval=major_locator_hours)
    minorHourLocator = dates.HourLocator(interval=minor_locator_hours)
    dfmt = dates.DateFormatter(xaxis_format)
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_major_locator(majorHourLocator)
    ax.xaxis.set_minor_locator(minorHourLocator)

    # xtickslocs = ax.get_xticks()

    # ax.xaxis.set_minor_locator(onehour)
    # ax.grid(b=True, which='minor', alpha=0.4, color='grey', linestyle='-')
    # ax.grid(b=True, which='major', alpha=0.4, color='grey', linestyle='-')
# 

    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.7)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)


    # LABELS
    ylabel = series[0].units if ylabel is None else ylabel
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    # LEGENDS
    ax.legend(loc=2, frameon=False, fontsize=14)
    if ax2series is not None:
        axr.legend(loc=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='x', labelsize=14)

    if not show_xticks:
        ax.set_xticklabels([])
        ax.set_xlabel(None)

    if minimal:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set(frame_on=False)
    # Plot Limits
    if ymid is None:
        ymid = findSeriesAverage(series)[0]
    if yrange is not None:
        yrange*=1.1
        ax.set_ylim([ymid-yrange/2,
                     ymid+yrange/2])
    if ylim is not None:
        yrange = ylim[1] - ylim[0]
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        s = series[0]
        time = s.datetime
        ax.set_xlim(time[0], time[-1])
    if ax2series is not None:
        if yrange is not None:
            ymid = 0
            axr.set_ylim([ymid-yrange/2, ymid+yrange/2])

    # ticks = [tick for tick in plt.gca().get_xticklabels()]
    # x_labels = list(ax.get_xticklabels(which='both'))
    # x_label_dict = dict([(x.get_text(), x.get_position()[0]) for x in x_labels])
    # print([x.get_text() for x in x_labels])
    # print(x_labels)
    # print(ticks)
    # ax.label_outer()
    # background_std = np.std(series.y)
    # ax.axhline(y=background_std, color='black', alpha=0.7, linestyle='--')
    # ax.axhline(y=-background_std, color='black', alpha=0.7, linestyle='--')


def plot_field_interpolation(FIELDS, gap_list, cal_list):
    ''' Plot interpolated and origina fields, together with gap
    and scas times '''

    clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0',
                   '#cf9bff', '#fdf33c',  '#ff396a']

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 26})
    plt.style.use('ggplot')
    for i, s in enumerate(FIELDS):
        cl = clist_field[i]
        plt.scatter(s.datetime_, s.y_, color=cl, marker='x', s=12,
                    alpha=0.9, label=s.name + ', Original')
        plt.plot(s.datetime, s.y, color=cl, ls='--', lw=1.5, marker='o',
                 ms=4, alpha=0.5, label=s.name + ', Resampled')
    for gap in gap_list:
        plt.axvspan(gap[0], gap[1], alpha=0.2, color='red')
    for gap in cal_list:
        plt.axvspan(gap[0], gap[1], alpha=0.2, color='gray')

    majorHourLocator = dates.HourLocator(interval=2)
    minorHourLocator = dates.HourLocator(interval=1)
    dfmt = dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_major_locator(majorHourLocator)
    ax.xaxis.set_minor_locator(minorHourLocator)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(b=True, which='major', color='black', linestyle='--', alpha=0.2)
    ax.grid(b=True, which='minor', color='black', linestyle=':', alpha=0.1)
    plt.legend()
    plt.title('Date: {} (24-hour Data Segment in KRTP Coords)'.format(FIELDS[0].datetime[0].strftime("%Y-%m-%d")))
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('B [nt]', fontsize=14)
    plt.show()

def plot_field_aligned_test(COORDS, FIELDS, BFIELDS=None):
    clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0',
                   '#cf9bff', '#fdf33c',  '#ff396a']

    # fig = plt.figure(figsize=(8,4))
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(14,10))
    plt.rcParams.update({'font.size': 26})
    plt.style.use('ggplot')
    for i, s in enumerate(FIELDS):
        cl = clist_field[i]
        ax1.plot(s.datetime, s.y, color=cl, marker='o', ms=1,
                    alpha=0.9, label=s.name)

        ax1.plot(s.datetime, s.run_avg, color=cl, ls='--',
                    alpha=0.8, label=s.name + ' BG')


    for i, s in enumerate(FIELDS):
        cl = clist_field[i]
        ax2.plot(s.datetime, s.dy, color=cl, marker='o', ms=1,
                    alpha=0.9, label=s.name + ' perturbation')

    if BFIELDS is not None:
        for i, s in enumerate(BFIELDS):
            cl = clist_field[i]
            ls = '-' if i<3 else '--'
            ax3.plot(FIELDS[0].datetime, s.y, color=cl, marker='o', ms=1,
                     ls=ls, alpha=0.9, label=s.name)

    majorHourLocator = dates.HourLocator(interval=2)
    minorHourLocator = dates.HourLocator(interval=1)
    dfmt = dates.DateFormatter("%H:%M")
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_major_locator(majorHourLocator)
        ax.xaxis.set_minor_locator(minorHourLocator)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(b=True, which='major', color='black', linestyle='--', alpha=0.2)
        ax.grid(b=True, which='minor', color='black', linestyle=':', alpha=0.1)
        ax.legend()
        ax.legend()
        ax.legend()
        ax.set_ylabel('B [nt]', fontsize=14)
    fig.suptitle('Transformation to Field-Aligned Coordinates for Synthetic Data')
    ax3.set_xlabel('Time', fontsize=14)
    plt.show()
