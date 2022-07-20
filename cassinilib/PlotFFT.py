import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from matplotlib import dates
from matplotlib import rc
from matplotlib import ticker
import matplotlib.style as mplstyle
import matplotlib.patches as mpatches
from matplotlib import rc
import matplotlib
import pandas as pd
import seaborn as sns
from scipy.integrate import simps
from numpy import trapz
from scipy import fft
from scipy.signal import blackman
from scipy.signal import hamming
from scipy.signal import bartlett
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.signal import ricker
from scipy.signal import kaiser
from scipy import signal
from scipy.fft import fftshift
from numpy.polynomial import Polynomial
from cassinilib import Plot
import copy as copy
import os
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as transforms
import kmag
# from cassinilib import *
from cassinilib.DatetimeFunctions import *
# import argparse
from cassinilib.Event import *
from wavesolver.sim import *
from wavesolver.fieldline import *
#=============================================================
#                       PLOT MAG DATA
#=============================================================
# clist = ['red', 'green', 'blue', 'white', 'yellow', 'cyan', 'grey', 'black', 'orange', 'magenta']
clist = ['#12d5ae', '#f29539', '#f26b59', '#fef0b3', '#38ff81', '#cf9bff', '#fdf33c', '#ff396a']
clist = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff', '#fdf33c',  '#ff396a']

def PlotFFT(ax,
            freq,
            data=None,   # TODO: Change
            data_list=None,
            data_list_labels=None,
            data_list_colors=None,
            data_bg=None,
            ylim=None,
            xlim=None,
            label=None,
            plotPeaks=False,
            properties=None,
            peaks=None,
            color=None,
            units=1e-3,
            minimal=True,
            noTicks=True,
            minimal_axis=False,
            periodLines=None,
            highlight_periods=None,
            highlight_periods2=None,
            edgecolor=None,
            show_all=False,
            minor_alpha=0.2,
            reference_FFT_freq=None,
            reference_FFT_pow=None,
            bg_FFT_freq=None,
            bg_FFT_pow=None,
            plot_power_diff=False,
            plot_equal_aspect=False,
            fit_inds = None,
            fit_out_inds = None,
            xlabel = 'Frequency [mHz]',
            # ylabel = r'Power density [nT$^2$ / Hz]',
            #  ylabel = r'Ratio, Signal to Background Power',
            ylabel = r'Fractional Change in Power',
            ):
    color = 'white' if color is None else color
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim((xlim[0]/units, xlim[1]/units))

    indstart = 0
    ind6h = np.argwhere(freq > (1/(6*60*60)))[0][0]
    ind30m = np.argwhere(freq > (1/(30*60)))[0][0]
    ind15m = np.argwhere(freq > (1/(15*60)))[0][0]
    ind5m  = np.argwhere(freq > (1/(5*60)))[0][0]
    indend  = len(freq) - 1
    seg_splits = [indstart, ind6h, ind30m, ind15m, ind5m, indend]
    seg_crit_split = [ind6h, ind15m]

    def plotSegs(ax, x, y, splits, alpha=1., **kwargs):
        # splits = [0, 6h, 30, 15, 5, end]
        alphas = np.asarray([0.1, 1, 0.1, 0.05, 0.03]) * alpha
        for i in range(len(splits)-1):
            kwargs_ = kwargs.copy()
            if 'label' in kwargs_ and i != 1:
                label = kwargs_.pop('label')
            ax.plot(x[splits[i]:splits[i+1]], y[splits[i]:splits[i+1]], alpha=alphas[i], **kwargs_)

    if data is not None:
        if data_bg is not None:
            ax.plot(freq/units, data_bg, ls='--', lw=2, color='white', alpha=0.2)

        # ax.plot(freq/units, data, label=label, color=color, lw=2, marker='o', ms=3, alpha=0.8)
        plotSegs(ax, freq/units, data, seg_splits,
                 alpha=0.8, label=label, color=color, lw=2, marker='o', ms=3)

    if reference_FFT_freq is not None:
        # ax.plot(reference_FFT_freq/units, reference_FFT_pow,
        plotSegs(ax, reference_FFT_freq/units, reference_FFT_pow, seg_splits,
                 label='Welch FFT', color='white', lw=1, marker='o',
                  ms=1.5, alpha=1)

    if bg_FFT_freq is not None:
        # ax.plot(bg_FFT_freq/units, bg_FFT_pow,
        plotSegs(ax, bg_FFT_freq/units, bg_FFT_pow, seg_splits,
                label='Background Power', color='orange', ls='--', lw=1,
                marker='o', ms=1.5, alpha=1)
    # ax.axhline(y=background_std, color='black', alpha=0.7, linestyle='--')
    # ax.hlines(y=1*background_std, xmin = freq[index_end], xmax = freq[index_start], color='black', alpha=0.7, linestyle='--', label=r'1 $\sigma$')
    # ax.hlines(y=2*background_std, xmin = freq[index_end], xmax = freq[index_start], color='blue', alpha=0.7, linestyle='--', label=r'2 $\sigma$')
    # ax.hlines(y=3*background_std, xmin = freq[index_end], xmax = freq[index_start], color='xkcd:greenish', alpha=0.7, linestyle='--', label=r'3 $\sigma$')

    # ax.plot(freq, data_bg + 1 * background_std, color='black', alpha=0.7, linestyle='--', label=r'1 $\sigma$')
    # ax.plot(freq, data_bg + 2 * background_std, color='blue', alpha=0.7, linestyle='--', label=r'2 $\sigma$')
    # ax.plot(freq, data_bg + 3 * background_std, color='xkcd:greenish', alpha=0.7, linestyle='--', label=r'3 $\sigma$')
    # ax.text(freq[index_end]-0.005, 1*background_std, r'1 $\sigma$', ha="center", va="bottom", fontsize=12, color='black', alpha=0.7)
    # ax.text(freq[index_end]-0.005, 2*background_std, r'2 $\sigma$', ha="center", va="bottom", fontsize=12, color='blue', alpha=0.7)
    # ax.text(freq[index_end]-0.005, 3*background_std, r'3 $\sigma$', ha="center", va="bottom", fontsize=12, color='xkcd:greenish', alpha=0.7)

    if data_list is not None:
        for n, data in enumerate(data_list):
            label = None
            if data_list_labels is not None:
                label = data_list_labels[n]
            cl = None
            if data_list_colors is not None:
                cl = data_list_colors[n]
            # show_all = True
            # ndata = len(data)
            BG_reference = True if 'BG' in label else False
            fits = None

            if len(data) > 1 and not BG_reference:
                # data = np.asarray(data)
                data_avg = np.average(data, axis=0)
                data_median = np.median(data, axis=0)
                data_q1 = np.quantile(data, q=0.25, axis=0)
                data_q3 = np.quantile(data, q=0.75, axis=0)

                if fit_inds is not None:
                    PowerLawSegmenter(data_median,
                                      ind_list=fit_inds,
                                      out_ind_list=fit_out_inds,
                                      deg=1)

                    fit_coefs = []
                    fits = []
                    fit_freq_ranges = []
                    for k, ind in enumerate(fit_inds):
                        out_ind = fit_out_inds[k]
                        coef, freq_fit_range, fit = PowerLawFit(freq, data_median,
                                                                indA=ind[0],
                                                                indB=ind[1],
                                                                outA=out_ind[0],
                                                                outB=out_ind[1],
                                                                deg=1,
                                                                returnSeries=True)
                        fits.append(fit)
                        fit_coefs.append(coef)
                        fit_freq_ranges.append(freq_fit_range)

            if show_all:
                for d in data:
                    lw = 2 if BG_reference else 1
                    ls = '--' if BG_reference else '-'
                    # ax.loglog(freq/units, d, color=cl, ls=ls, lw=lw, alpha=minor_alpha)
                    # ax.plot(freq/units, d,
                    plotSegs(ax, freq/units, d, seg_splits,
                            color=cl, ls=ls, lw=lw, alpha=minor_alpha)

                    # ax.semilogx(freq/units, data, label=label, color=color)
                    # ax.plot(freq, data, label=label, color=color)


            if data and len(data) > 1 and not BG_reference:
                ydata = data_median
                # ydata = data_avg
                # x = np.asarray(freq)/units
                # ax.fill_between((freq/units)[seg_crit_split], data_q1[seg_crit_split], data_q3[seg_crit_split], color=cl, alpha=0.3)
                ax.fill_between((freq/units)[ind6h:ind30m], data_q1[ind6h:ind30m], data_q3[ind6h:ind30m], color=cl, alpha=0.3)
                # ax.loglog(freq/units, ydata, lw=7, color='black', alpha=0.5)
                # ax.loglog(freq/units, ydata, lw=3, color=cl, marker='o', ms=3, alpha=1, label=label)

                # ax.plot(freq/units, ydata,
                plotSegs(ax, freq/units, ydata, seg_splits,
                        lw=3, color=cl, marker='o', ms=3, alpha=1, label=label)

            if fits is not None:
                for k, coef in enumerate(fit_coefs):
                    ax.plot(fit_freq_ranges[k]/units, fits[k], color='black', lw=3, alpha=0.3, ls='-')
                    ax.plot(fit_freq_ranges[k]/units, fits[k], color=cl, lw=2, alpha=0.9, ls='--')
                    mid_ind = len(fits[k]) // (len(data_list)+2) * n
                    # mid_ind = 2+n
                    ax.text(fit_freq_ranges[k][mid_ind]/units,
                            fits[k][mid_ind],
                            '{:.2f}'.format(coef[1]),
                            bbox=dict(facecolor='black', edgecolor=cl, boxstyle='round', alpha=0.5),
                            color=cl, ha="center", va="center", size=15)

    if peaks is not None:
        # timeaxis = freq
        # yaxis = data
        timeaxis = reference_FFT_freq
        yaxis = reference_FFT_pow
        ax.plot(timeaxis[peaks]/units, yaxis[peaks], "x", color=color, ms=10)
        ax.vlines(x=timeaxis[peaks]/units, ymin=yaxis[peaks] - properties["prominences"],
               ymax = yaxis[peaks], color = "orange")
        # ax.hlines(y=properties["width_heights"], xmin=timeaxis[np.round(properties["left_ips"]).astype(int)],
               # xmax=timeaxis[np.round(properties["right_ips"]).astype(int)], color = "red")
        ax.hlines(y=properties["width_heights"], xmin=timeaxis[np.round(properties["left_bases"]).astype(int)]/units,
               xmax=timeaxis[np.round(properties["right_bases"]).astype(int)]/units, color = "orange")

        for j, pk in enumerate(peaks):
        # if timeaxis[pk] > 1/120. and timeaxis[pk] < 1/20.:
            peak_f = 1./timeaxis[pk]
            # peak_power = properties["prominences"][j] * 1E3
            peak_amp = np.sqrt(yaxis[pk]) * np.sqrt(2)
            # ax.text(timeaxis[pk]/units, yaxis[pk],
                            # '%.0f min \n %.3f nT' %(peak_f/60, peak_amp),
                            # alpha=1.0, color='white', fontsize=16)

    # for i, pk in enumerate(peaks):
        # peak_f = 1./timeaxis[pk]
        # peak_power = properties["prominences"][i]
        # ax.text(timeaxis[pk], yaxis[pk],
                # '%.2f min \n %.0f pow' %(peak_height, peak_power),
                # alpha=0.7, color='red')
    # s.dfft_pow_bg *= 0.2

    # plotPeaks = True
    if plotPeaks:
        peak_lbase = np.round(properties["left_bases"]).astype(int)
        peak_rbase = np.ceil(properties["right_bases"]).astype(int)
        # for j in range(len(properties["width_heights"])):
        for j, pk in enumerate(peaks):
            if timeaxis[pk] > 1/120/60. and timeaxis[pk] < 1/20/60.:
                data_bg = np.full_like(yaxis, 1e-1)
                contor_bg = data_bg[peak_lbase[j]:peak_rbase[j]]
                contor = yaxis[peak_lbase[j]:peak_rbase[j]]
                contor_t = timeaxis[peak_lbase[j]:peak_rbase[j]]
                # contor_t = contor_t + [0.11]
                # contor_t = np.hstack((contor_t, [s.dfft_freq[peak_rbase[j]], s.dfft_freq[peak_lbase[j]]]))
                # contor = np.hstack((contor, [0,0]))
                contor_t = np.hstack((contor_t, np.flip(contor_t)))
                contor = np.hstack((contor, np.flip(contor_bg)))

                # Compute the area using the composite trapezoidal rule.
                dt = freq[1] - freq[0]
                area = trapz(contor, dx=dt)
                # Compute the area using the composite Simpson's rule.
                area = simps(contor, dx=dt)
                # print('area = ', area)
                ax.fill(contor_t/units, contor, "b")

                amp = np.sqrt(area) * np.sqrt(2)
                peak_f = 1./timeaxis[pk]
                peak_power = properties["prominences"][j] * 1E3
                ax.text(timeaxis[pk]/units, yaxis[pk],
                        '%.0f min \n %.3f nT ' %(peak_f/60, amp),
                        alpha=0.9, color='white', fontsize=16)

    if label is not None:
        ax.legend(loc=1, frameon=False)

    if highlight_periods is not None:
        loc = 0.8 * ylim[1] if ylim is None else 1
        drawPeriodLine(ax, highlight_periods, units=units, loc=loc, dir='vertical', color=clist[6], show_label=False, ls='-', alpha=0.3, lw=5, minimal=True)

    if highlight_periods2 is not None:
        loc = 0.8 * ylim[1] if ylim is None else 1
        drawPeriodLine(ax, highlight_periods2, units=units, loc=loc, dir='vertical', color=clist[7], show_label=True, text='RUN AVG', ls='-', alpha=0.5, lw=5, minimal=True)

    # periodLines = [5, 1
    if periodLines is not None:
        loc = 0.8 * ylim[1] if ylim is None else 5
        drawPeriodLine(ax, periodLines, units=units, loc=loc, dir='vertical', show_label=True, minimal=minimal, fontsize=19)

    # Plot.plot_periods(ax, Plot.Tlist, Plot.clist_boring)
    if xlim is not None: ax.set_xlim((xlim[0]/units, xlim[1]/units))
    if ylim is not None: ax.set_ylim(ylim)

    # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    if plot_power_diff:
        ax.set_xscale('log')
        ax.axhline(y=1, lw=3, alpha=0.6, ls='--', color='white')
        #  ax.set_yscale('symlog')
        #  ax.set_yscale('log')
        #  ax.set_ylim((1e-2, 1e2))
        #  ax.set_ylim((0., 1e2))
        ax.set_ylim((-1, 30))
        # ax.set_ylim((0., 8))
        # ax.set_ylim((0, 30))
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if plot_equal_aspect:
        ax.set_aspect('equal')

    if minimal:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_edgecolor(edgecolor)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['bottom'].set_linestyle('-')
        ax.spines['left'].set_visible(False)
        if not minimal_axis:
            ax.get_xaxis().set_ticks([])
        if noTicks:
            # ax.get_yaxis().set_ticks([])
            ax.tick_params(labelleft=False)
        # ax.set(frame_on=False)
        ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.7, lw=0.5)
        ax.grid(b=True, which='minor', color='white', linestyle=':', alpha=0.7)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.patch.set_alpha(0.0)
        if minimal_axis:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

def drawPeriodLine(ax, periods, units=1, loc=1, dir='horizontal',
                   color='white', lw=2, show_label=True, text=None, fontsize=12,
                   minimal=False,
                   alpha=0.5,
                   ls='--',):
    periods = np.atleast_1d(periods)
    if alpha is None:
        alpha = 0.5 if minimal else 0.6
    # fontsize = 10 if minimal else fs
    txt_d = {'color':color, 'alpha':alpha+0.2, 'fontsize':fontsize}
    line_d = {'color':color, 'alpha':alpha, 'ls':ls, 'lw':lw}
    for T in periods:
        f = 1 / T
        f = f / units
        if dir == 'horizontal':
            ax.axhline(y=f, **line_d)
            if show_label:
                if text is None:
                    if minimal:
                        label = r'%.0f$min' % (T/60)
                    else:
                        label = r'$T=%.0f$min' % (T/60)
                else:
                    label = text
                ax.text(loc, f, label, **txt_d)
        elif dir == 'vertical':
            ax.axvline(x=f, **line_d)
            if show_label:
                if text is None:
                    if minimal:
                        if T/60 > 90:
                            label = '%.0fh' % (T/60/60)
                        else:
                            label = '%.0fmin' % (T/60)
                    else:
                        label = r'$T=%.0f$min' % (T/60)
                else:
                    label = text
                ax.text(f, loc, label, rotation=90, horizontalalignment='center', **txt_d, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50,
              radius_inner=None, **kwargs):
    ''' Make and return a circular segment patch (or an annulus if provided
        an inner radius) '''
    if ax is None:
        ax = plt.gca()
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    xvals = radius*np.cos(theta) + center[0]
    yvals = radius*np.sin(theta) + center[1]
    if radius_inner is not None:
        xvals2 = radius_inner*np.cos(theta) + center[0]
        yvals2 = radius_inner*np.sin(theta) + center[1]
    else:
        xvals2 = 0
        yvals2 = 0
    xvals = np.append(xvals, np.flip(xvals2))
    yvals = np.append(yvals, np.flip(yvals2))
    points = np.vstack((xvals, yvals))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)
    return poly

def latitudeVisual(ax, lat_range=None, lat_values=None, color='orange',
                   minor_axis=[30, 60, -30, -60]):
    ''' Draw a visual to display latitudinal position in the x-z plane '''
    arc_patch((0,0), 0.4, -90, 90, ax=ax, fill=True, color='white', alpha=0.9)
    ax.plot((0.4, 1), (0,0), lw=1, ls='--', alpha=0.5, color='white')
    ax.arrow(0., -1, 0, 1.9, color='white', width=0.05, alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.axis("off")
    ax.text(0, -1.5, 'Latitude', color='white', clip_on=False, fontsize=16, alpha=0.6, ha='center')
    if lat_range is not None:
        arc_patch((0,0), 1, lat_range[0], lat_range[1], ax=ax, fill=True,
                  color='white', radius_inner=0.5, alpha=0.2)
    if lat_values is not None:
        for th in lat_values:
            ax.plot((0.5*np.cos(th), np.cos(th)),
                    (0.5*np.sin(th), np.sin(th)),
                    lw=0.5, color=color, alpha=0.9)
    if minor_axis is not None:
        for th in minor_axis:
            ax.plot((0.4*np.cos(th*np.pi/180), np.cos(th*np.pi/180)),
                    (0.4*np.sin(th*np.pi/180), np.sin(th*np.pi/180)),
                    lw=1, color='white', alpha=0.3, ls='--')


def rangeValueVisual(ax, data=None, data_range=None, xlim=None,
                     color='orange', facecolor='#171717', text_color='white',
                     edgecolor='white', label=None):
    ''' A cartoon to show some data values on a rectangular box '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_edgecolor(edgecolor)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_linestyle('-')
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.tick_params(labelleft=False)
    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.1, lw=0.5)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.05)
    ax.tick_params(axis='x', labelsize=13)
    if label is not None:
        ax.text(0.1, 0.15, label, color=text_color, fontsize=13)
    ax.set_facecolor(facecolor)
    if data_range is not None:
        r1 = data_range[0]
        r2 = data_range[1]
        ax.add_patch(Rectangle((r1, 0), (r2-r1), 1, color="white", alpha=0.2))
    if data is not None:
        for r in data:
            ax.axvline(x=r, lw=1, color=color, alpha=0.9)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(0,1)
    ax.set_xlabel('Radial Distance', alpha=0.6, fontsize=14)

def PowerVisual(ax, data=None, data_range=None, ylim=None,
                     color='orange', facecolor='#171717', text_color='white',
                     edgecolor='white', label=None):
    ''' A cartoon to show some data values on a rectangular box '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_edgecolor(edgecolor)
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_linestyle('-')
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.tick_params(labelbottom=False)
    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.1, lw=0.5)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.05)
    ax.tick_params(axis='y', labelsize=13)
    # if label is not None:
        # ax.text(-0.2, 1e-3, label, color=text_color, fontsize=13, clip_on=False, alpha=0.6, ha='center')
    ax.set_facecolor(facecolor)
    if data_range is not None:
        r1 = data_range[0]
        r2 = data_range[1]
        ax.add_patch(Rectangle((0, r1), 1, (r2-r1), color="white", alpha=0.2))
    if data is not None:
        for r in data:
            ax.axhline(y=r, lw=1, color=color, alpha=0.1)
        median = np.median(data)
        ax.axhline(y=median, lw=3, color=color, alpha=0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_ylabel(r'Median Power ($B_T$, 60m) [$nT^2/Hz$]', alpha=0.5, fontsize=14)
    ax.set_xlim(0,1)

def FieldVisual(ax, data=None, data_range=None, ylim=None,
                     color='orange', facecolor='#171717', text_color='white',
                     edgecolor='white', label=None):
    ''' A cartoon to show some data values on a rectangular box '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_edgecolor(edgecolor)
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_linestyle('-')
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.tick_params(labelbottom=False)
    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.1, lw=0.5)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.05)
    ax.tick_params(axis='y', labelsize=13)
    # if label is not None:
        # ax.text(-0.2, 1e-3, label, color=text_color, fontsize=13, clip_on=False, alpha=0.6, ha='center')
    ax.set_facecolor(facecolor)
    if data_range is not None:
        r1 = data_range[0]
        r2 = data_range[1]
        ax.add_patch(Rectangle((0, r1), 1, (r2-r1), color="white", alpha=0.2))
    if data is not None:
        for r in data:
            ax.axhline(y=r, lw=1, color=color, alpha=0.1)
        median = np.median(data)
        ax.axhline(y=median, lw=4, color=color, alpha=1)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_ylabel(r'Median Magnetic Field ($B_T$) [$nT$]', alpha=0.5, fontsize=14)
    ax.set_xlim(0,1)

def lt2degrees(lt):
    ''' Local Time to Degrees (can be < 0 and > 360) '''
    return (lt - 12) / 24 * 360

def localTimeVisual(ax, lt_range=None, lt_values=None, color='orange'):
    ''' A cartoon to show local time values around the planet '''
    axis_style = {'lw': 1, 'ls': '--', 'alpha': 0.4, 'color': 'white'}
    arc_patch((0,0), 0.4, 0, 360, ax=ax, fill=True,
              color='white', alpha=0.9)
    arc_patch((0,0), 0.4, 90, 270, ax=ax, fill=True,
              color='black', alpha=0.9)
    ax.text(0., -1.23, 'Local Time', color='white', clip_on=False, fontsize=16, alpha=0.6, ha='center')

    ax.plot((0.4, 1), (0,0), **axis_style)
    ax.plot((-1, -0.4), (0,0), **axis_style)
    ax.plot((0, 0), (0.4, 1), **axis_style)
    ax.plot((0, 0), (-1, -0.4), **axis_style)
    if lt_range is not None:
        th1 = lt2degrees(lt_range[0])
        th2 = lt2degrees(lt_range[1])
        arc_patch((0,0), 1, th1, th2, ax=ax, fill=True,
                  color='white', radius_inner=0.5, alpha=0.2)
    if lt_values is not None:
        for lt in lt_values:
            th = lt2degrees(lt) * np.pi/180
            ax.plot((0.5*np.cos(th), np.cos(th)),
                    (0.5*np.sin(th), np.sin(th)),
                    lw=0.5, color=color, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

def timeTicks(x, pos, datefrom=None):
    ''' Convert Seconds to Datetime Strings '''
    # d = datefrom + datetime.timedelta(seconds=x)
    # d = datetime.timedelta(seconds=x)
    return str(d)

def plotSpectrogram(ax,
                    tseg,
                    freq,
                    S,
                    data_datetimes,
                    textent = None,
                    ylim=None,
                    xlim=None,
                    plotPeaks=False,
                    units = 1e-3,
                    vmin = None,
                    vmax = None,
                    colorScale = 'log',
                    yScale = 'log',
                    showSegments = True,
                    periodLines = None,
                    cbar_label = r'Power Density (nT$^2$ / Hz)',
                    ylabel = 'Frequency (mHz)',
                    xlabel = None,
                    axisLocators = [1, 2],
                    xaxis_format = "%H:%M",
                    highlight_period = None,
                    snaps = None,
                    show_cbar=True,
                    show_xticks=True,
                    fontsize=10,
                    ):

    import copy as copy
    datefrom, dateto = data_datetimes[0], data_datetimes[1]
    cmap = copy.copy(plt.get_cmap('turbo'))
    cmap.set_under(color='k', alpha=None)
    tseg_dt = [datefrom + datetime.timedelta(seconds=t) for t in tseg]

    if textent is None:
        xmin, xmax = datefrom, dateto
    else:
        xmin, xmax = textent

    x_lims = dates.date2num([xmin, xmax])

    extent = x_lims[0], x_lims[1], freq[0] / units, freq[-1] / units
    # print('tmax:', datefrom+datetime.timedelta(seconds=xmax))
    # print('tmin:', datefrom+datetime.timedelta(seconds=xmin))

    if colorScale == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif colorScale == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Spectrogram rendering:
    im = ax.imshow(S,
                   cmap,
                   extent=extent,
                   norm=norm,
                   origin='lower',
                   interpolation='none',
                   )

    if show_cbar:
        fig = plt.gcf()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.outline.remove()
        cbar.ax.tick_params(axis='y', color='white', left=False, right=True)
        cbar.set_label(cbar_label)

    ax.xaxis_date()
    ax.xaxis.set_minor_locator(dates.HourLocator(interval=axisLocators[0]))
    ax.xaxis.set_major_locator(dates.HourLocator(interval=axisLocators[1]))
    dfmt = dates.DateFormatter(xaxis_format)
    ax.xaxis.set_major_formatter(dfmt)

    if periodLines is not None:
        drawPeriodLine(ax, periodLines, units=units, loc=datefrom, fontsize=fontsize)
    if highlight_period is not None:
        drawPeriodLine(ax, [highlight_period], units=units, loc=datefrom, color='white', lw=2, text='simulated period', fs=18)
    if showSegments == True:
        ypos = 1 if ylim is None else ylim[1] / 10 / units
        ax.scatter(tseg_dt, np.full_like(tseg_dt, ypos), marker='|', lw=2,
                   alpha=0.7, color='red')
    ax.set_facecolor('#171717')
    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.4)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if not show_xticks:
        ax.set_xticklabels([])
        ax.set_xlabel(None)

    # the x coords of this transformation are data, and the y coord are axes
    trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)

    if snaps is not None:
        ntsegs = len(tseg)
        ind_incr = int(ntsegs / (snaps-1))
        snap_pad = 0.02
        snap_width = (0.775-(snaps-1)*snap_pad) / snaps
        snap_height = 0.14
        snap_x = 0.1
        snap_y = 0.40
        ind = 0
        for n in range(snaps):
            snap_fft = S[:, ind]
            tseg_w_sec = tseg[1]-tseg[0] if len(tseg)>1 else 60
            w_scale = 1 if tseg_w_sec > 60*10 else 5
            tseg_w = datetime.timedelta(seconds=tseg_w_sec * w_scale)
            tseg_h = 0.04
            tseg_y = 1
            tseg_x = datefrom + datetime.timedelta(seconds=tseg[ind]-tseg_w_sec*w_scale/2)
            ax.add_patch(Rectangle((tseg_x, tseg_y), tseg_w, tseg_h,
                                   color=clist[n], alpha=0.7,
                                   transform=trans,
                                   clip_on=False))
            ax.axvline(x=datefrom+datetime.timedelta(seconds=tseg[ind]), ls='--', lw=2, color=clist[n], alpha=0.8)
            ind += ind_incr
            if n == snaps-2:
                ind = len(tseg)-1
            # cax.axis("auto")
            # cax.margins(0.05) # 5% padding in all direction

    # ax.axis("tight")
    ax.axis("auto")

    ax.set_yscale(yScale)
    if ylim is not None:
        ax.set_ylim(ylim[0]/units, ylim[1]/units)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    return im

def plotFFTSnaps(ax,
                 tseg,
                 freq,
                 S,
                 data_datetimes,
                 snaps = 4,
                 units = 1e-3,
                 vmin = None,
                 vmax = None,
                 periodLines = None,
                 highlight_periods = None,
                 highlight_periods2 = None,
                 yScale = 'log',
                 # ylabel = r'Power Density (nT$^2$ / Hz)',
                 ylabel = r'Power [nT$^2$]',
                 xlabel = 'Frequency [mHz)',
                 xlim = None,
                 reference_FFT_freq=None,
                 reference_FFT_pow=None,
                 bg_FFT_freq=None,
                 bg_FFT_pow=None,
                 peaks=None,
                 properties=None,
                 minimal_axis=True,
                ):
    ax.axis('off')
    datefrom, dateto = data_datetimes[0], data_datetimes[1]
    tseg_dt = [datefrom + datetime.timedelta(seconds=t) for t in tseg]
    snap_pad = 0.05
    if snaps > 1:
        ind_incr = int(len(tseg) / (snaps-1))
        snap_width = (1-(snaps-1)*snap_pad) / snaps
    else:
        ind_incr = 1
        snap_width = 1
    snap_height = 0.75
    snap_x = 0.1
    snap_y = 0
    ind = 0
    # axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])

    for n in range(snaps):
        snap_fft = S[:, ind]
        snap_x = 0. + n * (snap_width + snap_pad)
        axins = ax.inset_axes((snap_x, snap_y, snap_width , snap_height))
        PlotFFT(axins, freq, snap_fft,
                xlim=xlim,
                ylim=(vmin, vmax),
                units=units,
                noTicks=(True if n > 0 else False),
                periodLines = periodLines,
                highlight_periods = highlight_periods,
                highlight_periods2 = highlight_periods2,
                label=data_datetimes[0]+datetime.timedelta(seconds=tseg[ind]),
                edgecolor=clist[n],
                color=clist[n],
                reference_FFT_freq=reference_FFT_freq,
                reference_FFT_pow=reference_FFT_pow,
                bg_FFT_freq=bg_FFT_freq,
                bg_FFT_pow=bg_FFT_pow,
                peaks=peaks,
                properties=properties,
                minimal_axis=minimal_axis,
                xlabel=xlabel,
                ylabel=ylabel,
                )
        ind += ind_incr
        if n == snaps-2:
            ind = len(tseg)-1

# ============================================================================

def returnKMAG_Lshell(CurrentTime, Position, SIM):
    SIM.config["ETIME"] = date2timestamp(CurrentTime)
    SIM.config["IN_COORD"] = SIM.config["OUT_COORD"]
    Position = kmag.krot("KSM",
                         SIM.config["OUT_COORD"],
                         Position,
                         SIM.config["ETIME"],
                         SIM.config["EPOCH"])
    fl = traceFieldline(Position, SIM)
    ind = np.argwhere(fl.traceXYZ == fl.traceStart)[0][0]
    BT = fl.BT[ind]
    R1 = fl.traceRTP[0, 0]
    R2 = fl.traceRTP[-1, 0]
    TH1 = fl.traceRTP[0, 1]
    TH2 = fl.traceRTP[-1, 1]
    return fl.L, BT

def serialize(data):
    if len(data) > 0:
        freq = data[0].FIELDS[0].fft_freq
        ind_1h =    np.argwhere(freq > (1/(1*60*60)))[0][0]

    list_r = [x.info["median_coords"][0] for x in data]
    list_BT = [x.info["median_BT"] for x in data]
    #  list_BT_model = [x.info["BT_model"] for x in data]
    list_BT_model = [1 for x in data]
    #  list_L = [x.info["L"] for x in data]
    list_L = [1 for x in data]
    list_lt = [x.info["median_LT"] for x in data]
    list_th = [x.info["median_coords"][1] for x in data]
    list_P = [np.median(x.FIELDS[3].fft_bg[ind_1h-3:ind_1h+3]) for x in data]
    return list_r, list_L, list_BT, list_BT_model, list_lt, list_th, list_P

def sortDataByLocation(data):
    fft_list = [[] for i in range(4)]
    for n, d in enumerate(data):
        if d.info["location"] == 0:
            ind = 0
        if d.info["location"] == 1:
            ind = 1
        if d.info["location"] == 2:
            ind = 2
        if d.info["location"] == 9:
            ind = 3
        fft_list[ind].append(d)
    return fft_list[0], fft_list[1], fft_list[2], fft_list[3]

def value2bin(value, minValue=0., maxValue=1., bins=10, range=False):
    binSpacing = (maxValue - minValue) / bins
    index = np.floor_divide(value - minValue, binSpacing).astype(int)
    if not range:
      index = np.atleast_1d(index)
      index = np.where(index >= bins, index-1, index)

    return index[0] if (isinstance(index, np.ndarray) &
                        len(index) == 1) else index

def bin2value(value, minValue=0., maxValue=1., bins=10):
    binSpacing = (maxValue - minValue) / bins
    return (value * binSpacing + minValue +
              0.5 * binSpacing) # for bin center value

def sortDataByBins(data, minValue=2, maxValue=80, bins=10, SIM=None):
    bins = int((maxValue - minValue) / 1 + 1)
    fft_bins = [[] for i in range(bins)]
    fft_labels = np.linspace(minValue, maxValue, bins)
    for n, d in enumerate(data):
        y = d.r
        # y = returnKMAG_Lshell(d.datetime, d.pos_KSM, SIM)
        # print('L: ', y)
        ind = value2bin(y, minValue=minValue, maxValue=maxValue, bins=bins)
        if ind:
            if ind < len(fft_bins):
                fft_bins[ind].append(d)
    return fft_bins, fft_labels

def filterByProperties(data, arg='LT', minValue = 10, maxValue = 14):
    selection = []
    for n, x in enumerate(data):
        if arg == 'th':
            value = x.info["median_coords"][1]
        elif arg == 'r':
            value = x.info["median_coords"][0]
        elif arg == 'phi':
            value = x.info["median_coords"][2]
        elif arg == 'LT':
            value = x.info["median_LT"]
        else:
            # value = getattr(x, arg)
            value = x.info.get(arg)
        if value >= minValue and value < maxValue:
            selection.append(x)
    return selection

def selectByDatetime(data, dateFrom, dateTo):
    returnList = []
    for n, d in enumerate(data):
        if d.datetime >= dateFrom and d.datetime <= dateTo:
            returnList.append(d)
    return returnList

def PowerLawFit(x, y, indA=None, indB=None, deg=2, outA=None, outB=None,
                returnSeries=False):
    ''' Fit a Polynomial on a Log-Log scale '''
    if indA is None:
        indA = 1 if x[0] == 0 else 0  # Fix for f=0 in FFT
        indB = len(x) - 1

    # Take a segment of the data
    xdata = np.log(x[indA:indB])
    ydata = np.log(y[indA:indB])

    # Polynomial
    fit = Polynomial.fit(xdata, ydata, deg)
    coefs = fit.convert().coef
    # coefs = fit.coef
    if returnSeries:
        # Configure the output range
        if outA is not None and outB is not None:
            xout = np.log(x[outA:outB])
        else:
            xout = xdata

        pl = np.polynomial.polynomial.polyline(coefs[0], coefs[1])
        yfit = np.polynomial.polynomial.polyval(xout, pl)
        return coefs, np.exp(xout), np.exp(yfit)
    else:
        return coefs

def PowerLawSegmenter(fftlist, ind_list=None, out_ind_list=None, deg=1):
    ''' Fit power law segments for given ranges of the data '''
    fftlist = np.atleast_1d(fftlist)
    if out_ind_list is None:
        out_ind_list = ind_list
    # print('Starting to fit power law segments.')
    for f in fftlist:
        seg_coefs = []
        if hasattr(f, 'flag') and getattr(f, 'flag') is None:
            for n, ind in enumerate(ind_list):
                out_ind = out_ind_list[n]
                fft_coefs = []
                for y in [f.fft0, f.fft1, f.fft2, f.fft3]:
                    coef = PowerLawFit(f.freq, y,
                                       indA=ind[0],
                                       indB=ind[1],
                                       outA=out_ind[0],
                                       outB=out_ind[1],
                                       deg=deg,
                                       returnSeries=False)
                    fft_coefs.append(coef)
                seg_coefs.append(fft_coefs)
            f.fit_coefs = seg_coefs
            f.fit_ranges = out_ind_list
    # print('Finished fitting power law segments.')
    return fftlist

def powerBinner(fftlist,
                power_bins=[],
                bin_halfwidth=3):
    # fftlist = np.atleast_1d(fftlist)
    # print('Starting to estimate power in the designated bins.')
    for f in fftlist:
        if f.flag is None:
            bin_powers = []
            for pbin in power_bins:
                indA = pbin - bin_halfwidth
                indB = pbin + bin_halfwidth
                indA = 1 if indA < 1 else indA
                indB = len(f.FIELDS[0].dfft)-1 if indB > len(f.FIELDS[0].dfft)-1 else indB
                fft_powers = []
                for y in [f.FIELDS[0].dfft, f.FIELDS[1].dfft, f.FIELDS[2].dfft, f.FIELDS[3].dfft]:
                    median_power = np.median(y[indA:indB])
                    fft_powers.append(median_power)
                bin_powers.append(fft_powers)
            f.bin_powers = bin_powers
    # print('Finished estimating the power in the bins.')
    return fftlist

def filterByPower(data, component = 2):
    ''' Select Data Segments by Presence of Power in Selected Frequencies ''' 
    power_bins=[ind_4h, ind_3h, ind_2h, ind_90min, ind_75min,
                ind_1h, ind_45min, ind_30min, ind_15min]
    power_bin_labels=['4h', '3h', '2h', '90min', '75min',
                      '1h', '45min', '30min', '15min']

    powerBinner(data,
                power_bins=power_bins,
                bin_halfwidth=3)

    wave_selection = [[] for i in range(4)]
    for s in data:
        peaks_detected = False
        for i in [0, 1, 2, 3]:
            fft75_power = s.bin_powers[4][i]
            fft60_power = s.bin_powers[5][i]
            fft45_power = s.bin_powers[6][i]
            if (fft60_power > 1.2*fft75_power and
                fft60_power > 1.2*fft45_power and
                fft60_power > 1):
                peaks_detected = True
            # if peaks_detected and s.info["rKSM"][1] > 10:
            if (peaks_detected
                and s.info["LT"] > 18
                and s.info["LT"] < 24
                and s.info["rKSM"][1] > 30):
            # if peaks_detected:
                wave_selection[i].append(s)

    return wave_selection[component]

    #  power_bins=[ind_4h, ind_3h, ind_2h, ind_90min, ind_75min,
                #  ind_1h, ind_45min, ind_30min, ind_15min]
#
    #  power_bin_labels=['4h', '3h', '2h', '90min', '75min', '1h', '45min', '30min', '15min']

    # Bin the Powers
    # powerBinner(dataSelection,
                # power_bins=power_bins,
                # bin_halfwidth=3)
# 
    # wave_selection = [[] for i in range(4)]
    # for s in dataSelection:
        # peaks_detected = False
        # for i in [0, 1, 2, 3]:
            # fft75_power = s.bin_powers[4][i]
            # fft60_power = s.bin_powers[5][i]
            # fft45_power = s.bin_powers[6][i]
            # if (fft60_power > 2*fft75_power and
                # fft60_power > 2*fft45_power and
                # fft60_power > 2):
                # peaks_detected = True
            # if peaks_detected:
                # wave_selection[i].append(s)
# 
    # print('N of 60min Waves for FFT0:', len(wave_selection[0]))
    # print('N of 60min Waves for FFT1:', len(wave_selection[1]))
    # print('N of 60min Waves for FFT2:', len(wave_selection[2]))
    # print('N of 60min Waves for FFT3:', len(wave_selection[3]))
    # exit()

    # fft0_sel = [x.FIELDS[0].dfft for x in wave_selection[0]]
    # fft1_sel = [x.FIELDS[1].dfft for x in wave_selection[1]]
    # fft2_sel = [x.FIELDS[2].dfft for x in wave_selection[2]]
    # fft3_sel = [x.FIELDS[3].dfft for x in wave_selection[3]]
    # fft0_60min_power = [f[ind_1h] for f in fft0_sel]
    # fft1_60min_power = [f[ind_1h] for f in fft1_sel]
    # fft2_60min_power = [f[ind_1h] for f in fft2_sel]
    # fft3_60min_power = [f[ind_1h] for f in fft3_sel]
    # ind = np.argmax(fft2_60min_power)
# 
    # plot_series = [wave_selection[2][ind].COORDS[0].y,
                   # wave_selection[2][ind].COORDS[1].y,
                   # wave_selection[2][ind].COORDS[2].y,
                   # wave_selection[2][ind].COORDS[3].y]
    # plot_series_fft = [[wave_selection[2][ind].FIELDS[0].fft, wave_selection[2][ind].FIELDS[0].fft],
                       # [wave_selection[2][ind].FIELDS[1].fft, wave_selection[2][ind].FIELDS[1].fft],
                       # [wave_selection[2][ind].FIELDS[2].fft, wave_selection[2][ind].FIELDS[2].fft],
                       # [wave_selection[2][ind].FIELDS[3].fft, wave_selection[2][ind].FIELDS[3].fft]]
    # print(plot_series_fft[0])
    # plot_series_fft = high_power_selection[ind].fft2
    # plot_fft_freq = high_power_selection[ind].freq
    # exit()

def collectWaveEvents(signal, config={}, COORDS=None, info={},
    peak_separation = 3*60*60,
    min_duration=4):

    signals = []
    s = signal
    #  f.analyze(actions=['detrend', 'smooth', 'cwt', 'cwtm'], parameters={})
    s.analyze(actions=['detrend', 'smooth', 'cwtm'], parameters={})
    s.cwtm = np.abs(s.cwtm)
    s.cwtm = s.cwtm / np.max(s.cwtm)

    cwt_timeseries = []
    #  period_list_in_mins = [30, 35, 40, 45, 50, 55, 60, 65, 75, 80,
                           #  85, 90, 95, 100, 105, 110, 115, 120]
    period_list_in_mins = [25, 30, 35]
    #  period_list_in_mins = [55, 60, 65]
    #  period_list_in_mins = [115, 120, 125]
    for t in period_list_in_mins:
        ind = np.argwhere(s.cwtm_freq >= 1/(t*60))[0][0]
        cwt_timeseries.append(np.abs(s.cwtm[ind, :]))

    event_measure = np.linalg.norm(np.asarray(cwt_timeseries), axis=0)
    #  event_measures[j].append(event_measure)
    s.cwt = None
    s.cwtm = None
    s.cwtm_freq = None
    s.cwtm_tseg = None

    peaks, properties = find_peaks(event_measure,
                                   height = 0.05,
                                   distance = 60,
                                   prominence = 0.05,
                                   width = 100)
    for peak, prom, wh, l_ips, r_ips, l_base, r_base in zip(peaks, properties["prominences"], properties["width_heights"],
                                            properties["left_ips"], properties["right_ips"],
                                            properties["left_bases"], properties["right_bases"]):
        #  event_sep.append([sep/60/60, peaks[i][1], peaks[i][2], peaks[i][3]])
        #  sep_60 = pd.DataFrame(event_sep_60, columns=['sep', 'strength', 'from', 'to'])

        Field = signal.y[int(np.round(l_ips)): int(np.round(r_ips))]
        amplitude = np.max(Field)
        SLS5N = info["SLS5N"][int(floor(peak/10.))],
        SLS5S = info["SLS5S"][int(floor(peak/10.))],
        SLS5N2 = info["SLS5N2"][int(floor(peak/10.))],
        SLS5S2 = info["SLS5S2"][int(floor(peak/10.))],
        dateFrom=s.datetime[int(np.round(l_ips))]
        dateTo=s.datetime[int(np.round(r_ips))]
        duration = (dateTo - dateFrom).total_seconds() / 60/ 60

        if duration > min_duration:
            #  print(f'Duration = {duration:.1f} hours')
            #  print(f'Amplitude = {amplitude:.3f} nT')
            signals.append(WaveSignal(dateFrom=dateFrom,
                                  dateTo=dateTo,
                                  datePeak=s.datetime[peak],
                                  freq=None,
                                  amplitude=amplitude,
                                  duration=duration,
                                  power=prom,
                                  waveforms=None,
                                  intervals=None,
                                  amplitudes=None,
                                  slope=None,
                                  sls5_phase=[SLS5N, SLS5S, SLS5N2, SLS5S2],
                                  axis=None,
                                  LT=info["median_LT"],
                                  coord_KSM=None,
                                  coord_KRTP=[COORDS[0].y[peak], COORDS[1].y[peak], COORDS[2].y[peak]]))
    #  if showProgress:
        #  plt.plot(s.datetime, event_measure, color=clist_field[1], alpha=0.8, label='Event Measure')
        #  peaks, properties = find_peaks(event_measure, height=0.10, distance=1*60, prominence=0.10, width=100)
        #  for peak in peaks:
            #  plt.plot(s.datetime[peak], event_measure[peak], "x", color='yellow', ms=8)
        #  for peak, prom, wh, l_ips, r_ips, l_base, r_base in zip(peaks, properties["prominences"], properties["width_heights"],
                                                #  properties["left_ips"], properties["right_ips"],
                                                #  properties["left_bases"], properties["right_bases"]):
            #  plt.vlines(x=s.datetime[peak], ymin=event_measure[peak] - prom,
                       #  ymax = event_measure[peak], color = "C1")
            #  plt.hlines(y=wh, xmin=s.datetime[int(np.round(l_ips))],
                       #  xmax=s.datetime[int(np.round(r_ips))], color = "C1")
        #  plt.show()
    return signals


def calculateEventSeparation(data, loadData=False, showProgress=False):
    ''' Calculate Event (Wave Train) Separation based on Wavelet Transform '''
    clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff', '#fdf33c',  '#ff396a']

    if loadData:
        #  event_peaks_60 = np.load('event_peaks.npy', allow_pickle=True)
        #  event_measures_60 = np.load('event_measures.npy', allow_pickle=True)
        #  event_peaks_60 = np.load('event_peaks_strict_60.npy', allow_pickle=True)
        #  event_measures_60 = np.load('event_measures_strict_60.npy', allow_pickle=True)
        event_peaks_60 = np.load('event_peaks_60SLS.npy', allow_pickle=True)
        event_measures_60 = np.load('event_measures_60SLS.npy', allow_pickle=True)
        #  event_peaks_30 = np.load('event_peaks_30.npy', allow_pickle=True)
        #  event_measures_30 = np.load('event_measures_30.npy', allow_pickle=True)
        #  event_peaks_90 = np.load('event_peaks_90.npy', allow_pickle=True)
        #  event_measures_90 = np.load('event_measures_90.npy', allow_pickle=True)
        #  event_peaks_120 = np.load('event_peaks_120.npy', allow_pickle=True)
        #  event_measures_120 = np.load('event_measures_120.npy', allow_pickle=True)
    else:
        event_measures = [[] for i in range(4)]
        event_peaks = [[] for i in range(4)]
        prevTime = data[0].datetime[0]
        for i, s in enumerate(data):
            currTime = s.datetime[0]
            if (currTime - prevTime).total_seconds()/60/60 > 24:
                continuity_flag = 0
            else:
                continuity_flag = 1
            prevTime = currTime
            total_gaps = 0
            for gap in s.info["gaps"]:
                dur = (gap[1] - gap[0]).total_seconds() / 60. / 60.
                total_gaps += dur
            for j, f in enumerate(s.FIELDS):
                    


                #  f.analyze(actions=['detrend', 'smooth', 'cwt', 'cwtm'], parameters={})
                f.analyze(actions=['detrend', 'smooth', 'cwtm'], parameters={})
                f.cwtm = np.abs(f.cwtm)
                f.cwtm = f.cwtm / np.max(f.cwtm)

                cwt_timeseries = []
                #  period_list_in_mins = [30, 35, 40, 45, 50, 55, 60, 65, 75, 80,
                                       #  85, 90, 95, 100, 105, 110, 115, 120]
                #  period_list_in_mins = [25, 30, 35]
                period_list_in_mins = [55, 60, 65]
                #  period_list_in_mins = [115, 120, 125]
                for t in period_list_in_mins:
                    ind = np.argwhere(f.cwtm_freq >= 1/(t*60))[0][0]
                    cwt_timeseries.append(np.abs(f.cwtm[ind, :]))
                cwt_timeseries = np.asarray(cwt_timeseries)
                event_measure = np.linalg.norm(cwt_timeseries, axis=0)
                event_measures[j].append(event_measure)
                f.cwt = None
                f.cwtm = None
                f.cwtm_freq = None
                f.cwtm_tseg = None

                peaks, properties = find_peaks(event_measure,
                                               height = 0.05,
                                               distance = 60,
                                               prominence = 0.05,
                                               width = 100)
                for peak, prom, wh, l_ips, r_ips, l_base, r_base in zip(peaks, properties["prominences"], properties["width_heights"],
                                                        properties["left_ips"], properties["right_ips"],
                                                        properties["left_bases"], properties["right_bases"]):
                    peak_duplicate = False
                    if len(event_peaks[j]) > 1:
                        peak_sep = abs((s.datetime[peak] - event_peaks[j][-1][0]).total_seconds())
                        if peak_sep < 3 * 60 * 60:
                            peak_duplicate = True
                    #  if not peak_duplicate and event_measure[peak] < 0.5:
                    #  if not peak_duplicate and prom < 0.5:
                    if not peak_duplicate:
                        event_peaks[j].append([s.datetime[peak],
                                               prom,
                                               #  event_measure[peak],
                                               s.datetime[int(np.round(l_ips))],
                                               s.datetime[int(np.round(r_ips))],
                                               s.info["median_LT"],
                                               s.COORDS[0].y[peak],
                                               s.COORDS[1].y[peak],
                                               s.info["SLS5N"][int(floor(peak/10.))],
                                               s.info["SLS5S"][int(floor(peak/10.))],
                                               s.info["SLS5N2"][int(floor(peak/10.))],
                                               s.info["SLS5S2"][int(floor(peak/10.))],
                                               total_gaps,
                                               continuity_flag])
                if showProgress:
                    plt.plot(s.datetime, event_measure, color=clist_field[1], alpha=0.8, label='Event Measure')
                    peaks, properties = find_peaks(event_measure, height=0.10, distance=1*60, prominence=0.10, width=100)
                    for peak in peaks:
                        plt.plot(s.datetime[peak], event_measure[peak], "x", color='yellow', ms=8)
                    for peak, prom, wh, l_ips, r_ips, l_base, r_base in zip(peaks, properties["prominences"], properties["width_heights"],
                                                            properties["left_ips"], properties["right_ips"],
                                                            properties["left_bases"], properties["right_bases"]):
                        plt.vlines(x=s.datetime[peak], ymin=event_measure[peak] - prom,
                                   ymax = event_measure[peak], color = "C1")
                        plt.hlines(y=wh, xmin=s.datetime[int(np.round(l_ips))],
                                   xmax=s.datetime[int(np.round(r_ips))], color = "C1")
                    plt.show()
            if i%10 == 0:
                print('{:.1f}% completed'.format(i/len(data)*100))

        #  np.save('signals_MS.npy',  list_MS)
        np.save('event_peaks_60SLS.npy',  event_peaks)
        np.save('event_measures_60SLS.npy',  event_measures)

    #  for i in range(len(event_peaks_60[2])):
        #  if event_peaks_60[2][i][1] > 0.1:
            #  plt.scatter(event_peaks_60[2][i][-3], event_peaks_60[2][i][1])
    #  plt.show()
    #  exit()

    a = []
    for i in range(len(event_peaks_60[2])):
        dt = event_peaks_60[2][i][0]
        prom = event_peaks_60[2][i][1]
        LT = event_peaks_60[2][i][4]
        R = event_peaks_60[2][i][5]
        TH = event_peaks_60[2][i][6]
        SLS5N = event_peaks_60[2][i][7]
        SLS5S = event_peaks_60[2][i][8]
        SLS5N2 = event_peaks_60[2][i][9]
        SLS5S2 = event_peaks_60[2][i][10]
        if R > 18 and R < 22 and prom > 0.5:
            plt.scatter(SLS5S2, LT)
        #  plt.scatter(dt, SLS5N)
        if prom > 0.5:
            a.append(SLS5N2)
    #  _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    plt.xlabel('SLS5N2')
    plt.ylabel('Local Time (h)')
    #  plt.xlim(0, 360)
    #  plt.ylim(0, 24)
    plt.title('SLS5N Phase for 60-min Events, lat > 30')
    plt.show()
    exit()

    def calc_event_sep(event_peaks, filter_by=None, n=0, N=10):
        event_sep = []
        peaks_selection = event_peaks[2]
        peaks = []
        for p in peaks_selection:
            prom = p[1]
            if prom > 0.1:
                peaks.append(p)

        for i in range(1, len(peaks)):
            sep = peaks[i][0] - peaks[i-1][0]
            sep = sep.total_seconds()
            dur = (peaks[i][3] - peaks[i][2]).total_seconds()
            contCond0 = peaks[i-1][-1] == 1
            contCond1 = peaks[i][-1] == 1
            gapCond0 = peaks[i-1][-2] < 0.3
            gapCond1 = peaks[i][-2] < 0.3
            r_lim0 = 5 + n / N * 40 - 4
            r_lim1 = 5 + n / N * 40 + 4
            th_lim0 = (30 + n / N * 140 - 10 - 90) * np.pi / 180
            th_lim1 = (30 + n / N * 140 + 10 - 90) * np.pi / 180
            lt_lim0 = 2 + n / N * 22 - 2
            lt_lim1 = 2 + n / N * 22 + 2
            rCond = peaks[i][-4] > r_lim0 and peaks[i][-4] < r_lim1
            thCond = peaks[i][-3] > th_lim0 and peaks[i][-3] < th_lim1
            ltCond = peaks[i][-5] > lt_lim0 and peaks[i][-5] < lt_lim1
            txt = ''
            extraCond = True
            if filter_by is not None:
                if filter_by == 'r':
                    extraCond = rCond
                    txt = r"{:.1f} $\leq$ r $\leq$ {:.1f} [RS]".format(r_lim0, r_lim1)
                if filter_by == 'th':
                    extraCond = thCond
                    txt = r'{:.1f} $\leq$ latitude $\leq$ {:.1f} [degrees]'.format(th_lim0*180/np.pi, th_lim1*180/np.pi)
                if filter_by == 'lt':
                    extraCond = ltCond
                    txt = r'{:.1f} $\leq$ LT $\leq$ {:.1f} [h]'.format(lt_lim0, lt_lim1)
            meetConditions = (sep < 28*60*60
                              and sep > 5*60*60
                              and dur > 4*60*60
                              #  and contCond0
                              and contCond1
                              and gapCond0
                              and gapCond1
                              and extraCond)
            if meetConditions:
                event_sep.append([sep/60/60, peaks[i][1], peaks[i][2], peaks[i][3]])
        return event_sep, txt

    for n in range(1):
        print(' Plotting image', n)
        #  event_sep_30 = calc_event_sep(event_peaks_30)
        #  event_sep_60, title = calc_event_sep(event_peaks_60, filter_by='th', n=n, N=1)
        #  title += ' (N={:d})'.format(len(event_sep_60))
        event_sep_60, title = calc_event_sep(event_peaks_60, n=n, N=1)
        #  event_sep_90 = calc_event_sep(event_peaks_90)
        #  event_sep_120 = calc_event_sep(event_peaks_120)

        #  sep_30 = pd.DataFrame(event_sep_30, columns=['sep', 'strength', 'from', 'to'])
        sep_60 = pd.DataFrame(event_sep_60, columns=['sep', 'strength', 'from', 'to'])
        #  sep_90 = pd.DataFrame(event_sep_90, columns=['sep', 'strength', 'from', 'to'])
        #  sep_120 = pd.DataFrame(event_sep_120, columns=['sep', 'strength', 'from', 'to'])

        #  print('Selected peaks in 30 min: {:d}'.format(len(sep_30)))
        print('Selected peaks in 60 min: {:d}'.format(len(sep_60)))
        #  print('Selected peaks in 90 min: {:d}'.format(len(sep_90)))
        #  print('Selected peaks in 120 min: {:d}'.format(len(sep_120)))

        plt.style.use('default')
        plt.style.use('paper.mplstyle')
        fig, ax = plt.subplots(figsize=(10, 6))
        #  ax.set_title(r"Wave Activity Separation in Time ($\tau$)", pad = 10)
        ax.set_title(title, pad = 10)
        # for i, [duration, sep] in enumerate(zip([30, 60, 90],
                                             # [sep_30, sep_60, sep_90])):
        for i, [duration, sep] in enumerate(zip([60],
                                             [sep_60])):
            cl = clist_friendly[i+1]
            sep_median = sep["sep"].median()
            label = r'$\tau (%d\textrm{ min})$' % duration
            label = r'$%d\textrm{-min Wave Separation}$' % duration
            text = r'$\mu_{1/2} \textrm{(%d min)}$ = %.2f h' % (duration, sep_median)
            text = r'$\textrm{median } \textrm{(%d min)}$ = %.2f h' % (duration, sep_median)
            sep["sep"].plot(kind = "hist", density = True, bins = 24, color=cl, alpha=0.3, label='_nolegend_')
            sep["sep"].plot(kind = "kde", color=cl, lw=2, label=label)
            ax.axvline(x=sep_median, ls='--', lw=2, color=cl, alpha=0.8)
            ax.text(sep_median, 0.01+0.013*i, text, color='black', fontsize=16,
                    bbox=dict(facecolor='white', alpha=0.4, edgecolor=cl, lw=2, boxstyle='round'))
        ax.set_xlabel('Separation [h]')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 24)
        #  ax.set_ylim(0, 0.2)
        ax.set_xticks([3*n for n in range(0,9)])
        #  ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.9)
        ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.6)
        fullpath = os.path.join('Output', 'event_sep_XXX' + '.png')
        fullpath = fullpath.replace('XXX', "{0:03d}".format(n))
        #  plt.savefig(fullpath,
                    #  facecolor=fig.get_facecolor(),
                    #  edgecolor='none')
        #  plt.close(fig)
        plt.show()
        #  import pickle
        #  pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
        #  figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
        #  figx.show() # Show the figure, edit it, etc.!
        #  plt.show()

def calculateCorrelation(data, loadData=False, event_sep=None):
    if loadData:
        #  event_corr = np.load('event_corr_all.npy', allow_pickle=True)
        event_corr = np.load('event_corr.npy', allow_pickle=True)
    else:
        phase_lag = []
        Nlags = 20
        dt_lag = datetime.timedelta(seconds = int(24*60*60/Nlags))
        for i, s in enumerate(dataSelection):
            yy1 = s.FIELDS[1].y
            yy2 = s.FIELDS[2].y
            timeFrom = s.datetime[0]
            ccor_t = []
            ccor_dphase = []
            for ind in range(60, len(yy1)-60):
                tlag = s.datetime[ind]
                lag_hwidth = 61
                ind0 = ind - lag_hwidth
                ind1 = ind + lag_hwidth
                ind0 = 0 if ind0 < 0 else ind0
                ind1 = len(yy1) if ind1 > len(yy1) else ind1
                y1 = yy1[ind0:ind1]
                y2 = yy2[ind0:ind1]
                npts = len(y1)
                sr = 1./s.FIELDS[1].dt
                ccor = signal.correlate(y2, y1, mode='same', method='direct')
                delay_arr = np.linspace(-1*npts/sr, 1*npts/sr, npts)
                delay_arr = delay_arr / (60.*60.) * 180  # In degrees
                ind0 = np.argwhere(delay_arr < -180)[-1][0]
                ind1 = np.argwhere(delay_arr > 180)[0][0]
                ccor[:ind0] = 0
                ccor[ind1:] = 0
                delay = delay_arr[np.argmax(ccor)]
                ccor_t.append(tlag)
                ccor_dphase.append(delay)
            phase_lag.append([ccor_t, ccor_dphase])
            if i%10 == 0:
                print('{:.1f}% completed'.format(i/len(dataSelection)*100))
        np.save('event_corr.npy',  phase_lag)
        event_corr = phase_lag


    def collect_event_ccor(event_sep, event_corr):
        ccors = []
        for i, event in enumerate(event_sep):
            eventFrom = event[2]
            eventTo = event[3]
            for s in event_corr:
                if eventFrom >= s[0][0] and eventFrom <= s[0][-1]:
                    ind0 = np.argwhere(np.asarray(s[0]) >= eventFrom - datetime.timedelta(seconds=0*60*60))[0][0]
                    ind1 = np.argwhere(np.asarray(s[0]) <= eventTo + datetime.timedelta(seconds=0*60*60))[-1][0]
                    c = s[1][ind0:ind1]
                    N = len(c)
                    dateFrom = datetime.datetime.strptime('2000-01-01T00:00:00',
                                                          '%Y-%m-%dT%H:%M:%S')
                    TIME = [dateFrom + datetime.timedelta(seconds=60*n) for n in range(0,N)]
                    # if np.std(c) < 200:
                    ccors.append([TIME, c])
                break
        return ccors

    #  ccors_30 = collect_event_ccor(event_sep_30, event_corr)
    ccors = collect_event_ccor(event_sep, event_corr)
    #  ccors_90 = collect_event_ccor(event_sep_90, event_corr)

    # ccors_bg = []
    # for s in event_corr:
        # corrFrom = s[0][0]
        # corrTo = s[0][-1]
        # corr_mask = np.ones(len(s[0]))
        # n_events = 0
        # for i, event in enumerate(event_sep):
            # if event[2] >= corrFrom and event[2] <= corrTo:
                # ind0 = np.argwhere(np.asarray(s[0]) >= event[2])[0][0]
                # ind1 = np.argwhere(np.asarray(s[0]) <= event[3])[0][0]
                # corr_mask[ind0:ind1] = 0
                # n_events += 1
#
        # c = np.asarray(s[1])[corr_mask > 0]
        # N = len(s[1])
        # TIME = [dateFrom + datetime.timedelta(seconds=60*n) for n in range(0,N)]
        # TIME = np.asarray(TIME)[corr_mask > 0]
        # if n_events > 0:
            # ccors_bg.append([TIME, c])

    # ccors = ccors_bg

    total_ccor_30 = []
    total_ccor_60 = []
    total_ccor_90 = []
    # for ccor in ccors[:300]:
    for ccor in ccors_30:
        total_ccor_30.extend(ccor[1])
    for ccor in ccors_60:
        total_ccor_60.extend(ccor[1])
    for ccor in ccors_90:
        total_ccor_90.extend(ccor[1])

    # ccor_stack = []
    # for ccor in ccors:
        # ccor_stack.append(ccor[1][:360])
    # ccor_stack = np.asarray(ccor_stack)


    plt.style.use('paper.mplstyle')
    clist_friendly = ['#E1DAAE', '#FF934F', '#CC2D35', '#058ED9', '#848FA2', '#2D3142', '#CCCCCC' ]
    fig, ax = plt.subplots(figsize=(12,6))
    data_labels = ['Magnetosphere', 'Magnetosheath', 'Solar Wind', 'Else']
    data_cls = ['#19dfb7', '#f7bb38', '#790501']
    data_cls = ['#058ED9', '#FF934F', '#CC2D35']
    print('Len of data: ', len(data))
    # for s in data_flags:
        # ax.scatter(s.info["LT"], s.info["median_KRTP"][1]*180./np.pi, color='black', s=4)
    for s in data:
        ms = 16 if s.flag is None else 23
        alpha = 0.8 if s.flag is None else 1
        cl = 'black'
        cl = data_cls[0] if s.info["loc"] == 0 else cl
        cl = data_cls[1] if s.info["loc"] == 1 else cl
        cl = data_cls[2] if s.info["loc"] == 2 else cl
        cl = '#2D3142' if s.flag is not None else cl
        marker = 'o' if s.flag is None else 'x'
        loc_id = s.info["loc"]
        loc_id = 3 if loc_id < 0 else loc_id
        loc_id = 3 if loc_id > 3 else loc_id
        label = data_labels[loc_id]
        label = 'Flagged Data' if label == 'Else' else label
        ax.scatter(s.info["LT"], s.info["median_KRTP"][1]*180./np.pi, color=cl, s=ms, alpha=alpha, marker=marker, label=label)
    # plt.xlim(-90, 90)

    ax.set(xlim=(0, 24), xticks=[x*2 for x in range(0, 13)])
    ax.set(ylim=(-70, 70), yticks=[x*10 for x in range(-7, 8)])

    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.8)
    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.6)
    ax.set_xlabel('Local Time [h]', fontsize=16)
    ax.set_ylabel('Magnetic Latitude [deg]', fontsize=16)
    # ax.legend(loc=2, frameon=False)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=4, markerscale=2, columnspacing=1.)

    plt.show()

    #  mplstyle.use(['ggplot'])
    #  plt.style.use('dark_background')
    #  plt.hist(total_ccor_30, bins='auto', facecolor=clist_field[1], alpha=0.8)
    #  plt.hist(total_ccor_60, bins='auto', facecolor=clist_field[2], alpha=0.8)
    #  plt.hist(total_ccor_90, bins='auto', facecolor=clist_field[3], alpha=0.8)
#
    #  # plt.scatter(ccors[0][0][:360], np.mean(ccor_stack, axis=0))
    #  # for ccor in ccors:
        #  # plt.scatter(ccor[0], ccor[1])
        #  # plt.plot(ccor[0], ccor[1])
        #  # plt.title('std = {:.1f}'.format(np.std(ccor[1])))
    #  plt.xlim(-180, 180)
    #  # plt.ylim(-180, 180)
    #  plt.show()
    #  exit()


def calculateSlopes(data):
    slopes = []
    slope_dic = {'Slope': [], 'Segment': [], 'Location': [], 'Local Time': [], 'Latitude': []}

    for x in data:
            if x.mloc in ['MS', 'SH', 'SW']:
                loc_ind = np.argwhere(np.asarray(['MS', 'SH', 'SW']) == x.mloc)[0][0]
                loc = ['Magnetosphere', 'Magnetosheath', 'Solar Wind'][loc_ind]
                lt_string = int((x.lt + 3) // 6)
                lt_string = 0 if lt_string < 0 else lt_string
                lt_string = ['Midnight', 'Dawn', 'Noon', 'Dusk', 'Midnight'][lt_string]
                TH = abs(x.th*180/np.pi)
                if TH >= 0. and TH < 10: lat_string = "Low Lat"
                if TH >= 10 and TH < 30: lat_string = "Mid Lat"
                if TH >= 30: lat_string = "High Lat"
                for i in range(4):
                    for k, seg in enumerate(['Low Frequencies', 'High Frequencies']):
                        slope_dic["Slope"].append(x.fit_coefs[k][i][1])
                        slope_dic["Segment"].append(seg)
                        slope_dic["Location"].append(loc)
                        slope_dic["Local Time"].append(lt_string)
                        slope_dic["Latitude"].append(lat_string)

    slopes = pd.DataFrame(data=slope_dic)
    for col in ['Segment', 'Location', 'Local Time', 'Latitude']:
            slopes[col] = slopes[col].astype('category')
    print(slopes.head())
    print(slopes.tail())
    print(slopes.dtypes)
    def annotate(data, **kws):
            n = np.median(data.Slope)
            ax = plt.gca()
            ax.text(.1, .6, "m = {:.2f}".format(n), transform=ax.transAxes)
    sns.set_theme(style="whitegrid")
    sns.set_theme(style="ticks", palette="pastel")
    sns.set_theme()
    sns.axes_style("darkgrid")
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    tips[(tips.day=='Thur') & (tips.sex=='Female') ] = np.nan
    g = sns.FacetGrid(slopes, col="Local Time", height=4.5, aspect=1.85,
                          col_order=['Midnight', 'Dawn', 'Noon', 'Dusk'],
                          hue_order=['Low Frequencies', 'High Frequencies'],
                          row_order=['Midnight', 'Dawn', 'Noon', 'Dusk'],
                          despine=True,
                          sharex=True,
                          sharey=True,
                          legend_out=True,
                          col_wrap=2,
                          )
    #  Bigger than normal fonts
    sns.set(font_scale=1.2)

    g.map(sns.swarmplot, x="Slope", y="Location", hue="Segment", dodge=True, data=slopes,
              order=['Magnetosphere', 'Magnetosheath', 'Solar Wind'],
              color=".5",
              edgecolor="gray",
              linewidth=1,
              alpha=0.8,
              size=0.1,
              palette="Set2",
              )
    g.map(sns.stripplot,x="Slope", y="Location", hue="Segment",
              order=['Magnetosphere', 'Magnetosheath', 'Solar Wind'],
              hue_order=['Low Frequencies', 'High Frequencies'],
              data=slopes, dodge=True, alpha=.1, zorder=1,
              edgecolor='orange', linewidth=1, size=3,
              jitter=0.3, palette="Set2")

    g.map(sns.boxplot, x="Slope", y="Location", hue="Segment", data=slopes, whis=np.inf,
              order=['Magnetosphere', 'Magnetosheath', 'Solar Wind'],
              hue_order=['Low Frequencies', 'High Frequencies'],
              palette="Set2",
              )

    g = sns.catplot(x="Slope", y="Location",
                        hue="Segment", row="Local Time",
                        data=slopes,
                        orient="h", height=2.5, aspect=4, palette="Set2",
                        hue_order=['Low Frequencies', 'High Frequencies'],
                        row_order=['Midnight', 'Dawn', 'Noon', 'Dusk'],
                        order=['Magnetosphere', 'Magnetosheath', 'Solar Wind'],
                        kind="box", dodge=True,)
    g.map_dataframe(annotate)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
    g.set(xlim=(-5, 0), xticks=[-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0])
    #  Tweak the supporting aspects of the plot
    sns.despine(offset=10, trim=True)
    sns.despine(offset=1)
    g.set_axis_labels("Power Law Slope", "")
    g.tight_layout()
    g.add_legend()
    sns.despine(left=False)
    plt.show()

    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    tips = sns.load_dataset("tips")
    tips[(tips.day=='Thur') & (tips.sex=='Female') ] = np.nan
    print(sns.__version__)
    print(tips.head())
    #  Bigger than normal fonts
    sns.set(font_scale=1.5)
    sns.boxplot(x="total_bill", y="day", data=tips, whis=np.inf, ax=ax)
    sns.swarmplot(x="total_bill", y="day", data=tips, color=".2", ax=ax)
    plt.show()



class WaveSignal():
    ''' Class for a detected wave signal in the data '''
    def __init__(self,
                 dateFrom=None,
                 dateTo=None,
                 datePeak=None,
                 freq=None,
                 amplitude=None,
                 duration=None,
                 power=power,
                 waveforms=None,
                 intervals=None,
                 amplitudes=None,
                 slope=None,
                 sls5_phase=None,
                 axis=None,
                 LT=None,
                 coord_KSM=None,
                 coord_KRTP=None,
                 ):
        self.dateFrom = dateFrom
        self.dateTo = dateTo
        self.datePeak = datePeak
        self.freq = freq
        self.amplitude = amplitude
        self.duration = duration
        self.power = power
        self.waveforms = waveforms
        self.intervals = intervals
        self.amplitudes = amplitudes
        self.slope = slope
        self.sls5_phase = sls5_phase
        self.axis = axis
        self.LT = LT
        self.coord_KSM = coord_KSM
        self.coord_KRTP = coord_KRTP



def readAndPlotFFT(data=None,
                   year=None,
                   coord=None,
                   yearFrom=2004,
                   yearTo=2017,
                   component=0,
                   location='MS',
                   show_all=True,
                   dataFile=None,
                   title=None,
                   save_plots=True,
                   splitByBins=False,
                   splitByTime=False,
                   plot_fft=True,
                   plot_test=False,
                   plot_type = 'components',
                   periodLines=None,
                   vmin=1e-4,
                   vmax=1e4,
                   units=1e3,
                   plot_spec_ylim = (0.5e-4, 0.5e-2),
                   plot_spec_xlim = None,
                   highlight_periods=None,
                   highlight_periods2=None,
                   output_dir=['Output'],
                   filename_out='Cassini_III_FFT_COORD_YYYY_ID',
                   ):

    # Plot Paameters
    rc("text", usetex=False)
    clist = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff', '#fdf33c',  '#ff396a']
    clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff', '#fdf33c',  '#ff396a']
    clist_friendly = ['#E1DAAE', '#FF934F', '#CC2D35', '#058ED9', '#848FA2', '#2D3142']
    clist_friendly = ['#2D3142', '#CC2D35', '#058ED9', '#058ED9', '#848FA2', '#2D3142']
    clist_friendly = ['#FF934F', '#CC2D35', '#058ED9', '#058ED9', '#848FA2', '#2D3142']
    clist_friendly2 = ['#553a9c', '#594095', '#5d468f', '#604b89', '#635183',
                      '#67567e', '#6a5a79', '#6e5e75', '#726271', '#77666d',
                      '#7d696b', '#846c69', '#8d6e69', '#976e6a', '#a56d6d',
                      '#b66975', '#cf5c82', '#ff00a5']
    if periodLines is None:
        periodLines = [T*60 for T in [5, 15, 30, 45, 60, 90, 120, 180]]

    print('--------------------------------------------------------')
    data_noflags = [x for x in data if x.flag is None]
    data_flags = [x for x in data if x.flag is not None]
    print('Data records with no flags: ', len(data_noflags))
    print('TOTAL records: ', len(data))
    print('--------------------------------------------------------')

    # Frequency list for all FFTs
    #  freq = data_noflags[0].FIELDS[0].fft_freq
    #  ind_12h =   np.argwhere(freq > (1/(12*60*60)))[0][0]
    #  ind_10h =   np.argwhere(freq > (1/(10*60*60)))[0][0]
    #  ind_8h =    np.argwhere(freq > (1/(8*60*60)))[0][0]
    #  ind_6h =    np.argwhere(freq > (1/(6*60*60)))[0][0]
    #  ind_4h =    np.argwhere(freq > (1/(4*60*60)))[0][0]
    #  ind_3h =    np.argwhere(freq > (1/(3*60*60)))[0][0]
    #  ind_2h =    np.argwhere(freq > (1/(2*60*60)))[0][0]
    #  ind_1h =    np.argwhere(freq > (1/(1*60*60)))[0][0]
    #  ind_90min = np.argwhere(freq > (1/(90*60)))[0][0]
    #  ind_75min = np.argwhere(freq > (1/(75*60)))[0][0]
    #  ind_45min = np.argwhere(freq > (1/(45*60)))[0][0]
    #  ind_30min = np.argwhere(freq > (1/(30*60)))[0][0]
    #  ind_15min = np.argwhere(freq > (1/(15*60)))[0][0]
    #  ind_10min = np.argwhere(freq > (1/(10*60)))[0][0]
    #  ind_5min =  np.argwhere(freq > (1/(5*60)))[0][0]
    #  ind_end = len(freq) - 1

    # Power Law fit ranges in terms of frequencies
    #  fit_inds = [[ind_8h, ind_3h], [ind_90min, ind_5min]]
    #  fit_out_inds = [[ind_12h, ind_2h], [ind_2h, ind_5min]]
    #  PowerLawSegmenter(data_noflags,
                      #  ind_list=[[ind_10h, ind_2h], [ind_1h, ind_5min]],
                      #  out_ind_list=[[1, ind_2h], [ind_2h, ind_end]],
                      #  deg=1)

    #  list_r, list_L, list_BT, list_BT_model, list_lt, list_th, list_P = serialize(data)
    fft_MS, fft_SH, fft_SW, fft_XX = [], [], [], []
    list_MS, list_SH, list_SW, list_XX = [], [], [], []

    n = 0
    nbins = 200
    BT_bins = np.round(np.exp(np.linspace(np.log(1e-2), np.log(1e1), nbins+1)))
    power_bins = np.exp(np.linspace(np.log(1e0), np.log(1e3), nbins+1))
    BT_bins = np.linspace(0.1, 100., nbins+1)
    th_bins = np.linspace(-80*np.pi/180., 80*np.pi/180., nbins+1)
    LT_bins = np.linspace(0, 24, nbins+1)
    L_bins = np.linspace(4, 25, nbins+1)

    filename_out = filename_out + '_XXX'
    splitByBins = True

    continueProcessing = True
    while continueProcessing:
        dataSelection = data_noflags

        # dataSelection = filterByProperties(dataSelection,
                                           # arg='L',
                                           # minValue = L_bin - 10,
                                           # maxValue = L_bin + 10,
                                           # minValue = L_bins[n],
                                           # maxValue = L_bins[n+1],
                                           # )
        # minValue = L_bins[n]
        # maxValue = L_bins[n+1]

        #  minValue = LT_bins[n]-3
        #  maxValue = LT_bins[n]+3
        #  dataSelection = filterByProperties(dataSelection, arg='LT',
                                           #  minValue = minValue,
                                           #  maxValue = maxValue)
        #  if minValue < 0:
            #  dataSelection0 = filterByProperties(data_noflags, arg='LT',
                                                #  minValue = 24+minValue,
                                                #  maxValue = 24)
            #  dataSelection += dataSelection0
        #  if maxValue > 24:
            #  dataSelection1 = filterByProperties(data_noflags, arg='LT',
                                                #  minValue = 0,
                                                #  maxValue = 0 + (maxValue-24))
            #  dataSelection += dataSelection1

        dataSelection = filterByProperties(dataSelection, arg='r',
                                           minValue = 2,
                                           maxValue = 50,
                                           )

        # dataSelection = filterByProperties(dataSelection, arg='lt',
                                           # minValue = 12-3,
                                           # maxValue = 12+3,
                                           # )
# 
        # dataSelection = filterByProperties(dataSelection,
                                           # arg='BT',
                                           # minValue = 0,
                                           # maxValue = 5,
                                           # minValue = BT_bins[n],
                                           # maxValue = BT_bins[n+1],
                                           # )

        #  dataSelection = filterByProperties(dataSelection,
                                           #  arg='th',
                                           #  minValue = -10*np.pi/180,
                                           #  maxValue = 10*np.pi/180,
                                           #  )
        #  dataSelection = filterByProperties(dataSelection,
                                           #  arg='th',
                                           #  minValue = -10*np.pi/180,
                                           #  maxValue = 10*np.pi/180,
                                           #  )
        # dataSelection += dataSelection2

        # Narrow down the data by location
        list_MS, list_SH, list_SW, list_XX = sortDataByLocation(dataSelection)
        dataSelection = list_MS
        print('data N:', len(dataSelection))
        #  dataSelection = filterByPower(dataSelection, component=2)
        print('Data in MS:', len(list_MS))
        print('Data in SH:', len(list_SH))
        print('Data in SW:', len(list_SW))



        wave_packets = [[] for i in range(4)]
        #  data = data[:20]
#
        #  for i, signal in enumerate(data):
            #  currTime = signal.datetime[0]
            #  for j, s in enumerate(signal.FIELDS):
                #  s.datetime = signal.datetime
                #  waves = collectWaveEvents(s, config={}, COORDS=signal.COORDS, info=signal.info,
                                          #  peak_separation = 3*60*60)
                #  wave_packets[j].extend(waves)
                #  s.datetime = []
#
            #  if i%10 == 0:
                #  print('{:.1f}% completed'.format(i/len(data)*100))
        #  np.save('wave_packets_30.npy',  wave_packets)
        #  exit()

        #  wave_packets = np.load('wave_packets_30.npy', allow_pickle=True)
        wave_packets = np.load('wave_packets_60.npy', allow_pickle=True)

        def LT_2_phi(LT):
            return (LT-12)*np.pi/12

        wp = wave_packets[3]
        phi = []
        r = []
        th = []
        amp = []
        prom = []
        for w in wp:
            if w.amplitude > 0.01 and w.amplitude < 8 and w.power > 0.3:
                phi.append(LT_2_phi(w.LT))
                r.append(w.coord_KRTP[0])
                amp.append(w.amplitude)
                prom.append(w.power)
                th.append(w.coord_KRTP[1]*180/np.pi)
        phi = np.asarray(phi)
        r = np.asarray(r)
        th = np.asarray(th)
        amp = np.asarray(amp)
        prom = np.asarray(prom)


        mplstyle.use(['ggplot'])
        plt.style.use('dark_background')
        matplotlib.rcParams.update({'font.size': 17})
        #  fig, ax = plt.subplots(figsize=(9, 7))
        fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': 'polar'})
        fig.set_facecolor('#171717')

        c = amp
        cmap = plt.cm.get_cmap('turbo')

        #  sc = ax.scatter(r, amp, c=c, cmap=cmap)
        sc = ax.scatter(phi, r, s=30, c=c, cmap=cmap)
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('Amplitude (nT)')
        ax.set_rmax(45)
        ax.set_rticks([10, 20, 30, 40])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.1)
        ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.1)
        ax.set_facecolor('#171717')
        ax.set_xticklabels(['12', '', '18', '', '0', '', '6', ''])

        ax.set_title("55-65 min Wave Events in the Magnetosphere", va='bottom')
        #  ax.set_title("25-35 min Wave Events in the Magnetosphere", va='bottom')
        plt.show()
        exit()


        #  calculateEventSeparation(dataSelection, loadData=True)
        #  calculateCorrelation(data, loadData=False, event_sep=None):
        #  calculateSlopes(data)
        #  print(wave_packets[1][0].LT)
        #  print(wave_packets[1][1].LT)
        #  print(wave_packets[1][0].coord_KRTP)
        #  print(wave_packets[1][1].coord_KRTP)
        exit()

        # Narrow down the data by components
        if plot_type == 'locations':
            fft0 = [x.FIELDS[component].dfft for x in list_MS]
            fft1 = [x.FIELDS[component].dfft for x in list_SH]
            fft2 = [x.FIELDS[component].dfft for x in list_SW]
            fft3 = [x.FIELDS[component].dfft for x in list_XX]
            data_list = [fft0, fft1, fft2]
            data_list_labels = ['Magnetosphere', 'Magnetosheath', 'Solar Wind']
            data_list_colors = ['#19dfb7', '#f7bb38', '#790501']
            # print('MS: ', len(fft0), '| SH: ', len(fft1), 'SW: ',
                  # len(fft2), 'XX: ', len(fft3))

        plot_series = None
        if plot_type == 'components':
            ind = np.argwhere(np.asarray(['MS','SH','SW', 'XX']) == location)[0][0]
            dataSelection = [list_MS, list_SH, list_SW, list_XX][ind]
            fft0 = [x.FIELDS[0].dfft for x in dataSelection]
            fft1 = [x.FIELDS[1].dfft for x in dataSelection]
            fft2 = [x.FIELDS[2].dfft for x in dataSelection]
            fft3 = [x.FIELDS[3].dfft for x in dataSelection]
            data_list = [fft0, fft1, fft2, fft3]
            # data_list_labels = ['Br', 'Bth', 'Bphi', 'Btotal']
            data_list_labels = ['Bpar', 'Bperp1', 'Bperp2', 'Btotal']
            data_list_labels = [location + ' ' + x for x in data_list_labels]
            data_list_colors = [clist[0], clist[1], clist[6], clist[3]]

            power_bins=[ind_4h, ind_3h, ind_2h, ind_90min, ind_75min,
                        ind_1h, ind_45min, ind_30min, ind_15min]

            power_bin_labels=['4h', '3h', '2h', '90min', '75min', '1h', '45min', '30min', '15min']

        if plot_type == 'single':
            fft0 = [x.FIELDS[component].dfft for x in selection]
            data_list = [fft0]
            data_list_labels = [['Br', 'Bth', 'Bphi', 'Btotal'][component]]
            data_list_labels = [location + ' ' + x for x in data_list_labels]
            data_list_colors = [[clist[0], clist[1],
                                clist[6], clist[3]][component]]


        # fft_bg = [getattr(x, 'bg_power_B'+str(component)) for x in selection]
        # data_list.append(fft_bg)
        # data_list_labels.append(['Br BG', 'Bth BG', 'Bphi BG', 'Btotal BG'][component])
        # data_list_colors.append([clist[0], clist[1], clist[6], clist[3]][component])

        list_r, list_L, list_BT, list_BT_model, list_lt, list_th, list_P = serialize(dataSelection)
        BT_median = np.median(list_BT)
        TH_median = np.median(list_th) * 180/np.pi
        R_median = np.median(list_r)
        L_median = np.median(list_L)
        LT_median = np.median(list_lt)
        Nsegments = len(dataSelection)
        # stat_summary = r'Data Segments: {:5d} / Median $B_T$: {:6.1f}nT / Median $\theta$: {:6.0f}deg / Median $R$: {:4.0f}$R_S$ / Median $L$: {:4.0f}$R_S$ / Median LT: {:3.1f}h'.format(Nsegments, BT_median, TH_median, R_median, L_median, LT_median)
        stat_summary = r'Data Segments: {:5d} / Median $B_T$: {:6.1f}nT / Median $\theta$: {:6.0f}deg / Median $R$: {:4.0f}$R_S$ / Median LT: {:3.1f}h'.format(Nsegments, BT_median, TH_median, R_median, LT_median)       # if splitByBins:
            # data_list_labels = ['r=' + str(bin_label) + ' '
                                # + x for x in data_list_labels]

        titleLT = 'LT Bin = {:.1f}'.format(LT_bins[n]) + r'$\pm3$h'
        # titleL = 'L Shell = {:.1f}'.format(L_bins[n]) + r'$\pm3$ $R_S$'
        # titleL = 'L Shell = {:.1f}-{:.1f}'.format(L_bins[n], L_bins[n+1]) + r'$R_S$'
        # titleLT = r'LT Bin = {:.0f}$\pm 3$h'.format(lt_bins[n], lt_bins[n+1])
        # titleLT = 'LT Bin = {:.0f}-{:.0f}h'.format(lt_bins[n], lt_bins[n+1])
        # titleR = 'R Bin = {:.0f}-{:.0f}RS'.format(R_bins[n], R_bins[n+1])
        # titleBT = 'BT Bin = {:.1f}-{:.1f}nT'.format(BT_bins[n], BT_bins[n+1])
        # titleTH = 'TH Bin = {:.0f}-{:.0f}deg'.format(lt_bins[n], lt_bins[n+1])
        # titlePower = r'Power Bin = {:.1e}-{:.1e} $nT^2/Hz$'.format(power_bins[n], power_bins[n+1])
        title = titleLT
        # title = titlePower
        # title = titleL

        mplstyle.use(['ggplot'])
        plt.style.use('dark_background')
        matplotlib.rcParams.update({'font.size': 17})
        # fig, axes = plt.subplots(3, figsize=(18, 9))
        # ax = axes[0]
        # ax2 = axes[1]
        # ax3 = axes[2]
        fig, ax = plt.subplots(figsize=(18, 9))
        fig.set_facecolor('#171717')

        axin1 = ax.inset_axes([0.0, 0.05, 0.18, 0.18])
        axin2 = ax.inset_axes([0.08, 0.05, 0.18, 0.18])
        axin3 = ax.inset_axes([0.24, 0.11, 0.4, 0.03])
        axin4 = ax.inset_axes([0.86, 0.06, 0.03, 0.4])
        #  axin5 = ax.inset_axes([0.96, 0.06, 0.03, 0.4])

        # plot_series = True
        plot_single_fft = False

        if plot_fft:
            # localTimeVisual(axin1, lt_range=None, lt_values=list_lt, color='orange')
            localTimeVisual(axin1, lt_range=[minValue, maxValue], lt_values=list_lt, color='orange')
            latitudeVisual(axin2, lat_range=None, lat_values=list_th, color='orange')
            rangeValueVisual(axin3, data=list_r, data_range=None,
                             xlim=[0, 60], color='orange', label='R')
            PowerVisual(axin4, data=list_P, data_range=None,
                             ylim=[1e-1, 1e4], color='orange', label='Median\nPower\nDensity\n(BT 60m)')
            #  FieldVisual(axin5, data=list_BT, data_range=None,
                             #  ylim=[1, 1e4], color=clist_field[3], label='Median\nMagnetic\nField\n(nT)')


            #  mode_label = dict(units=units, loc=30, lw=2, alpha=0.8, dir='vertical', show_label=True, minimal=True, fontsize=18)
            #  drawPeriodLine(ax, 115*60, color=clist_friendly[0], text='m=2', **mode_label)
            #  drawPeriodLine(ax, 80*60,  color=clist_friendly[1], text='m=3', **mode_label)
            #  drawPeriodLine(ax, 50*60,  color=clist_friendly[2], text='m=4', **mode_label)
            #  drawPeriodLine(ax, 38*60,  color=clist_friendly[3], text='m=5', **mode_label)
            #  drawPeriodLine(ax, 30*60,  color=clist_friendly[4], text='m=6', **mode_label)
#
            ax.text(0.01, 0.97, stat_summary, color='white', alpha=0.6,
                    ha='left', va='center', transform=ax.transAxes)

            PlotFFT(ax,
                    freq,
                    data_list = data_list,
                    data_list_labels = data_list_labels,
                    data_list_colors = data_list_colors,
                    xlim=plot_spec_ylim,
                    ylim=(vmin, vmax),
                    units=units,
                    periodLines = periodLines,
                    highlight_periods=highlight_periods,
                    highlight_periods2=highlight_periods2,
                    label='',
                    color=clist_field[0],
                    noTicks=False,
                    minimal_axis=True,
                    show_all=show_all,
                    # show_all=False,
                    minor_alpha=0.03,
                    # minor_alpha=0.85,
                    # fit_inds = fit_inds,
                    # fit_out_inds = fit_out_inds,
                    plot_power_diff=True,
                    )
        # plot_series = None
        plot_single_fft = None
        if plot_single_fft is not None:
            PlotFFT(axes[1],
                    freq,
                    data_list = plot_series_fft,
                    data_list_labels = data_list_labels,
                    data_list_colors = data_list_colors,
                    xlim=plot_spec_ylim,
                    ylim=(vmin, vmax),
                    units=units,
                    periodLines = periodLines,
                    highlight_periods=highlight_periods,
                    highlight_periods2=highlight_periods2,
                    label='',
                    color=clist_field[0],
                    noTicks=False,
                    minimal_axis=True,
                    show_all=True,
                    # show_all=False,
                    minor_alpha=0.01,
                    # fit_inds = fit_inds,
                    # fit_out_inds = fit_out_inds,
                    )
        if plot_series is not None:

            ax2 = axes[2]
            dateFrom = datetime.datetime.strptime('2000-01-01T00:00:00',
                                                  '%Y-%m-%dT%H:%M:%S')
            dt = 60
            N = len(plot_series[0])
            TIME = [dateFrom + datetime.timedelta(seconds=dt*n) for n in range(0,N)]

            # ax2.plot(TIME, plot_series[0], label='Br', color=clist[0])
            ax2.plot(TIME, plot_series[1], label='Bth', color=clist[1])
            ax2.plot(TIME, plot_series[2], label='Bphi', color=clist[2])
            # ax2.plot(TIME, plot_series[3], label='BT', color=clist[3])

            # Formatting and Styling
            # GRID
            xaxis_format = "%H:%M"
            majorHourLocator = dates.HourLocator(interval=2)
            minorHourLocator = dates.HourLocator(interval=1)
            dfmt = dates.DateFormatter(xaxis_format)
            ax2.xaxis.set_major_formatter(dfmt)
            ax2.xaxis.set_major_locator(majorHourLocator)
            ax2.xaxis.set_minor_locator(minorHourLocator)

            ax2.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.2)
            ax2.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.1)
            # ax2.set_xlabel('T, fontsize=fs)
            ax2.set_ylabel('[nT]', fontsize=16)
            # LEGENDS
            ax2.legend(loc=2, frameon=False)
            # ax2.set(frame_on=False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(True)
            ax2.spines['left'].set_visible(False)
            # ax2.tick_params(axis='y', labelsize=10)
            # ax2.tick_params(axis='x', labelsize=10)
            ax2.patch.set_alpha(0.0)

        if plot_test:
            # fft_bins_count = [len(f) for f in fft_bins]
            # ax.plot(fft_labels, fft_bins_count)
            # ax.plot(bin_label, len(selection))
            print('Plotting Test')

            list_r = np.asarray(list_r)
            list_BT = np.asarray(list_BT)
            list_BT_model = np.asarray(list_BT_model)
            list_lt = np.asarray(list_lt)
            list_th = np.asarray(list_th)
# 
            # xdata = list_r
            # xdata = list_th*180./np.pi
            xdata = list_lt
            ydata = list_BT
            ydata2 = list_BT_model
            # ydata = list_th*180./np.pi
            # ydata = list_r
            # label = r'Cassini $B_T$'
            # label = None
            # label2 = r"Khurana Model $B_T$"
            # ax.scatter(xdata, ydata, color=clist[1], label=label, alpha=0.6, s=8)
            # ax.scatter(xdata, ydata2, color=clist[0], label=label2, alpha=0.6, s=8)

            nbins = 1
            counts = []
            bin_labels = []
            L_bins = np.linspace(0, 80, nbins+1)
            while continueProcessing:
                dataSelection = filterByProperties(data,
                                                   arg='r',
                                                   minValue = L_bins[n],
                                                   maxValue = L_bins[n+1],
                                                   )
                counts.append(len(dataSelection))
                bin_labels.append(L_bins[n])
                n += 1
                if n > nbins-2:
                    continueProcessing = False

            # ax.scatter(bin_labels, counts, color=clist[0], alpha=0.6, s=8)
            ax.bar(bin_labels, counts, color=clist[1], alpha=0.8)

            ax.set_xlabel('R (RS)')
            # ax.set_xlabel(r'L Shell ($R_S$)')
            # ax.set_xlabel('TH [deg]')
            # ax.set_xlabel('Local Time [h]')
            # ax.set_xlabel('LT [h]')
            # ax.set_ylabel('BT (nT)')
            # ax.set_ylabel('TH [deg]')
            # ax.set_ylabel('R [RS]')
            # ax.set_yscale('log')
            # ax.set_ylabel('Counts')
            ax.set_ylabel('Days')
            # ax.legend()

        if title is not None:
            fig.suptitle(title)
        else:
            # fig.suptitle(r'$B_{\phi}$ Power Density | ' + dateFrom.strftime('%Y-%m-%d') + ' - ' + dateTo.strftime('%Y-%m-%d'))
            fig.suptitle(r'$B_{\phi}$ Power Density | ' + data_list_labels[0])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_linestyle('-')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_linestyle('-')
        # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.5, lw=0.5)
        ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.3)
        ax.patch.set_alpha(0.0)

        if save_plots:
            output_dir = np.atleast_1d(output_dir)
            if not os.path.exists(os.path.join(*output_dir)):
                os.makedirs(os.path.join(*output_dir))
            fullpath = os.path.join(*output_dir, filename_out + '.png')
            # fullpath = fullpath.replace('YYYY', str(yearList[0]))
            fullpath = fullpath.replace('XXX', "{0:03d}".format(n))
            plt.savefig(fullpath,
                        facecolor=fig.get_facecolor(),
                        edgecolor='none')
            fig.clear()
            plt.close(fig)
        else:
            plt.show()

        n += 1
        # print('N: ', n, ' | ', dateFrom.strftime('%Y-%m-%d')
              # + ' - ' + dateTo.strftime('%Y-%m-%d'))
        print('N: ', n, ' | ', title)

        # if n > 0:
            # continueProcessing = False

        #  if splitByTime and dateTo >= dateMissionTo:
            #  continueProcessing = False
        # if splitByBins and n > len(fft_bins)-1:
            # continueProcessing = False
        #  if splitByBins and n > nbins-1:
        if n > nbins-1:
            continueProcessing = False
        #  if n > 0 and not splitByBins and not splitByTime:
            #  continueProcessing = False
