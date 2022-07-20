import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from collections import OrderedDict
from matplotlib import dates
import copy
from cassinilib import *
from scipy.integrate import simps
from numpy import trapz
import matplotlib.style as mplstyle
import argparse
from cassinilib.Event import *
from matplotlib import rc
from scipy import signal
from scipy.signal import find_peaks
rc('text', usetex=True)  # For more extensive LaTeX features beyond mathtex
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

clist_field = ['#fc4267', '#0acaff', '#0aff9d', 'white']
clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff',
               '#fdf33c',  '#ff396a', 'red', 'green', 'blue', 'black', 'white',
               '#fc4267', '#0acaff', '#0aff9d', 'white', 'cyan', 'grey', 'yellow']

class Interval():
    def __init__(self,
                 dateFrom=None,
                 dateTo=None,
                 comment=None,
                 ):
        """ Interval class to keep track of data discontinuities """
        self.dateFrom = dateFrom
        self.dateTo = dateTo
        self.comment = comment


class FFT_list():
    def __init__(self,
                 datetime=None,
                 freq=None,
                 fft0=None,
                 fft1=None,
                 fft2=None,
                 fft3=None,
                 name=None,
                 flag=None,
                 NaNs=None,
                 B0=None,
                 B1=None,
                 B2=None,
                 B3=None,
                 R0=None,
                 R1=None,
                 R2=None,
                 r=None,
                 lt=None,
                 th=None,
                 pos_KSM=None,
                 BT=None,
                 mloc=None,
                 ):
        """ Collection of FFT spectra for a given time """
        self.datetime = datetime
        self.freq = freq
        self.fft0 = fft0
        self.fft1 = fft1
        self.fft2 = fft2
        self.fft3 = fft3
        self.name = name
        self.B0 = B0
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.flag = flag
        self.NaNs = NaNs
        self.r = r
        self.lt = lt
        self.th = th
        self.pos_KSM = pos_KSM
        self.BT = BT
        self.mloc = mloc


class SignalSnapshot():
    def __init__(self,
                 datetime=None,
                 FIELDS=None,
                 COORDS=None,
                 flag=None,
                 info={},
                 ):
        ''' Storage for all signals (fields/coords) for a give time period '''
        self.datetime = datetime
        self.FIELDS = FIELDS
        self.COORDS = COORDS
        self.flag = flag
        self.info = info

def str2datetime(string, format='%Y-%m-%dT%H:%M:%S'):
    ''' String to a datetime object, using a given format '''
    return datetime.datetime.strptime(string, format)


def check_for_scas_times(dateFrom, dateTo, cal_treshold=60*60):
    ''' Check if there are calibration events within a datetime interval '''
    # Cassini SCAS (Calibration) Time File
    SCAS_TIMES = np.load('SCAS_TIMES.npy', allow_pickle=True)
    cal_list = []

    flag = None
    for scas in SCAS_TIMES:
        if (scas.dateFrom < dateFrom and scas.dateTo > dateFrom and scas.dateTo < dateTo):
            cal_list.append([dateFrom, scas.dateTo, scas.comment2])
        if (scas.dateFrom > dateFrom and scas.dateTo < dateTo):
            cal_list.append([scas.dateFrom, scas.dateTo, scas.comment2])
        if (scas.dateTo > dateTo and scas.dateFrom > dateFrom and scas.dateFrom < dateTo):
            cal_list.append([scas.dateFrom, dateTo, scas.comment2])
    for cal in cal_list:
        cal_dur = (cal[1] - cal[0]).total_seconds()
        if cal_dur > cal_threshold:
            flag = cal[2] if flag is None else flag + ' ' + cal[2]
    return cal_list, flag


def find_gaps_in_data(time, dateFrom, dateTo, gap_threshold):
    ''' Find data gaps in a datetime array based on the threshold '''
    gap_list = []
    flag = None

    # Check if the first datapoint is much past the expected time
    if (time[0] - dateFrom).total_seconds() > gap_threshold:
        gap_list.append([dateFrom, time[0], 'GAP'])

    time_disc = [(time[n+1]-time[n]).total_seconds() for n in range(len(time)-1)]
    for i, disc in enumerate(time_disc):
        if disc > gap_threshold:
            gap_list.append([time[i], time[i+1], 'GAP'])

    # Check if the last point much earlier than the expected time
    if (dateTo - time[-1]).total_seconds() > gap_threshold:
        gap_list.append([time[-1], dateTo, 'GAP'])

    # Finally, check if the any of the gaps are critically big
    for gap in gap_list:
        gap_dur = (gap[1] - gap[0]).total_seconds()
        if gap_dur > gap_flag_threshold:
            flag = gap[2] if flag is None else flag + ' ' + gap[2]
    return gap_list, flag

# --------------------------------------------------------------------------
# Command line aguments
parser = argparse.ArgumentParser(description='Analyze Data Gaps')
parser.add_argument('--readOnly', dest='readOnly',
                    action='store_true', help='Process the Data')
parser.add_argument('--synthYear', dest='synthYear',
                    action='store_true', help='Synthesize a whole year')
parser.add_argument('--save', dest='save_plots',
                    action='store_true', help='Save plots instead of showing')
parser.add_argument('--plot', dest='plot',
                    action='store_true', help='Plot the Data and Spectra')
parser.add_argument('--plot_all', dest='plot_all',
                    action='store_true', help='Plot FFTs for the whole time')
parser.add_argument('--plot_all_type', dest='plot_all_type', type=str,
                    choices=['single', 'locations', 'components'],
                    help='The type of FFT component / location data to plot')
parser.add_argument('--verbose', dest='verbose',
                    action='store_true', help='Print progress')
parser.add_argument('--Bcoord', dest='Bcoord',
                    action='store_true', help='Transform to B-aligned coords')
parser.add_argument('--coord', dest='coord', type=str, default='KSM',
                    choices=['KSM', 'KRTP', 'KSO', 'RTN'],
                    help='Coordinate System')
parser.add_argument('--source', dest='dataSource', type=str, default='synth',
                    help='Event data, the whole mission, or synthetic data',
                    choices=['synth', 'synthfull', 'events', 'events2', 'mission'])
parser.add_argument('--year', dest='year', type=int,
                    help='Year to analyze the data')
parser.add_argument('--dateFrom', dest='dateFrom', type=str,
                    help='Datetime string to start (e.g., 2000-01-01)')
parser.add_argument('--dateTo', dest='dateTo', type=str,
                    help='Datetime string to end (e.g., 2000-02-01)')
parser.add_argument('--component', dest='component', type=int,
                    help='Component of the field to plot')
parser.add_argument('--location', dest='location', type=str,
                    choices=['MS', 'SH', 'SW', 'XX'],
                    help='Position within/outside the Magnetosphere')
parser.add_argument('--window', dest='window', type=float,
                    help='Length of an FFT window in seconds')
parser.add_argument('--Tavg', dest='Tavg', type=float,
                    help='Moving average time')
parser.add_argument('--ID', dest='ID', type=int,
                    help='ID for distinguishing runs / plots')
args = parser.parse_args()

#============================== Constants ============================

hour = 60*60
mins = 60

#============================== Parameters ===========================

dataInstrument = 'MAG'
dataMeasurement = '1min'
if args.dataSource in ['synthfull']:
    dataInstrument = 'SIM'
    dataMeasurement = '1min'
dataMargin = 0
dataReadInSegments = True

# SIGNAL PARAMETERS
signal_padding = 4*hour
window_size = (24*hour+2*signal_padding) if args.window is None else args.window
signal_dt = 60
signal_interval = window_size
resample_interval = window_size
time_margin = 0. # Margin to each side of the data interval 
N_interval = int(signal_interval // signal_dt)

# FFT PARAMETERS
noverlap_method = 'half'
signal_seg = window_size // 4
N_seg = int(signal_seg // signal_dt)
NFFT = int(4 * N_seg)
Tavg = 3 * hour if args.Tavg is None else args.Tavg
freq_lim = (1/(Tavg), 1/(hour/6))

# SIGNAL MODIFICATION (USED FOR SYNTHETIC SIGNALS)
analyzeData = False
add_noise = False
add_data_gaps = False
generateLongSignalToFile = args.synthYear
artificialNoiseAmp = 0.01
NaN_count_max = 2 * 60

COORD = args.coord
YEAR = args.year
DATEFROM = args.dateFrom
DATETO = args.dateTo
COMPONENT = 0 if args.component is None else args.component
LOCATION = 'MS' if args.location is None else args.location
dataSource = args.dataSource
transform_to_B_coords = args.Bcoord
continueProcessing = not args.readOnly
VERBOSE = args.verbose

# OPTIONS TO BE SENT TO THE SIGNAL PROCESSOR
signal_options = {
    'dt' : signal_dt,
    'coord' : COORD,
    'resample_total_time' : resample_interval,
    #  'actions' : ['resample', 'fft'],
    'actions' : ['resample'],
    'freq_lim': freq_lim,
    'window': ('blackman'),
    'nperseg' : N_seg,
    'NFFT' : NFFT,
    'noverlap_method' : noverlap_method,
    'NaN_method' : 'cubic',
}

# synth = syntheticEvents()

# FFT COLLECTION
collect_FFTs = True if not args.readOnly else False
collect_FFTs = True
plot_all_FFTs = args.plot_all
plot_all_fft_type = 'components' if args.plot_all_type is None else args.plot_all_type
# plot_all_fft_type = 'components'
# plot_all_fft_type = 'locations'

#=========================== Plot Parameters =========================

save_plots = args.save_plots
plot_segments = args.plot
plot_fft_snaps = False
plot_cwt = False
plot_cwt_timeseries = False
plot_event_measure = False
plot_spec = False
plot_phase = False
plot_wspec = False
plot_org_series = True
plot_dfft = False
plot_infoaxis = True
plot_component = 'Bx'  # 'Bx', 'All'
plot_fft_nsnaps = 4
# plot_spec_ylim = (1e-4, 0.5e-1) # Hz
# plot_spec_ylim = (1e-5, 0.5e-2) # Hz
plot_spec_ylim = (0.2e-4, 0.2e-2) # Hz
plot_spec_xlim = None
plot_units = 1e-3  # mHz
plot_vmin = 1e-3
plot_vmax = 1e5
# plot_vmin = 1e-9 # default
# plot_vmax = 5e8 # default
# plot_vmin = -1e4
# plot_vmax = 1e4
# plot_vmax = 5e9
# plot_periodLines = [T*60 for T in [5, 10, 15, 30, 60, 90, 120, 180, 6*60, 12*60, 24*60]]
plot_periodLines = [T*60 for T in [10, 15, 30, 60, 90, 120, 180, 6*60, 12*60]]
highlight_periods = []
highlight_periods2 = []
FILENAME_FFT_PLOT = 'Cassini_III_FFT_COORD_YYYY_XXX'
FILENAME_FFT_ARCHIVE = 'Cassini_III_FFT_COORD_YYYY'
FILENAME_FFT_ARCHIVE_PLOT = 'Cassini_III_FFT_COORD_YYYY'
ID = args.ID
if ID is not None:
    FILENAME_FFT_ARCHIVE_PLOT += '_ID'
OUTPUT_DIR = ['Output']

title = None
# title = 'Running Average of {:.0f} min'.format(Tavg/60)
title = 'Window of {:.1f} h'.format(window_size/60/60)
# title = 'Wave Period: {:.1f}min.'.format(synth_options["period"])
# title = ("Event {0:0=3d}".format(n+1) + ': ' +
         # interval.dateFrom.strftime('%Y-%m-%d %H:%M') +
         # ' - ' +
         # interval.dateTo.strftime('%Y-%m-%d %H:%M'))

#======================================================================
#               Cassini Location File (collected every hour)
#======================================================================

MLoc_file = np.load('CassiniLocation_KSM.npy', allow_pickle=True)
MLoc_time = MLoc_file[:, 0]
# B1 = MLoc_file[:, 1]
# B2 = MLoc_file[:, 2]
# B3 = MLoc_file[:, 3]
MLoc_BT = MLoc_file[:, 4]
MLoc_R1 = MLoc_file[:, 5]
MLoc_R2 = MLoc_file[:, 6]
MLoc_R3 = MLoc_file[:, 7]
# G =  MLoc_file[:, -3]
# Dir =MLoc_file[:, -2]
MLoc = MLoc_file[:, -1]
MLoc = np.asarray(MLoc)

CROSSINGS = np.load('CROSSINGS.npy', allow_pickle=True)
# Cassini SCAS (Calibration) Time File
# SCAS_TIMES = np.load('SCAS_TIMES.npy', allow_pickle=True)
#=====================================================================

if generateLongSignalToFile:
    generateLongSignal()

#=====================================================================
#                            Simulated Signals
#=====================================================================
synth = {}
# synth['periods'] =    [9*hour,  6*hour, 3*hour, 2*hour, 60*mins, 30*mins, 15*mins, 10*mins, 5*mins, ]
# synth['amplitudes'] = [0.3,       0.1,      0.2,    0.2,    0.05,     0.2,     0.2,     0.2,    0.2,    ]
# synth['shifts'] =     [3*hour,  0*hour, 0*hour, 12*hour,3*hour,  6*hour,  10*hour, 8*hour, 3*hour, ]
# synth['periods'] =    [60*mins, 15*mins, 90*mins, 30*mins, 4*hour]
# synth['amplitudes'] = [0.5,     0.2,     1,       0.1,     2]
# synth['shifts'] =     [3*hour,  12*hour, 18*hour, 12*hour, 12*hour]
synth['periods'] =    [60*mins, 30*mins, 60*mins]
synth['amplitudes'] = [0.5,     0.3   ,  0.4]
synth['shifts'] =     [2*hour,  18*hour, 12*hour]
synth['phases'] =     [0 * period for period in synth['periods']]
synth['decays'] =     [2*period for period in synth['periods']]
synth['cutoffs'] =    [None for period in synth['periods']]
synth['types'] =      ['sine' for period in synth['periods']]
#=====================================================================
#                           Process Data Times
#=====================================================================
intervals = list()

if dataSource == 'events':
    eventFile = 'events_proposal.dat'
    # eventFile = 'events_scas.dat'
    try:
        f = np.loadtxt(eventFile, usecols=(0,1), dtype='str')
        for n in range(len(f)):
            # if n > 0: break
            dateFrom = str2datetime(f[n, 0])
            dateTo = str2datetime(f[n, 1])
            intervals.append(Interval(dateFrom, dateTo))
    except FileNotFoundError:
        print("file {} does not exist".format(fname))

if dataSource == 'events2':
    # intervals.append(Interval(str2datetime('2007-02-01T00:00:00'), str2datetime('2007-02-02T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-02T00:00:00'), str2datetime('2007-02-03T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-03T00:00:00'), str2datetime('2007-02-04T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-04T00:00:00'), str2datetime('2007-02-05T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-05T00:00:00'), str2datetime('2007-02-06T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-06T00:00:00'), str2datetime('2007-02-07T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-07T00:00:00'), str2datetime('2007-02-08T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-08T00:00:00'), str2datetime('2007-02-09T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-09T00:00:00'), str2datetime('2007-02-10T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-10T00:00:00'), str2datetime('2007-02-11T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-11T00:00:00'), str2datetime('2007-02-12T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-12T00:00:00'), str2datetime('2007-02-13T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-13T00:00:00'), str2datetime('2007-02-14T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-14T00:00:00'), str2datetime('2007-02-15T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-15T00:00:00'), str2datetime('2007-02-16T00:00:00')))

    # intervals.append(Interval(str2datetime('2007-02-16T00:00:00'), str2datetime('2007-02-17T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-17T00:00:00'), str2datetime('2007-02-18T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-18T00:00:00'), str2datetime('2007-02-19T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-19T00:00:00'), str2datetime('2007-02-20T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-20T00:00:00'), str2datetime('2007-02-21T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-21T00:00:00'), str2datetime('2007-02-22T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-22T00:00:00'), str2datetime('2007-02-23T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-23T00:00:00'), str2datetime('2007-02-24T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-24T00:00:00'), str2datetime('2007-02-25T00:00:00')))

    # intervals.append(Interval(str2datetime('2006-09-27T00:00:00'), str2datetime('2006-09-28T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-11-10T00:00:00'), str2datetime('2006-11-11T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-11-23T00:00:00'), str2datetime('2006-11-24T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-12-06T00:00:00'), str2datetime('2006-12-07T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-12-18T00:00:00'), str2datetime('2006-12-19T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-12-21T00:00:00'), str2datetime('2006-12-22T00:00:00')))
    # intervals.append(Interval(str2datetime('2006-12-22T00:00:00'), str2datetime('2006-12-23T00:00:00')))
    # intervals.append(Interval(str2datetime('2007-02-08T00:00:00'), str2datetime('2007-02-09T00:00:00')))
    intervals.append(Interval(str2datetime('2008-02-29T00:00:00'), str2datetime('2008-03-01T00:00:00')))
    # intervals.append(Interval(str2datetime('2008-08-10T00:00:00'), str2datetime('2008-08-11T00:00:00')))
    # intervals.append(Interval(str2datetime('2008-09-09T00:00:00'), str2datetime('2008-09-10T00:00:00')))
    # intervals.append(Interval(str2datetime('2009-02-23T00:00:00'), str2datetime('2009-02-24T00:00:00')))
    # intervals.append(Interval(str2datetime('2012-10-18T00:00:00'), str2datetime('2012-10-19T00:00:00')))
    # intervals.append(Interval(str2datetime('2013-08-16T00:00:00'), str2datetime('2013-08-17T00:00:00')))

elif dataSource == 'synth':
    T0 = '2000-01-01T00:00:00'
    T1 = '2000-01-01T10:00:00'
    dateFrom = str2datetime(T0)
    dateTo = str2datetime(T1)
    intervals.append(Interval(dateFrom, dateTo))
elif dataSource in ['mission', 'synthfull']:
    if dataSource == 'synthfull':
        T0 = '2000-01-01T00:00:00'
        T1 = '2000-03-15T00:00:00'
    else:
        if YEAR == 2004:
            T0 = '2004-06-30T00:00:00'
            T1 = '2005-01-01T00:00:00'
        elif YEAR == 2017:
            T0 = '2017-01-01T00:00:00'
            # T1 = '2017-09-15T00:00:00'
            #  T1 = '2017-08-30T00:00:00'
            T1 = '2017-09-13T00:00:00'
        else:
            if YEAR is None:
                T0 = '2004-06-30T00:00:00'
                # T1 = '2017-09-15T00:00:00'
                #  T1 = '2017-08-30T00:00:00'
                #  T0 = '2017-08-13T00:00:00'
                T1 = '2017-09-13T00:00:00'
                #  T1 = '2004-08-30T00:00:00'
            else:
                T0 = str(YEAR) + '-01-01T00:00:00'
                # T1 = str(YEAR) + '-03-15T00:00:00'
                T1 = str(YEAR+1) + '-01-01T00:00:00'
    if DATEFROM is not None and DATETO is not None:
        T0 = DATEFROM + 'T00:00:00'
        T1 = DATETO + 'T00:00:00'
    dateMissionFrom = str2datetime(T0)
    dateMissionTo = str2datetime(T1)
    missionDuration = (dateMissionTo - dateMissionFrom).total_seconds()
    #  print('Mission Days = ', missionDuration / 24 / 60 / 60)
    dateFrom = dateMissionFrom - datetime.timedelta(seconds=signal_padding)
    dateTo = dateFrom + datetime.timedelta(seconds=signal_interval)
    intervals.append(Interval(dateFrom, dateTo))

#=====================================================================
#                          START THE PROCESSING 
#=====================================================================
timeseries = []
signals = []

n_max = 10000
#  n_max = 0
n = 0

continueProcessing = False
while continueProcessing:
    # Readjust data times
    if dataSource in ['mission', 'synthfull']:
        dateFrom = dateTo - datetime.timedelta(seconds=2*signal_padding)
        dateTo = dateFrom + datetime.timedelta(seconds=signal_interval)
    else:
        ind = n if n < len(intervals) else 0
        interval = intervals[ind]
        dateFrom = interval.dateFrom
        dateTo = interval.dateTo
    if dataMargin:
        dateFrom = dateFrom - datetime.timedelta(seconds=dataMargin)
        dateTo = dateTo + datetime.timedelta(seconds=dataMargin)
    if dataReadInSegments:
        dateTo = dateFrom + datetime.timedelta(seconds=signal_interval)
    #  print(dateFrom, ' -- ', dateTo)
    #=================================================================
    #                    LOAD OR SYNTHESIZE THE DATA
    #=================================================================

    if dataSource == 'synth':
        timeseries = simulateSignal(N=N_interval,
                                    # periods=synth['periods'],
                                    # amplitudes=synth['amplitudes'],
                                    # shifts=synth['shifts'],
                                    # decays=synth['decays'],
                                    # phases=synth['phases'],
                                    # cutoffs=synth['cutoffs'],
                                    # types=synth['types'],
                                    **signal_options)
    else:
        timeseries = readSignal(dateFrom,
                                dateTo,
                                instrument = dataInstrument,
                                measurement = dataMeasurement,
                                file = None,
                                **signal_options)

    #=================================================================
    #      ANALYSIS - CALCUTATE RUNAVG, FFT, AND FIND THE PEAKS
    #=================================================================
    # Let's first assume data is corrupted
    NaN_count = 0
    flag = None

    if timeseries:
        COORDS = [s for s in timeseries if s.kind == 'Coord']
        LTS =    [s for s in timeseries if s.kind == 'lt']
        FIELDS = [s for s in timeseries if s.kind == 'Field']

        cal_list = []
        gap_list = []
        cal_threshold = 60*60
        gap_threshold = 10*60
        gap_flag_threshold = 60*60

        data = FIELDS[0].y
        time = FIELDS[0].datetime
        # Check for a total number of NaNs (or missing data)
        NaN_count = (signal_interval/signal_dt
                     - np.count_nonzero(~np.isnan(data)))

        # Check for SCAS MAG Calibration events
        if dataSource not in ['synth', 'synthfull']:
            cal_list, cal_flag = check_for_scas_times(dateFrom, dateTo, cal_threshold)
            if cal_flag is not None:
                flag = cal_flag if flag is None else flag + ' ' + cal_flag
                processData = False
                print('** Skipping due to {0} CAL'.format(flag))

        # If a sufficient number of missing/NaN points, collect invalid times
        if NaN_count > gap_threshold // FIELDS[0].dt:
            gap_list, gap_flag = find_gaps_in_data(time, dateFrom, dateTo, gap_threshold)
            if gap_flag is not None:
                flag = gap_flag if flag is None else flag + ' ' + gap_flag
                processData = False
                print('** Skipping due to NaN count: {:d}'.format(int(NaN_count)))

        for s in FIELDS:
            # s.y_, s.datetime_ = s.y, s.datetime
            s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)
        for s in COORDS:
            s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)
        for s in LTS:
            s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)

        TIME = FIELDS[0].datetime

        if collect_FFTs:
            # No need to store the same datetime array for every component
            for s in COORDS:
                s.datetime = []
            for s in FIELDS:
                s.datetime = []
            for s in LTS:
                s.datetime = []

            R1 = np.median(COORDS[0].y)
            R2 = np.median(COORDS[1].y)
            R3 = np.median(COORDS[2].y)
            BT = np.median(FIELDS[3].y)
            BTstd = np.std(FIELDS[3].y)
            LT = np.median(LTS[0].y) if LTS else None
            loc, cl, ind = magnetospherePosition(dateFrom,
                                                 MLoc_time,
                                                 MLoc,
                                                 returnIndex=True)
            rKSM = [MLoc_R1[ind], MLoc_R2[ind], MLoc_R3[ind]]
            median_KRTP = [R1, R2, R3]
            signals.append(SignalSnapshot(
                datetime = TIME,
                FIELDS = FIELDS,
                COORDS = COORDS,
                flag = flag,
                info = dict(BT = BT,
                            BTstd = BTstd,
                            LT = LT,
                            loc = loc,
                            rKSM = rKSM,
                            median_KRTP = median_KRTP,
                            NaNs = NaN_count,
                            gap_list = gap_list,
                            cal_list = cal_list,
                            MAGrange = None,
                            comment = None,
                )))

        # TEST INTERPOLATION
        # if NaN_count >= 2*60 and NaN_count < 8*60:
            # plot_field_interpolation(FIELDS, gap_list, cal_list)

        processData = False
        if processData:
            #=============== TRANSFORM TO MAGNETIC COORDINATES ===========
            # TODO: Check the field transformation

            if transform_to_B_coords:
                Navg = int(1/freq_lim[0] // signal_dt)
                for s in FIELDS:
                    run_avg = s.detrender(Navg=Navg, mode='exclude', returnOnly=True)
                    s.dy = s.y - run_avg
                    s.run_avg = run_avg
                BFIELDS = fieldTransform(COORDS, FIELDS, 'KSM')
                for i in range(4):
                    FIELDS[i].y = BFIELDS[i].y
                    FIELDS[i].name = BFIELDS[i].name
                    FIELDS[i].dy = []
                    FIELDS[i].run_avg = []
                    # plot_field_aligned_test(COORDS, FIELDS, BFIELDS=BFIELDS)

            for s in FIELDS:
                if add_noise:
                    s.add_noise(simple=True, sigma=artificialNoiseAmp)
                if add_data_gaps:
                    s.add_NaNs(data_gap_from, data_gap_to)
                if analyzeData:
                    # s.analyze()
                    s.analyze(actions=['all'])
            #  for s in FIELDS:
                #  s.dfft = np.exp(np.log(s.fft) - np.log(FIELDS[3].fft_bg))

            if plot_segments and plot_fft_snaps:
                if plot_fft_nsnaps > len(FIELDS[0].S_tseg):
                    plot_fft_nsnaps = len(FIELDS[0].S_tseg)
                if plot_fft_nsnaps < 2:
                    plot_fft_nsnaps = 1

        #=============================================================
        #                       PLOT THE DATA
        #=============================================================

        if plot_segments:
            if plot_component == 'All':
                if COORD == 'KRTP':
                    plotSeries = [['Br'], ['Bth'], ['Bphi'], ['Btot']]
                else:
                    plotSeries = [['Bx'], ['By'], ['Bz'], ['Btot']]
                if transform_to_B_coords:
                    plotSeries = [['Bpar'], ['Bperp1'], ['Bperp2'], ['Btot']]
                ncols = 2 if (plot_fft_snaps or plot_spec or plot_wspec) else 1
            else:
                if plot_component is None:
                    if COORD == 'KRTP':
                        plot_component = 'Bphi'
                    else:
                        plot_component = 'Bx'
                    if transform_to_B_coords:
                        plot_component = 'Bperp1'
                # plotSeries = [[plot_component]]
                if COORD == 'KRTP':
                    plotSeries = [['Br', 'Bth', 'Bphi', 'Btot']]
                if COORD == 'KSM':
                    plotSeries = [['Bx', 'By', 'Bz', 'Btot']]
                if transform_to_B_coords:
                    plotSeries = [['Bpar', 'Bperp1', 'Bperp2']]
                # ncols = 2 if (plot_fft_snaps) else 1
                ncols = 1

            nrows = len(plotSeries)
            plot_extras = []
            height_ratios = [4] * len(plotSeries)
            if plot_fft_snaps and plot_component != 'All':
                plot_extras.append('snaps')
                height_ratios.append(4)
            if plot_spec and plot_component != 'All':
                plot_extras.append('spec')
                height_ratios.append(4)
            if plot_phase and plot_component != 'All':
                plot_extras.append('phase')
                height_ratios.append(4)
            if plot_wspec and plot_component != 'All':
                plot_extras.append('wspec')
                height_ratios.append(4)
            if plot_cwt and plot_component != 'All':
                plot_extras.append('cwt')
                height_ratios.append(4)
            if plot_cwt_timeseries and plot_component != 'All':
                plot_extras.append('cwt_timeseries')
                height_ratios.append(4)
            if plot_event_measure and plot_component != 'All':
                plot_extras.append('event_measure')
                height_ratios.append(4)
            if plot_dfft and plot_component != 'All':
                plot_extras.append('dfft')
                height_ratios.append(4)
            if plot_infoaxis:
                plot_extras.append('info')
                height_ratios.append(1)
            nrows += len(plot_extras)

            extras = [plot_fft_snaps, plot_spec, plot_phase, plot_wspec, plot_cwt, plot_cwt_timeseries, plot_event_measure, plot_dfft, plot_infoaxis]
            extras_labels = ['snaps', 'spec', 'phase', 'wspec', 'cwt', 'cwt_timeseries', 'event_measure', 'dfft', 'info']
            plot_extras = [extras_labels[i] for i, e in enumerate(extras)
                           if e ==True]

            hspace = 0.18 if plot_component == 'All' else 0.18

            # Start a figure
            plt.style.use('default')
            plt.style.use('paper.mplstyle')
            # mplstyle.use(['ggplot'])
            # plt.style.use('dark_background')
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     # figsize=(20, 9),
                                     figsize=(12, 5),
                                     gridspec_kw={'height_ratios': height_ratios})
            fig.subplots_adjust(bottom=0.03, top=0.90, left=0.07, right=0.94,
                    wspace=0.10, hspace=hspace)
            # fig.set_facecolor('#171717')
            plot_kwds = {'lw' : 2, 'alpha' : 0.8, 'color' : 'orange'}
            plot2_kwds = {'lw' : 1}


            title = dateFrom.strftime('%Y-%m-%d')
            # Find series y-limits:
            yrange = findSeriesRange(FIELDS)

            for i, ax in enumerate(axes.flat):
                if plot_component == 'All' and plot_dfft:
                    plotSeriesID = i // 2
                    current_plot = 'timeseries' if i%2 == 0 else 'fft'
                    if plot_infoaxis and i // 2 > len(plotSeries)-1:
                        current_plot = 'info'
                        plotSeriesID = 0
                    plot_fft_snaps = False
                    show_xticks = False if i//2<len(plotSeries)-1 else True
                    plot_kwds['color'] = clist_field[plotSeriesID]
                else:
                    if i <= len(plotSeries)-1:
                        current_plot = 'timeseries'
                        plotSeriesID = i
                        # plot_kwds['color'] = clist_field[plotSeriesID]
                    else:
                        plotSeriesID = 0
                        if len(plot_extras) > 0:
                            current_plot = plot_extras.pop(0)
                    show_xticks = True

                plotSelection = plotSeries[plotSeriesID]
                selectSeries = [s for s in timeseries if s.name in plotSelection]

                # Select the time series
                s = selectSeries[0]
                seriesName = s.name
                datefrom = s.datetime[0]
                dateto = s.datetime[-1]

                # Spectrogram plot limit adjustment due to any overlap
                padding = 0 if s.S_pad is None else s.S_pad
                padding = datetime.timedelta(seconds=padding)
                seg_start = datetime.timedelta(seconds=s.S_tseg[0])
                seg_end = datetime.timedelta(seconds=s.S_tseg[-1])
                datefrom_lim = datefrom + seg_start - padding
                dateto_lim = datefrom + seg_end + padding

                # correlation calc
                # def lag_finder(y1, y2, sr):
                Nlags = 20
                phase_lag_time = []
                phase_lag = []
                yy1 = selectSeries[0].y
                yy2 = selectSeries[1].y
                dt_lag = datetime.timedelta(seconds = int(24*60*60/Nlags))
                timeFrom = selectSeries[0].datetime[0]

                def normalizeCrossings(y):
                    zero_crossings = np.where(np.diff(np.sign(y)))[0]
                    for i in range(0, len(zero_crossings)-1):
                        zc_prev = zero_crossings[i]
                        zc_next = zero_crossings[i+1]
                        ymax = np.mean(np.abs(y[zc_prev : zc_next]))
                        y[zc_prev:zc_next] = y[zc_prev:zc_next] / ymax
                    y[0:zero_crossings[0]] = 0
                    y[zero_crossings[-1]:] = 0
                    return y

                # yy1 = normalizeCrossings(yy1)
                # yy2 = normalizeCrossings(yy2)

                for ind in range(60, len(yy1)-60):
                    # ind = i * len(yy1) // Nlags
                    # ind = i
                    # print(ind)
                    # tlag = timeFrom + dt_lag
                    tlag = selectSeries[0].datetime[ind]
                    lag_hwidth = 61
                    ind0 = ind - lag_hwidth
                    ind1 = ind + lag_hwidth
                    ind0 = 0 if ind0 < 0 else ind0
                    ind1 = len(yy1) if ind1 > len(yy1) else ind1
                    y1 = yy1[ind0:ind1]
                    y2 = yy2[ind0:ind1]
                    # y1 = y1 / np.max(y1)
                    # y2 = y2 / np.max(y2)
                    # exit()
                    npts = len(y1)
                    sr = 1./selectSeries[0].dt
                    ccor = signal.correlate(y2, y1, mode='same', method='direct')
                    # ccor = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(npts/2)] * signal.correlate(y2, y2, mode='same')[int(npts/2)])
                    # ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='same')
                    # ccor = ccov / (npts * y1.std() * y2.std())
                    delay_arr = np.linspace(-1*npts/sr, 1*npts/sr, npts)
                    delay_arr = delay_arr / (60.*60.) * 180  # In degrees
                    # delay_arr = delay_arr / (60)
                    ind0 = np.argwhere(delay_arr < -180)[-1][0]
                    ind1 = np.argwhere(delay_arr > 180)[0][0]
                    ccor[:ind0] = 0
                    ccor[ind1:] = 0
                    # delay = delay_arr[np.argmax(np.abs(ccor))]
                    delay = delay_arr[np.argmax(ccor)]

                    # fig, axs = plt.subplots(2)
                    # fig.suptitle('delay = {:.2f} deg'.format(delay))
                    # axs[0].plot(y1)
                    # axs[0].plot(y2)
                    # axs[1].plot(delay_arr, ccor)
                    # axs[1].axvline(x=delay, color='yellow')
                    # plt.show()
                    # plt.plot(y1)
                    # plt.plot(y2)
                    # plt.plot(delay_arr, corr)
                    # plt.show()
                    # exit()
                    # print(delay)
                    phase_lag.append(delay)
                    phase_lag_time.append(tlag)

                if current_plot == 'org_timeseries':
                    ts = PlotTimeseries(ax,
                        selectSeriesOrg,
                        # plot_kwds = {'color': 'yellow'},
                        plot_kwds=plot_kwds,
                        plot_peaks=False,
                        xlim=(datefrom, dateto),
                        yrange=yrange if plot_component == 'All' else None,
                        # ylim=(-1.5, 1.5),
                        # highlight=datefrom + datetime.timedelta(seconds=3*hour),
                        # highlight_color='white',
                        minimal=True,
                        show_xticks=show_xticks,
                        )
                if current_plot == 'timeseries':
                    ts = PlotTimeseries(ax,
                        selectSeries,
                        # plot_kwds = {'color': 'yellow'},
                        plot_kwds=plot_kwds,
                        plot_peaks=False,
                        xlim=(datefrom, dateto),
                        yrange=yrange if plot_component == 'All' else None,
                        ylim=(-0.3, 0.3),
                        # highlight=datefrom + datetime.timedelta(seconds=3*hour),
                        # highlight_color='white',
                        minimal=True,
                        plotZeroAxis=True,
                        show_xticks=show_xticks,
                        fs=14,
                        )
                s.cwtm = np.abs(s.cwtm)
                s.cwtm = s.cwtm / np.max(s.cwtm)
                if current_plot == 'cwt_timeseries':
                    cwt_labels = []
                    cwt_timeseries = []
                    for t in [30, 35, 40, 45, 50, 55, 60, 65, 75, 80, 85, 90,
                              95, 100, 105, 110, 115, 120,
                              ]:
                        ind = np.argwhere(s.cwtm_freq >= 1/(t*60))[0][0]
                        cwt_labels.append(str(t) + ' min CWT')
                        cwt_timeseries.append(np.abs(s.cwtm[ind, :]))
                        # cwt_timeseries.append(s.cwtm[ind, :])
# 
                    for i, ts in enumerate(cwt_timeseries):
                        ax.plot(s.datetime, ts, color=clist_field[i], alpha=0.8, label=cwt_labels[i])
                    # ax.plot(delay_arr, corr, color=clist_field[1])

                    # ax.plot(phase_lag_time, phase_lag, color=clist_field[1])
                    # ax.set_ylim(-180,180)
                    # ax.axhline(y=0, ls='--', lw=2, color='white', alpha=0.5)
                    # ax.axhline(y=-180, ls='--', lw=2, color='orange', alpha=0.3)
                    # ax.axhline(y=-90, ls='--', lw=2, color='yellow', alpha=0.3)
                    # ax.axhline(y=90, ls='--', lw=2, color='yellow', alpha=0.3)
                    # ax.axhline(y=180, ls='--', lw=2, color='orange', alpha=0.3)
                    # ax.set_yticks((-180, -135, -90, -45, 0, 45, 90, 135, 180))

                    # ax.set_ylim(
                    ax.set_xlim(s.datetime[0], s.datetime[-1])
                    majorHourLocator = dates.HourLocator(interval=2)
                    minorHourLocator = dates.HourLocator(interval=1)
                    dfmt = dates.DateFormatter('%H:%M')
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.xaxis.set_major_locator(majorHourLocator)
                    ax.xaxis.set_minor_locator(minorHourLocator)

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set(frame_on=False)
                    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.2)
                    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.1)
                    ax.legend()

                if current_plot == 'event_measure':
                    cwt_labels = []
                    cwt_timeseries = []
                    for t in [30, 35, 40, 45, 50, 55, 60, 65, 75, 80, 85, 90,
                              95, 100, 105, 110, 115, 120,
                              ]:
                        ind = np.argwhere(s.cwtm_freq >= 1/(t*60))[0][0]
                        cwt_labels.append(str(t) + ' min CWT')
                        cwt_timeseries.append(np.abs(s.cwtm[ind, :]))

                    # event_measure = np.zeros(len(cwt_timeseries[0]))
                    # for i, ts in enumerate(cwt_timeseries):
                        # event_measure += ts

                    cwt_timeseries = np.asarray(cwt_timeseries)
                    event_measure = np.linalg.norm(cwt_timeseries, axis=0)
                    # event_measure = event_measure / np.max(event_measure)
                    # event_measure = np.mean(cwt_timeseries, axis=0)

                    ax.plot(s.datetime, event_measure, color=clist_field[1], alpha=0.8, label='Event Measure')
                    peaks, properties = find_peaks(event_measure, height=0.10, distance=1*60, prominence=0.10, width=100)
                    for peak in peaks:
                        ax.plot(s.datetime[peak], event_measure[peak], "x", color='yellow', ms=8)
                    for peak, prom, wh, l_ips, r_ips, l_base, r_base in zip(peaks, properties["prominences"], properties["width_heights"],
                                                            properties["left_ips"], properties["right_ips"],
                                                            properties["left_bases"], properties["right_bases"]):
                        ax.vlines(x=s.datetime[peak], ymin=event_measure[peak] - prom,
                                   ymax = event_measure[peak], color = "C1")
                        ax.hlines(y=wh, xmin=s.datetime[int(np.round(l_ips))],
                                   xmax=s.datetime[int(np.round(r_ips))], color = "C1")
                        # ax.hlines(y=wh, xmin=s.datetime[int(np.round(l_base))],
                                   # xmax=s.datetime[int(np.round(r_base))], color = "C1")

                    # ax.set_ylim(
                    ax.set_xlim(s.datetime[0], s.datetime[-1])
                    majorHourLocator = dates.HourLocator(interval=2)
                    minorHourLocator = dates.HourLocator(interval=1)
                    dfmt = dates.DateFormatter('%H:%M')
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.xaxis.set_major_locator(majorHourLocator)
                    ax.xaxis.set_minor_locator(minorHourLocator)

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set(frame_on=False)
                    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='--', alpha=0.2)
                    ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.1)
                    ax.legend()

                if current_plot == 'phase':
                    im = plotSpectrogram(ax,
                        s.Phase_tseg,
                        s.Phase_freq,
                        s.Phase,
                        (datefrom, dateto),
                        textent = (datefrom_lim, dateto_lim),
                        ylim=plot_spec_ylim,
                        xlim=(datefrom, dateto),
                        plotPeaks=False,
                        units = plot_units,  # mHz
                        vmin = -20*np.pi,
                        vmax = 20*np.pi,
                        colorScale = 'linear',
                        # yScale = 'log',
                        showSegments = False,
                        periodLines=plot_periodLines,
                        # highlight_period=hightlight_periods,
                        # snaps = None if plot_component == 'All' else plot_fft_nsnaps,
                        show_cbar=False,
                        show_xticks=show_xticks,
                        fontsize = 11 if plot_component == 'All' else 12
                        )
                if current_plot == 'spec':
                    im = plotSpectrogram(ax,
                        s.S_tseg,
                        s.S_freq,
                        s.S,
                        (datefrom, dateto),
                        textent = (datefrom_lim, dateto_lim),
                        ylim=plot_spec_ylim,
                        xlim=(datefrom, dateto),
                        plotPeaks=False,
                        units = plot_units,  # mHz
                        vmin = plot_vmin,
                        vmax = plot_vmax,
                        colorScale = 'log',
                        yScale = 'log',
                        showSegments = False,
                        periodLines=plot_periodLines,
                        # highlight_period=hightlight_periods,
                        snaps = None if plot_component == 'All' else plot_fft_nsnaps,
                        show_cbar=False,
                        show_xticks=show_xticks,
                        fontsize = 11 if plot_component == 'All' else 12
                        )
                if current_plot == 'wspec':
                    im2 = plotSpectrogram(ax,
                        s.cwtm_tseg,
                        s.cwtm_freq,
                        np.abs(s.cwtm),
                        (datefrom, dateto),
                        ylim=plot_spec_ylim,
                        xlim=(datefrom, dateto),
                        plotPeaks=False,
                        units = plot_units,  # mHz
                        # colorScale = 'log',
                        # vmin = 1e-2,
                        # vmax = 1e1,
                        colorScale = 'linear',
                        vmin = 1e-2,
                        vmax = 1,
                        # yScale = 'log',
                        showSegments = False,
                        periodLines=plot_periodLines,
                        # highlight_period=hightlight_periods,
                        # snaps = None if plot_component == 'All' else plot_fft_nsnaps,
                        show_cbar=False,
                        show_xticks=show_xticks,
                        fontsize = 11 if plot_component == 'All' else 12
                        )
                if current_plot == 'dfft':
                    PlotFFT(ax,
                            s.fft_freq,
                            np.exp(np.log(s.fft) - np.log(s.fft_bg)),
                            # data_list = data_list,
                            # data_list_labels = data_list_labels,
                            # data_list_colors = data_list_colors,
                            xlim=plot_spec_ylim,
                            ylim=(plot_vmin, plot_vmax),
                            units=plot_units,
                            periodLines = plot_periodLines,
                            highlight_periods=highlight_periods,
                            highlight_periods2=highlight_periods2,
                            label='Power Residual from Background',
                            color=clist_field[0],
                            edgecolor='orange',
                            noTicks=False,
                            minimal_axis=True,
                            show_all=False,
                            # show_all=False,
                            minor_alpha=0.01,
                            # fit_inds = fit_inds,
                            # fit_out_inds = fit_out_inds,
                            plot_power_diff=True,
                           )
                if current_plot == 'info':
                    ia = plotInfoAxis(ax,
                        # selectSeries,
                        timeseries,
                        # plotSelection,
                        coords=COORD,
                        lt=LTS,
                        fontsize = 13 if plot_component == 'All' else 13,
                        # xlim=(datefrom, dateto),
                        MLoc=MLoc,
                        MLoc_time=MLoc_time,
                        )
                if current_plot == 'snaps':
                    fs = plotFFTSnaps(ax,
                         s.S_tseg,
                         s.S_freq,
                         s.S,
                         (datefrom, dateto),
                         snaps=plot_fft_nsnaps,
                         units = plot_units,
                         vmin = plot_vmin,
                         vmax = plot_vmax,
                         periodLines = plot_periodLines,
                         highlight_periods=highlight_periods,
                         highlight_periods2=highlight_periods2,
                         yScale = 'log',
                         # ylabel = r'Power Density (nT$^2$ / Hz)',
                         ylabel = r'Power [nT$^2$]',
                         xlabel = 'Frequency [mHz]',
                         xlim = plot_spec_ylim,
                         reference_FFT_freq = s.fft_freq,
                         reference_FFT_pow = s.fft,
                         bg_FFT_freq = s.fft_freq,
                         bg_FFT_pow = s.fft_bg,
                         # peaks = s.peaksf,
                         # properties = s.propertiesf,
                         minimal_axis = True,
                         )
                if current_plot == 'cwt':
                    # data_list = [[np.abs(s.cwt)]]
                    data_list = [[np.abs(s.cwtm[:,0])]]
                    data_list_labels = ['CWT']
                    data_list_colors = [clist_field[1]]
                    fs = PlotFFT(ax,
                            s.cwtm_freq,
                            data_list = data_list,
                            data_list_labels = data_list_labels,
                            data_list_colors = data_list_colors,
                            # xlim=(1e-6, 1e-1),
                            xlim = plot_spec_ylim,
                            ylim=(1e-3, 1e0),
                            units=plot_units,
                            label='',
                            color=clist_field[0],
                            noTicks=False,
                            minimal_axis=True,
                            show_all=True,
                            minor_alpha=0.9,
                            )
                # if current_plot == 'fft_broadview':
                    # fs = PlotFFT(ax,
                            # s.fft_freq,
                            # data_list = data_list,
                            # data_list_labels = data_list_labels,
                            # data_list_colors = data_list_colors,
                            # xlim=(1e-6, 1e-1),
                            # ylim=(1e-2, 1e7),
                            # units=1,
                            # label='',
                            # color=clist_field[0],
                            # noTicks=False,
                            # minimal_axis=True,
                            # show_all=False,
                            # minor_alpha=0.1,
                            # )

            if plot_spec:
                if plot_component == 'All':
                    loc = [0.955, 0.05, 0.01, 0.85]
                else:
                    pos1 = axes[1].get_position() # get the original position 
                    loc = [pos1.x0 + 0.9, pos1.y0 + 0.,  0.01, pos1.height]
                cax = fig.add_axes(loc)
                cbar = fig.colorbar(im, cax=cax)
               # cbar.outline.remove()
                cbar.ax.tick_params(axis='y', color='white',
                                    left=False, right=True)
                cbar.set_label(r'Power Density (nT$^2$ / Hz)')
            if plot_wspec:
                pos1 = axes[2].get_position() # get the original position 
                loc = [pos1.x0 + 0.9, pos1.y0 + 0.,  0.01, pos1.height]
                cax = fig.add_axes(loc)
                cbar = fig.colorbar(im2, cax=cax)
               # cbar.outline.remove()
                cbar.ax.tick_params(axis='y', color='white',
                                    left=False, right=True)
                cbar.set_label(r'Wavelet Coeff')

            if title is not None:
                plt.suptitle(title, fontsize=22)
            if save_plots:
                OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
                if not os.path.exists(os.path.join(*OUTPUT_DIR)):
                    os.makedirs(os.path.join(*OUTPUT_DIR))
                fullpath = os.path.join(*OUTPUT_DIR,
                                        FILENAME_FFT_PLOT + '.png')
                fullpath.replace('YYYY', str(YEAR))
                fullpath.replace('ID', "{0:03d}".format(ID))
                fullpath.replace('XXX', "{0:03d}".format(n))
                fullpath.replace('III', dataInstrument)
                fullpath.replace('COORD', COORD)
                plt.savefig(fullpath,
                            facecolor=fig.get_facecolor(),
                            edgecolor='none')
                plt.close()
            else:
                plt.show()
    else:
        print('No time series data found.')

    n += 1
    if n > n_max:
        continueProcessing = False
        print('** Reached max iteration count.')
    if dataSource in ['mission', 'synthfull']:
        timeElapsed = (dateTo - dateMissionFrom).total_seconds()
        if n % 20 == 0:
            print('N: {:d} ({:.1f}%) | {:s}-{:s} | NaN: {:d} | n(signals): {:d}'.format(n,
                timeElapsed/missionDuration*100,
                dateFrom.strftime('%Y-%m-%d'),
                dateTo.strftime('%Y-%m-%d'), int(NaN_count),
                len(signals)))

        if dateTo >= dateMissionTo:
            print('** Reached the end of mission.')
            continueProcessing = False
    else:
        print('N: {:d}'.format(n))
    if len(intervals) > 1 and n > len(intervals) - 1:
        continueProcessing = False
        print('** Reached the end of input intervals.')

save_signals = True
if save_signals:
    OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
    if not os.path.exists(os.path.join(*OUTPUT_DIR)):
        os.makedirs(os.path.join(*OUTPUT_DIR))
    fullpath = os.path.join(*OUTPUT_DIR, FILENAME_FFT_ARCHIVE + '.npy')
    fullpath = fullpath.replace('YYYY', str(YEAR))
    if ID is not None:
        fullpath = fullpath.replace('ID', "{0:03d}".format(ID))
    fullpath = fullpath.replace('COORD', COORD)
    fullpath = fullpath.replace('III', dataInstrument)
    np.save(fullpath, signals)
exit()

load_signals = False
if load_signals:
    #  signals = np.load('Output/Cassini_MAG_DATA_KRTP_ALL_24H.npy', allow_pickle=True)
    signals = np.load('Output/Cassini_MAG_DATA_KRTP_ALL_32H.npy', allow_pickle=True)
    # signals = signals[len(signals) // 2 :]
    for n, signal in enumerate(signals):
        FIELDS = signal.FIELDS
        COORDS = signal.COORDS
        transform_to_B_coords = True
        analyzeData = True
        if transform_to_B_coords:
            Navg = int(1/freq_lim[0] // signal_dt)
            for s in FIELDS:
                run_avg = s.detrender(Navg=Navg, mode='exclude', returnOnly=True)
                s.dy = s.y - run_avg
                s.run_avg = run_avg
            BFIELDS = fieldTransform(COORDS, FIELDS, 'KRTP')
            for i in range(4):
                FIELDS[i].y = BFIELDS[i].y
                FIELDS[i].name = BFIELDS[i].name
                FIELDS[i].dy = None
                FIELDS[i].run_avg = None
            # plot_field_aligned_test(COORDS, FIELDS, BFIELDS=BFIELDS)
        for s in FIELDS:
            if analyzeData:
                s.datetime = signal.datetime
                s.detrend_repeat = 1
                s.analyze(actions = ['resample', 'fft'])
                s.datetime = None
        if analyzeData:
            for s in FIELDS:
                s.dfft = np.exp(np.log(s.fft) - np.log(FIELDS[3].fft_bg))
                # s.dfft = np.exp(np.log(s.fft) - np.log(s.fft_bg))

        ind0 = np.argwhere(CROSSINGS[0] >= signal.datetime[0])[0][0]
        ind1 = np.argwhere(CROSSINGS[0] >= signal.datetime[-1])[0][0]
        signal.info['locs'] = CROSSINGS[1][ind0:ind1]
        signal.info['flag_times'] = CROSSINGS[0][ind0:ind1]
        LOCS = np.asarray(CROSSINGS[1], dtype=int)
        # signal.info['loc'] = np.argmax(np.bincount(CROSSINGS[1][ind0:ind1]))
        signal.info['loc'] = np.argmax(np.bincount(LOCS[ind0:ind1]))
        signal.info['cros'] = CROSSINGS[2][ind0:ind1]
        if n % 100 == 0:
            print('N: {:d} ({:.1f}%)'.format(n, n/len(signals)*100))

# exit()
save_signals = False
if save_signals and collect_FFTs:
    OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
    if not os.path.exists(os.path.join(*OUTPUT_DIR)):
        os.makedirs(os.path.join(*OUTPUT_DIR))
    fullpath = os.path.join(*OUTPUT_DIR, FILENAME_FFT_ARCHIVE + '.npy')
    fullpath = fullpath.replace('COORD', COORD)
    fullpath = fullpath.replace('III', dataInstrument)
    np.save(fullpath, signals)
#  exit()

print('Starting to import signal data.')
load_signals = True
if load_signals:
    # signals = np.load('Output/Cassini_MAG_FFT_KRTP_ALL_24H.npy', allow_pickle=True)
    # signals = np.load('Output/Cassini_MAG_FFT_KRTP.npy', allow_pickle=True)
    # signals = np.load('Output/Cassini_MAG_FFT_KRTP_LOWFIT.npy', allow_pickle=True)
    # signals = np.load('Output/Cassini_MAG_FFT_KRTP_HIGHFIT.npy', allow_pickle=True)
    # signals = np.load('Output/Cassini_MAG_FFT_KRTP_SELFRATIO.npy', allow_pickle=True)
    # signals = np.load('Output/Cassini_MAG_FFT_BFIELD.npy', allow_pickle=True)
    #  signals = np.load('Output/Cassini_MAG_FFT_BFIELD_LOWFIT.npy', allow_pickle=True)
    signals = np.load('Output/Cassini_MAG_FFT_BFIELD_32H.npy', allow_pickle=True)
    # np.save('Output/Cassini_MAG_FFT_BFIELD_LOWFIT_SMALL.npy', signals[360:720])
    # signals = np.load('Output/Cassini_MAG_FFT_BFIELD_LOWFIT_SMALL.npy', allow_pickle=True)
    print('Data Loaded.')

    # print('Recalculating ...')
    # for signal in signals:
        # if signal.flag is None:
            # for s in signal.FIELDS:
                # s.dfft = np.exp(np.log(s.fft) - np.log(signal.FIELDS[3].fft_bg))
                # s.dfft = np.exp(np.log(s.fft) - np.log(s.fft_bg))
                # s.dfft = np.exp(np.log(s.fft_bg) - np.log(signal.FIELDS[3].fft_bg))
    # print('Done recomputing')
    # data_list = []
    # print(signals[0].info)
    # print(signals[0].info['rKSM'])
    # for n, signal in enumerate(signals):
        # data_list.append(signal.info["loc"])
        # data_list.append(signal.info["median_KRTP"][1])
        # data_list.append(signal.info["NaNs"]/60)
        # data_list.append(signal.COORDS[0].y[0])
    # plt.plot(data_list)
    # plt.show()

# exit()
#  for s in signals:


if plot_all_FFTs:
    fn_in = FILENAME_FFT_ARCHIVE
    fn_out = FILENAME_FFT_ARCHIVE_PLOT
    if ID is not None:
        fn_out = fn_out.replace('ID', "{0:03d}".format(ID))
    fn_out = fn_out.replace('COORD', COORD)
    fn_out = fn_out.replace('III', dataInstrument)
    fn_in = fn_in.replace('COORD', COORD)
    fn_in = fn_in.replace('III', dataInstrument)
    # print(fn_in)
    readAndPlotFFT(data=signals,
                   year=YEAR,
                   coord=COORD,
                   component=COMPONENT,
                   location=LOCATION,
                   show_all=True,
                   title=title,
                   save_plots=save_plots,
                   splitByBins=False,
                   splitByTime=False,
                   plot_fft=False,
                   plot_test=True,
                   plot_type = plot_all_fft_type,
                   periodLines=plot_periodLines,
                   vmin=plot_vmin,
                   vmax=plot_vmax,
                   units=plot_units,
                   plot_spec_ylim = plot_spec_ylim,
                   plot_spec_xlim = plot_spec_xlim,
                   highlight_periods=highlight_periods,
                   highlight_periods2=highlight_periods2,
                   output_dir=OUTPUT_DIR,
                   # filename_in=fn_in,
                   filename_out=fn_out,
                   )
