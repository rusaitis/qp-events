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
from cassinilib import DataPaths
rc('text', usetex=True)  # For more extensive LaTeX features beyond mathtex
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


class Interval():
    """ Interval class to keep track of data discontinuities """
    def __init__(self,
                 dateFrom=None,
                 dateTo=None,
                 comment=None,
                 ):
        self.dateFrom = dateFrom
        self.dateTo = dateTo
        self.comment = comment


class SignalSnapshot():
    ''' Storage for all signals (fields/coords) for a give time period '''
    def __init__(self,
                 datetime=None,
                 FIELDS=None,
                 COORDS=None,
                 flag=None,
                 info={},
                 ):
        self.datetime = datetime
        self.FIELDS = FIELDS
        self.COORDS = COORDS
        self.flag = flag
        self.info = info


def str2datetime(string, format='%Y-%m-%dT%H:%M:%S'):
    ''' String to a datetime object, using a given format '''
    return datetime.datetime.strptime(string, format)

def check_for_scas_times(dateFrom, dateTo, file, threshold=60*60):
    ''' Check if there are calibration events within a datetime interval '''
    # Cassini SCAS (Calibration) Time File
    SCAS_TIMES = np.load(file, allow_pickle=True)
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
        if cal_dur > threshold:
            flag = cal[2] if flag is None else flag + ' ' + cal[2]
    return cal_list, flag


def find_gaps_in_data(time, dateFrom, dateTo, gap_threshold, gap_flag_threshold):
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
            flag = 'GAP' if flag is None else flag + ' ' + 'GAP'
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

class dataConfig():
    ''' Parameters for Data Loading and Processing '''
    def __init__(self,
                 source = 'mission',  # mission/synth/synthfull/events
                 coord = 'KRTP',  # KRTP/KSM/KSO
                 instrument = '',
                 measurement = '',
                 interval = 24*60*60,
                 segment_interval = 12*60*60,
                 averaging_interval = 3*60*60,
                 dt = 60,
                 margin = 0,
                 year = None,
                 date_from = None,
                 date_to = None,
                 date_mission_from = None,
                 date_mission_to = None,
                 max_read_segments = 1e6,
                 calibration_flag_threshold = None,
                 gap_threshold = None,
                 gap_flag_threshold = None,
                 zero_padding_factor = 1,
                 noverlap_method = None,
                 analyze = False,
                 field_aligned_coords = True,
                 freq_lim = None,
                 synth = None,
                 window = ('blackman'),
                 NaN_interp_method = 'cubic',
                 ):
        self.source = source
        self.coord = coord
        self.instrument = instrument
        self.measurement = measurement
        self.interval = interval
        self.segment_interval = segment_interval
        self.averaging_interval = averaging_interval
        self.dt = dt
        self.margin = margin
        self.year = year
        self.date_from = date_from
        self.date_to = date_to
        self.date_mission_from = date_mission_from
        self.date_mission_to = date_mission_to
        self.mission_duration = None
        self.max_read_segments = max_read_segments
        self.calibration_flag_threshold = calibration_flag_threshold
        self.gap_threshold = gap_threshold
        self.gap_flag_threshold = gap_flag_threshold
        # Signal Analysis
        self.analyze = analyze
        self.field_aligned_coords = field_aligned_coords
        self.zero_padding_factor = zero_padding_factor
        self.noverlap_method = noverlap_method
        self.NaN_interp_method = NaN_interp_method
        self.freq_lim = freq_lim
        self.window = window
        self.synth = synth


def read_data_segments(config):

    # For synthetic data, adjust 'instrument' to find correct files
    if config.source in ['synthfull']:
        config.instrument = 'SIM'
        config.measurement = '1min'

    # If data margin is specified, adjust the signal length
    if config.margin is not None:
        config.interval += 2*config.margin

    # Signal options for the signal processor (mostly for FFT)
    nperseg = int(config.segment_interval // config.dt)
    NFFT = int(nperseg * config.zero_padding_factor)
    signal_options = {
        'dt' : config.dt,
        'coord' : config.coord,
        'resample_total_time' : config.interval,
        'actions' : ['resample'],
        'freq_lim': config.freq_lim,
        'window': config.window,
        'nperseg' : nperseg,
        'NFFT' : NFFT,
        'noverlap_method' : config.noverlap_method,
        'NaN_method' : config.NaN_interp_method,
    }

    #=====================================================================
    #                        Process Datetimes
    #=====================================================================
    intervals = list()

    if config.source == 'events':
        try:
            f = np.loadtxt(DataPaths.eventFile, usecols=(0,1), dtype='str')
            for n in range(len(f)):
                dateFrom = str2datetime(f[n, 0])
                dateTo = str2datetime(f[n, 1])
                intervals.append(Interval(dateFrom, dateTo))
        except FileNotFoundError:
            print("file {} does not exist".format(fname))

    if config.source == 'events_selected':
        for eventFrom in DataPaths.events_selected:
            intervals.append(Interval(str2datetime(eventFrom),
                str2datetime(eventFrom)+datetime.timedelta(seconds=24*60*60)))

    elif config.source == 'synth':
        T0 = DataPaths.testFull_dateMissionFrom
        T1 = DataPaths.testFull_dateMissionTo
        dateFrom = str2datetime(T0)
        dateTo = str2datetime(T1)
        intervals.append(Interval(dateFrom, dateTo))
    elif config.source in ['mission', 'synthfull']:
        if config.source == 'synthfull':
            T0 = DataPaths.test_dateMissionFrom
            T1 = DataPaths.test_dateMissionTo
        else:
            if config.year is None:
                T0 = DataPaths.dateMissionFrom
                T1 = DataPaths.dateMissionTo
            else:
                T0 = str(config.year) + '-01-01T00:00:00'
                T1 = str(config.year + 1) + '-01-01T00:00:00'
        if config.date_from is not None and config.date_to is not None:
            T0 = config.date_from + 'T00:00:00'
            T1 = config.date_to + 'T00:00:00'
        config.date_mission_from = str2datetime(T0)
        config.date_mission_to = str2datetime(T1)
        config.mission_duration = (config.date_mission_to
                                  - config.date_mission_from).total_seconds()
        dateTo = config.date_mission_from + datetime.timedelta(seconds=config.margin)
        dateFrom = dateTo - datetime.timedelta(seconds=config.interval)
        intervals.append(Interval(dateFrom, dateTo))

    #=====================================================================
    #                        DATA LOADING AND PROCESSING
    #=====================================================================
    timeseries = []
    signals = []

    n = 0
    continueProcessing = True
    while continueProcessing:
        if config.source in ['mission', 'synthfull']:
            # Readjust datetimes for mission data for a sequatial read
            dateFrom = dateTo - datetime.timedelta(seconds = 2*config.margin)
            dateTo = dateFrom + datetime.timedelta(seconds = config.interval)
        else:
            ind = n if n < len(intervals) else 0
            interval = intervals[ind]
            dateFrom = interval.dateFrom
            dateTo = interval.dateTo
            if config.margin is not None:
                dateFrom -= datetime.timedelta(seconds=config.margin)
                dateTo += datetime.timedelta(seconds=config.margin)
        #=================================================================
        #                    LOAD OR SYNTHESIZE THE DATA
        #=================================================================

        if config.source == 'synth':
            N = int(config.interval // config.dt)
            timeseries = simulateSignal(N = N,
                                        periods=config.synth['periods'],
                                        amplitudes=config.synth['amplitudes'],
                                        shifts=config.synth['shifts'],
                                        decays=config.synth['decays'],
                                        phases=config.synth['phases'],
                                        cutoffs=config.synth['cutoffs'],
                                        types=config.synth['types'],
                                        **signal_options)
        else:
            timeseries = readSignal(dateFrom,
                                    dateTo,
                                    instrument = config.instrument,
                                    measurement = config.measurement,
                                    file = None,
                                    **signal_options)

        NaN_count = 0
        flag = None

        if timeseries:
            COORDS = [s for s in timeseries if s.kind == 'Coord']
            LTS =    [s for s in timeseries if s.kind == 'lt']
            FIELDS = [s for s in timeseries if s.kind == 'Field']

            calibrations = []
            gaps = []
            data = FIELDS[0].y  # Take the first component of the field
            time = FIELDS[0].datetime  # Take the datetime array of the field

            # Check for SCAS MAG Calibration events
            if config.source not in ['synth', 'synthfull']:
                calibrations, calibration_flag = check_for_scas_times(
                        dateFrom,
                        dateTo,
                        DataPaths.SCAS_FILE,
                        config.calibration_flag_threshold)
                if calibration_flag is not None:
                    flag = calibration_flag if flag is None else flag + ' ' + calibration_flag
                    print('** Skipping due to {0} CALIBRATION'.format(flag))

            # If a great number of missing or NaN points, save invalid times
            NaN_count = (config.interval/config.dt
                         - np.count_nonzero(~np.isnan(data)))
            if NaN_count > config.gap_threshold // config.dt:
                gaps, gap_flag = find_gaps_in_data(time,
                                                   dateFrom,
                                                   dateTo,
                                                   config.gap_threshold,
                                                   config.gap_flag_threshold)
                if gap_flag is not None:
                    flag = gap_flag if flag is None else flag + ' ' + gap_flag
                    print('** Skipping due to NaN count: {:d}'.format(int(NaN_count)))

            # Resample the Fields to have uniformly sampled data
            for s in FIELDS:
                s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)
            for s in COORDS:
                s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)
            for s in LTS:
                s.uniform_resampler(resample_from=dateFrom, resample_to=dateTo)

            TIME = FIELDS[0].datetime
            # No need to store the same datetime array for every field comp
            for s in COORDS:
                s.datetime = []
            for s in FIELDS:
                s.datetime = []
            for s in LTS:
                s.datetime = []

            median_BT = np.median(FIELDS[3].y)
            BTstd = np.std(FIELDS[3].y)
            median_LT = np.median(LTS[0].y) if LTS else None
            median_coords = [np.median(COORDS[0].y),
                             np.median(COORDS[1].y),
                             np.median(COORDS[2].y)]

            signals.append(SignalSnapshot(
                datetime = TIME,
                FIELDS = FIELDS,
                COORDS = COORDS,
                flag = flag,
                info = dict(
                    gaps = gaps,
                    calibrations = calibrations,
                    NaN_count = NaN_count,
                    BTstd = BTstd,
                    median_LT = median_LT,
                    median_BT = median_BT,
                    median_coords = median_coords,
                    location = None,
                    instrument_range = None,
                    comment = None,
                )))

            # TEST INTERPOLATION
            # if NaN_count >= 2*60 and NaN_count < 8*60:
                # plot_field_interpolation(FIELDS, gap_list, cal_list)
        else:
            print('No time series data found.')

        n += 1
        #  print('N: {:d} | {:s}-{:s} | NaN: {:d} | n(signals): {:d}'.format(n,
            #  dateFrom.strftime('%Y-%m-%d-%H'),
            #  dateTo.strftime('%Y-%m-%d-%H'), int(NaN_count),
            #  len(signals)))
        if n > config.max_read_segments:
            continueProcessing = False
            print('** Reached max iteration count.')
        if config.source in ['mission', 'synthfull']:
            timeElapsed = (dateTo - config.date_mission_from).total_seconds()
            if n % 20 == 0:
                print('N: {:d} ({:.1f}%) | {:s}-{:s} | NaN: {:d} | n(signals): {:d}'.format(n,
                    timeElapsed/config.mission_duration * 100,
                    dateFrom.strftime('%Y-%m-%d'),
                    dateTo.strftime('%Y-%m-%d'), int(NaN_count),
                    len(signals)))

            if dateTo >= config.date_mission_to:
                print('** Reached the end of mission.')
                continueProcessing = False
        else:
            print('N: {:d}'.format(n))
        if len(intervals) > 1 and n > len(intervals) - 1:
            continueProcessing = False
            print('** Reached the end of input intervals.')
    return signals



def export_sls5(
        directory = None,
        filename = 'SLS5_2004-2018.txt',
        output = 'SLS5_2004-2018.npy'): 

    sls5 = []
    TIMES = []
    C1 = []
    C2 = []
    C3 = []
    C4 = []
    fullpath = os.path.join(directory, filename)
    with open(fullpath) as fp:
        line = fp.readline()  # Skip the first row (column names)
        line = fp.readline()
        cnt = 1
        while line:
            nl = line.split()
            dt = str(nl[0])
            SLS5N = float(nl[1])
            SLS5S = float(nl[2])
            SLS5N2 = float(nl[3])
            SLS5S2 = float(nl[4])
            dt = datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S')
            TIMES.append(dt)
            C1.append(SLS5N)
            C2.append(SLS5S)
            C3.append(SLS5N2)
            C4.append(SLS5S2)
            line = fp.readline()
            cnt += 1

    SLS5_STACK = np.vstack((TIMES, C1, C2, C3, C4))

    fullpath = os.path.join(directory, output)
    np.save(fullpath, SLS5_STACK)

if __name__ == "__main__":




    #=====================================================================
    #                            Simulated Signals
    #=====================================================================
    synthesize_full_year = False
    if synthesize_full_year:
        generateLongSignal()

    hour = 60*60
    mins = 60
    # synth = syntheticEvents()
    synth = {}
    #  synth['periods'] =    [9*hour,  6*hour, 3*hour, 2*hour, 60*mins, 30*mins, 15*mins, 10*mins, 5*mins, ]
    #  synth['amplitudes'] = [0.3,       0.1,      0.2,    0.2,    0.05,     0.2,     0.2,     0.2,    0.2,    ]
    #  synth['shifts'] =     [3*hour,  0*hour, 0*hour, 12*hour,3*hour,  6*hour,  10*hour, 8*hour, 3*hour, ]
    #  synth['periods'] =    [60*mins, 15*mins, 90*mins, 30*mins, 4*hour]
    #  synth['amplitudes'] = [0.5,     0.2,     1,       0.1,     2]
    #  synth['shifts'] =     [3*hour,  12*hour, 18*hour, 12*hour, 12*hour]
    synth['periods'] =    [60*mins, 30*mins, 60*mins]
    synth['amplitudes'] = [0.5,     0.3   ,  0.4]
    synth['shifts'] =     [2*hour,  18*hour, 12*hour]
    synth['phases'] =     [0 * period for period in synth['periods']]
    synth['decays'] =     [2*period for period in synth['periods']]
    synth['cutoffs'] =    [None for period in synth['periods']]
    synth['types'] =      ['sine' for period in synth['periods']]


    config = dataConfig(source = 'mission',
                        instrument = 'MAG',
                        measurement = '1min',
                        coord = 'KRTP',
                        dt = 60,
                        interval = 24*60*60,
                        segment_interval = 12*60*60,
                        margin = 6*60*60,
                        max_read_segments = 1e5,
                        gap_threshold = 10*60,
                        gap_flag_threshold = 30*60,
                        calibration_flag_threshold = 30*60,
                        noverlap_method = 'half',
                        freq_lim = (1/(3*60*60), 1/(10*60)),
                        NaN_interp_method = 'cubic',
                        zero_padding_factor = 4,
                        window = ('blackman'),
                        synth = synth,
                        )

    FILENAME_DATA_ARCHIVE = 'Cassini_III_COORD'
    output_comment = '_36H'
    load_data = True
    save_data = True
    process_data = False
    #  datafile = 'Cassini_MAG_KRTP_24H.npy'
    datafile = 'Cassini_MAG_MFA_36H.npy'
    OUTPUT_DIR = DataPaths.OUTPUT_DIR

    if load_data:
        fullpath = os.path.join(OUTPUT_DIR, datafile)
        #  export_sls5(directory = OUTPUT_DIR, filename = 'SLS5_2004-2018.txt', output = 'SLS5_2004-2018.npy')

        signals = np.load(fullpath, allow_pickle=True)
        print('Signals loaded.')

        CROSSINGS_FILE = os.path.join(OUTPUT_DIR, 'CROSSINGS.npy')
        CROSSINGS = np.load(CROSSINGS_FILE, allow_pickle=True)
        print('BS-MS Crossings loaded.')
        SLS5_FILE = os.path.join(OUTPUT_DIR, 'SLS5_2004-2018.npy')
        SLS5 = np.load(SLS5_FILE, allow_pickle=True)
        print('SLS5 loaded.')
    else:
        signals = read_data_segments(config)

    if process_data:
        print(f"Number of data segments loaded: {len(signals)}")
        process_metadata = False
        transform_to_field_aligned = True

        if process_metadata:
            for n, signal in enumerate(signals):
                # Attach BS/MP Boundary data and flag times
                ind0 = np.argwhere(CROSSINGS[0] >= signal.datetime[0])[0][0]
                ind1 = np.argwhere(CROSSINGS[0] >= signal.datetime[-1])[0][0]
                signal.info['locations'] = CROSSINGS[1][ind0:ind1]
                signal.info['flag_times'] = CROSSINGS[0][ind0:ind1]
                LOCS = np.asarray(CROSSINGS[1], dtype=int)
                signal.info['location'] = np.argmax(np.bincount(LOCS[ind0:ind1]))
                signal.info['crossings'] = CROSSINGS[2][ind0:ind1]

                # Attach SLS5 data
                ind0 = np.argwhere(SLS5[0] >= signal.datetime[0])[0][0]
                ind1 = np.argwhere(SLS5[0] >= signal.datetime[-1])[0][0]
                signal.info['SLS5N'] = SLS5[1][ind0:ind1]
                signal.info['SLS5S'] = SLS5[2][ind0:ind1]
                signal.info['SLS5N2'] = SLS5[3][ind0:ind1]
                signal.info['SLS5S2'] = SLS5[4][ind0:ind1]
                ll = len(SLS5[1][ind0:ind1])
                #  print(f"{n}: len(sls5): {ll}")
                if n % 100 == 0:
                    print('N: {:d} ({:.1f}%)'.format(n, n/len(signals)*100))
                if ll != 216:
                    print(f"ERROR {n}: len(sls5): {ll}")
                    exit()


        config.field_aligned_coords = True
        if config.field_aligned_coords:
            for n, signal in enumerate(signals):
                FIELDS = signal.FIELDS
                COORDS = signal.COORDS
                Navg = int(1/config.freq_lim[0] // config.dt)
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
                if n % 100 == 0:
                    print('N: {:d} ({:.1f}%)'.format(n, n/len(signals)*100))
            # plot_field_aligned_test(COORDS, FIELDS, BFIELDS=BFIELDS)

        config.analyze = False
        if config.analyze:
            for n, signal in enumerate(signals):
                FIELDS = signal.FIELDS
                COORDS = signal.COORDS
                for s in FIELDS:
                    s.datetime = signal.datetime
                    s.detrend_repeat = 1
                    s.analyze(actions = ['resample', 'fft'])
                    s.datetime = None
                for s in FIELDS:
                    s.dfft = np.exp(np.log(s.fft) - np.log(FIELDS[3].fft_bg))
                    s.dfft = np.exp(np.log(s.fft) - np.log(s.fft_bg))
                if n % 100 == 0:
                    print('N: {:d} ({:.1f}%)'.format(n, n/len(signals)*100))

            #  for s in FIELDS:
                #  if config.analyze:
                    #  s.datetime = signal.datetime
                    #  s.detrend_repeat = 1
                    #  s.analyze(actions = ['resample', 'fft'])
                    #  s.datetime = None
            #  if config.analyze:
                #  for s in FIELDS:
                    #  s.dfft = np.exp(np.log(s.fft) - np.log(FIELDS[3].fft_bg))
                    # s.dfft = np.exp(np.log(s.fft) - np.log(s.fft_bg))
    #  print('Events Processed.')

    #  datafile = 'Cassini_MAG_MFA_KRTP_36H.npy'
    #  fullpath = os.path.join(OUTPUT_DIR, datafile)
    #  np.save(fullpath, signals)
    #  print('Saved Data File successfully.')
    #  exit()

    #  if save_data:
        #  OUTPUT_DIR = DataPaths.OUTPUT_DIR
        #  OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
        #  if not os.path.exists(os.path.join(*OUTPUT_DIR)):
            #  os.makedirs(os.path.join(*OUTPUT_DIR))
        #  fullpath = os.path.join(*OUTPUT_DIR, FILENAME_DATA_ARCHIVE + output_comment + '.npy')
        #  fullpath = fullpath.replace('COORD', config.coord)
        #  fullpath = fullpath.replace('III', config.instrument)
        #  np.save(fullpath, signals)
        #  print('Saved Data File successfully.')
    #  exit()



    COMPONENT = 2
    LOCATION = 'MS'
    VERBOSE = False
    COORD = 'MFA'

    #  save_signals = False
    #  if save_signals and collect_FFTs:
        #  OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
        #  if not os.path.exists(os.path.join(*OUTPUT_DIR)):
            #  os.makedirs(os.path.join(*OUTPUT_DIR))
        #  fullpath = os.path.join(*OUTPUT_DIR, FILENAME_FFT_ARCHIVE + '.npy')
        #  fullpath = fullpath.replace('COORD', COORD)
        #  fullpath = fullpath.replace('III', config.instrument)
        #  np.save(fullpath, signals)
    #  exit()



    # PLOTTING PARAMETERS
    plot_all_data = args.plot_all
    plot_type = 'components' if args.plot_all_type is None else args.plot_all_type

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
    # title = 'Running Average of {:.0f} min'.format(config.averaging_interval/60)
    title = 'Window of {:.1f} h'.format(config.interval/60/60)
    # title = 'Wave Period: {:.1f}min.'.format(synth_options["period"])
    # title = ("Event {0:0=3d}".format(n+1) + ': ' +
                 # interval.dateFrom.strftime('%Y-%m-%d %H:%M') +
                 # ' - ' +
                 # interval.dateTo.strftime('%Y-%m-%d %H:%M'))


    #  if load_signals:
        # signals = np.load('Output/Cassini_MAG_FFT_KRTP_ALL_24H.npy', allow_pickle=True)
        # signals = np.load('Output/Cassini_MAG_FFT_KRTP.npy', allow_pickle=True)
        # signals = np.load('Output/Cassini_MAG_FFT_KRTP_LOWFIT.npy', allow_pickle=True)
        # signals = np.load('Output/Cassini_MAG_FFT_KRTP_HIGHFIT.npy', allow_pickle=True)
        # signals = np.load('Output/Cassini_MAG_FFT_KRTP_SELFRATIO.npy', allow_pickle=True)
        # signals = np.load('Output/Cassini_MAG_FFT_BFIELD.npy', allow_pickle=True)
        #  signals = np.load('Output/Cassini_MAG_FFT_BFIELD_LOWFIT.npy', allow_pickle=True)
        #  signals = np.load('Output/Cassini_MAG_FFT_BFIELD_32H.npy', allow_pickle=True)
        #  signals = np.load('Output/Cassini_MAG_BCOORDS_36H.npy', allow_pickle=True)
        #  signals = np.load('Output/Cassini_MAG_FFT_BCOORDS_36H.npy', allow_pickle=True)
        #  signals = np.load('Output/Cassini_MAG_FFT_BCOORDS_36H_2.npy', allow_pickle=True)
        # np.save('Output/Cassini_MAG_FFT_BFIELD_LOWFIT_SMALL.npy', signals[360:720])
        # signals = np.load('Output/Cassini_MAG_FFT_BFIELD_LOWFIT_SMALL.npy', allow_pickle=True)
        #  print('Data Loaded.')
        #  for n, signal in enumerate(signals):
            #  FIELDS = signal.FIELDS
            #  COORDS = signal.COORDS
            #  for s in FIELDS:
                #  s.nperseg = int(12*60*60 // s.dt)
                #  s.noverlap_method == 'half'
                #  s.datetime = signal.datetime
                #  s.analyze(actions = ['resample', 'fft'])
                #  s.datetime = None
            #  if n % 100 == 0:
                #  print('N: {:d} ({:.1f}%)'.format(n, n/len(signals)*100))
        #  for signal in signals:
            #  if signal.flag is None:
                #  for s in signal.FIELDS:
                    #  s.dfft = (s.fft - signal.FIELDS[3].fft_bg)/(signal.FIELDS[3].fft_bg)
                    #  s.dfft = np.exp(np.log(s.fft) - np.log(signal.FIELDS[3].fft_bg))
                    #  s.dfft = np.exp(np.log(s.fft) - np.log(s.fft_bg))
                    #  s.dfft = np.exp(np.log(s.fft_bg) - np.log(signal.FIELDS[3].fft_bg))
        #  print('Done recomputing')
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
    #  save_signals = True
    #  if save_signals:
        #  OUTPUT_DIR = np.atleast_1d(OUTPUT_DIR)
        #  if not os.path.exists(os.path.join(*OUTPUT_DIR)):
            #  os.makedirs(os.path.join(*OUTPUT_DIR))
        #  fullpath = os.path.join(*OUTPUT_DIR, FILENAME_FFT_ARCHIVE + '.npy')
        #  fullpath = fullpath.replace('COORD', COORD)
        #  fullpath = fullpath.replace('III', config.instrument)
        #  np.save(fullpath, signals)
    #  exit()



    #  # Load Saturn configuration (browse configurations.py for more options)
    #  SIM = None
    #  trace_step = 0.1  #RS
    #  trace_method = 'euler'
    #  trace_maxR = 100
    #  SIM = loadsim(configSaturnNominal)
    #  SIM.config["ETIME"] = 1483228900  # (01 Jan 2017)
    #  SIM.config["IN_COORD"] = 'KSM'  # S3C/DIS/DIP/KSM/KSO
    #  SIM.config["OUT_COORD"] = 'DIS'  # S3C/DIS/DIP/KSM/KSO
    #  SIM.config["IN_CARSPH"] = 'CAR'  # INPUT IN SPH/CAR
    #  SIM.config["OUT_CARSPH"] = 'CAR'  # OUTPUT IN SPH/CAR
    #  SIM.config["step"] = float(trace_step)
    #  SIM.config["maxIter"] = 1E6
    #  SIM.config["method"] = trace_method
    #  SIM.config["maxR"] = trace_maxR
    #  SIM.step = float(0.1)
    #
    #  for n, x in enumerate(data):
        #  # L, BT_model = returnKMAG_Lshell(x.datetime[0], x.pos_KSM, SIM)
        #  L, BT_model = 1, 1
        #  if L is None or L == 0:
            #  L = 0
            #  print('Overwriting L')
        #  # if n % 100 == 0:
            #  # print('Processing ({}/{}): L={:.1f} RS'.format(n+1, len(data), L) + ' | POSITION=', x.pos_KSM, '| BT_MODEL =', BT_model)
        #  x.info["L"] = L
        #  x.info["BT_model"] = BT_model

    plot_all_data = True

    if plot_all_data:
        fn_out = FILENAME_FFT_ARCHIVE_PLOT
        if ID is not None:
            fn_out = fn_out.replace('ID', "{0:03d}".format(ID))
        fn_out = fn_out.replace('COORD', COORD)
        fn_out = fn_out.replace('III', config.instrument)
        # print(fn_in)
        readAndPlotFFT(data=signals,
                       year=config.year,
                       coord=COORD,
                       component=COMPONENT,
                       location=LOCATION,
                       show_all=True,
                       title=title,
                       save_plots=save_plots,
                       splitByBins=False,
                       splitByTime=False,
                       plot_fft=True,
                       plot_test=False,
                       plot_type = plot_type,
                       periodLines=plot_periodLines,
                       vmin=plot_vmin,
                       vmax=plot_vmax,
                       units=plot_units,
                       plot_spec_ylim = plot_spec_ylim,
                       plot_spec_xlim = plot_spec_xlim,
                       highlight_periods=highlight_periods,
                       highlight_periods2=highlight_periods2,
                       output_dir=OUTPUT_DIR,
                       filename_out=fn_out,
                       )
