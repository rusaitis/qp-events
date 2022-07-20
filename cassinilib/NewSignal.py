""" Signal Classes and functions for detecting waves in time series data

"""

__version__ = '0.1'
__author__ = 'Liutauras Rusaitis'

import numpy as np
import math
from pylab import *
from scipy import fft
from scipy import signal
from itertools import repeat
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.integrate import simps
from numpy import trapz
import numpy.ma as ma  # Masked Arrays
from scipy.interpolate import interp1d
from cassinilib.DataPaths import *
from cassinilib import SelectData
from cassinilib.Wave import Wave
from cassinilib import DatetimeFunctions
import copy

def round_up_to_odd(f):
    ''' Round a float to the nearest odd int '''
    return int(np.ceil(f) // 2 * 2 + 1)

def fft_helper(N, data0, dt=1, window=None):
    ''' Calculate FFT amplitude and power spectrum '''
    A = 2.0 / N
    # A = 8.0 / N
    # N = int(N // 2)
    N = N
    data = copy.copy(data0)
    # FREQUENCY SPACE
    # freq = rfftfreq(N, d=dt)[0:N]
    freq = rfftfreq(N, d=dt)

    if window is not None:
        data *= window

    fft = rfft(data)
    # fft_amp = A * np.abs(fft[0:N])
    # fft_pow = A * np.power(np.abs(fft[0:N]), 2)
    fft_amp = A * np.abs(fft)
    fft_pow = A * np.power(np.abs(fft), 2) / (1/60)

    return fft, fft_amp, fft_pow, freq


def calculateSpectrogram(data,
                         dt=1,  # Sampling rate
                         nperseg=None,
                         noverlap=None,
                         **kwargs):
    ''' Return a spectrogram for the given data '''

    fs = 1./dt
    if nperseg is None:
        nperseg = len(data)
    if noverlap is None:
        noverlap = 0

    freq, tseg, S  = signal.spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,  # nperseg // 8 (default)
        **kwargs
        )

    NFFT = int(nperseg)
    # Padding is needed for first and last segment:
    pad_t = (NFFT - noverlap) * dt / 2

    return freq, tseg, S, pad_t

def calculateWaveletSpectrogram(data,
                                dt = 60,
                                w = 30 * 60,
                                freq_lim = 1 * 60 * 60,
                                **kwargs):
    ''' Return a Wavelet spectrogram for the given data '''

    w = 10
    # w = 10 * 60
    freq_lim = 4 * 60 * 60

    fs = 1./dt
    tseg = np.linspace(0, (len(data)-1)*dt, len(data))
    freq = np.linspace(1/freq_lim, fs/2, 500)
    widths = w*fs / (2*freq*np.pi)

    cwtm = signal.cwt(data, signal.morlet2, widths, w=w)
    # cwtm = signal.cwt(data, signal.ricker, widths)

    return freq, tseg, cwtm

def calculateWelch(data,
                   dt=1,  # Sampling rate
                   nperseg=None,
                   noverlap=None,
                   **kwargs):
    ''' Return a Welch (averaged) FFT '''

    fs = 1./dt
    if nperseg is None:
        nperseg = len(data)

    if noverlap is None:
        noverlap = 0

    freq, fft = signal.welch(data,
                           fs=fs,
                           nperseg=nperseg,
                           noverlap=noverlap,
                           average='mean',
                           **kwargs
                           )
    # Return Power spectral density or power spectrum of x.
    return freq, fft

# def customSpectrogram():
    # Spectrogram estimation:
    # N = NFFT
    # S = []
    # for k in range(0, data.shape[0]+1, N):
        # x = fft.fftshift(fft.fft(data[k:k+N], n=N))[N//2:N]
        # assert np.allclose(np.imag(x*np.conj(x)), 0)(
        # Pxx = np.real(x*np.conj(x))
        # S.append(Pxx)
    # S = np.array(S)
    # f = fft.fftshift(fft.fftfreq(N, d=1/fs))[N//2:N]

def printProgress(message, variable, verbose):
    if verbose:
        print(message, variable)


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


class NewSignal():
    ''' Class to hold signal data, with useful analysis functions '''
    def __init__(self,
                 data=None,     # Signal Data
                 datetime=None,     # Signal Datetime
                 dt=1,          # Time interval between data pts (in mins) 
                 # t=None,        # Time in dt units starting from 0 at data[0]
                 N=None,        # Size of the signal
                 name=None,
                 window=('blackman'),
                 color='black',
                 units=None,
                 kind=None,     # Data kind: Field or Coord?
                 freq_lim=None,
                 nperseg=256,
                 noverlap=0,
                 noverlap_method=None,
                 NFFT=None,
                 resample_total_time = None,
                 NaN_method = 'zero',
                 actions = ['resample', 'fft'],
                 parameters = {},
                 ):
        if data is not None:
            self.y = data
            self.N = size(data)
        else:
            if N is not None:
                self.N = int(N)
                self.y = np.zeros(self.N)
            else:
                self.y = None
                self.N = None
        self.dt = dt
        self.datetime = datetime
        self.datetime_lim = []
        self.name = name
        self.kind = kind
        self.units = units
        self.color = color
        # Analysis
        self.actions = actions
        self.parameters = parameters
        self.dfft = None
        self.fft = None
        self.fft_freq = None
        self.S = None
        self.S_freq = None
        self.S_tseg = None
        self.S_pad = None
        self.cwt = None
        self.cwtm = None
        self.cwtm_freq = None
        self.cwtm_tseg = None
        # Analysis parameters
        self.freq_lim = freq_lim
        self.window = window
        self.nperseg = nperseg
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.noverlap_method = noverlap_method
        self.waves = []
        self.peaks = []
        self.peaksf = []
        self.properties = []
        self.propertiesf = []
        self.peakstrengths = []
        self.resample_total_time = resample_total_time
        self.NaN_method = NaN_method
    # Access the data as array elements:
    # def __iter__(self):
        # return iter(self.y)

    # def aslist(self):
        # return self.y

    # def __len__(self):
        # return len(self.y)

    def __getitem__(self, idt):
        return self.y[idt]

    def wave_finder(self):
        ''' Find the peaks in the data '''
        data = self.y

        self.peaks, self.properties = signal.find_peaks(data,
                                                 # height=(0.015,2),
                                                 prominence=(0.015, 3.0),
                                                 width=(10,30),
                                                 wlen=30,
                                                 distance=20,
                                                 # rel_height=0.5,
                                                 )
        self.peaksCWT = signal.find_peaks_cwt(data,
                                       # np.arange(4,70),
                                       np.arange(7,30),
                                       )
        if self.calculate_fft:
            self.peaksf, self.propertiesf = find_peaks(
                self.fft,
                # self.S[0],
                # height=(0.01, None), # [-1,1]
                prominence=(0.00001, 1e4), # (None, 0.6)
                width=(4,30), # (1,30)
                # wlen=2, # 20
                distance=5, # 5
            )

        timeaxis = self.datetime
        properties = self.properties
        peaks = self.peaks

        peak_heights = []
        for i, pk in enumerate(peaks):
            peak_height = properties["prominences"][i]
            peak_heights.append(peak_height)
        peak_heights = np.asarray(peak_heights)
        avg_peak_height = None
        std_peak_height = None
        if peak_heights.size > 1:
            avg_peak_height = np.average(peak_heights)
            std_peak_height = np.std(peak_heights)

        peaks = self.peaksCWT
        intervals = []
        for i in range(1, len(peaks)):
            interval = (timeaxis[peaks[i]]
                        - timeaxis[peaks[i-1]]).total_seconds()
            if interval > 15*60 and interval < 90*60:
                intervals.append(interval)

        intervals = np.asarray(intervals)
        avg_interval = None
        std_interval = None
        if intervals.size > 2:
            avg_interval = np.average(intervals)
            std_interval = np.std(intervals)

        power = 1
        waveSignal = WaveSignal(dateFrom=timeaxis[peaks[0]],
            dateTo=timeaxis[peaks[-1]],
            freq=1/avg_interval,
            amplitude=avg_peak_height,
            duration=(timeaxis[-1]-timeaxis[0]).total_seconds(),
            power=power,
            waveforms=None,
            intervals=intervals,
            amplitudes=peak_heights,
            slope=None,
            ppo_phase=None,
            axis=self.name,)

        self.waves.append(waveSignal)
        return waveSignal

    def printWaveSignals(self, returnOnly=False):
        string = ''
        for wave in self.waves:
            s = ("Detected a {:.1f}min signal".format(1/wave.freq/60) + ': '
                + wave.start_datetime.strftime('%Y-%m-%dT%H-%M-%S')
                + ' - '
                + wave.end_datetime.strftime('%Y-%m-%dT%H-%M-%S')
                + 'A={:.1f} nT, '.format(wave.amplitude)
                + 'dt={:.1f} h, '.format(wave.duration/60/60)
                + 'P={:.1f} nT^2, '.format(wave.power) + '\n')
            string += s
        if not returnOnly: print(string)
        return string

    def uniform_resampler(self, interval=None, resample_from=None,
                          resample_to=None, threshold=600):
        if interval is None:
            if self.resample_total_time is not None:
                interval = self.resample_total_time
            else:
                if all(x is not None for x in [resample_from, resample_to]):
                    interval = resample_to - resample_from
                else:
                    interval = self.datetime[-1] - self.datetime[0]
        N = int(interval // self.dt)
        data = self.y
        if resample_from is not None:
            start_time = resample_from
        else:
            start_time = self.datetime[0]

        # Deal with any present NaNs in the data
        if self.NaN_method == 'zero':
            data = np.nan_to_num(data)
        elif self.NaN_method == 'mean':
            data = np.where(np.isnan(data),
                            ma.array(data,
                                     mask=np.isnan(data)).mean(axis=0),
                            data)
        elif self.NaN_method == 'linear':
            ind = np.argwhere(np.isnan(data))
            indexL = ind[0][0] - 1
            indexR = ind[-1][0] + 1
            # print(indexL, indexR)
            newValues = np.linspace(data[indexL], data[indexR], indexR-indexL)
            data[indexL:indexR] = newValues
            # print(np.argwhere(np.isnan(data)))
            # exit()
        elif self.NaN_method == 'cubic':
            ind = np.argwhere(np.isnan(data))
            if len(ind) > 0:
                indexL = ind[0][0]-1
                indexR = ind[-1][0]+1
                time = [DatetimeFunctions.UnixTime(t) for t in self.datetime]
                data = data.tolist()
                del time[indexL:indexR]
                del data[indexL:indexR]
                f = interp1d(time, data, kind='cubic', fill_value="extrapolate")
                time = [time[0] + self.dt*n for n in range(0, N)]
                timeDatetime = [self.datetime[0]
                                   + datetime.timedelta(seconds=self.dt*n)
                                   for n in range(0, N)]
                data = f(time)

        # Convert to ctime so resampler could work on time in seconds
        time = [DatetimeFunctions.UnixTime(t) for t in self.datetime]

        # f = interp1d(time, data, kind='cubic', fill_value="extrapolate")
        f = interp1d(time, data, kind='linear',
                     fill_value=(data[0], data[-1]), bounds_error=False)
        # f = interp1d(time, data, kind='linear', fill_value='extrapolate')

        start_time_unix = DatetimeFunctions.UnixTime(start_time)
        newTime = [start_time_unix + self.dt*n for n in range(0, N)]
        newTimeDatetime = [start_time
                           + datetime.timedelta(seconds=self.dt*n)
                           for n in range(0, N)]

        resampled_data = f(newTime)

        self.y = resampled_data
        self.N = len(resampled_data)
        self.datetime = newTimeDatetime

    def run_averager(self, Navg=None, returnOnly=False):
        ''' Calculate the running average '''

        if Navg is None:
            time_to_avg = 1 / self.freq_lim[0]
            Navg = round_up_to_odd(time_to_avg / self.dt)

        run_avg = uniform_filter1d(self.y,
                                   size=Navg,
                                   mode='nearest',
                                   # origin=-(N//2),
                                   )

        if not returnOnly:
            self.run_avg = run_avg
            self.Navg = Navg
        return run_avg

    def detrender(self, Navg=None, returnOnly=False,
                  mode='normal', saveAvg=False):
        ''' Detrend the data with a running average '''
        run_avg = self.run_averager(Navg=Navg, returnOnly=returnOnly)
        dy = self.y - run_avg

        if mode == 'normal':
            result = dy
        elif mode == 'exclude':
            result = run_avg

        if not returnOnly:
            self.y = result
        if saveAvg:
            self.run_avg = run_avg

        return result

    def smoother(self, Navg=None, order=3,
                 deriv=None, returnOnly=False, data=None):
        ''' Smooth the data '''
        data = self.y if data is None else data
        smoothed = signal.savgol_filter(data,
                                        Navg,
                                        order,
                                        deriv = deriv,
                                        )
        if not returnOnly:
            self.y = smoothed
        return smoothed

    def estimate_fft_background(self, data, Navg=17, order=3, deriv=0,
                                mode='nearest'):
        ''' Estimate the background fft amplitude or power '''
        y = np.log(data)
        freq = self.fft_freq
        freq[0] = 1e-6
        x = np.log(freq)

        # f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        # new_x = np.linspace(x[0], x[-1], 200)
        # data = f(new_x)
        # fft_bg = savgol_filter(data, Navg, order, deriv=deriv, mode=mode)
        # f = interp1d(new_x, fft_bg, kind='cubic', fill_value="extrapolate")
        # fft_bg = f(x)
        # fft_bg = np.exp(fft_bg)

        f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        Nresample = Navg*3
        # Nresample = Navg*6
        # Nresample = Navg*10

        new_x = np.linspace(x[0], x[-1], Nresample)
        data = f(new_x)

        # fft_bg = np.gradient(data, 0.4)
        fft_bg = savgol_filter(data, Navg, order, deriv=deriv, mode=mode)


        ind_low = np.argwhere(self.fft_freq > (1/(3*60*60)))[0][0]
        ind_high = np.argwhere(self.fft_freq > (1/(30*60)))[0][0]

        med_bg = np.median(fft_bg[ind_low:ind_high])
        med_fft = np.median(y[ind_low:ind_high])
        med_dif = med_fft - med_bg
        N = len(data[ind_low:ind_high])
        # print('N crit region: ', N)
        dfft = data - fft_bg
        Nabove = np.count_nonzero(dfft[ind_low:ind_high] > 0)
        score = 0.8 - Nabove / N
        sign = -1 if score > 0 else 1
        sign_prev = sign
        bg_adjust = abs(med_dif / 10)
        n = 0
        dfft_prev = data - fft_bg
        score_target = 0.5
        score_tresh = 0.01
        n_max = 50

        while True:
            n += 1
            dfft = data - fft_bg
            Nabove = np.count_nonzero(dfft[ind_low:ind_high] > 0)
            score = score_target - Nabove / N
            sign = -1 if score > 0 else 1
            signflip = sign_prev * sign < 0
            sign_prev = sign
            if abs(score) < score_tresh or n > n_max:
                break
            else:
                fft_bg += sign * bg_adjust
            if signflip:
                bg_adjust = bg_adjust / 2

        med_bg = np.median(fft_bg[ind_low:ind_high])
        med_fft = np.median(data[ind_low:ind_high])
        # print('N adjustments:', n)
        # print('Med bg:', med_bg)
        # print('Med fft:', med_fft)

        f = interp1d(new_x, fft_bg, kind='cubic', fill_value="extrapolate")
        fft_bg = f(x)
        fft_bg = np.exp(fft_bg)
        return fft_bg

    def returnDetrended(self, data=None, Navg=None, repeat=1):
        ''' Compute a smoothed time series '''
        data = self.y if data is None else data

        for _ in repeat(None, repeat):
            data = self.smoother(data=data, returnOnly=True)
            # self.detrender(Navg=Nwindow,
                           # mode='exlude')
        return data

    def analyze(self, actions=None, parameters=None):
        ''' Analyze the signal: perform a running average, fft,
        and find peaks '''

        if actions is None:
            actions = self.actions
        if parameters is None:
            parameters = self.parameters

        if 'all' in actions:
            actions = ['resample', 'detrend', 'phase', 'fft', 'spec', 'cwt', 'cwtm']
            # actions = ['resample', 'phase', 'fft', 'spec', 'cwt', 'cwtm']
            # actions = ['resample', 'fft', 'spec', 'cwt', 'cwtm']

        # Resample the data interval evenly for FFT
        if 'resample' in actions:
            self.uniform_resampler()

        if 'detrend' in actions:
            if 'detrend_interval' in parameters:
                detrend_interval = parameters['detrend_interval']
            elif self.freq_lim is not None:
                detrend_interval = 1 / self.freq_lim[0]
            else:
                detrend_interval = self.N * self.dt // 10
            detrend_repeat = (parameters['detrend_repeat']
                if 'detrend_repeat' in parameters else 1)
            # detrend_interval = 120*60
            detrend_interval = 180*60
            Navg = int(detrend_interval // self.dt)
            for _ in repeat(None, detrend_repeat):
                self.detrender(Navg = Navg, mode='normal')

        # Smooth the signals (easier for peak detection)
        if 'smooth' in actions:
            if 'smooth_interval' in parameters:
                smooth_interval = parameters['smooth_interval']
            elif self.freq_lim is not None:
                smooth_interval = 1 / self.freq_lim[1]
            else:
               smooth_interval = self.N * self.dt // 50
            Nwindow = round_up_to_odd(smooth_interval / self.dt)
            order = (parameters['smooth_order']
                             if 'smooth_order' in parameters else 3)
            savgol_params = [Nwindow, order]
            detrend_repeat = (parameters['detrend_repeat']
                             if 'detrend_repeat' in parameters else 1)
            smooth_method = (parameters['smooth_method']
                             if 'smooth_method' in parameters else 'run_avg')
            for _ in repeat(None, detrend_repeat):
                if smooth_method == 'savgol':
                    self.smoother(Navg=Nwindow, order=order, deriv=None)
                elif smooth_method == 'run_avg':
                    self.detrender(Navg = Nwindow, mode='exclude')

        # Calculate FFT
        if 'fft' in actions:
            if self.nperseg is None:
                time_interval = (self.datetime[-1]
                                 -self.datetime[0]).total_seconds()
                self.nperseg = int(time_interval // 2 // s.dt)
            if self.NFFT is None:
                self.NFFT = self.nperseg
            if self.noverlap_method is not None:
                if self.noverlap_method == 'half':
                    self.noverlap = int(self.nperseg // 2)
                elif self.noverlap_method == 'eight':
                    self.noverlap = int(self.nperseg // 8)
                elif self.noverlap_method == 'zero':
                    self.noverlap = 0
                elif self.noverlap_method == 'max':
                    self.noverlap = int(self.nperseg - 1)

            # Adjust a smaller data "window" for Welch to average
            nperseg = self.nperseg // 3 * 2
            noverlap = nperseg // 2
            self.fft_freq, self.fft = calculateWelch(
                self.y,
                dt = self.dt,
                nperseg = nperseg,
                noverlap = noverlap,
                window = self.window,
                nfft = self.NFFT,
                detrend = False,  # False, 'linear', 'constant' (default)
                scaling = 'density',  # 'spectrum', 'density' (default)
            )

            # Custom FFT Calculation
            # self.fft, self.fft_amp, self.fft_pow, self.fft_freq = fft_helper(
                # self.N,
                # self.y,
                # dt=self.dt,
                # window=signal.get_window(self.window, self.N)
                # )

            # Calculate power spectra background level
            self.fft_bg = self.estimate_fft_background(self.fft)
            # self.fft = self.fft - self.fft_bg
            # self.fft = np.exp(np.log(self.fft) - np.log(self.fft_bg))

        # Calculate Spectrogram
        if 'spec' in actions:
            (self.S_freq,
             self.S_tseg,
             self.S,
             self.S_pad) = calculateSpectrogram(
                self.y,
                dt = self.dt,
                nperseg = self.nperseg,
                noverlap = self.noverlap,
                window = self.window,
                nfft = self.NFFT,
                detrend = False,  # False, 'linear', 'constant' (default)
                scaling ='density',  # 'spectrum', 'density' (default)
                # mode = 'psd' # 'psd','magnitude','angle','phase','complex'
            )

        # Calculate Spectrogram
        if 'phase' in actions:
            (self.Phase_freq,
             self.Phase_tseg,
             self.Phase,
             self.Phase_pad) = calculateSpectrogram(
                self.y,
                dt = self.dt,
                nperseg = self.nperseg,
                noverlap = self.noverlap,
                window = self.window,
                nfft = self.NFFT,
                # detrend = 'linear',  # False, 'linear', 'constant' (default)
                mode = 'phase' # 'psd','magnitude','angle','phase','complex'
            )

        # Calculate Wavelet Spectrogram
        if 'cwt' in actions or 'cwtm' in actions:
            w = (parameters['wavelet_width']
                 if 'wavelet_width' in parameters else 30 * self.dt)
            freq_lim = (parameters['wavelet_freqlim']
                 if 'wavelet_freqlim' in parameters else 4 * 60 * self.dt)
            (w_freq,
             w_t, w) = calculateWaveletSpectrogram(self.y,
                                                   dt = self.dt,
                                                   w = w,
                                                   freq_lim = freq_lim,
                                                   # wavelet = 'morlet2',
                                                   )
            if 'cwtm' in actions:
                self.cwtm_freq, self.cwtm_tseg, self.cwtm = w_freq, w_t, w
            if 'cwt' in actions:
                self.cwt = np.average(self.cwtm, axis=-1)

        # Find peaks in the signal
        if 'waves' in actions:
            self.wave_finder()

    def add_data(self, data):
        ''' Add data to the signal data '''
        self.y += data

    def add_noise(self, mu=0., sigma=0.01, simple=False):
        ''' Add normal noise '''
        self.y += np.random.default_rng(seed=42).normal(mu, sigma*0.1, self.N)
        self.y += np.random.default_rng(seed=42).normal(mu, sigma, self.N)
        self.y += np.random.default_rng(seed=41).normal(mu, sigma*5, self.N)
        self.y += np.random.default_rng(seed=40).normal(mu, sigma*3, self.N)
        # self.y += np.random.default_rng(self.N).normal(mu, sigma, self.N)
        if not simple:
            self.add_wave(period=10*60, amplitude=0.01, phase=0.3,
                          shift=12*60*60, type='sawtooth',decay_width=6*60*60)
            self.add_wave(period=5*60, amplitude=0.01, phase=0.7,
                          shift=12*60*60, type='sawtooth',decay_width=6*60*60)
            self.add_wave(period=12*60, amplitude=0.01, phase=0.9,
                          shift=12*60*60, type='sawtooth',decay_width=6*60*60)
            self.add_wave(period=20*60, amplitude=0.02, phase=0.6,
                          shift=12*60*60, type='sawtooth',decay_width=6*60*60)

    def add_NaNs(self, dateFrom=None, dateTo=None):
        if dateFrom is None:
            dateFrom = self.datetime[0] + datetime.timedelta(seconds=10*60*60)
            dateTo = self.datetime[0] + datetime.timedelta(seconds=14*60*60)
        i_from = int((datefrom - self.datetime[0]).total_seconds() // self.dt)
        i_to = int((dateto - self.datetime[0]).total_seconds() // self.dt)
        # index_from = np.argwhere(self.datetime[:] > datefrom)[0][0]
        self.y[i_from:i_to] = np.nan

    def add_wave(self,
                 period=60*60,
                 amplitude=1.,
                 phase=0.,
                 shift=0.,
                 cutoff=None,
                 type='sine',
                 decay_width=None,
                 ):
        ''' Add a periodic signal to the signal '''

        f = 1.0 / period
        w = 2.0 * np.pi * f

        # Shift the time axis if needed

        t = np.arange(0, int(np.ceil(self.N * self.dt)), int(np.ceil(self.dt)))
        time = t - int(shift)

        if type == 'sine':
            waveform = np.sin(w * time - phase)
        elif type == 'sawtooth':
            # Periodic sawtooth or triangle waveform 
            # arg: width (of the rising ramp as a prop of the total cycle)
            # 1 produced a rising ramp, 0 produces a falling ramp
            waveform = signal.sawtooth(w * time - phase, width=0.8)
        elif type == 'square':
            # Periodic square-wave waveform
            # If duty is an array, causes wave shape to change over time
            waveform = signal.square(w * time - phase, duty=0.2)
        elif type == 'gausspulse':
            # Gaussian modulated sinusoid
            waveform = signal.gausspulse(time, fc=f)
        elif type == 'poly1d':
            # Frequency-swept cos generator, with a time-dependent frequency
            p = np.poly1d([1E-4 * w, w])
            waveform = signal.sweep_poly(time, p)

        # Add a gaussian envelope for amplitude decay 
        if decay_width:
            gauss_envelope = np.exp(-1.0  / 2.0 / pow(decay_width, 2) *
                                    np.power(time, 2))
            waveform *= gauss_envelope

        # Scale to amplitude
        waveform *= amplitude

        # print(cutoff)
        if cutoff is not None:
            # print('cutoff:', cutoff)
            # print('cutoff ind0 = ', cutoff[0] // self.dt)
            # print('cutoff ind0 = ', cutoff[1] // self.dt)
            # print('length of waveform:', len(waveform))
            # exit()
            waveform[0:int(cutoff[0]//self.dt)] = 0.
            waveform[int(cutoff[1]//self.dt):] = 0.

        self.y += waveform

    def add_waves(self, *waves):
        for wave in waves:
            self.add_wave(period=wave.period,
                          amplitude=wave.amplitude,
                          phase=wave.phase,
                          type=wave.type,
                          decay_width=wave.decay_width,
                          shift=wave.shift,
                          cutoff=wave.cutoff,
                          )

def saveSignal(source, DATE_FROM, DATE_TO, coord, file=None, dt=1):
    ''' Save the signal data to a Numpy file '''
    PROCEED = True
    series = []
    if source == 'MAG':
        dataFilePath = dataFileName(instrument='MAG', measurement='1min',
                                    coords='KRTP', dataDirectory=DATA_DIR)
    else:
        PROCEED = False

    DATA = SelectData(dataFilePath, DATA_SOURCE_FILE_START[0],
                      DATA_SOURCE_FILE_END[0], DATA_SOURCE_TIME_FORMAT[0],
                      0, DATE_FROM, DATE_TO, 0)
    if DATA and PROCEED:
        DATA = np.array(DATA)
        np.savetxt(file, DATA, fmt='%s')

def readSignal(dateFrom,
               dateTo,
               instrument = 'MAG',
               measurement = '1min',
               coord = 'KRTP',
               file = None,
               marginInSec = 0,
               datetimeFormat = '%Y-%m-%dT%H:%M:%S',
               dt = 1,
               **kwargs):
    ''' Read Signal Data for a given time period from Data Files '''

    timeseries = []
    TIME = None

    # Read from a given file, or figure out the data directory
    if file:
        DATA = np.loadtxt(file, dtype = 'str')
    else:
        dataFilePath = dataFileName(instrument = instrument,
                                    measurement = measurement,
                                    coords = coord,
                                    dataDirectory = DATA_DIR)
        DATA = SelectData(dataFilePath,
                          dateFrom,
                          dateTo,
                          datetimeFormat = datetimeFormat,
                          marginInSec = marginInSec,
                          resolutionInSec = dt)

    if DATA:
        DataFrame = DataFields[instrument][coord]
        DATA = np.array(DATA)

        # If the wanted resolution is lower than sourced, do sampling
        Nstep = int(round(dt // DataFrame[0].dt))
        N = len(DATA[:,0])

        if Nstep > 1: N = (N // Nstep + 1) if N % Nstep > 0 else N // Nstep
        for DF in DataFrame:
            DATA_FILTERED = DATA[:, DF.column].astype(DF.type)
            #  if dt > DF.dt:
                #  if DF.Averaged == True:
                    #  # Average the data before sampling (don't use this yet)
                    #  DATA_FILTERED = Core.AvgArray(DATA_FILTERED, Nstep)
                #  else:
                    # Sample the data every Nstep
                #  DATA_FILTERED = DATA_FILTERED[::Nstep]
            if dt < DF.dt:
                print('Requested resolution exceeds data source.')
                sys.exit()
            if DF.kind == 'Time':
                if DF.format is not None:
                    # Convert into a datetime object
                    DATA_FILTERED = [datetime.datetime.strptime(time, DF.format)
                                     for time in DATA[:,DF.column]]
                #  if Nstep > 1: DATA_FILTERED = DATA_FILTERED[::Nstep]
                TIME = DATA_FILTERED
            if DF.units == 'deg':
                DATA_FILTERED = np.deg2rad(DATA_FILTERED)
            timeseries.append(NewSignal(data=DATA_FILTERED,
                                        dt=dt,
                                        name=DF.name,
                                        color=DF.color,
                                        units=DF.units,
                                        kind=DF.kind,
                                        **kwargs))
        if TIME:
            # Copy the time to other columns of the data
            for t in timeseries:
                t.datetime = np.asarray(TIME)
    return timeseries

def simulateSignal(N=600,
                   dt=60,
                   coord='KRTP',
                   periods=None,
                   amplitudes=None,
                   decays=None,
                   shifts=None,
                   cutoffs=None,
                   phases=None,
                   types=None,
                   # fields=['Bx', 'By', 'Bz', 'Btot'],
                   fields=['Br', 'Bth', 'Bphi', 'Btot'],
                   # coords=['X', 'Y', 'Z'],
                   coords=['r', 'th', 'phi'],
                   # colors=['red', 'green', 'blue', 'black'],
                   colors=['#DC267F', '#FFB000', '#FE6100', '#648FFF'],
                   dateFrom = None,
                   **kwargs
                   ):
    ''' Produce an artificial signal with specified periodicities '''

    if dateFrom is None:
        dateFrom = datetime.datetime.strptime('2000-01-01T00:00:00',
                                              '%Y-%m-%dT%H:%M:%S')
    TIME = [dateFrom + datetime.timedelta(seconds=dt*n) for n in range(0,N)]
    hour = 60*60

    signals = []
    for i, name in enumerate(fields):
        signals.append(NewSignal(N=N,
                                 dt=dt,
                                 name=name,
                                 color=colors[i],
                                 units='nT',
                                 kind='Field',
                                 **kwargs
                                 ))
    for i, name in enumerate(coords):
        if i == 0:
            # data = np.asarray([20] * N)
            data = np.asarray([20] * N)
        if i == 1:
            # data = np.linspace(-1, 1, N)
            data = np.linspace(85, 95, N)
        if i == 2:
            # data = np.asarray([0] * N)
            data = np.asarray([0] * N)
        signals.append(NewSignal(N=N,
                                 dt=dt,
                                 data=data,
                                 name=name,
                                 color=colors[i],
                                 units='R_S',
                                 kind='Coord',
                                 **kwargs
                                 ))
    if periods is not None:
        for i, period in enumerate(periods):
            if amplitudes is not None:
                A = amplitudes[i]
            else:
                A = 1.
            if decays is not None:
                decay = decays[i]
            else:
                decay = None
            if shifts is not None:
                shift = shifts[i]
            else:
                shift = N * dt / 2
            if phases is not None:
                phase = phases[i]
            else:
                phase = 0.
            if cutoffs is not None:
                cutoff = cutoffs[i]
            else:
                cutoff = None
            if types is not None:
                type = types[i]
            else:
                type = 'sine'
            signals[0].add_waves(Wave(period=period, amplitude=A, phase=phase,
                                      type=type, decay_width=decay,
                                      shift=shift, cutoff=cutoff))
            signals[1].add_waves(Wave(period=period, amplitude=A, phase=phase+np.pi/4,
                                      type=type, decay_width=decay,
                                      shift=shift, cutoff=cutoff))
            signals[2].add_waves(Wave(period=period, amplitude=A, phase=phase+np.pi/2,
                                      type=type, decay_width=decay,
                                      shift=shift, cutoff=cutoff))
    else:
        wave_48hour = Wave(period=65*hour, amplitude=0.2, phase=0, type='sine',
                          decay_width=80*hour, shift=-6*hour)
        wave_6hour = Wave(period=6*hour, amplitude=0.5, phase=0, type='sine',
                          decay_width=12*hour, shift=6*hour)
        wave_90min = Wave(period=90*60, amplitude=0.1, phase=0, type='sine',
                          decay_width=3*hour, shift=15*hour)
        wave_1hour = Wave(period=1*hour, amplitude=0.1, phase=0, type='sine',
                          decay_width=3*hour, shift=0*hour)
        wave_1hour_phased = Wave(period=1*hour, amplitude=0.15, phase=np.pi, type='sine',
                          decay_width=3.1*hour, shift=0*hour)
        wave_2hour = Wave(period=2*hour, amplitude=0.1, phase=0, type='sine',
                          decay_width=2*hour, shift=9*hour)
        wave_3hour = Wave(period=3*hour, amplitude=0.1, phase=0, type='sine',
                          decay_width=6*hour, shift=15*hour)
        wave_60min2 = Wave(period=hour, amplitude=0.06, phase=0, type='sine',
                          decay_width=1*hour, shift=18*hour)
        wave_60min_phased2 = Wave(period=1*hour, amplitude=0.04, phase=-np.pi/2, type='sine',
                          decay_width=1*hour, shift=18*hour)
        wave_60min3 = Wave(period=hour, amplitude=0.06, phase=0, type='sine',
                          decay_width=1*hour, shift=12*hour)
        wave_60min_phased3 = Wave(period=1*hour, amplitude=0.06, phase=0, type='sine',
                          decay_width=1*hour, shift=12*hour)
        wave_15min = Wave(period=15*60, amplitude=0.05, phase=0, type='sine',
                          decay_width=1*hour, shift=10*hour)
        wave_5min = Wave(period=5*60, amplitude=0.03, phase=np.pi/5, type='sine',
                          decay_width=1*hour, shift=12*hour)
        wave_2min = Wave(period=2*60, amplitude=0.05, phase=0, type='sine',
                          decay_width=0.5*hour, shift=17*hour)
        signals[0].add_waves(
                              # wave_6hour,
                              wave_1hour,
                              # wave_90min,
                              # wave_2hour,
                              # wave_3hour,
                              # wave_30min,
                              # wave_15min,
                              # wave_5min,
                              # wave_2min,
                              )

        signals[1].add_waves(wave_1hour, wave_60min2, wave_60min3)
        signals[2].add_waves(wave_1hour_phased, wave_60min_phased2, wave_60min_phased3)

    wave_1hour = Wave(period=1*hour, amplitude=0.1, phase=0, type='sine',
                      decay_width=3*hour, shift=3*hour)
    # signals[1].add_waves(wave_1hour)
    # signals[2].add_waves(wave_1hour)
    # signals[2] = copy.deepcopy(signal[1])
    # signals[0].add_noise(0, 0.02)
    # signals[0].y += np.random.normal(0, 0.01, N)
    # signals[1].y += np.random.normal(0, 0.01, N)
    # signals[2].y += np.random.normal(0, 0.01, N)
    # The last one is simply the magnitude (e.g., Btot)
    signals[3].y = np.linalg.norm([signals[0].y,
                                   signals[1].y,
                                   signals[2].y],
                                  axis=0)
    for signal in signals:
        signal.datetime = TIME

    return signals

def saveSignals(signals,
                dataDirectory='/Users/leo/DATA/CASSINI_DATA/',
                coord='KSM',
                ):
    COORDS = [s for s in signals if s.kind == 'Coord']
    FIELDS = [s for s in signals if s.kind == 'Field']

    year = signals[0].datetime[0].strftime('%Y')
    dt_format = '%Y-%m-%dT%H:%M:%S'
    time = signals[0].datetime
    time = [t.strftime(dt_format) for t in time]
    Bx = FIELDS[0].y*1000
    By = FIELDS[1].y
    Bz = FIELDS[2].y
    Bt = FIELDS[3].y
    x = COORDS[0].y
    y = COORDS[1].y
    z = COORDS[2].y

    # Bx = 
    # np.set_printoptions(precision=3, suppress=True)
    Bx = [round(a, 4) for a in Bx]
    By = [round(a, 4) for a in By]
    Bz = [round(a, 4) for a in Bz]
    Bt = [round(a, 4) for a in Bt]
    x =  [round(a, 4) for a in x]
    y =  [round(a, 4) for a in y]
    z =  [round(a, 4) for a in z]

    aa = np.stack((time, Bx, By, Bz, Bt, x, y, z), axis=1)

    DATA_SOURCE_NAME = 'SIM_1MINAVG'
    dataFilePath = os.path.join(dataDirectory, DATA_SOURCE_NAME, year, (year + '_'+ coord + '_1M.TAB'))
    np.savetxt(dataFilePath, aa, fmt='%19s'+'%10.7s'*7)


def generateLongSignal(dateFrom='2000-01-01T00:00:00',
                       dateTo='2000-06-15T00:00:00',
                       coord='KSM',
                       signal_dt=60):


    dateFrom = datetime.datetime.strptime(dateFrom, '%Y-%m-%dT%H:%M:%S')
    dateTo = datetime.datetime.strptime(dateTo, '%Y-%m-%dT%H:%M:%S')
    signal_interval = (dateTo - dateFrom).total_seconds()
    N_interval = int(signal_interval // signal_dt)

    signal_options = {
        'dt' : signal_dt,
        'coord' : coord,
        'dateFrom' : dateFrom,
    }

    hour = 60*60
    mins = 60

    periods =    []
    amplitudes = []
    shifts =     []
    decays =     []
    phases =     []
    cutoffs =    []
    types =      []


    # periods =    [3*hour]
    # amplitudes = [0.2]
    # shifts =     [0*hour]
    # decays =     None
    # phases =     [0]
    # cutoffs =    [[10*hour, 14*hour]]
# 
    # periods =    np.atleast_1d(synth_options['period'])
    # amplitudes = np.atleast_1d(synth_options['amplitude'])
    # shifts =     np.atleast_1d(synth_options['shift'])
    # decays =     np.atleast_1d(synth_options['decay'])
    # phases =     np.atleast_1d(synth_options['phase'])
    # cutoffs =    np.atleast_1d(synth_options['cutoff'])

    # periods =    [9*hour,  4*hour, 2*hour, 60*mins, 30*mins, 15*mins, 10*mins, 5*mins, ]
    # amplitudes = [1,       1,      0.2,    0.2,     0.2,     0.2,     0.2,    0.2,    ]
    # shifts =     [3*hour,  0*hour, 12*hour,3*hour,  6*hour,  10*hour, 8*hour, 3*hour, ]

    # periods =    [9*hour,   60*mins,    30*mins, 15*mins]
    # amplitudes = [4,        0.1,        0.04,     0.05]
    # shifts =     [3*hour,   12*hour,    12*hour, 18*hour]
    # periods =    [10.7*hour, 3*hour,   60*mins,    10*mins,    15*mins]
    # amplitudes = [4,         1,        0.5,        0.1,       0.1]
    # shifts =     [0*hour,    6*hour,   12*hour,    16*hour,    19*hour]
    # periods =    [40*mins, 30*mins,    20*mins,    10*mins]
    # amplitudes = [0.5,    0.5,        0.5,        0.5]
    # phases =     [0, 0, 3*np.pi/3, 0]
    # phases =     [0 * period for period in periods]
    # decays =     [2 * period for period in periods]
    # decays =     [1*hour,  1.5*hour,     0.5*hour]
    # cutoffs =    [[shift-1*hour, shift+1*hour] for shift in shifts]
    # cutoffs =    [None for period in periods]
    # types =      ['sine', 'sawtooth', 'sawtooth']
    # types =      ['sine' for period in periods]

    synth = {}
    synth["periods"]=[]
    synth["amplitudes"]=[]
    synth["shifts"]=[]
    synth["phases"]=[]
    synth["decays"]=[]
    synth["cutoffs"]=[]
    synth["types"]=[]

    def sprinkleWaves(dateFrom, dateTo, dt=None, T=60*60, A=1, phi=0,
                      d=None, cutoff=None, type='sine', dic={}, rand=False):
        currentTime = dateFrom
        current_shift = 0
        while currentTime < dateTo:
            if dt is None:
                currentTime = dateTo
            else:
                current_shift += dt + rand * np.random.normal(0, dt/2)
                currentTime = dateFrom + datetime.timedelta(seconds=current_shift)
            dic["periods"].append(T + rand * np.random.normal(0, T/10))
            dic["amplitudes"].append(A + rand * np.random.normal(0, A/2))
            dic["shifts"].append(current_shift)
            dic["phases"].append(phi + rand * np.random.rand() * np.pi)
            dic["decays"].append(d)
            dic["cutoffs"].append(cutoff)
            dic["types"].append('sine')
        return dic

    def sprinkleRandomWaves(dateFrom, dateTo, dt=20*hour, dic={}):
        currentTime = dateFrom
        current_shift = 0
        while currentTime < dateTo:
            current_shift += dt + np.random.normal(0, dt/2)
            currentTime = dateFrom + datetime.timedelta(seconds=current_shift)
            dic["periods"].append(np.random.uniform(5*mins, 5*hour))
            dic["amplitudes"].append(np.random.uniform(0.01, 0.5))
            dic["shifts"].append(current_shift)
            dic["phases"].append(np.random.rand() * np.pi)
            dic["decays"].append(np.random.uniform(10*mins, 3*hour))
            dic["cutoffs"].append(None)
            dic["types"].append('sine')
        return dic

    synth = sprinkleWaves(dateFrom, dateTo, dt=None, T=200*24*hour, A=1000,
                          phi=0, d=None, type='sine', dic=synth)
    synth = sprinkleWaves(dateFrom, dateTo, dt=None, T=10.7*hour, A=2,
                          phi=0, d=None, type='sine', dic=synth)
    synth = sprinkleWaves(dateFrom, dateTo, dt=55*hour, T=3*hour, A=0.05,
                          phi=0, d=4*hour, type='sine', dic=synth, rand=True)
    synth = sprinkleWaves(dateFrom, dateTo, dt=72*hour, T=60*mins, A=0.02,
                          phi=0, d=3*hour, type='sine', dic=synth, rand=True)
    synth = sprinkleWaves(dateFrom, dateTo, dt=48*hour, T=30*mins, A=0.02,
                          phi=0, d=1*hour, type='sine', dic=synth, rand=True)
    synth = sprinkleRandomWaves(dateFrom, dateTo, dic=synth)


    timeseries = simulateSignal(N=N_interval,
                                periods=synth['periods'],
                                amplitudes=synth['amplitudes'],
                                shifts=synth['shifts'],
                                decays=synth['decays'],
                                phases=synth['phases'],
                                cutoffs=synth['cutoffs'],
                                types=synth['types'],
                                **signal_options)
    saveSignals(timeseries)
    print('Series saved!')
