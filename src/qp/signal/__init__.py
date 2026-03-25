"""Signal processing: FFT, wavelet, power ratios, cross-correlation."""

from qp.signal.fft import welch_psd, spectrogram, estimate_background
from qp.signal.wavelet import morlet_cwt, cwt_power, cwt_averaged_spectrum
from qp.signal.power_ratio import compute_power_ratios, freq_to_period_minutes
from qp.signal.cross_correlation import (
    cross_correlate,
    phase_shift,
    classify_polarization,
)
from qp.signal.timeseries import (
    block_average,
    uniform_resample,
    running_average,
    detrend,
    smooth_savgol,
    detrend_for_fft,
)
from qp.signal.pipeline import SpectralResult, analyze_segment
from qp.signal.synthetic import (
    simulate_signal,
    simulate_multi_component,
    generate_long_signal,
)
