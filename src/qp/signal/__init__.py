"""Signal processing: FFT, wavelet, power ratios, polarization."""

from qp.signal.cross_correlation import (
    classify_polarization,
    cross_correlate,
    phase_shift,
)
from qp.signal.fft import (
    estimate_background,
    estimate_background_powerlaw,
    spectrogram,
    welch_psd,
)
from qp.signal.pipeline import SpectralResult, analyze_segment
from qp.signal.polarization import (
    degree_of_polarization,
    ellipticity_inclination,
    ellipticity_inclination_from_stokes,
    ellipticity_inclination_tapered,
    per_oscillation_ellipticity,
    stokes_parameters,
    stokes_parameters_real,
    stokes_parameters_tapered,
)
from qp.signal.power_ratio import compute_power_ratios
from qp.signal.synthetic import (
    generate_long_signal,
    simulate_multi_component,
    simulate_signal,
)
from qp.signal.timeseries import (
    block_average,
    detrend,
    detrend_for_fft,
    running_average,
    smooth_savgol,
    uniform_resample,
)
from qp.signal.wavelet import cwt_power, morlet_cwt
