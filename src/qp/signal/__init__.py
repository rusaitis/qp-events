"""Signal processing: FFT, wavelet, power ratios, polarization."""

from qp.signal.fft import (
    welch_psd,
    spectrogram,
    estimate_background,
    estimate_background_powerlaw,
)
from qp.signal.wavelet import morlet_cwt, cwt_power
from qp.signal.power_ratio import compute_power_ratios
from qp.signal.cross_correlation import (
    cross_correlate,
    phase_shift,
    classify_polarization,
)
from qp.signal.polarization import (
    stokes_parameters,
    stokes_parameters_real,
    stokes_parameters_tapered,
    degree_of_polarization,
    ellipticity_inclination,
    ellipticity_inclination_from_stokes,
    ellipticity_inclination_tapered,
    per_oscillation_ellipticity,
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
