"""Data analysis: filtering, binning, spectral fitting, correlation."""

from qp.analysis.filtering import (
    bin_centers as bin_centers,
    bin_edges as bin_edges,
    bin_to_value as bin_to_value,
    filter_by_datetime as filter_by_datetime,
    filter_by_property as filter_by_property,
    group_by_bins as group_by_bins,
    value_to_bin as value_to_bin,
)
from qp.analysis.spectral_fitting import (
    bin_power_spectra as bin_power_spectra,
    power_law_fit as power_law_fit,
    spectral_slopes as spectral_slopes,
)
from qp.analysis.correlation import (
    event_property_correlation as event_property_correlation,
    sliding_phase_lag as sliding_phase_lag,
)
