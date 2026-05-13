"""Data analysis: filtering, binning, spectral fitting, correlation."""

from qp.analysis.correlation import (
    event_property_correlation as event_property_correlation,
)
from qp.analysis.correlation import (
    sliding_phase_lag as sliding_phase_lag,
)
from qp.analysis.filtering import (
    bin_centers as bin_centers,
)
from qp.analysis.filtering import (
    bin_edges as bin_edges,
)
from qp.analysis.filtering import (
    bin_to_value as bin_to_value,
)
from qp.analysis.filtering import (
    filter_by_datetime as filter_by_datetime,
)
from qp.analysis.filtering import (
    filter_by_property as filter_by_property,
)
from qp.analysis.filtering import (
    group_by_bins as group_by_bins,
)
from qp.analysis.filtering import (
    value_to_bin as value_to_bin,
)
from qp.analysis.spectral_fitting import (
    bin_power_spectra as bin_power_spectra,
)
from qp.analysis.spectral_fitting import (
    power_law_fit as power_law_fit,
)
from qp.analysis.spectral_fitting import (
    spectral_slopes as spectral_slopes,
)
