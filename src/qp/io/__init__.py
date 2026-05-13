"""Data I/O: PDS readers, pre-computed product loaders, boundary crossings."""

from qp.io.crossings import (
    MS,
    SH,
    SW,
    UNKNOWN,
    BoundaryCrossing,
    build_crossing_timeseries,
    export_crossings,
    parse_crossing_list,
)
from qp.io.legacy_pickle import register_stubs as register_legacy_pickle_stubs
from qp.io.mag_reader import (
    MagSegment,
    load_segment,
    read_mag_segment,
    save_segment,
)
from qp.io.pds import (
    COLUMNS,
    DATETIME_FMT,
    FIELD_COLORS,
    MISSION_END,
    MISSION_START,
    ColumnDef,
    mag_filepath,
    mag_filepaths_for_range,
    read_timeseries_file,
    select_data,
)
from qp.io.products import (
    load_spacecraft_position,
)
