"""Data I/O: PDS readers, pre-computed product loaders, boundary crossings."""

from qp.io.pds import (
    select_data,
    read_timeseries_file,
    mag_filepath,
    mag_filepaths_for_range,
    ColumnDef,
    COLUMNS,
    FIELD_COLORS,
    DATETIME_FMT,
    MISSION_START,
    MISSION_END,
)
from qp.io.products import (
    load_spacecraft_position,
)
from qp.io.mag_reader import (
    MagSegment,
    load_segment,
    read_mag_segment,
    save_segment,
)
from qp.io.crossings import (
    parse_crossing_list,
    build_crossing_timeseries,
    export_crossings,
    BoundaryCrossing,
    MS,
    SH,
    SW,
    UNKNOWN,
)
