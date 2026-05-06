"""Label building and shared direction/depth utilities."""

from prosper.labels.depth import (  # noqa: F401 – public API
    DEPTH_BIN_LABELS,
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIRECTION_CLASSES,
    DIR_TO_IDX,
    IDX_TO_DIR,
    HorizonSpec,
    N_DEPTH_BINS,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
