"""Label building and shared direction/depth utilities."""

from prosper.labels.depth import (  # noqa: F401 – public API
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DEPTH_BIN_LABELS,
    DIR_TO_IDX,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    N_DEPTH_BINS,
    HorizonSpec,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
