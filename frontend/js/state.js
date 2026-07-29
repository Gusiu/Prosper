/**
 * State shared between more than one module.
 *
 * Everything else stays private to the module that owns it — this object is
 * deliberately small, and anything added here should be genuinely cross-cutting.
 * ES module bindings are read-only for importers, so shared mutable values live
 * as properties of one object rather than as exported `let`s.
 */
export const state = {
  /** Inventory rows from /api/data/inventory, keyed by nothing — read by most tabs. */
  globalInventory: [],
  /** Mirrors the runner: true while the server reports a task in flight.
   *  The queue itself lives on the server — see /api/task/queue. */
  isPipelineRunning: false,
  /** Absolute cursor into the runner's log buffer; the server trims its window. */
  lastLogIdx: 0,
};

/** Bar sizes in ascending order, used to sort and rank interval pickers. */
export const ANALYSIS_INTERVALS = [
  "1m",
  "3m",
  "5m",
  "15m",
  "30m",
  "1h",
  "2h",
  "4h",
  "6h",
  "8h",
  "12h",
  "1d",
  "3d",
  "1w",
];

// Depth-bin labels are deliberately not duplicated here. Prediction payloads
// name their own bins, so views sort the keys they are given; a copy of the
// backend scheme would go stale the moment the bins change.
