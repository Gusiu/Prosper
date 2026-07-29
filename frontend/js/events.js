/**
 * A minimal publish/subscribe bus.
 *
 * It exists to break dependency cycles: the queue used to import the three tab
 * modules just to refresh them when a task finished, and the models tab
 * imported the evaluation and backtest tabs to repopulate their run pickers.
 * Publishing an event instead keeps the dependency graph acyclic, so a module
 * only imports what it actually calls.
 */

const listeners = new Map();

/** Subscribe to *event*; returns an unsubscribe function. */
export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
  return () => listeners.get(event)?.delete(handler);
}

/** Notify every subscriber. One failing handler must not stop the others. */
export function emit(event, payload) {
  for (const handler of listeners.get(event) ?? []) {
    try {
      handler(payload);
    } catch (error) {
      console.error(`handler for "${event}" failed`, error);
    }
  }
}

export const EVENTS = {
  /** A background task finished (successfully or not). */
  TASK_FINISHED: "task:finished",
  /** /api/data/inventory has been reloaded into state.globalInventory. */
  INVENTORY_LOADED: "inventory:loaded",
  /** /api/ai/models has been reloaded; payload is the run list. */
  RUNS_LOADED: "runs:loaded",
};
