/**
 * One delegated click handler for the whole document.
 *
 * Markup declares intent with `data-action` (plus `data-*` parameters) instead
 * of `onclick="fn(...)"`. That matters for the rows the JS builds itself:
 * interpolating a symbol straight into an attribute string would let any value
 * containing a quote break - or inject - markup. It also means no function has
 * to be reachable as a global.
 */

const handlers = new Map();

/** Register a handler for `data-action="name"`. */
export function registerAction(name, handler) {
  handlers.set(name, handler);
}

export function registerActions(map) {
  for (const [name, handler] of Object.entries(map)) registerAction(name, handler);
}

/**
 * Start listening. Clicks bubble to `document`, so elements added later work
 * without re-binding.
 */
export function startActionDispatch(root = document) {
  root.addEventListener("click", (event) => {
    const target = event.target.closest("[data-action]");
    if (!target || !root.contains(target)) return;

    const handler = handlers.get(target.dataset.action);
    if (!handler) {
      console.warn(`no handler for data-action="${target.dataset.action}"`);
      return;
    }
    event.preventDefault();
    handler(target.dataset, target, event);
  });
}
