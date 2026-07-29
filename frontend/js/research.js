/** Research mode: strict / deterministic / seed / metadata flags. */

let researchMode = false;

export function initResearchMode() {
  const saved = localStorage.getItem("prosper_research_mode");
  researchMode = saved === "true";
  applyResearchModeUI();

  // Restore individual flag states
  const flags = JSON.parse(
    localStorage.getItem("prosper_research_flags") || "{}",
  );
  if (flags.strict !== undefined)
    document.getElementById("research-strict").checked = flags.strict;
  if (flags.deterministic !== undefined)
    document.getElementById("research-deterministic").checked =
      flags.deterministic;
  if (flags.seed !== undefined)
    document.getElementById("research-seed").value = flags.seed;
  if (flags.metadata !== undefined)
    document.getElementById("research-metadata").checked = flags.metadata;
}


export function toggleResearchMode() {
  const switchEl = document.getElementById("research-switch");
  const optionsEl = document.getElementById("research-options");
  const badgeEl = document.getElementById("research-badge");

  const isResearch = switchEl.classList.toggle("active");
  researchMode = isResearch;
  localStorage.setItem("prosper_research_mode", String(researchMode));

  if (isResearch) {
    optionsEl.style.display = "flex";
    badgeEl.innerText = "ON";
    badgeEl.className = "research-badge on";
  } else {
    optionsEl.style.display = "none";
    badgeEl.innerText = "OFF";
    badgeEl.className = "research-badge off";
  }
}


export function applyResearchModeUI() {
  const toggle = document.getElementById("research-toggle");
  const sw = document.getElementById("research-switch");
  const badge = document.getElementById("research-badge");
  const options = document.getElementById("research-options");

  if (researchMode) {
    toggle.classList.add("active");
    sw.classList.add("active");
    badge.innerText = "ON";
    badge.className = "research-badge on";
    options.style.display = "flex";
  } else {
    toggle.classList.remove("active");
    sw.classList.remove("active");
    badge.innerText = "OFF";
    badge.className = "research-badge off";
    options.style.display = "none";
  }
}


export function saveResearchFlags() {
  const flags = {
    strict: document.getElementById("research-strict").checked,
    deterministic: document.getElementById("research-deterministic").checked,
    seed: document.getElementById("research-seed").value,
    metadata: document.getElementById("research-metadata").checked,
  };
  localStorage.setItem("prosper_research_flags", JSON.stringify(flags));
}

/** Persist a flag whenever the user toggles it. Called once from main. */
export function bindResearchFlagInputs() {
  [
    "research-strict",
    "research-deterministic",
    "research-seed",
    "research-metadata",
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", saveResearchFlags);
  });
}

/**
 * Research flags as a structured object. The server turns these into CLI
 * flags; the browser no longer assembles command strings.
 */


export function getResearchFlags() {
  if (!researchMode) {
    return { strict: false, deterministic: false, seed: null, save_metadata: false };
  }
  saveResearchFlags();

  const seed = document.getElementById("research-seed").value;
  return {
    strict: document.getElementById("research-strict").checked,
    deterministic: document.getElementById("research-deterministic").checked,
    seed: seed === "" ? null : Number(seed),
    save_metadata: document.getElementById("research-metadata").checked,
  };
}

