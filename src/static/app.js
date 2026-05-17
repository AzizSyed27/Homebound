"use strict";

/* Bicycle Theft Return Predictor — frontend logic.
 * Talks to the REAL Flask backend: GET /health, GET /meta, POST /predict.
 * Design ported from the Claude Design handoff (cobalt & coral, bar meter). */

const CATS = ["BIKE_TYPE", "BIKE_COLOUR", "DIVISION", "LOCATION_TYPE", "PREMISES_TYPE", "PRIMARY_OFFENCE"];

const MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const DOW_FULL = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];

// Valid real model codes so the example never folds to "Other".
const EXAMPLE = {
  BIKE_COST: 1450,
  BIKE_TYPE: "RC",
  BIKE_COLOUR: "BLK",
  DIVISION: "D14",
  LOCATION_TYPE: "Streets, Roads, Highways (Bicycle Path, Private Road)",
  PREMISES_TYPE: "Outside",
  PRIMARY_OFFENCE: "THEFT UNDER",
};

let META = null;
let timingMode = "dt";          // "dt" | "parts"
let showAllErrors = false;
let lastSubmittedCats = {};
let lastResult = null;

const $ = (s) => document.querySelector(s);

/* ---------------- result state machine ---------------- */
const STATE_LABEL = { idle: "Idle", loading: "Predicting", success: "Success", error: "Error" };
function setState(state) {
  $("#result-state-pill").textContent = `[ ${STATE_LABEL[state]} ]`;
  $("#result").querySelectorAll("[data-show]").forEach((el) => {
    el.hidden = el.getAttribute("data-show") !== state;
  });
}

/* ---------------- startup ---------------- */
async function init() {
  bindEvents();
  applyTimingMode();
  setState("idle");
  await Promise.all([loadHealth(), loadMeta()]);
}

async function loadHealth() {
  const dot = $("#health-dot");
  const text = $("#health-text");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("health " + res.status);
    const data = await res.json();
    dot.className = "health-dot";
    text.textContent = `online · ${data.model || "—"}`;
  } catch {
    dot.className = "health-dot offline";
    text.textContent = "offline";
  }
}

async function loadMeta() {
  try {
    const res = await fetch("/meta");
    if (!res.ok) throw new Error("/meta returned HTTP " + res.status);
    META = await res.json();
    populateCategoryDropdowns();
    populateDowSelect();
  } catch (err) {
    $("#meta-error-msg").textContent =
      "Could not load form options from the server (/meta). The model may not be running. " +
      (err && err.message ? err.message : "");
    $("#meta-error").hidden = false;
    $("#predict-form").querySelectorAll("input, select, button").forEach((el) => { el.disabled = true; });
  }
}

function fillSelect(sel, options, placeholder) {
  sel.innerHTML = "";
  const ph = new Option(placeholder, "", true, true);
  ph.disabled = true;
  sel.add(ph);
  options.forEach((o) => sel.add(new Option(o.label, o.value)));
}

function populateCategoryDropdowns() {
  CATS.forEach((col) => {
    const sel = document.querySelector(`select[data-cat="${col}"]`);
    if (sel) fillSelect(sel, META.categories[col] || [], "Select…");
  });
}

function populateDowSelect() {
  const sel = $("#occ_dow");
  sel.innerHTML = "";
  const ph = new Option("Pick a day…", "", true, true);
  ph.disabled = true;
  sel.add(ph);
  DOW_FULL.forEach((name, i) => sel.add(new Option(`${i} · ${name}`, String(i))));
}

function labelFor(col, value) {
  if (!META) return value;
  const opt = (META.categories[col] || []).find((o) => o.value === value);
  return opt ? opt.label : value;
}

/* ---------------- timing mode toggle ---------------- */
function applyTimingMode() {
  const dt = timingMode === "dt";
  $("#mode-dt").classList.toggle("active", dt);
  $("#mode-parts").classList.toggle("active", !dt);
  $("#timing-dt").hidden = !dt;
  $("#timing-parts").hidden = dt;
  $("#occ_dt").disabled = !dt;
  ["#occ_month", "#occ_hour", "#occ_dow"].forEach((s) => { $(s).disabled = dt; });
}

/* ---------------- validation ---------------- */
function validate() {
  const e = {};
  if (timingMode === "dt") {
    if (!$("#occ_dt").value) e.occ_dt = "Required";
  } else {
    const m = Number($("#occ_month").value);
    if ($("#occ_month").value === "" || isNaN(m) || m < 1 || m > 12) e.occ_month = "1–12";
    const h = Number($("#occ_hour").value);
    if ($("#occ_hour").value === "" || isNaN(h) || h < 0 || h > 23) e.occ_hour = "0–23";
    const d = Number($("#occ_dow").value);
    if ($("#occ_dow").value === "" || isNaN(d) || d < 0 || d > 6) e.occ_dow = "Pick a day";
  }
  if ($("#report_dt").value && timingMode === "dt" && $("#occ_dt").value) {
    if (new Date($("#report_dt").value) < new Date($("#occ_dt").value)) {
      e.report_dt = "Must be on/after occurrence";
    }
  }
  const c = Number($("#BIKE_COST").value);
  if ($("#BIKE_COST").value === "" || isNaN(c) || c < 0) e.BIKE_COST = "Required, ≥ 0";
  CATS.forEach((k) => { if (!document.querySelector(`select[data-cat="${k}"]`).value) e[k] = "Required"; });
  return e;
}

function renderErrors(errs) {
  document.querySelectorAll("[data-err]").forEach((span) => {
    const key = span.getAttribute("data-err");
    const msg = showAllErrors ? errs[key] : null;
    span.textContent = msg || "";
    span.hidden = !msg;
    // helper sits right after the err span's sibling; hide helper when error shows
    const field = span.closest(".field");
    if (field) {
      const helper = field.querySelector(".helper");
      if (helper) helper.hidden = !!msg;
      const ctrl = field.querySelector(".control");
      if (ctrl) ctrl.classList.toggle("invalid", !!msg);
    }
  });
}

/* ---------------- submit ---------------- */
async function onSubmit(evt) {
  evt.preventDefault();
  showAllErrors = true;
  const errs = validate();
  renderErrors(errs);
  if (Object.keys(errs).length) return;

  const payload = { BIKE_COST: Number($("#BIKE_COST").value) };
  if (timingMode === "dt") {
    payload.occ_dt = $("#occ_dt").value;
  } else {
    payload.occ_month = Number($("#occ_month").value);
    payload.occ_hour = Number($("#occ_hour").value);
    payload.occ_dow = Number($("#occ_dow").value);
  }
  if ($("#report_dt").value) payload.report_dt = $("#report_dt").value;
  lastSubmittedCats = {};
  CATS.forEach((c) => {
    const v = document.querySelector(`select[data-cat="${c}"]`).value;
    payload[c] = v;
    lastSubmittedCats[c] = v;
  });

  setState("loading");
  const btn = $("#predict-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      renderError(data.error || `Request failed (HTTP ${res.status}).`);
    } else {
      renderSuccess(data);
    }
  } catch (err) {
    renderError("Network error: " + (err && err.message ? err.message : err));
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Predict outcome &rarr;";
  }
}

/* ---------------- result rendering ---------------- */
function renderError(message) {
  $("#error-message").textContent = message;
  setState("error");
}

function renderSuccess(data) {
  lastResult = data;
  const f = data.features || {};
  const returned = data.prediction === 1;
  const p = typeof data.proba === "number" ? data.proba : null;

  const verdict = $("#verdict");
  verdict.className = "verdict " + (returned ? "returned" : "not-returned");
  $("#verdict-emoji").textContent = returned ? "🎉" : "🚲";
  $("#verdict-badge").textContent = `Verdict · prediction = ${data.prediction}`;
  $("#verdict-label").textContent = returned ? "Likely returned" : "Likely not returned";

  if (p === null) {
    $("#prob-value").innerHTML = '—';
    $("#meter-fill").style.width = "0%";
  } else {
    const pct = (p * 100).toFixed(p < 0.05 ? 2 : 1);
    $("#prob-value").innerHTML = `${pct}<span class="pct">%</span>`;
    $("#meter-fill").style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
  }

  // Unsolicited "Other" fold detection
  const folded = CATS.filter((c) =>
    lastSubmittedCats[c] && lastSubmittedCats[c] !== "Other" && f[c] === "Other");
  const warn = $("#fold-warning");
  if (folded.length) {
    warn.hidden = false;
    warn.innerHTML =
      `<span>⚠️</span><div><strong>Heads up:</strong> ` +
      `${folded.length === 1 ? "one input wasn't" : folded.length + " inputs weren't"} ` +
      `a recognised category and the model substituted <code>"Other"</code> for ` +
      folded.map((c) => `<code>${c}</code>`).join(", ") +
      `. Pick from the dropdown to use a known value.</div>`;
  } else {
    warn.hidden = true;
    warn.innerHTML = "";
  }

  // Echoed features
  const monthName = MONTHS_SHORT[(f.occ_month || 1) - 1] || f.occ_month;
  const dowName = DOW_FULL[f.occ_dow] || f.occ_dow;
  const dl = $("#features-list");
  dl.innerHTML = "";
  const add = (dt, ddHtml, foldedFlag) => {
    const dtEl = document.createElement("dt");
    dtEl.textContent = dt;
    const ddEl = document.createElement("dd");
    if (foldedFlag) ddEl.className = "folded";
    ddEl.innerHTML = ddHtml;
    dl.append(dtEl, ddEl);
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (m) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m]));

  add("occ_month", `${f.occ_month} · ${monthName}`);
  add("occ_hour", `${String(f.occ_hour).padStart(2, "0")}:00`);
  add("occ_dow", `${f.occ_dow} · ${dowName}`);
  add("report_delay_h", `${f.report_delay_hours}h`);
  add("BIKE_COST", `$${Number(f.BIKE_COST).toLocaleString()}`);
  CATS.forEach((c) => {
    const code = f[c];
    const label = labelFor(c, code);
    const codeTag = label !== code
      ? ` <span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px;margin-left:6px">· ${esc(code)}</span>`
      : "";
    add(c, `${esc(label)}${codeTag}`, folded.includes(c));
  });

  $("#raw-json").textContent = JSON.stringify(data, null, 2);
  $("#raw-json").hidden = true;
  $("#raw-toggle").classList.remove("open");
  setState("success");
}

/* ---------------- actions ---------------- */
function applyExample() {
  timingMode = "dt";
  applyTimingMode();
  const d = new Date();
  d.setDate(d.getDate() - 3);
  d.setHours(22, 15, 0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  $("#occ_dt").value =
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const r = new Date();
  r.setDate(r.getDate() - 2);
  r.setHours(8, 30, 0, 0);
  $("#report_dt").value =
    `${r.getFullYear()}-${pad(r.getMonth() + 1)}-${pad(r.getDate())}T${pad(r.getHours())}:${pad(r.getMinutes())}`;
  $("#BIKE_COST").value = EXAMPLE.BIKE_COST;
  CATS.forEach((c) => {
    const sel = document.querySelector(`select[data-cat="${c}"]`);
    if (Array.from(sel.options).some((o) => o.value === EXAMPLE[c])) sel.value = EXAMPLE[c];
  });
  showAllErrors = false;
  renderErrors({});
}

function resetAll() {
  $("#predict-form").reset();
  timingMode = "dt";
  applyTimingMode();
  showAllErrors = false;
  renderErrors({});
  lastResult = null;
  setState("idle");
}

/* ---------------- events ---------------- */
function bindEvents() {
  $("#predict-form").addEventListener("submit", onSubmit);
  $("#example-btn").addEventListener("click", applyExample);
  $("#reset-btn").addEventListener("click", resetAll);
  $("#mode-dt").addEventListener("click", () => { timingMode = "dt"; applyTimingMode(); if (showAllErrors) renderErrors(validate()); });
  $("#mode-parts").addEventListener("click", () => { timingMode = "parts"; applyTimingMode(); if (showAllErrors) renderErrors(validate()); });
  $("#banner-dismiss").addEventListener("click", () => $("#banner").remove());
  $("#raw-toggle").addEventListener("click", () => {
    const pre = $("#raw-json");
    pre.hidden = !pre.hidden;
    $("#raw-toggle").classList.toggle("open", !pre.hidden);
  });
  $("#predict-form").addEventListener("input", () => { if (showAllErrors) renderErrors(validate()); });
  $("#predict-form").addEventListener("change", () => { if (showAllErrors) renderErrors(validate()); });
}

document.addEventListener("DOMContentLoaded", init);
