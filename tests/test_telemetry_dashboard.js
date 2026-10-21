"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const filename = require.resolve("../src/force_vla/pi05/dashboard_assets/dashboard.js");
const {fieldValue, visibleRows, plotRange} = require(filename);
assert.equal(fieldValue({tau: [null]}, "tau", 0), null);
assert.equal(fieldValue({tau: [0]}, "tau", 0), 0);
assert.equal(plotRange([{tau: null}], [{key: "tau"}], 0), null);
assert.deepEqual(visibleRows([{seconds: 0}, {seconds: 10}, {seconds: 20}], 10), [{seconds: 10}, {seconds: 20}]);

class Element {
  constructor() {
    this.children = []; this.listeners = {}; this.style = {}; this.dataset = {};
    this.classList = {toggle() {}}; this.value = "30"; this.clientWidth = 400; this.textContent = "";
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  remove() {}
  setAttribute() {}
  addEventListener(name, handler) { this.listeners[name] = handler; }
  getContext() { return new Proxy({}, {get: () => () => {}, set: () => true}); }
}
const elements = new Map();
const tabs = ["position", "orientation", "joints", "action", "torque", "external"].map(tab => {
  const element = new Element(); element.dataset.tab = tab; return element;
});
const body = new Element();
const document = {body, createElement: () => new Element(), createTextNode: text => text,
  getElementById: id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
  querySelectorAll: () => tabs};
let now = Date.now();
let tick;
const window = {devicePixelRatio: 1, addEventListener() {}};
vm.runInNewContext(fs.readFileSync(filename, "utf8"), {window, document,
  Date: {now: () => now}, setInterval: callback => {tick = callback;}, console});
assert.equal(elements.get("charts").children.length, 3);
const payload = {schema: 1, generated_at_ms: now, status: "live", log_age_s: .1, log: "/local/log", full_action_available: true,
  latest: {step: 9, status: "policy", action_index: 5, inference_ms: 250},
  rows: [{seconds: 0, raw: Array(10).fill(.2), requested: Array(10).fill(.3), measured_state: Array(10).fill(.25),
    raw_xyz_mm: [200, 200, 200], requested_xyz_mm: [300, 300, 300], measured_xyz_mm: [250, 250, 250],
    raw_rpy_deg: [0, 0, 0], requested_rpy_deg: [1, 1, 1], measured_rpy_deg: [2, 2, 2],
    joint_degrees: Array(7).fill(30), teleop_active: true,
    tau: Array(7).fill(1), predicted_tau: Array(7).fill(1.2), external_tau: Array(7).fill(.3)}]};
window.receiveTelemetry(payload);
assert.equal(elements.get("step").textContent, 9);
assert.equal(elements.get("action-index").textContent, "5 / 0–49");
elements.get("pause").listeners.click();
window.receiveTelemetry({...payload, latest: {...payload.latest, step: 10}});
assert.equal(elements.get("step").textContent, 9);
tabs[1].listeners.click();
assert.equal(elements.get("charts").children.length, 3);
assert.equal(elements.get("step").textContent, 9);
elements.get("pause").listeners.click();
assert.equal(elements.get("step").textContent, 10);
tabs[2].listeners.click();
assert.equal(elements.get("charts").children.length, 7);
tabs[3].listeners.click();
assert.equal(elements.get("charts").children.length, 10);
tabs[4].listeners.click();
assert.equal(elements.get("charts").children.length, 7);
tabs[5].listeners.click();
assert.equal(elements.get("charts").children.length, 7);
window.receiveTelemetry({...payload, latest: {status: "teleop", teleop_active: true}});
assert.equal(elements.get("control-status").textContent, "手柄接管");
now += 4000;
tick();
assert.match(elements.get("freshness").textContent, /未更新/);
assert.match(body.children[0].src, /^telemetry\.js\?refresh=/);
console.log("Passed: chart data, missing values, tabs, pause, stale monitor and local script polling");
