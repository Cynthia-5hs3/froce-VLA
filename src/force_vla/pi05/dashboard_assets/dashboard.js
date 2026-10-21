"use strict";

function fieldValue(row, key, component) {
  const value = row[key] && row[key][component];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function visibleRows(rows, seconds) {
  if (!rows.length) return [];
  const newest = rows[rows.length - 1].seconds;
  return rows.filter(row => row.seconds >= newest - seconds);
}

function plotRange(rows, series, component) {
  const values = [];
  for (const row of rows) for (const entry of series) {
    const value = fieldValue(row, entry.key, component);
    if (value !== null) values.push(value);
  }
  if (!values.length) return null;
  const low = Math.min(...values), high = Math.max(...values);
  const margin = Math.max((high - low) * .12, Math.abs(high) * .0001, .0001);
  return [low - margin, high + margin];
}

if (typeof module !== "undefined") module.exports = {fieldValue, visibleRows, plotRange};

if (typeof window !== "undefined") {
  const descriptions = {
    position: "基坐标下TCP的XYZ位置，单位毫米。对比模型预测、请求目标和实测反馈；人工接管期间预测线留空，请求线显示手柄目标。",
    orientation: "TCP相对基坐标的Roll/Pitch/Yaw，单位度；从rotation6D正交化后转换。仅改变显示方式，不改变模型输入或控制动作。",
    joints: "机器人反馈的7个关节位置q，弧度转换为度。模型输出TCP位姿，不输出关节角目标；旧日志没有q时留空。",
    action: "显示50步动作块中当前选中索引对应的10D动作随时间变化，不是整块未来轨迹。对比原始预测、请求目标和实测状态；夹爪统一显示开度比例。",
    torque: "7个关节的实测tau_J与当前选中索引的预测力矩。预测线仅作趋势参考，与同列传感器值并非严格同一物理时刻，不能直接视为力矩误差。",
    external: "机器人估计并滤波的7维外部关节力矩tau_ext_hat_filtered，单位Nm。success_30k外部力矩模型使用它作为历史输入；旧A组7维模型不使用。"
  };
  const seriesByTab = {
    position: [{key: "raw_xyz_mm", label: "模型预测", color: "#7953bb"},
      {key: "requested_xyz_mm", label: "请求目标", color: "#147fc0"},
      {key: "measured_xyz_mm", label: "实测位置", color: "#179476"}],
    orientation: [{key: "raw_rpy_deg", label: "模型预测", color: "#7953bb"},
      {key: "requested_rpy_deg", label: "请求目标", color: "#147fc0"},
      {key: "measured_rpy_deg", label: "实测姿态", color: "#179476"}],
    joints: [{key: "joint_degrees", label: "实测关节角", color: "#179476"}],
    action: [{key: "raw", label: "模型原始预测", color: "#7953bb"},
      {key: "requested", label: "请求目标（非实际执行位姿）", color: "#147fc0"},
      {key: "measured_state", label: "实测反馈", color: "#179476"}],
    torque: [{key: "tau", label: "传感器tau_J", color: "#147fc0"},
      {key: "predicted_tau", label: "预测力矩", color: "#df8324"}],
    external: [{key: "external_tau", label: "外部关节力矩估计", color: "#179476"}]
  };
  const actionNames = ["TCP X · m", "TCP Y · m", "TCP Z · m", "旋转6D · col0 X", "旋转6D · col0 Y",
    "旋转6D · col0 Z", "旋转6D · col1 X", "旋转6D · col1 Y", "旋转6D · col1 Z", "夹爪开度 · 0关闭 / 1打开"];
  let payload = null, frozenPayload = null, paused = false, tab = "position", lastReceived = 0;
  const chartElements = [];
  const element = id => document.getElementById(id);

  function buildCharts() {
    element("description").textContent = descriptions[tab];
    element("charts").replaceChildren();
    chartElements.length = 0;
    const titles = tab === "action" ? actionNames : tab === "position" ? ["TCP X · mm", "TCP Y · mm", "TCP Z · mm"] :
      tab === "orientation" ? ["Roll · °", "Pitch · °", "Yaw · °"] :
      Array.from({length: 7}, (_, index) => `关节 J${index + 1} · ${tab === "joints" ? "°" : "Nm"}`);
    for (const title of titles) {
      const card = document.createElement("section"), heading = document.createElement("h2"), canvas = document.createElement("canvas");
      card.className = "chart";
      heading.textContent = title;
      canvas.setAttribute("aria-label", title + " 随时间变化曲线");
      card.append(heading, canvas);
      element("charts").append(card);
      chartElements.push(canvas);
    }
    element("legend").replaceChildren();
    for (const entry of seriesByTab[tab]) {
      const label = document.createElement("span"), dot = document.createElement("i");
      dot.className = "legend-dot";
      dot.style.background = entry.color;
      label.append(dot, document.createTextNode(entry.label));
      element("legend").append(label);
    }
  }

  function draw(canvas, rows, component) {
    const width = canvas.clientWidth || 360, height = 200, ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const left = 63, top = 12, right = width - 14, bottom = height - 30;
    context.clearRect(0, 0, width, height);
    const series = seriesByTab[tab], range = plotRange(rows, series, component);
    context.font = "11px sans-serif";
    if (!range) {
      context.fillStyle = "#8190a1";
      context.fillText("等待此通道的数据", left, 85);
      return;
    }
    const seconds = Number(element("window").value), newest = rows[rows.length - 1].seconds;
    const start = Math.max(0, newest - seconds), end = Math.max(start + .1, newest);
    const horizontal = value => left + (value - start) / (end - start) * (right - left);
    const vertical = value => bottom - (value - range[0]) / (range[1] - range[0]) * (bottom - top);
    context.fillStyle = "#fff0dc";
    rows.forEach((row, index) => {
      if (!row.teleop_active) return;
      const until = Math.min(end, rows[index + 1]?.seconds ?? end, row.seconds + .25);
      context.fillRect(horizontal(row.seconds), top, Math.max(0, horizontal(until) - horizontal(row.seconds)), bottom - top);
    });
    context.lineWidth = 1;
    for (let index = 0; index <= 4; index++) {
      const value = range[0] + (range[1] - range[0]) * index / 4;
      const position = vertical(value);
      context.strokeStyle = "#e9eef4";
      context.beginPath(); context.moveTo(left, position); context.lineTo(right, position); context.stroke();
      context.fillStyle = "#66788b";
      context.textAlign = "right";
      context.fillText(value.toFixed(tab === "action" ? 4 : 2), left - 8, position + 4);
    }
    context.textAlign = "left";
    context.fillText(`${(start - newest).toFixed(1)} s`, left, height - 9);
    context.textAlign = "right";
    context.fillText("最新记录 0 s", right, height - 9);
    for (const entry of series) {
      context.strokeStyle = entry.color;
      context.lineWidth = 1.8;
      context.setLineDash(entry.key === "raw" || entry.key.startsWith("raw_") || entry.key === "predicted_tau" ? [5, 3] : []);
      context.beginPath();
      let connected = false, previousTime = null, previousValue = null;
      for (const row of rows) {
        const value = fieldValue(row, entry.key, component);
        if (value === null) { connected = false; continue; }
        if (previousTime !== null && row.seconds - previousTime > .25) connected = false;
        if (tab === "orientation" && previousValue !== null && Math.abs(value - previousValue) > 180) connected = false;
        if (connected) context.lineTo(horizontal(row.seconds), vertical(value));
        else context.moveTo(horizontal(row.seconds), vertical(value));
        connected = true;
        previousTime = row.seconds;
        previousValue = value;
      }
      context.stroke();
      context.setLineDash([]);
    }
  }

  function updateFreshness() {
    const badge = element("freshness");
    const lost = !lastReceived || Date.now() - lastReceived > 2000 || (payload && Date.now() - payload.generated_at_ms > 2500);
    const status = lost ? "monitor_stopped" : payload.status;
    const labels = {live: "日志更新中", stale: "无新日志 · 推理可能已停止", waiting_file: "等待日志文件",
      waiting_data: "等待传感器记录", monitor_stopped: "监视数据未更新"};
    badge.textContent = labels[status] + (paused ? " · 画面暂停" : "");
    badge.className = "badge " + (status === "live" ? "live" : status.startsWith("waiting") ? "waiting" : "stale");
  }

  function render() {
    updateFreshness();
    const displayed = paused ? frozenPayload : payload;
    if (!displayed) return;
    const latest = displayed.latest || {};
    const names = {policy: "模型控制", teleop: "手柄接管", waiting_for_history_or_chunk: "等待历史 / 预测", inference_restart: "重启等待", deployment_session: "加载会话"};
    element("control-status").textContent = names[latest.status] || latest.status || "等待";
    element("step").textContent = latest.step ?? "—";
    element("action-index").textContent = latest.action_index === null || latest.action_index === undefined ? "—" : `${latest.action_index} / 0–49`;
    element("latency").textContent = Number.isFinite(latest.inference_ms) ? `${latest.inference_ms.toFixed(0)} ms` : "—";
    element("age").textContent = Number.isFinite(displayed.log_age_s) ? `${displayed.log_age_s.toFixed(1)} s` : "—";
    element("log-path").textContent = displayed.log;
    element("warning").hidden = tab !== "action" || displayed.full_action_available || !displayed.rows.length || latest.teleop_active;
    element("warning").textContent = "当前记录未包含完整原始10D预测：仅显示日志中实际存在的通道。重启更新后的推理程序可记录完整数据。";
    const rows = visibleRows(displayed.rows, Number(element("window").value));
    chartElements.forEach((canvas, index) => draw(canvas, rows, index));
  }

  window.receiveTelemetry = value => {
    if (!value || value.schema !== 1 || !Array.isArray(value.rows)) return;
    payload = value;
    lastReceived = Date.now();
    render();
  };
  let pending = false;
  function poll() {
    updateFreshness();
    if (pending) return;
    pending = true;
    const script = document.createElement("script");
    script.src = `telemetry.js?refresh=${Date.now()}`;
    const done = () => { pending = false; script.remove(); };
    script.onload = done;
    script.onerror = done;
    document.body.append(script);
  }
  document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => {
    tab = button.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach(item => item.classList.toggle("active", item === button));
    buildCharts(); render();
  }));
  element("pause").addEventListener("click", () => {
    paused = !paused;
    frozenPayload = paused ? payload : null;
    element("pause").textContent = paused ? "继续画面" : "暂停画面";
    render();
  });
  element("window").addEventListener("change", render);
  window.addEventListener("resize", render);
  buildCharts();
  poll();
  setInterval(poll, 250);
}
