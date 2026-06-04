"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let currentMode = "images";
let lastResults = null;     // {task, columns, results}
let lastAmountMismatch = null;  // {task, columns, results} for amount mismatch
let lastVinMismatch = null;     // {task, columns, results} for VIN mismatch
let lastParkingAnomaly = null; // {task, columns, results} for parking fee anomaly subset

const HINTS = {
  invoice: "每张图片/每行 Excel 各自识别一个实付金额；Excel 含 supply_money 列时自动比对并标红不一致。",
  parking: "停车为「一组图片 = 一次停车事件」：上传图片/URL 视为同一次停车；批量请用 Excel（每行一组，cost_images 为逗号分隔直链）。",
};
const IMG_HINTS = {
  invoice: "可多选；每张独立识别。",
  parking: "可多选；本次上传的所有图片视为同一次停车。",
};
const EXCEL_HINTS = {
  invoice: "需含列 pic_url（图片直链）；可选 supply_money（原始金额，用于比对）。",
  parking: "需含列 cost_images（逗号分隔直链）；可选 cost、vin（用于比对）。",
};

const PARKING_HIGH = 25;
const PARKING_LOW  = 0.1;

function task() { return document.querySelector('input[name=task]:checked').value; }

function refreshHints() {
  $("#task-hint").textContent = HINTS[task()];
  $("#images-hint").textContent = IMG_HINTS[task()];
  $("#excel-hint").textContent = EXCEL_HINTS[task()];
}

$$('input[name=task]').forEach((r) => r.addEventListener("change", refreshHints));

$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  currentMode = t.dataset.mode;
  $$(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== currentMode));
}));

// 异常面板折叠
document.addEventListener("click", (e) => {
  if (!e.target.classList.contains("anomaly-toggle")) return;
  const body = $("#" + e.target.dataset.target);
  if (!body) return;
  body.classList.toggle("collapsed");
  e.target.textContent = body.classList.contains("collapsed") ? "展开" : "收起";
});

async function health() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    $("#backend-badge").textContent = "后端：" + (j.backend || (j.model_loaded ? "已加载" : "待加载"));
  } catch { $("#backend-badge").textContent = "后端：连接失败"; }
}

function setStatus(msg, busy = false) {
  $("#status").textContent = msg || "";
  $("#run").disabled = busy;
}

// ── 通用导出函数 ────────────────────────────────────────
async function exportToExcel(payload, filename) {
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) return alert("导出失败");
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

$("#run").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("task", task());
  fd.append("mode", currentMode);
  if (currentMode === "images") {
    const fs = $("#imgs").files;
    if (!fs.length) return alert("请选择图片");
    for (const f of fs) fd.append("files", f);
  } else if (currentMode === "urls") {
    const t = $("#urls").value.trim();
    if (!t) return alert("请粘贴图片 URL");
    fd.append("urls", t);
  } else {
    const f = $("#excel").files[0];
    if (!f) return alert("请选择 Excel 文件");
    fd.append("excel", f);
  }
  setStatus("识别中…（首次会加载模型，可能较久）", true);
  try {
    const r = await fetch("/api/recognize", { method: "POST", body: fd });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
    const j = await r.json();
    lastResults = { task: j.task, columns: j.columns, results: j.results };
    renderAll(j);
    setStatus(`完成：${j.n} 条`);
  } catch (e) {
    setStatus("出错：" + e.message);
  } finally { $("#run").disabled = false; }
});

// 主结果表导出
$("#export-main").addEventListener("click", () => {
  if (!lastResults) return;
  const name = (lastResults.task === "invoice" ? "发票" : "停车") + "识别结果_全部.xlsx";
  exportToExcel(lastResults, name);
});

// 金额不匹配导出
$("#export-amount-mismatch").addEventListener("click", () => {
  if (!lastAmountMismatch) return;
  const name = (lastAmountMismatch.task === "invoice" ? "发票" : "停车") + "识别结果_金额不匹配.xlsx";
  exportToExcel(lastAmountMismatch, name);
});

// 车架号不匹配导出
$("#export-vin-mismatch").addEventListener("click", () => {
  if (!lastVinMismatch) return;
  exportToExcel(lastVinMismatch, "停车识别结果_车架号不匹配.xlsx");
});

// 停车费异常导出
$("#export-parking-anomaly").addEventListener("click", () => {
  if (!lastParkingAnomaly) return;
  exportToExcel(lastParkingAnomaly, "停车识别结果_费用异常.xlsx");
});

// ── 渲染入口 ────────────────────────────────────────────
function renderAll(j) {
  renderTable(j);
  renderSummary(j);
  renderAmountMismatch(j);
  renderVinMismatch(j);
  renderParkingAnomaly(j);
}

// ── 概览条 ──────────────────────────────────────────────
function renderSummary(j) {
  const sec = $("#summary-section");
  const bar = $("#summary-bar");
  const total = j.results.length;
  if (!total) { sec.classList.add("hidden"); return; }

  const amtMiss = j.results.filter(r => r.amount_match === "✗");
  const vinMiss = j.results.filter(r => r.vin_match === "✗");
  const matched = j.results.filter(r => r.amount_match === "✓" || r.vin_match === "✓");

  let html = `<span class="summary-chip chip-total">共 ${total} 条</span>`;
  if (matched.length)  html += `<span class="summary-chip chip-ok">匹配 ${matched.length}</span>`;
  if (amtMiss.length)  html += `<span class="summary-chip chip-miss">金额不匹配 ${amtMiss.length}</span>`;
  if (vinMiss.length)  html += `<span class="summary-chip chip-vin">VIN不匹配 ${vinMiss.length}</span>`;

  if (j.task === "parking") {
    const anomalies = getParkingAnomalies(j.results);
    if (anomalies.length) html += `<span class="summary-chip chip-warn">费用异常 ${anomalies.length}</span>`;
  }

  bar.innerHTML = html;
  sec.classList.remove("hidden");
}

// ── 主结果表 ────────────────────────────────────────────
function renderTable(j) {
  const cols = j.columns.filter((c) => j.results.some((r) => r[c] !== undefined));
  const thead = $("#result-table thead"), tbody = $("#result-table tbody");
  thead.innerHTML = "<tr>" + cols.map((c) => `<th>${colLabel(c)}</th>`).join("") + "</tr>";
  tbody.innerHTML = j.results.map((row) => {
    const bad = row.amount_match === "✗" || row.vin_match === "✗";
    const tds = cols.map((c) => {
      let v = row[c]; v = (v === null || v === undefined) ? "" : v;
      const cls = ((c === "amount_match" || c === "vin_match") && v === "✗") ? "miss"
        : ((c === "amount_match" || c === "vin_match") && v === "✓") ? "ok" : "";
      return `<td class="${cls}">${String(v)}</td>`;
    }).join("");
    return `<tr class="${bad ? "row-miss" : ""}">${tds}</tr>`;
  }).join("");

  const banner = $("#banner");
  if (j.n_mismatch > 0) {
    banner.textContent = `⚠ 有 ${j.n_mismatch} 条识别金额/车架号与原始标注不一致（已标红）`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }

  $("#export-main").disabled = j.results.length === 0;
}

// ── 异常面板：金额不匹配 ─────────────────────────────────
function renderAmountMismatch(j) {
  const sec = $("#amount-mismatch-section");
  const rows = j.results.filter(r => r.amount_match === "✗");
  if (!rows.length) {
    sec.classList.add("hidden");
    lastAmountMismatch = null;
    return;
  }

  $("#amount-mismatch-count").textContent = `${rows.length} 条`;

  const cols = j.task === "invoice"
    ? ["source", "pred_amount", "label", "amount_match", "error"]
    : ["source", "cost", "payment_amount", "amount_match", "error"];
  const usedCols = cols.filter(c => rows.some(r => r[c] !== undefined && r[c] !== null && r[c] !== ""));

  lastAmountMismatch = { task: j.task, columns: usedCols, results: rows };

  const thead = $("#amount-mismatch-table thead");
  const tbody = $("#amount-mismatch-table tbody");
  thead.innerHTML = "<tr>" + usedCols.map(c => `<th>${colLabel(c)}</th>`).join("") + "</tr>";
  tbody.innerHTML = rows.map(row => {
    const tds = usedCols.map(c => {
      let v = row[c]; v = (v === null || v === undefined) ? "" : v;
      const cls = (c === "amount_match" && v === "✗") ? "miss" : (c === "amount_match" && v === "✓") ? "ok" : "";
      return `<td class="${cls}">${String(v)}</td>`;
    }).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  $("#amount-mismatch-body").classList.remove("collapsed");
  $('[data-target="amount-mismatch-body"]').textContent = "收起";
  sec.classList.remove("hidden");
}

// ── 异常面板：车架号不匹配 ───────────────────────────────
function renderVinMismatch(j) {
  const sec = $("#vin-mismatch-section");
  const rows = j.results.filter(r => r.vin_match === "✗");
  if (!rows.length || j.task === "invoice") {
    sec.classList.add("hidden");
    lastVinMismatch = null;
    return;
  }

  $("#vin-mismatch-count").textContent = `${rows.length} 条`;

  const cols = ["source", "vin_label", "vin_detected", "vin_match", "error"];
  const usedCols = cols.filter(c => rows.some(r => r[c] !== undefined && r[c] !== null && r[c] !== ""));

  lastVinMismatch = { task: j.task, columns: usedCols, results: rows };

  const thead = $("#vin-mismatch-table thead");
  const tbody = $("#vin-mismatch-table tbody");
  thead.innerHTML = "<tr>" + usedCols.map(c => `<th>${colLabel(c)}</th>`).join("") + "</tr>";
  tbody.innerHTML = rows.map(row => {
    const tds = usedCols.map(c => {
      let v = row[c]; v = (v === null || v === undefined) ? "" : v;
      const cls = (c === "vin_match" && v === "✗") ? "miss" : (c === "vin_match" && v === "✓") ? "ok" : "";
      return `<td class="${cls}">${String(v)}</td>`;
    }).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  $("#vin-mismatch-body").classList.remove("collapsed");
  $('[data-target="vin-mismatch-body"]').textContent = "收起";
  sec.classList.remove("hidden");
}

// ── 异常面板：停车费异常 ─────────────────────────────────
function getParkingAnomalies(results) {
  return results.filter(r => {
    const price = parseFloat(r.unit_price_yuan_per_h);
    return !isNaN(price) && price > 0 && (price > PARKING_HIGH || price < PARKING_LOW);
  });
}

function renderParkingAnomaly(j) {
  const sec = $("#parking-anomaly-section");
  if (j.task !== "parking") {
    sec.classList.add("hidden");
    lastParkingAnomaly = null;
    return;
  }

  const rows = getParkingAnomalies(j.results);
  if (!rows.length) {
    sec.classList.add("hidden");
    lastParkingAnomaly = null;
    return;
  }

  $("#parking-anomaly-count").textContent = `${rows.length} 条`;

  const baseCols = ["source", "entry_time", "exit_time", "park_hours",
                    "unit_price_yuan_per_h", "payment_amount", "cost"];
  const usedCols = baseCols.filter(c => rows.some(r => r[c] !== undefined && r[c] !== null && r[c] !== ""));

  // 导出数据增加异常原因列
  const exportRows = rows.map(r => {
    const price = parseFloat(r.unit_price_yuan_per_h);
    return { ...r, anomaly_reason: price > PARKING_HIGH ? `单价过高(>${PARKING_HIGH}元/h)` : `单价过低(<${PARKING_LOW}元/h)` };
  });
  lastParkingAnomaly = { task: j.task, columns: [...usedCols, "anomaly_reason"], results: exportRows };

  const displayCols = [...usedCols, "_reason"];
  const thead = $("#parking-anomaly-table thead");
  const tbody = $("#parking-anomaly-table tbody");
  thead.innerHTML = "<tr>" + displayCols.map(c => `<th>${colLabel(c)}</th>`).join("") + "</tr>";
  tbody.innerHTML = rows.map(row => {
    const price = parseFloat(row.unit_price_yuan_per_h);
    const isHigh = price > PARKING_HIGH;
    const tds = displayCols.map(c => {
      if (c === "_reason") {
        if (isHigh) return `<td><span class="reason-tag reason-tag--high">单价过高 (&gt;${PARKING_HIGH}元/h)</span></td>`;
        return `<td><span class="reason-tag reason-tag--low">单价过低 (&lt;${PARKING_LOW}元/h)</span></td>`;
      }
      let v = row[c]; v = (v === null || v === undefined) ? "" : v;
      const cls = c === "unit_price_yuan_per_h" ? "miss" : "";
      return `<td class="${cls}">${String(v)}</td>`;
    }).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  $("#parking-anomaly-body").classList.remove("collapsed");
  $('[data-target="parking-anomaly-body"]').textContent = "收起";
  sec.classList.remove("hidden");
}

// ── 列名友好映射 ────────────────────────────────────────
function colLabel(key) {
  const map = {
    source: "来源",
    pred_amount: "识别金额",
    label: "标签金额",
    amount_match: "金额匹配",
    error: "错误",
    cost: "标签费用",
    vin_label: "标签VIN",
    num_images: "图片数",
    time_source: "时间来源",
    entry_time: "进场时间",
    exit_time: "出场时间",
    park_hours: "停车时长(h)",
    unit_price_yuan_per_h: "单价(元/h)",
    vin_detected: "识别VIN",
    vin_match: "VIN匹配",
    payment_amount: "识别金额",
    plate: "车牌",
    _reason: "异常原因",
    anomaly_reason: "异常原因",
  };
  return map[key] || key;
}

refreshHints();
health();
