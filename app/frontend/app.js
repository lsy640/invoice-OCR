"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let currentMode = "images";
let lastResults = null;     // {task, columns, results}

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
    renderTable(j);
    $("#export").disabled = j.results.length === 0;
    setStatus(`完成：${j.n} 条`);
  } catch (e) {
    setStatus("出错：" + e.message);
  } finally { $("#run").disabled = false; }
});

$("#export").addEventListener("click", async () => {
  if (!lastResults) return;
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lastResults),
  });
  if (!r.ok) return alert("导出失败");
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (lastResults.task === "invoice" ? "发票" : "停车") + "识别结果.xlsx";
  a.click();
  URL.revokeObjectURL(url);
});

function renderTable(j) {
  const cols = j.columns.filter((c) => j.results.some((r) => r[c] !== undefined));
  const thead = $("#result-table thead"), tbody = $("#result-table tbody");
  thead.innerHTML = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
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
}

refreshHints();
health();
