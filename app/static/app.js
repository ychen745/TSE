// ---------- DOM helpers ----------
const $ = (q) => document.querySelector(q);

// Tabs & controls
const tabAnalyze = $("#tabAnalyze");
const tabExplain = $("#tabExplain");
const tabBacktest = $("#tabBacktest");
const runBtn = $("#runBtn");
const statusEl = $("#status");
const codeEl = $("#code");
const parsedEl = $("#parsed");
const explanationEl = $("#explanation");

const uploadBtn = $("#uploadBtn");
const uploadStatusEl = $("#uploadStatus");
const fileEl = $("#file");

// Backtest UI
const backtestInputs = $("#backtestInputs");
const backtestPanel = $("#backtestPanel");
const btSymbol = $("#btSymbol");
const btStart = $("#btStart");
const btEnd = $("#btEnd");
const btCash = $("#btCash");
const btMetrics = $("#btMetrics");
const btChart = $("#btChart");
const btDaily = $("#btDaily");
const downloadBtn = $("#downloadBtn");
const btAllowShorts = document.querySelector("#btAllowShorts");

let mode = "analyze"; // "analyze" | "explain" | "backtest"

// ---------- Tabs ----------
function setTab(newMode){
  mode = newMode;
  tabAnalyze.classList.toggle("active", mode==="analyze");
  tabExplain.classList.toggle("active", mode==="explain");
  tabBacktest.classList.toggle("active", mode==="backtest");
  // Panels
  backtestInputs.hidden = mode!=="backtest";
  backtestPanel.hidden = mode!=="backtest";
  if (mode !== "explain") showExplanation("", "");
}
tabAnalyze.addEventListener("click", ()=>setTab("analyze"));
tabExplain.addEventListener("click", ()=>setTab("explain"));
tabBacktest.addEventListener("click", ()=>setTab("backtest"));

// ---------- Status & render ----------
function showStatus(msg, kind="muted"){
  statusEl.className = "status " + kind;
  statusEl.textContent = msg;
}
function showParsed(obj){
  parsedEl.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}
function showExplanation(md, html){
  if (html) {
    explanationEl.innerHTML = html; // server-side sanitized HTML
  } else if (md) {
    explanationEl.innerHTML = mdToHtml(md); // fallback client renderer
  } else {
    explanationEl.innerHTML = "";
  }
}
function renderBtMetrics(bt){
  const num = (x, d=2) => (typeof x === "number" ? x.toFixed(d) : x);
  const pct = (x, d=2) => (typeof x === "number" ? (x*100).toFixed(d)+"%" : x);
  btMetrics.innerHTML = `
    <table class="table">
      <tbody>
        <tr><th>Symbol</th><td>${bt.symbol}</td></tr>
        <tr><th>Period</th><td>${bt.start} → ${bt.end}</td></tr>
        <tr><th>Initial Cash</th><td>$${num(bt.initial_cash,0)}</td></tr>
        <tr><th>Final Equity</th><td>$${num(bt.final_equity,0)}</td></tr>
        <tr><th>CAGR</th><td>${pct(bt.cagr)}</td></tr>
        <tr><th>Sharpe</th><td>${num(bt.sharpe,2)}</td></tr>
        <tr><th>Max Drawdown</th><td>${pct(bt.max_drawdown)}</td></tr>
        <tr><th>Approx. Trades</th><td>${bt.trades_approx}</td></tr>
      </tbody>
    </table>
  `;
  btChart.src = bt.equity_curve || "";
  // show first 20 daily rows
  if (bt.daily && bt.daily.dates) {
    const n = Math.min(20, bt.daily.dates.length);
    const lines = ["date,equity,position,return"];
    for (let i=0;i<n;i++){
      lines.push(`${bt.daily.dates[i]},${bt.daily.equity[i]},${bt.daily.position[i]},${bt.daily.returns[i]}`);
    }
    btDaily.textContent = lines.join("\n");
  } else {
    btDaily.textContent = "(no daily series)";
  }
}

// ---------- Minimal client-side MD fallback ----------
function escapeHtml(s){ return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function mdToHtml(md){
  md = md.replace(/```([\s\S]*?)```/g, (_, code)=>`<pre><code>${escapeHtml(code)}</code></pre>`);
  md = md.replace(/^### (.*)$/gm, '<h3>$1</h3>');
  md = md.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  md = md.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  md = md.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
  md = md.replace(/(?:^|\n)(<li>[\s\S]*?<\/li>)(?=\n|$)/g, '<ul>$1</ul>');
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');
  md = md.replace(/`([^`]+)`/g, '<code>$1</code>');
  md = md.replace(/^(?!<h\d>|<ul>|<pre>|<li>|<\/li>|<code>|<\/code>)(.+)$/gm, '<p>$1</p>');
  return md;
}

// ---------- Robust JSON fetch ----------
async function fetchJson(url, payload){
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify(payload)
  });

  const resClone = res.clone();
  let data;
  try { data = await res.json(); }
  catch {
    const text = await resClone.text(); // safe: reading the clone
    throw new Error(`Non-JSON response (${res.status}). First bytes: ${text.slice(0,120)}`);
  }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ---------- Run button handler ----------
runBtn.addEventListener("click", async ()=>{
  const code = codeEl.value.trim();
  if (!code){ showStatus("Please paste some code.", "error"); return; }
  runBtn.disabled = true;
  showParsed("(waiting…)"); showExplanation("", "");
  btMetrics.innerHTML = ""; btChart.src = ""; btDaily.textContent = "";

  try{
    if (mode === "analyze") {
      showStatus("Analyzing…");
      const a = await fetchJson("/analyze", { code });
      showParsed(a.parsed_logic || a);
      showStatus("Done.", "success");
    } else if (mode === "explain") {
      // step 1: analyze fast -> show parsed immediately
      showStatus("Analyzing (step 1/2)…");
      const a = await fetchJson("/analyze", { code });
      showParsed(a.parsed_logic || a);
      // step 2: explanation
      showStatus("Generating explanation (step 2/2)…");
      const e = await fetchJson("/analyze/explain", { code });
      if (e.parsed_logic) showParsed(e.parsed_logic);
      showExplanation(e.markdown, e.html);
      showStatus("Done.", "success");
    } else {
      // backtest
      showStatus("Analyzing for backtest…");
      const a = await fetchJson("/analyze", { code });
      showParsed(a.parsed_logic || a);

      showStatus("Running backtest…");
      const payload = {
        code,
        symbol: (btSymbol.value || "SPY").trim(),
        start: (btStart.value || "2018-01-01").trim(),
        end: (btEnd.value || "").trim() || undefined,
        initial_cash: Number(btCash.value || 100000),
        allow_shorts: !!(btAllowShorts && btAllowShorts.checked)
      };
      const r = await fetchJson("/backtest/run", payload);
      if (r.parsed_logic) showParsed(r.parsed_logic);
      if (r.backtest) renderBtMetrics(r.backtest);
      showStatus("Done.", "success");
    }
  }catch(err){
    showStatus(err.message, "error");
  }finally{
    runBtn.disabled = false;
  }
});

// ---------- Upload .py ----------
uploadBtn.addEventListener("click", async () => {
  const f = fileEl.files && fileEl.files[0];
  if (!f) { uploadStatusEl.className="status error"; uploadStatusEl.textContent="Pick a .py file first."; return; }
  uploadBtn.disabled = true;
  uploadStatusEl.className="status muted"; uploadStatusEl.textContent="Uploading…";
  const fd = new FormData(); fd.append("file", f);
  try {
    const res = await fetch("/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    uploadStatusEl.className="status success"; uploadStatusEl.textContent="Uploaded: " + (data.path || f.name);
  } catch (e) {
    uploadStatusEl.className="status error"; uploadStatusEl.textContent=e.message;
  } finally {
    uploadBtn.disabled = false;
  }
});

async function postAndDownload(url, payload){
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/octet-stream" },
    body: JSON.stringify(payload)
  });
  if (!res.ok){
    // try to read JSON error if present
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.error || msg; } catch {}
    throw new Error(msg);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const match = cd.match(/filename="?([^"]+)"?/i);
  const filename = match ? match[1] : "strategy_report.html";
  const urlObj = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = urlObj; a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(()=>{ URL.revokeObjectURL(urlObj); a.remove(); }, 0);
}

downloadBtn.addEventListener("click", async ()=>{
  const code = codeEl.value.trim();
  if (!code){ showStatus("Paste code before downloading a report.", "error"); return; }
  downloadBtn.disabled = true; showStatus("Building report…");
  try {
    const include_backtest = (mode === "backtest"); // if you're on the Backtest tab, include it
    await postAndDownload("/report/build", {
      code,
      include_explain: true,
      include_backtest,
      format: "html",           // or "md"
      symbol: (btSymbol?.value || "SPY"),
      start: (btStart?.value || "2018-01-01"),
      end: (btEnd?.value || ""),
      initial_cash: Number(btCash?.value || 100000)
    });
    showStatus("Report downloaded.", "success");
  } catch(e){
    showStatus("Report error: " + e.message, "error");
  } finally {
    downloadBtn.disabled = false;
  }
});


// ---------- Init ----------
setTab("analyze");
