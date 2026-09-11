/* ==========================================================================
   F1OPT 赛车调教优化助手 —— 前端交互逻辑（纯原生 JS，无框架）
   -------------------------------------------------------------------------
   职责：
     1. 赛道选择 → 加载 SVG + 弯道热区
     2. 热区点击 → 反馈面板（12 症状 4 类 + 强度 0-5）
     3. WebSocket 订阅遥测/当前弯高亮/建议结果/连接状态
     4. 建议报告渲染（23 参数表格 + 联动/出处/置信度/tradeoff）
     5. 迭代历史前后对比
   关键约定：
     - 后端响应统一信封 {code, message, data}，取 data 字段
     - feedback.corner_number：NULL=全局症状；非 NULL=具体弯道反馈
       （来源：2026-09-10-implicit-normal-feedback-semantics）
     - 未点击的弯道默认正常，前端不提交任何反馈
     - SVG viewBox 800×600，锚点 anchor_x/y 归一化 0~1
   ========================================================================== */
(function () {
  "use strict";

  /* ---------- 常量 ---------- */
  const API_BASE = "/api/v1";
  // WebSocket 端点：与页面同源，/api/v1/ws
  const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${API_BASE}/ws`;
  // SVG 静态资源前缀（ui/ 为静态根，tracks/ 在其下）
  const SVG_PREFIX = "/tracks/";
  // 热区直径（px），在 800px 画布上约 3.5%
  const HOTZONE_DIAMETER = 28;
  // WebSocket 自动重连配置：断开后 3 秒重连，最多 5 次
  const WS_RECONNECT_DELAY = 3000;
  const WS_RECONNECT_MAX = 5;
  // 症状 → 类别映射（与 domain/symptoms.py 逐字对齐）
  const SYMPTOM_CATEGORY = {
    understeer: "entry", oversteer: "entry", turnin_unresponsive: "entry",
    brake_long: "entry", lockup: "entry",
    midcorner_unstable: "apex", midcorner_traction: "apex",
    exit_wheelspin: "exit",
    bottoming: "global", tyre_wear: "global", straight_slow: "global", lap_slow: "global",
  };
  const GLOBAL_CATEGORY = "global";

  /* ---------- DOM 引用 ---------- */
  const $ = (id) => document.getElementById(id);
  const dom = {
    trackSelect: $("track-select"),
    trackMeta: $("track-meta"),
    mapWrap: $("track-map-wrap"),
    mapEmpty: $("track-map-empty"),
    wsDot: $("ws-dot"),
    wsText: $("ws-status-text"),
    telSpeed: $("tel-speed"), telThrottle: $("tel-throttle"), telBrake: $("tel-brake"),
    telGear: $("tel-gear"), telRpm: $("tel-rpm"), telLaptime: $("tel-laptime"),
    telSector: $("tel-sector"), telCorner: $("tel-corner"), sectorTag: $("telemetry-sector-tag"),
    btnSuggest: $("btn-generate-suggest"),
    reportSummary: $("report-summary"),
    reportWrap: $("report-table-wrap"),
    btnHistory: $("btn-refresh-history"),
    historyWrap: $("history-wrap"),
    toastContainer: $("toast-container"),
    // 反馈面板
    fbOverlay: $("feedback-overlay"), fbClose: $("fb-close"), fbSubmit: $("fb-submit"),
    fbCornerNum: $("fb-corner-number"), fbCornerName: $("fb-corner-name"),
    fbStrength: $("fb-strength-range"), fbStrengthVal: $("fb-strength-value"),
    fbError: $("fb-error"),
  };

  /* ---------- 运行时状态 ---------- */
  const state = {
    tracks: [],            // 24 赛道列表
    currentTrackId: null,  // 当前选中赛道 id
    corners: [],           // 当前赛道弯道列表 [{corner_number, anchor_x, anchor_y, name, ...}]
    feedbackCorners: new Set(), // 已提交反馈的弯道编号集合
    selectedCorner: null,  // 反馈面板当前选中的弯道 {number, name}
    currentCorner: null,   // 遥测高亮的当前弯道编号
    ws: null,              // WebSocket 实例
    wsReconnectCount: 0,   // WebSocket 重连次数计数
  };

  /* ---------- 工具函数 ---------- */

  /** 封装 fetch + JSON 解析，处理 {code, message, data} 信封。 */
  async function fetchJSON(path, options) {
    const resp = await fetch(API_BASE + path, options);
    let body;
    try { body = await resp.json(); } catch (_) { body = null; }
    if (!resp.ok || !body || body.code !== 0) {
      const msg = (body && body.message) || `请求失败（HTTP ${resp.status}）`;
      const err = new Error(msg);
      err.status = resp.status;
      err.body = body;
      throw err;
    }
    return body.data;
  }

  /** Toast 通知：在顶部 toast-container 中堆叠显示，3.2 秒后滑出移除。 */
  function showToast(msg, type) {
    if (!dom.toastContainer) return;
    const el = document.createElement("div");
    el.className = "toast" + (type ? ` toast-${type}` : "");
    el.textContent = msg;
    dom.toastContainer.appendChild(el);
    setTimeout(() => {
      el.classList.add("toast-leaving");
      el.addEventListener("animationend", () => el.remove(), { once: true });
      // 兜底：动画事件未触发时 350ms 后强制移除
      setTimeout(() => { if (el.parentNode) el.remove(); }, 350);
    }, 3200);
  }

  /** 在指定容器内显示 loading spinner。
   *  @param {HTMLElement} container - 目标容器
   *  @param {string} [text] - spinner 下方提示文字
   */
  function showLoading(container, text) {
    const wrap = document.createElement("div");
    wrap.className = "spinner-wrap";
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    spinner.setAttribute("role", "status");
    spinner.setAttribute("aria-label", "加载中");
    wrap.appendChild(spinner);
    if (text) {
      const label = document.createElement("span");
      label.textContent = text;
      wrap.appendChild(label);
    }
    container.innerHTML = "";
    container.appendChild(wrap);
  }

  /** 隐藏 loading spinner（清空容器）。 */
  function hideLoading(container) {
    const wrap = container.querySelector(".spinner-wrap");
    if (wrap) wrap.remove();
  }

  /** 格式化圈速（ms → m:ss.mmm）。 */
  function fmtLapTime(ms) {
    if (ms == null || ms <= 0) return "—";
    const totalSec = ms / 1000;
    const m = Math.floor(totalSec / 60);
    const s = (totalSec - m * 60).toFixed(3);
    return `${m}:${s.padStart(6, "0")}`;
  }

  /** 转义 HTML，防止注入。 */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /** 给遥测数值添加 flash 过渡 class（短暂高亮后移除）。 */
  function flashValue(el) {
    if (!el) return;
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 300);
  }

  /* ========================================================================
     1. 赛道选择
     ======================================================================== */

  async function loadTracks() {
    try {
      const data = await fetchJSON("/tracks");
      state.tracks = Array.isArray(data) ? data : (data && data.items) || [];
      dom.trackSelect.innerHTML = '<option value="">— 请选择赛道 —</option>';
      state.tracks.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t.track_id;
        opt.textContent = `${t.official_name} · ${t.circuit_name}`;
        dom.trackSelect.appendChild(opt);
      });
    } catch (e) {
      dom.trackSelect.innerHTML = '<option value="">赛道加载失败</option>';
      showToast("赛道列表加载失败：" + e.message, "error");
    }
  }

  function onTrackChange() {
    const trackId = dom.trackSelect.value;
    if (!trackId) { return; }
    selectTrack(trackId);
  }

  async function selectTrack(trackId) {
    state.currentTrackId = trackId;
    state.feedbackCorners.clear();
    state.currentCorner = null;

    // 赛道图区域显示 loading spinner
    showLoading(dom.mapWrap, "加载赛道数据…");

    // 通知后端当前赛道
    try {
      await fetchJSON("/tracks/current", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId }),
      });
    } catch (_) { /* 非致命，继续渲染 */ }

    // 获取赛道详情 + 弯道锚点
    try {
      const data = await fetchJSON(`/tracks/${encodeURIComponent(trackId)}`);
      const track = data.track || data;
      const corners = data.corners || track.corners || [];
      state.corners = corners;
      // 更新元信息
      dom.trackMeta.textContent = `${track.circuit_name || ""} · ${track.country || ""} · ${(track.length_m / 1000).toFixed(3)} km · ${corners.length} 弯`;
      renderTrackMap(trackId, corners);
      // 加载已有反馈，标记已反馈弯道
      loadExistingFeedback(trackId);
      // 刷新建议报告（若已有）
      loadLatestSuggestion(trackId);
      // 刷新迭代历史
      loadIterationHistory(trackId);
    } catch (e) {
      dom.mapWrap.innerHTML = '<div class="track-map-empty">赛道数据加载失败：' + esc(e.message) + "</div>";
      showToast("赛道数据加载失败：" + e.message, "error");
    }
  }

  /* ========================================================================
     2. SVG 赛道图 + 弯道热区
     ======================================================================== */

  async function renderTrackMap(trackId, corners) {
    dom.mapWrap.innerHTML = "";
    // 赛道图加载中显示 spinner
    showLoading(dom.mapWrap, "加载赛道图…");

    // 加载 SVG
    let svgText;
    try {
      const resp = await fetch(SVG_PREFIX + encodeURIComponent(trackId) + ".svg");
      if (!resp.ok) throw new Error(`SVG ${resp.status}`);
      svgText = await resp.text();
    } catch (e) {
      dom.mapWrap.innerHTML = '<div class="track-map-empty">赛道图加载失败</div>';
      showToast("赛道图加载失败", "error");
      return;
    }

    // 清除 loading，开始渲染
    dom.mapWrap.innerHTML = "";
    dom.mapWrap.classList.add("is-active");

    // 容器：SVG + 热区层
    const box = document.createElement("div");
    box.className = "track-svg-box";

    // 注入 SVG（移除原有 width/height 让其自适应）
    const tmp = document.createElement("div");
    tmp.innerHTML = svgText;
    const svg = tmp.querySelector("svg");
    if (!svg) { dom.mapWrap.innerHTML = '<div class="track-map-empty">SVG 格式异常</div>'; return; }
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = "100%";
    svg.style.height = "100%";
    box.appendChild(svg);

    // 热区层（百分比定位，覆盖在 SVG 上）
    const layer = document.createElement("div");
    layer.className = "hotzone-layer";

    corners.forEach((c) => {
      const ax = c.anchor_x != null ? c.anchor_x : 0.5;
      const ay = c.anchor_y != null ? c.anchor_y : 0.5;
      const num = c.corner_number;

      // 热区圆形（透明可点击）
      const hz = document.createElement("div");
      hz.className = "hotzone";
      hz.style.left = (ax * 100) + "%";
      hz.style.top = (ay * 100) + "%";
      hz.style.width = HOTZONE_DIAMETER + "px";
      hz.style.height = HOTZONE_DIAMETER + "px";
      hz.dataset.corner = num;
      hz.title = `弯道 ${num}${c.name ? " · " + c.name : ""}`;
      hz.setAttribute("role", "button");
      hz.setAttribute("aria-label", `弯道 ${num}${c.name ? " " + c.name : ""}，点击录入反馈`);
      hz.textContent = num;
      hz.addEventListener("click", () => {
        // 弯道点击 pulse 动画（1 秒后移除）
        hz.classList.add("pulse");
        setTimeout(() => hz.classList.remove("pulse"), 1000);
        openFeedbackPanel(num, c.name);
      });
      layer.appendChild(hz);
    });

    box.appendChild(layer);
    dom.mapWrap.appendChild(box);
  }

  /** 标记某弯道热区为「已反馈」。 */
  function markCornerFeedbackDone(cornerNumber) {
    state.feedbackCorners.add(cornerNumber);
    const hz = dom.mapWrap.querySelector(`.hotzone[data-corner="${cornerNumber}"]`);
    if (hz) hz.classList.add("feedback-done");
  }

  /** 高亮当前弯道（遥测驱动）。 */
  function highlightCurrentCorner(cornerNumber) {
    // 清除旧高亮
    const prev = dom.mapWrap.querySelector(".hotzone.current-corner");
    if (prev) prev.classList.remove("current-corner");
    state.currentCorner = cornerNumber;
    if (cornerNumber == null) return;
    const hz = dom.mapWrap.querySelector(`.hotzone[data-corner="${cornerNumber}"]`);
    if (hz) hz.classList.add("current-corner");
  }

  /* ========================================================================
     3. 反馈面板
     ======================================================================== */

  function openFeedbackPanel(cornerNumber, cornerName) {
    state.selectedCorner = { number: cornerNumber, name: cornerName || "" };
    dom.fbCornerNum.textContent = `T${cornerNumber}`;
    dom.fbCornerName.textContent = cornerName || "";
    // 重置选择
    const checked = dom.fbOverlay.querySelector('input[name="symptom"]:checked');
    if (checked) checked.checked = false;
    dom.fbStrength.value = "3";
    dom.fbStrengthVal.textContent = "3";
    dom.fbError.textContent = "";
    // 默认展开第一组（入弯）
    const groups = dom.fbOverlay.querySelectorAll(".sym-group");
    groups.forEach((g, i) => g.classList.toggle("expanded", i === 0));
    dom.fbOverlay.hidden = false;
  }

  function closeFeedbackPanel() {
    dom.fbOverlay.hidden = true;
    state.selectedCorner = null;
  }

  /** 提交反馈。
   *  关键：全局类症状 → corner_number = null；其余 → 选中弯道编号。
   *  （来源：2026-09-10-implicit-normal-feedback-semantics）
   */
  async function submitFeedback() {
    const symInput = dom.fbOverlay.querySelector('input[name="symptom"]:checked');
    if (!symInput) {
      dom.fbError.textContent = "请先选择一个症状";
      return;
    }
    const symptom = symInput.value;
    const category = SYMPTOM_CATEGORY[symptom];
    const strength = parseInt(dom.fbStrength.value, 10);
    const corner = state.selectedCorner;

    // 全局类症状不绑定具体弯道（corner_number = null）
    const cornerNumber = category === GLOBAL_CATEGORY ? null : corner.number;

    const payload = {
      track_id: state.currentTrackId,
      symptom: symptom,
      strength: strength,
    };
    if (cornerNumber != null) payload.corner_number = cornerNumber;

    dom.fbSubmit.disabled = true;
    dom.fbError.textContent = "";
    try {
      await fetchJSON("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      // 标记热区已反馈（仅具体弯道）
      if (cornerNumber != null) markCornerFeedbackDone(cornerNumber);
      closeFeedbackPanel();
      showToast("反馈已提交", "success");
    } catch (e) {
      dom.fbError.textContent = e.message;
      showToast("反馈提交失败：" + e.message, "error");
    } finally {
      dom.fbSubmit.disabled = false;
    }
  }

  /** 加载已有反馈，标记已反馈弯道。 */
  async function loadExistingFeedback(trackId) {
    try {
      const data = await fetchJSON(`/feedback?track_id=${encodeURIComponent(trackId)}`);
      const list = Array.isArray(data) ? data : (data && data.items) || [];
      list.forEach((f) => {
        // 仅具体弯道反馈标记热区；全局症状（corner_number=null）不标记
        if (f.corner_number != null) markCornerFeedbackDone(f.corner_number);
      });
    } catch (_) { /* 非致命 */ }
  }

  /* ========================================================================
     4. WebSocket（遥测 / 当前弯 / 建议 / 连接状态）
        自动重连：断开后 3 秒重连，最多 5 次；重连成功后计数归零。
     ======================================================================== */

  function connectWebSocket() {
    // 超过最大重连次数则停止
    if (state.wsReconnectCount >= WS_RECONNECT_MAX) {
      setWsStatus(false);
      showToast("遥测连接已断开，请刷新页面重试", "error");
      return;
    }
    try {
      state.ws = new WebSocket(WS_URL);
    } catch (e) {
      setWsStatus(false);
      scheduleReconnect();
      return;
    }
    state.ws.onopen = () => {
      // 重连成功，计数归零
      state.wsReconnectCount = 0;
    };
    state.ws.onclose = () => {
      setWsStatus(false);
      scheduleReconnect();
    };
    state.ws.onerror = () => { /* onclose 会处理 */ };
    state.ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch (_) { return; }
      if (!msg || !msg.event) return;
      switch (msg.event) {
        case "telemetry": onTelemetry(msg.data || msg.payload || msg); break;
        case "corner": onCornerEvent(msg.data || msg.payload || msg); break;
        case "suggestion": onSuggestionEvent(msg.data || msg.payload || msg); break;
        case "telemetry_status": setWsStatus((msg.data || msg.payload || msg).connected === true); break;
        default: break;
      }
    };
  }

  /** 安排重连：3 秒后重连，计数 +1。 */
  function scheduleReconnect() {
    state.wsReconnectCount += 1;
    if (state.wsReconnectCount <= WS_RECONNECT_MAX) {
      setTimeout(connectWebSocket, WS_RECONNECT_DELAY);
    }
  }

  function setWsStatus(connected) {
    if (connected) {
      dom.wsDot.className = "status-dot status-online";
      dom.wsText.textContent = "遥测已连接";
    } else {
      dom.wsDot.className = "status-dot status-offline";
      dom.wsText.textContent = "遥测未连接";
    }
  }

  /** 遥测事件 → 更新显示（数值变化添加 flash 过渡）。 */
  function onTelemetry(t) {
    if (t.speed != null) { dom.telSpeed.textContent = Math.round(t.speed); flashValue(dom.telSpeed); }
    if (t.throttle != null) { dom.telThrottle.textContent = Math.round(t.throttle * 100); flashValue(dom.telThrottle); }
    if (t.brake != null) { dom.telBrake.textContent = Math.round(t.brake * 100); flashValue(dom.telBrake); }
    if (t.gear != null) { dom.telGear.textContent = t.gear < 0 ? "R" : t.gear; flashValue(dom.telGear); }
    if (t.rpm != null) { dom.telRpm.textContent = Math.round(t.rpm); flashValue(dom.telRpm); }
    if (t.lap_time_ms != null) { dom.telLaptime.textContent = fmtLapTime(t.lap_time_ms); flashValue(dom.telLaptime); }
    else if (t.last_lap_time_ms != null) { dom.telLaptime.textContent = fmtLapTime(t.last_lap_time_ms); flashValue(dom.telLaptime); }
    if (t.sector != null) {
      dom.telSector.textContent = `S${t.sector}`;
      dom.sectorTag.textContent = `扇区 ${t.sector}`;
    }
  }

  /** 当前弯事件 → 高亮 + 遥测区显示。 */
  function onCornerEvent(c) {
    const num = c.corner_number;
    highlightCurrentCorner(num);
    dom.telCorner.textContent = num != null ? `T${num}` : "—";
    if (c.sector != null) {
      dom.telSector.textContent = `S${c.sector}`;
      dom.sectorTag.textContent = `扇区 ${c.sector}`;
    }
  }

  /** 建议结果事件 → 自动刷新报告。 */
  function onSuggestionEvent(s) {
    const report = s.report_json || s.report;
    if (report) renderReport(report);
    showToast("新建议已生成", "success");
  }

  /* ========================================================================
     5. 建议报告
     ======================================================================== */

  async function generateSuggestion() {
    if (!state.currentTrackId) {
      showToast("请先选择赛道", "error");
      return;
    }
    dom.btnSuggest.disabled = true;
    const oldText = dom.btnSuggest.textContent;
    // 按钮显示 loading spinner + 文字
    dom.btnSuggest.innerHTML = '<span class="spinner-inline"></span>生成中…';
    try {
      const data = await fetchJSON("/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: state.currentTrackId }),
      });
      const report = data && data.report_json ? data.report_json : data;
      if (report) renderReport(report);
      else loadLatestSuggestion(state.currentTrackId);
      showToast("建议已生成", "success");
    } catch (e) {
      // 400 无反馈 → 引导录入
      if (e.status === 400) {
        showToast("请先点击赛道图上的问题弯道并录入反馈", "error");
      } else {
        showToast("生成建议失败：" + e.message, "error");
      }
    } finally {
      dom.btnSuggest.disabled = false;
      dom.btnSuggest.textContent = oldText;
    }
  }

  /** 读取最新建议并渲染。 */
  async function loadLatestSuggestion(trackId) {
    try {
      const data = await fetchJSON(`/suggest/latest?track_id=${encodeURIComponent(trackId)}`);
      const report = data && data.report_json ? data.report_json : data;
      if (report && report.parameters) renderReport(report);
    } catch (_) { /* 无建议，保持空态 */ }
  }

  /** 渲染建议报告（23 参数表格）。
   *  报告结构（design.md 2.7.7）：
   *  { track_id, generated_at, parameters: [{param, current, setup_delta,
   *    linkages, linked_notes, source, confidence, tradeoff}], summary }
   */
  function renderReport(report) {
    const params = report.parameters || [];
    if (!params.length) {
      dom.reportSummary.classList.remove("visible");
      dom.reportWrap.innerHTML = '<div class="report-empty">建议报告为空</div>';
      return;
    }

    // 摘要
    if (report.summary) {
      dom.reportSummary.innerHTML = `<div>${esc(report.summary)}</div>` +
        (report.generated_at ? `<div class="meta">生成时间：${esc(report.generated_at)}</div>` : "");
      dom.reportSummary.classList.add("visible");
    } else {
      dom.reportSummary.classList.remove("visible");
    }

    // 表格（参数调整可视化条形图）
    const rows = params.map((p) => {
      const delta = p.setup_delta;
      const deltaCls = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "delta-zero";
      const deltaSign = delta > 0 ? "+" : "";
      const conf = p.confidence || "medium";
      const confCls = conf === "high" ? "conf-high" : conf === "low" ? "conf-low" : "conf-medium";
      const suggestVal = p.suggested != null ? p.suggested : (p.current != null && delta != null ? p.current + delta : "—");
      const linkages = Array.isArray(p.linkages) ? p.linkages.join("、") : (p.linkages || "");
      const tradeoff = p.tradeoff ? `<span class="tradeoff-text">${esc(p.tradeoff)}</span>` : "—";
      // 可视化条形图：以 delta 比例填充
      const barFill = renderDeltaBar(delta);

      return `<tr>
        <td class="col-param">${esc(p.param)}</td>
        <td class="col-current">${esc(p.current)}</td>
        <td class="col-delta ${deltaCls}">${deltaSign}${esc(delta)}${barFill}</td>
        <td class="col-suggest">${esc(suggestVal)}</td>
        <td class="col-link">${esc(linkages)}</td>
        <td class="col-note">${esc(p.linked_notes || "")}</td>
        <td class="col-source">${esc(p.source || "")}</td>
        <td><span class="confidence-tag ${confCls}">${esc(conf)}</span></td>
        <td>${tradeoff}</td>
      </tr>`;
    }).join("");

    dom.reportWrap.innerHTML = `
      <table class="report-table">
        <thead>
          <tr>
            <th>参数</th>
            <th>当前值</th>
            <th>调整量</th>
            <th>建议值</th>
            <th>联动维度</th>
            <th>联动说明</th>
            <th>出处</th>
            <th>置信度</th>
            <th>权衡</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  /** 渲染参数调整可视化条形图（delta 比例填充）。 */
  function renderDeltaBar(delta) {
    if (delta == null || delta === 0) return "";
    // 将 delta 映射到 -100% ~ 100%，以 50% 为中点
    const ratio = Math.min(Math.abs(delta) / 10, 1) * 50; // 假设 delta 量级 ≤10
    const dir = delta > 0 ? "up" : "down";
    const width = ratio.toFixed(1);
    const left = delta > 0 ? "50%" : `${(50 - ratio).toFixed(1)}%`;
    return `<span class="delta-bar"><span class="delta-bar-fill ${dir}" style="left:${left};width:${width}%;"></span></span>`;
  }

  /* ========================================================================
     6. 迭代历史
     ======================================================================== */

  async function loadIterationHistory(trackId) {
    if (!trackId) return;
    try {
      const data = await fetchJSON(`/iteration/history?track_id=${encodeURIComponent(trackId)}`);
      const list = Array.isArray(data) ? data : (data && data.items) || [];
      renderHistory(list);
    } catch (_) {
      dom.historyWrap.innerHTML = '<div class="history-empty">迭代历史加载失败</div>';
    }
  }

  function renderHistory(list) {
    if (!list.length) {
      dom.historyWrap.innerHTML = '<div class="history-empty">暂无迭代记录</div>';
      return;
    }
    dom.historyWrap.innerHTML = list.map((it) => {
      const round = it.round_no != null ? `第 ${it.round_no} 轮` : "迭代";
      const time = it.created_at || "";
      // 前后对比：before_params / after_params 或 before_setup / after_setup
      const before = it.before_params || (it.before_setup && it.before_setup.params) || {};
      const after = it.after_params || (it.after_setup && it.after_setup.params) || {};
      const diffs = Object.keys(after).filter((k) => before[k] != null && after[k] !== before[k]);
      const diffRows = diffs.length ? diffs.map((k) => {
        const d = after[k] - before[k];
        const sign = d > 0 ? "+" : "";
        return `<div class="diff-row"><span>${esc(k)}</span><span class="dv">${sign}${esc(d)}</span></div>`;
      }).join("") : '<div class="diff-row"><span>无参数变化</span></div>';

      return `<div class="history-item">
        <div class="history-item-head">
          <span class="history-round">${esc(round)}</span>
          <span class="history-time">${esc(time)}</span>
        </div>
        <div class="history-diff">${diffRows}</div>
      </div>`;
    }).join("");
  }

  /* ========================================================================
     7. 事件绑定 & 初始化
     ======================================================================== */

  function bindEvents() {
    // 赛道选择
    dom.trackSelect.addEventListener("change", onTrackChange);

    // 生成建议
    dom.btnSuggest.addEventListener("click", generateSuggestion);

    // 刷新历史
    dom.btnHistory.addEventListener("click", () => loadIterationHistory(state.currentTrackId));

    // 反馈面板
    dom.fbClose.addEventListener("click", closeFeedbackPanel);
    dom.fbSubmit.addEventListener("click", submitFeedback);
    dom.fbOverlay.addEventListener("click", (e) => {
      if (e.target === dom.fbOverlay) closeFeedbackPanel();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !dom.fbOverlay.hidden) closeFeedbackPanel();
    });

    // 强度滑块实时显示
    dom.fbStrength.addEventListener("input", () => {
      dom.fbStrengthVal.textContent = dom.fbStrength.value;
    });

    // 症状分组展开/折叠
    document.querySelectorAll(".sym-group-head").forEach((head) => {
      head.addEventListener("click", () => {
        head.parentElement.classList.toggle("expanded");
      });
    });
  }

  /** 入口。 */
  function init() {
    bindEvents();
    loadTracks();
    connectWebSocket();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
