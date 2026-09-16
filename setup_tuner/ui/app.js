/* ==========================================================================
   F1OPT 赛车调教优化助手 —— 前端交互逻辑（纯原生 JS，无框架）
   -------------------------------------------------------------------------
   职责：
      1. 赛道选择 → 加载 SVG + 弯道热区（SVG 缓存）
      2. 热区点击 → 反馈面板（17 症状 4 类 checkbox 多选 + 每症状独立强度 1-3）
       3. 21 项调教参数手动输入面板（6 大类分组，滑块+数值输入，保存/重置/导入）
       4. WebSocket 订阅遥测/当前弯高亮/建议结果/连接状态（指数退避重连）
       5. 建议报告渲染（21 参数表格 + 可视化条形图 + 中文参数名映射）
      6. 迭代历史前后对比
    关键约定：
      - 后端响应统一信封 {code, message, data}，取 data 字段
      - 后端 SuggestionView 返回字段名为 report（非 report_json）——已修复 3 处
      - feedback.corner_number：NULL=全局症状；非 NULL=具体弯道反馈
      - SVG viewBox 800×600，锚点 anchor_x/y 归一化 0~1
    性能优化：
      - 遥测数据更新防抖（debounce 100ms）
      - 报告表格渲染使用 DocumentFragment 批量插入
      - 赛道图 SVG 加载后缓存，切换赛道先显示缓存
      - WebSocket 重连指数退避（1s, 2s, 4s, 8s, 16s，上限 30s）
    UI 增强（消除机械感）：
      - Tab 切换滑动指示器（requestAnimationFrame 驱动位置动画）
      - Range input 渐变轨道填充（CSS 变量 --range-fill 实时更新）
      - 卡片 shimmer 扫光效果（CSS hover 触发）
      - 微交互过渡（hover/active/focus 精细化反馈）
   ========================================================================== */
(function () {
  "use strict";

  /* ---------- 常量 ---------- */
  const API_BASE = "/api/v1";
  const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${API_BASE}/ws`;
  const SVG_PREFIX = "/static/tracks/";
  const HOTZONE_DIAMETER = 28;
  // WebSocket 指数退避重连：1s, 2s, 4s, 8s, 16s（上限 30s），最多 8 次
  const WS_RECONNECT_BASE = 1000;
  const WS_RECONNECT_MAX = 8;
  const WS_RECONNECT_CAP = 30000;
  // 遥测防抖间隔（ms）
  const TELEMETRY_DEBOUNCE = 100;
  // Toast 显示时长（ms）与离场动画兜底时长（ms）
  const TOAST_DURATION_MS = 3200;
  const TOAST_LEAVE_FALLBACK_MS = 350;
  // 数值 flash 高亮持续时间（ms）
  const FLASH_DURATION_MS = 200;
  // 热区 pulse 动画持续时间（ms）
  const PULSE_DURATION_MS = 1000;
  // 默认症状强度（1-3 档位初始值：2=明显）
  const DEFAULT_STRENGTH = 2;
  // 速度仪表盘最大值（km/h，用于进度条计算）
  const SPEED_MAX = 350;
  // 调整量条形图缩放因子与半宽百分比
  const DELTA_BAR_SCALE = 10;
  const DELTA_BAR_HALF_PCT = 50;
  // 浮点比较 epsilon（判断参数是否相对默认值已修改）
  const FLOAT_EPSILON = 1e-9;
  // 百分比最大值（油门/刹车等百分比类遥测数据的上限）
  const PERCENT_MAX = 100;
  // 默认锚点位置（弯道锚点缺失时的归一化中心位置）
  const DEFAULT_ANCHOR = 0.5;
  // 症状强度滑块范围
  // task-61：强度收敛为 1-3（轻微/明显/严重）
  const STRENGTH_MIN = 1;
  const STRENGTH_MAX = 3;

  // 症状 → 类别映射（与 domain/symptoms.py 逐字对齐）
  const SYMPTOM_CATEGORY = {
    understeer: "entry", oversteer: "entry", turnin_unresponsive: "entry",
    brake_long: "entry", lockup: "entry",
    midcorner_understeer: "apex", midcorner_unstable: "apex", midcorner_traction: "apex",
    exit_wheelspin: "exit", exit_oversteer: "exit",
    exit_understeer: "exit", exit_unstable: "exit",
    bottoming: "global", tyre_wear: "global", straight_slow: "global",
    // lap_slow 已降级为综合结论，不再出现在反馈面板；映射保留兼容旧数据
    lap_slow: "global", high_speed_instability: "global", tyre_overheat: "global",
  };
  const GLOBAL_CATEGORY = "global";

  // 症状中文显示名映射
  const SYMPTOM_LABEL_ZH = {
    understeer: "转向不足", oversteer: "转向过度", turnin_unresponsive: "转向不灵敏",
    brake_long: "刹车距离长", lockup: "轮胎锁死",
    midcorner_understeer: "弯中推头", midcorner_unstable: "车身不稳定", midcorner_traction: "弯中不能稳定加速",
    exit_wheelspin: "出弯打滑", exit_oversteer: "出弯甩尾",
    exit_understeer: "出弯转向不足", exit_unstable: "出弯车身不稳",
    bottoming: "直道刮底", tyre_wear: "胎耗偏高", straight_slow: "直道速度低",
    // lap_slow 已从 UI 可勾列表降级为综合结论；标签保留用于展示旧数据
    lap_slow: "圈速不高", high_speed_instability: "高速不稳", tyre_overheat: "胎温过高",
  };

  // 21 参数中文显示名映射（前端 fallback，优先使用 GET /api/v1/setup/fields 返回的 label_zh）
  const PARAM_LABEL_ZH = {
    front_wing: "前翼", rear_wing: "后翼",
    on_throttle_diff: "差速器（开油门）", off_throttle_diff: "差速器（松油门）",
    front_camber: "前外倾角", rear_camber: "后外倾角", front_toe: "前束角", rear_toe: "后束角",
    front_suspension: "前悬挂", rear_suspension: "后悬挂",
    front_anti_roll_bar: "前防倾杆", rear_anti_roll_bar: "后防倾杆",
    front_ride_height: "前行驶高度", rear_ride_height: "后行驶高度",
    brake_pressure: "刹车压力", brake_bias: "刹车偏置",
    engine_braking: "引擎制动",
    front_left_tyre_pressure: "前左胎压", front_right_tyre_pressure: "前右胎压",
    rear_left_tyre_pressure: "后左胎压", rear_right_tyre_pressure: "后右胎压",
  };

  // 6 大类中文显示名映射
  const GROUP_LABEL_ZH = {
    "Aerodynamics": "空气动力学", "Transmission": "变速箱",
    "Suspension Geometry": "悬挂几何", "Suspension": "悬挂",
    "Brakes": "刹车", "Tyres": "轮胎",
  };

  // 21 参数前端 fallback 定义（与后端 setup.py ALL_SETUP_FIELDS 对齐；API 不可用时使用）
  const FALLBACK_SETUP_FIELDS = [
    { name: "front_wing", group: "Aerodynamics", label: "前翼", min: 0, max: 50, step: 1, default: 25, unit: "级" },
    { name: "rear_wing", group: "Aerodynamics", label: "后翼", min: 0, max: 50, step: 1, default: 25, unit: "级" },
    { name: "on_throttle_diff", group: "Transmission", label: "差速器（开油门）", min: 10, max: 100, step: 1, default: 50, unit: "%" },
    { name: "off_throttle_diff", group: "Transmission", label: "差速器（松油门）", min: 10, max: 100, step: 1, default: 50, unit: "%" },
    { name: "front_camber", group: "Suspension Geometry", label: "前外倾角", min: -3.5, max: -2.5, step: 0.1, default: -3.5, unit: "°" },
    { name: "rear_camber", group: "Suspension Geometry", label: "后外倾角", min: -2.0, max: -1.0, step: 0.1, default: -1.5, unit: "°" },
    { name: "front_toe", group: "Suspension Geometry", label: "前束角", min: 0.0, max: 0.2, step: 0.01, default: 0.0, unit: "°" },
    { name: "rear_toe", group: "Suspension Geometry", label: "后束角", min: 0.1, max: 0.35, step: 0.01, default: 0.2, unit: "°" },
    { name: "front_suspension", group: "Suspension", label: "前悬挂", min: 1, max: 41, step: 1, default: 6, unit: "级" },
    { name: "rear_suspension", group: "Suspension", label: "后悬挂", min: 1, max: 41, step: 1, default: 6, unit: "级" },
    { name: "front_anti_roll_bar", group: "Suspension", label: "前防倾杆", min: 1, max: 21, step: 1, default: 6, unit: "级" },
    { name: "rear_anti_roll_bar", group: "Suspension", label: "后防倾杆", min: 1, max: 21, step: 1, default: 6, unit: "级" },
    { name: "front_ride_height", group: "Suspension", label: "前行驶高度", min: 15, max: 35, step: 1, default: 25, unit: "级" },
    { name: "rear_ride_height", group: "Suspension", label: "后行驶高度", min: 40, max: 60, step: 1, default: 50, unit: "级" },
    { name: "brake_pressure", group: "Brakes", label: "刹车压力", min: 80, max: 100, step: 1, default: 90, unit: "%" },
    { name: "brake_bias", group: "Brakes", label: "刹车偏置", min: 50, max: 70, step: 1, default: 58, unit: "%" },
    { name: "engine_braking", group: "Brakes", label: "引擎制动", min: 0, max: 100, step: 1, default: 50, unit: "%" },
    { name: "front_left_tyre_pressure", group: "Tyres", label: "前左胎压", min: 22.5, max: 29.5, step: 0.1, default: 23.5, unit: "psi" },
    { name: "front_right_tyre_pressure", group: "Tyres", label: "前右胎压", min: 22.5, max: 29.5, step: 0.1, default: 23.5, unit: "psi" },
    { name: "rear_left_tyre_pressure", group: "Tyres", label: "后左胎压", min: 20.5, max: 26.5, step: 0.1, default: 22.0, unit: "psi" },
    { name: "rear_right_tyre_pressure", group: "Tyres", label: "后右胎压", min: 20.5, max: 26.5, step: 0.1, default: 22.0, unit: "psi" },
  ];

  /* ---------- DOM 引用 ---------- */
  const $ = (id) => document.getElementById(id);
  const dom = {
    trackSelect: $("track-select"),
    trackMeta: $("track-meta"),
    mapWrap: $("track-map-wrap"),
    mapEmpty: $("track-map-empty"),
    wsDot: $("ws-dot"),
    wsText: $("ws-status-text"),
    topbarMode: $("topbar-mode"),
    telSpeed: $("tel-speed"), telThrottle: $("tel-throttle"), telBrake: $("tel-brake"),
    telGear: $("tel-gear"), telRpm: $("tel-rpm"), telLaptime: $("tel-laptime"),
    telSector: $("tel-sector"), telCorner: $("tel-corner"), sectorTag: $("telemetry-sector-tag"),
    telSpeedBar: $("tel-speed-bar"), telThrottleBar: $("tel-throttle-bar"), telBrakeBar: $("tel-brake-bar"),
    btnSuggest: $("btn-generate-suggest"),
    modelTypeSelect: $("model-type-select"),
    reportSummary: $("report-summary"),
    reportWrap: $("report-table-wrap"),
    btnHistory: $("btn-refresh-history"),
    historyWrap: $("history-wrap"),
    toastContainer: $("toast-container"),
    // 21 参数面板
    setupFieldsWrap: $("setup-fields-wrap"),
    btnSaveSetup: $("btn-save-setup"),
    btnResetDefault: $("btn-reset-default"),
    btnImportCurrent: $("btn-import-current"),
    btnImportTelemetry: $("btn-import-telemetry"),
    // 反馈摘要
    feedbackSummaryWrap: $("feedback-summary-wrap"),
    btnClearFeedback: $("btn-clear-feedback"),
    // 反馈面板
    fbOverlay: $("feedback-overlay"), fbClose: $("fb-close"), fbSubmit: $("fb-submit"),
    fbTitle: $("fb-title"), fbTip: $("fb-tip"),
    fbCornerNum: $("fb-corner-number"), fbCornerName: $("fb-corner-name"),
    fbStrengthList: $("fb-strength-list"),
    fbSelectedCount: $("fb-selected-count"),
    fbError: $("fb-error"),
    btnTrackFeedback: $("track-feedback-btn"),
    // 遥测录制
    btnRecord: $("btn-record-toggle"),
    btnListenerToggle: $("btn-listener-toggle"),
  };

  /* ---------- 运行时状态 ---------- */
  const state = {
    tracks: [],
    currentTrackId: null,
    corners: [],
    feedbackCorners: new Set(),
    selectedCorner: null,
    feedbackMode: null, // "corner" | "track"
    currentCorner: null,
    ws: null,
    wsReconnectCount: 0,
    // 21 参数定义（从 API 加载或 fallback）
    setupFields: [],
    // 当前参数值（param_name → value）
    setupValues: {},
    // 参数是否已修改（相对默认值）
    setupModified: {},
    // SVG 缓存（track_id → svgText）
    svgCache: {},
    // 遥测防抖定时器
    telemetryDebounceTimer: null,
    // 遥测最新数据（防抖用）
    latestTelemetry: null,
    // 已提交反馈列表（用于摘要显示）
    feedbacks: [],
    // 遥测录制状态
    isRecording: false,
    isListening: true,  // 默认监听器已启动
  };

  /* ---------- 工具函数 ---------- */

  /** 封装 fetch + JSON 解析，处理 {code, message, data} 信封。
   *  @param {string} path — API 路径（相对 API_BASE）
   *  @param {RequestInit} [options] — fetch options
   *  @returns {Promise<any>} — body.data
   *  @throws {Error & {status:number, body:any}}
   */
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

  /** Toast 通知。
   *  @param {string} msg — 通知文本
   *  @param {("success"|"error"|"")} [type] — 通知类型
   */
  function showToast(msg, type) {
    if (!dom.toastContainer) return;
    const el = document.createElement("div");
    el.className = "toast" + (type ? ` toast-${type}` : "");
    el.textContent = msg;
    dom.toastContainer.appendChild(el);
    setTimeout(() => {
      el.classList.add("toast-leaving");
      el.addEventListener("animationend", () => el.remove(), { once: true });
      setTimeout(() => { if (el.parentNode) el.remove(); }, TOAST_LEAVE_FALLBACK_MS);
    }, TOAST_DURATION_MS);
  }

  /** Loading spinner。 */
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

  /** 格式化圈速（ms → m:ss.mmm）。
   *  @param {number|null} ms — 圈速毫秒数
   *  @returns {string}
   */
  function fmtLapTime(ms) {
    if (ms == null || ms <= 0) return "—";
    const totalSec = ms / 1000;
    const m = Math.floor(totalSec / 60);
    const s = (totalSec - m * 60).toFixed(3);
    return `${m}:${s.padStart(6, "0")}`;
  }

  /** 转义 HTML（防止 XSS）。
   *  @param {string|null|undefined} s — 原始文本
   *  @returns {string}
   */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /** flash 高亮过渡（使用 requestAnimationFrame 减少重绘，仅在值变化时触发）。 */
  function flashValue(el) {
    if (!el) return;
    // 使用 rAF 确保在下一帧统一应用，避免布局抖动
    requestAnimationFrame(() => {
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), FLASH_DURATION_MS);
    });
  }

  /** 仅当文本内容变化时才更新 DOM + flash（减少无谓重绘）。
   *  @param {HTMLElement} el — 目标元素
   *  @param {string} newVal — 新文本值
   */
  function updateTelValue(el, newVal) {
    if (!el || el.textContent === newVal) return;
    el.textContent = newVal;
    flashValue(el);
  }

  /** 防抖函数。 */
  function debounce(fn, delay) {
    let timer = null;
    return function (...args) {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { timer = null; fn.apply(this, args); }, delay);
    };
  }

  /** 更新仪表盘进度条（仅在宽度变化时写入 style，减少重排）。
   *  @param {HTMLElement} barEl — 进度条填充元素
   *  @param {number} value — 当前值
   *  @param {number} max — 最大值
   */
  function setGaugeBar(barEl, value, max) {
    if (!barEl) return;
    const pct = Math.max(0, Math.min(100, (value / max) * 100));
    const newWidth = pct.toFixed(1) + "%";
    if (barEl.style.width !== newWidth) {
      barEl.style.width = newWidth;
    }
  }

  /* ========================================================================
     1. 赛道选择
     ======================================================================== */

  /** 加载赛道列表并填充下拉选择器。
   *  @returns {Promise<void>}
   */
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

  /** 赛道选择变更事件处理。
   *  @returns {void}
   */
  function onTrackChange() {
    const trackId = dom.trackSelect.value;
    if (!trackId) { return; }
    selectTrack(trackId);
  }

  /** 重置赛道相关运行时状态。
   *  @param {string} trackId — 赛道 ID
   *  @returns {void}
   */
  function resetTrackState(trackId) {
    state.currentTrackId = trackId;
    state.feedbackCorners.clear();
    state.currentCorner = null;
    state.feedbacks = [];
  }

  /** 显示缓存的赛道图或 loading 占位。
   *  @param {string} trackId — 赛道 ID
   *  @returns {void}
   */
  function showCachedTrackOrLoading(trackId) {
    if (state.svgCache[trackId] && state.corners.length) {
      renderTrackMapFromCache(trackId);
    } else {
      showLoading(dom.mapWrap, "加载赛道数据…");
    }
  }

  /** 通知后端当前选中的赛道（非致命）。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
  async function notifyBackendCurrentTrack(trackId) {
    try {
      await fetchJSON("/tracks/current", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: trackId }),
      });
    } catch (_) { /* 非致命 */ }
  }

  /** 加载赛道详情并渲染所有关联面板（赛道图、反馈、建议、历史）。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
  async function loadTrackDetailsAndRender(trackId) {
    try {
      const data = await fetchJSON(`/tracks/${encodeURIComponent(trackId)}`);
      const track = data.track || data;
      const corners = data.corners || track.corners || [];
      state.corners = corners;
      dom.trackMeta.textContent = `${track.circuit_name || ""} · ${track.country || ""} · ${(track.length_m / 1000).toFixed(3)} km · ${corners.length} 弯`;
      renderTrackMap(trackId, corners);
      loadExistingFeedback(trackId);
      loadLatestSuggestion(trackId);
      loadIterationHistory(trackId);
      renderFeedbackSummary();
    } catch (e) {
      dom.mapWrap.innerHTML = '<div class="track-map-empty">赛道数据加载失败：' + esc(e.message) + "</div>";
      showToast("赛道数据加载失败：" + e.message, "error");
    }
  }

  /** 选择并加载指定赛道的详情、弯道、反馈、建议和历史。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
  async function selectTrack(trackId) {
    resetTrackState(trackId);
    showCachedTrackOrLoading(trackId);
    await notifyBackendCurrentTrack(trackId);
    await loadTrackDetailsAndRender(trackId);
  }

  /* ========================================================================
     2. SVG 赛道图 + 弯道热区（带缓存）
     ======================================================================== */

  /** 从 SVG 文本 + 弯道列表构建赛道图 box 节点（SVG + 热区层）。
   *  @param {string} svgText — SVG 文本
   *  @param {Array} corners — 弯道锚点列表
   *  @returns {HTMLDivElement|null} — box 节点；SVG 格式异常时返回 null
   */
  function buildTrackSvgBox(svgText, corners) {
    const box = document.createElement("div");
    box.className = "track-svg-box";
    const tmp = document.createElement("div");
    tmp.innerHTML = svgText;
    const svg = tmp.querySelector("svg");
    if (!svg) return null;
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = "100%";
    svg.style.height = "100%";
    box.appendChild(svg);
    const layer = document.createElement("div");
    layer.className = "hotzone-layer";
    corners.forEach((c) => appendHotzone(layer, c));
    box.appendChild(layer);
    return box;
  }

  /** 从缓存渲染赛道图（切换赛道时先显示，避免闪烁）。
   *  @param {string} trackId — 赛道 ID
   *  @returns {void}
   */
  function renderTrackMapFromCache(trackId) {
    const svgText = state.svgCache[trackId];
    if (!svgText) return;
    dom.mapWrap.innerHTML = "";
    dom.mapWrap.classList.add("is-active");
    const box = buildTrackSvgBox(svgText, state.corners);
    if (!box) return;
    dom.mapWrap.appendChild(box);
    // 恢复已反馈标记
    state.feedbackCorners.forEach((num) => markCornerFeedbackDone(num));
  }

  /** 给热区层追加一个弯道热区圆点。 */
  /** 给热区层追加一个弯道热区圆点。
   *  @param {HTMLElement} layer — 热区层容器
   *  @param {Object} c — 弯道锚点数据
   *  @returns {void}
   */
  function appendHotzone(layer, c) {
    const ax = c.anchor_x != null ? c.anchor_x : DEFAULT_ANCHOR;
    const ay = c.anchor_y != null ? c.anchor_y : DEFAULT_ANCHOR;
    const num = c.corner_number;
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
      hz.classList.add("pulse");
      setTimeout(() => hz.classList.remove("pulse"), PULSE_DURATION_MS);
      openFeedbackPanel(num, c.name, "corner");
    });
    layer.appendChild(hz);
  }

  /** 异步加载并渲染赛道图（带 SVG 缓存）。
   *  @param {string} trackId — 赛道 ID
   *  @param {Array} corners — 弯道锚点列表
   *  @returns {Promise<void>}
   */
  async function renderTrackMap(trackId, corners) {
    // 若已有缓存且已渲染，跳过
    if (state.svgCache[trackId] && dom.mapWrap.querySelector(".track-svg-box")) {
      return;
    }
    if (!state.svgCache[trackId]) {
      showLoading(dom.mapWrap, "加载赛道图…");
    }

    let svgText = state.svgCache[trackId];
    if (!svgText) {
      try {
        const resp = await fetch(SVG_PREFIX + encodeURIComponent(trackId) + ".svg");
        if (!resp.ok) throw new Error(`SVG ${resp.status}`);
        svgText = await resp.text();
        state.svgCache[trackId] = svgText; // 缓存
      } catch (e) {
        dom.mapWrap.innerHTML = '<div class="track-map-empty">赛道图加载失败</div>';
        showToast("赛道图加载失败", "error");
        return;
      }
    }

    dom.mapWrap.innerHTML = "";
    dom.mapWrap.classList.add("is-active");
    const box = buildTrackSvgBox(svgText, corners);
    if (!box) { dom.mapWrap.innerHTML = '<div class="track-map-empty">SVG 格式异常</div>'; return; }
    dom.mapWrap.appendChild(box);
  }

  /** 标记某弯道已反馈（添加视觉标记）。
   *  @param {number} cornerNumber — 弯道编号
   *  @returns {void}
   */
  function markCornerFeedbackDone(cornerNumber) {
    state.feedbackCorners.add(cornerNumber);
    const hz = dom.mapWrap.querySelector(`.hotzone[data-corner="${cornerNumber}"]`);
    if (hz) hz.classList.add("feedback-done");
  }

  /** 高亮当前弯道（移除旧高亮，设置新高亮）。
   *  @param {number|null} cornerNumber — 弯道编号，null 表示清除高亮
   *  @returns {void}
   */
  function highlightCurrentCorner(cornerNumber) {
    const prev = dom.mapWrap.querySelector(".hotzone.current-corner");
    if (prev) prev.classList.remove("current-corner");
    state.currentCorner = cornerNumber;
    if (cornerNumber == null) return;
    const hz = dom.mapWrap.querySelector(`.hotzone[data-corner="${cornerNumber}"]`);
    if (hz) hz.classList.add("current-corner");
  }

  /* ========================================================================
     3. 反馈面板（checkbox 多选 + 每症状独立强度）
     ======================================================================== */

  /** 配置弯道模式反馈面板 UI（隐藏全局组，显示弯道相关组）。
   *  @param {number} cornerNumber — 弯道编号
   *  @param {string|null} cornerName — 弯道名称
   *  @returns {void}
   */
  function configureCornerMode(cornerNumber, cornerName) {
    state.selectedCorner = { number: cornerNumber, name: cornerName || "" };
    dom.fbCornerNum.textContent = `T${cornerNumber}`;
    dom.fbCornerName.textContent = cornerName || "";
    if (dom.fbTitle) dom.fbTitle.textContent = "弯道反馈 · 多症状录入";
    if (dom.fbTip) dom.fbTip.textContent = "可同时勾选多个症状（checkbox 多选）。入弯/弯中/出弯类症状绑定当前弯道。每个选中症状都有独立的强度档位（1-3）。";
    dom.fbOverlay.querySelectorAll('.sym-group').forEach((g) => {
      g.style.display = g.dataset.category === 'global' ? 'none' : '';
    });
  }

  /** 配置赛道模式反馈面板 UI（仅显示全局组）。
   *  @returns {void}
   */
  function configureTrackMode() {
    state.selectedCorner = { number: null, name: "赛道级" };
    dom.fbCornerNum.textContent = "—";
    dom.fbCornerName.textContent = "赛道级";
    if (dom.fbTitle) dom.fbTitle.textContent = "赛道反馈 · 全局症状";
    if (dom.fbTip) dom.fbTip.textContent = "可同时勾选多个全局症状（checkbox 多选）。全局类症状作用于整条赛道（corner_number=null）。每个选中症状都有独立的强度档位（1-3）。";
    dom.fbOverlay.querySelectorAll('.sym-group').forEach((g) => {
      g.style.display = g.dataset.category === 'global' ? '' : 'none';
    });
  }

  /** 打开反馈面板（弯道模式或赛道模式）。
   *  @param {number|null} cornerNumber — 弯道编号（赛道模式为 null）
   *  @param {string|null} cornerName — 弯道名称
   *  @param {("corner"|"track")} mode — 反馈模式
   *  @returns {void}
   */
  function openFeedbackPanel(cornerNumber, cornerName, mode) {
    const fbMode = mode === "track" ? "track" : "corner";
    state.feedbackMode = fbMode;

    // Guard clause：按模式分别配置，避免嵌套 if/else
    if (fbMode === "track") {
      configureTrackMode();
    } else {
      configureCornerMode(cornerNumber, cornerName);
    }

    // 重置所有 checkbox
    dom.fbOverlay.querySelectorAll('input[name="symptom"]').forEach((cb) => { cb.checked = false; });
    dom.fbError.textContent = "";
    updateFbSelectedCount();
    renderStrengthSliders();
    // 默认展开第一组（可见组中的第一个）
    const visibleGroups = Array.from(dom.fbOverlay.querySelectorAll(".sym-group")).filter((g) => g.style.display !== "none");
    visibleGroups.forEach((g, i) => g.classList.toggle("expanded", i === 0));
    dom.fbOverlay.hidden = false;
  }

  /** 关闭反馈面板并重置状态。
   *  @returns {void}
   */
  function closeFeedbackPanel() {
    dom.fbOverlay.hidden = true;
    state.selectedCorner = null;
    state.feedbackMode = null;
    // 恢复所有症状组显示，避免下次打开时残留 display 状态
    dom.fbOverlay.querySelectorAll('.sym-group').forEach((g) => { g.style.display = ''; });
  }

  /** 更新已选症状计数。
   *  @returns {void}
   */
  function updateFbSelectedCount() {
    const checked = dom.fbOverlay.querySelectorAll('input[name="symptom"]:checked');
    const count = checked.length;
    dom.fbSelectedCount.textContent = `已选 ${count} 个症状`;
    // 更新每组计数
    dom.fbOverlay.querySelectorAll(".sym-group").forEach((g) => {
      const cat = g.dataset.category;
      const cnt = g.querySelectorAll('input[name="symptom"]:checked').length;
      const badge = g.querySelector(`[data-count-for="${cat}"]`);
      if (badge) badge.textContent = cnt;
    });
  }

  /** 渲染每个选中症状的独立强度滑块。
   *  @returns {void}
   */
  function renderStrengthSliders() {
    const checked = dom.fbOverlay.querySelectorAll('input[name="symptom"]:checked');
    if (!checked.length) {
      dom.fbStrengthList.innerHTML = '<div class="fb-strength-empty">勾选症状后，此处显示每个症状的强度滑块</div>';
      return;
    }
    // 使用 DocumentFragment 批量插入
    const frag = document.createDocumentFragment();
    checked.forEach((cb) => {
      const symptom = cb.value;
      const label = SYMPTOM_LABEL_ZH[symptom] || symptom;
      const item = document.createElement("div");
      item.className = "fb-strength-item";
      item.dataset.symptom = symptom;
      item.innerHTML = `
        <span class="fb-strength-item-name">${esc(label)}</span>
        <input type="range" class="fb-strength-item-range" min="${STRENGTH_MIN}" max="${STRENGTH_MAX}" step="1" value="${DEFAULT_STRENGTH}" aria-label="${esc(label)} 强度" />
        <span class="fb-strength-item-val">${DEFAULT_STRENGTH}</span>`;
      const range = item.querySelector(".fb-strength-item-range");
      const val = item.querySelector(".fb-strength-item-val");
      range.addEventListener("input", () => { val.textContent = range.value; });
      frag.appendChild(item);
    });
    dom.fbStrengthList.innerHTML = "";
    dom.fbStrengthList.appendChild(frag);
  }

  /** 收集面板中勾选的症状与强度，组装为 feedback 数组。
   *  全局类症状 → corner_number = null；其余 → 选中弯道编号。
   *  @returns {Array<{symptom:string, strength:number, corner_number?:number}>}
   */
  function collectFeedbacks() {
    const checked = dom.fbOverlay.querySelectorAll('input[name="symptom"]:checked');
    const corner = state.selectedCorner;
    const feedbacks = [];
    checked.forEach((cb) => {
      const symptom = cb.value;
      const category = SYMPTOM_CATEGORY[symptom];
      const strengthItem = dom.fbStrengthList.querySelector(`.fb-strength-item[data-symptom="${symptom}"]`);
      const strength = strengthItem ? parseInt(strengthItem.querySelector(".fb-strength-item-range").value, 10) : DEFAULT_STRENGTH;
      const cornerNumber = category === GLOBAL_CATEGORY ? null : corner.number;
      const fb = { symptom, strength };
      if (cornerNumber != null) fb.corner_number = cornerNumber;
      feedbacks.push(fb);
    });
    return feedbacks;
  }

  /** 反馈提交成功后更新 UI：标记热区、追加本地摘要、关闭面板、toast。
   *  @param {Object} corner — 当前选中弯道信息
   *  @param {Array} feedbacks — 反馈列表
   *  @param {boolean} isFallback — 是否为降级逐条模式
   *  @returns {void}
   */
  function applyFeedbackSuccess(corner, feedbacks, isFallback) {
    if (corner.number != null) markCornerFeedbackDone(corner.number);
    feedbacks.forEach((f) => {
      state.feedbacks.push({ corner_number: f.corner_number, symptom: f.symptom, strength: f.strength });
    });
    renderFeedbackSummary();
    closeFeedbackPanel();
    const suffix = isFallback ? "（逐条模式）" : "";
    showToast(`已提交 ${feedbacks.length} 条反馈${suffix}`, "success");
  }

  /** 提交反馈（批量多症状）。
   *  API 契约：POST /api/v1/feedback {track_id, feedbacks: [{corner_number, symptom, strength}]}
   *  @returns {Promise<void>}
   */
  async function submitFeedback() {
    const checked = dom.fbOverlay.querySelectorAll('input[name="symptom"]:checked');
    if (!checked.length) {
      dom.fbError.textContent = "请至少勾选一个症状";
      return;
    }
    const corner = state.selectedCorner;
    const feedbacks = collectFeedbacks();
    const payload = { track_id: state.currentTrackId, feedbacks: feedbacks };

    dom.fbSubmit.disabled = true;
    dom.fbError.textContent = "";
    try {
      // 尝试批量接口
      await fetchJSON("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      applyFeedbackSuccess(corner, feedbacks, false);
    } catch (e) {
      // 若批量接口不支持（422/400），降级为逐条提交
      if (e.status === 422 || e.status === 400) {
        try {
          await submitFeedbackFallback(feedbacks);
          applyFeedbackSuccess(corner, feedbacks, true);
          return;
        } catch (e2) {
          dom.fbError.textContent = e2.message;
          showToast("反馈提交失败：" + e2.message, "error");
        }
      } else {
        dom.fbError.textContent = e.message;
        showToast("反馈提交失败：" + e.message, "error");
      }
    } finally {
      dom.fbSubmit.disabled = false;
    }
  }

  /** 降级：逐条提交反馈（兼容旧版单条 POST /feedback 接口）。
   *  @param {Array<{symptom:string, strength:number, corner_number?:number}>} feedbacks — 反馈列表
   *  @returns {Promise<void>}
   */
  async function submitFeedbackFallback(feedbacks) {
    for (const f of feedbacks) {
      const payload = {
        track_id: state.currentTrackId,
        symptom: f.symptom,
        strength: f.strength,
      };
      if (f.corner_number != null) payload.corner_number = f.corner_number;
      await fetchJSON("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }
  }

  /** 加载已有反馈，标记已反馈弯道。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
  async function loadExistingFeedback(trackId) {
    try {
      const data = await fetchJSON(`/feedback?track_id=${encodeURIComponent(trackId)}`);
      const list = Array.isArray(data) ? data : (data && data.items) || [];
      state.feedbacks = list;
      list.forEach((f) => {
        if (f.corner_number != null) markCornerFeedbackDone(f.corner_number);
      });
      renderFeedbackSummary();
    } catch (_) { /* 非致命 */ }
  }

  /** 渲染反馈摘要面板。
   *  @returns {void}
   */
  function renderFeedbackSummary() {
    if (!state.feedbacks.length) {
      dom.feedbackSummaryWrap.innerHTML = '<div class="feedback-summary-empty">暂无反馈，点击赛道图弯道圆点开始录入</div>';
      return;
    }
    // 按弯道分组（全局症状单独一组）
    const groups = {};
    state.feedbacks.forEach((f) => {
      const key = f.corner_number != null ? `T${f.corner_number}` : "全局";
      if (!groups[key]) groups[key] = [];
      groups[key].push(f);
    });
    // 使用 DocumentFragment 批量插入
    const frag = document.createDocumentFragment();
    Object.keys(groups).sort().forEach((key) => {
      const item = document.createElement("div");
      item.className = "fb-summary-item";
      const tags = groups[key].map((f) => {
        const label = SYMPTOM_LABEL_ZH[f.symptom] || f.symptom;
        return `<span class="fb-summary-symptom-tag">${esc(label)}·${f.strength}</span>`;
      }).join("");
      item.innerHTML = `<span class="fb-summary-corner">${esc(key)}</span><span class="fb-summary-symptoms">${tags}</span>`;
      frag.appendChild(item);
    });
    dom.feedbackSummaryWrap.innerHTML = "";
    dom.feedbackSummaryWrap.appendChild(frag);
  }

  /** 清空当前赛道反馈（仅前端展示，后端清空需另行调用）。
   *  @returns {void}
   */
  function clearFeedbackDisplay() {
    state.feedbacks = [];
    state.feedbackCorners.clear();
    dom.mapWrap.querySelectorAll(".hotzone.feedback-done").forEach((hz) => hz.classList.remove("feedback-done"));
    renderFeedbackSummary();
    showToast("已清空前端反馈显示（后端数据需重新加载赛道刷新）", "success");
  }

  /* ========================================================================
     3.5 遥测录制控制（POST /api/v1/telemetry/record/toggle）
     ======================================================================== */

  /** 切换遥测录制状态（开始/停止）。
   *  @returns {Promise<void>}
   */
  async function toggleRecording() {
    if (!dom.btnRecord) return;
    dom.btnRecord.disabled = true;
    try {
      const data = await fetchJSON("/telemetry/record/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      // 后端返回 {recording: bool, session_id: str?}
      state.isRecording = data && data.recording === true;
      updateRecordingUI();
      if (state.isRecording) {
        showToast("遥测录制已开始", "success");
      } else {
        showToast("遥测录制已停止", "success");
      }
    } catch (e) {
      showToast("录制控制失败：" + e.message, "error");
    } finally {
      dom.btnRecord.disabled = false;
    }
  }

  /** 更新录制按钮 UI 状态（文字与样式）。
   *  @returns {void}
   */
  function updateRecordingUI() {
    if (!dom.btnRecord) return;
    dom.btnRecord.classList.toggle("recording", state.isRecording);
    const label = dom.btnRecord.querySelector(".record-label");
    if (label) label.textContent = state.isRecording ? "停止" : "录制";
  }

  /** 切换遥测收集开关 → POST /api/v1/telemetry/listener/toggle
   *  @returns {Promise<void>}
   */
  async function toggleListener() {
    if (!dom.btnListenerToggle) return;
    dom.btnListenerToggle.disabled = true;
    try {
      const action = state.isListening ? "stop" : "start";
      const data = await fetchJSON("/telemetry/listener/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      state.isListening = data && data.listening === true;
      updateListenerUI();
      if (state.isListening) {
        showToast("遥测收集已开启", "success");
      } else {
        showToast("遥测收集已关闭", "success");
      }
    } catch (e) {
      showToast("遥测开关失败：" + e.message, "error");
    } finally {
      dom.btnListenerToggle.disabled = false;
    }
  }

  /** 更新遥测收集按钮 UI 状态。
   *  @returns {void}
   */
  function updateListenerUI() {
    if (!dom.btnListenerToggle) return;
    dom.btnListenerToggle.classList.toggle("listening", state.isListening);
    const label = dom.btnListenerToggle.querySelector(".listener-label");
    if (label) label.textContent = state.isListening ? "收集" : "关闭";
  }

  /* ========================================================================
     4. WebSocket（遥测 / 当前弯 / 建议 / 连接状态）
        指数退避重连：1s, 2s, 4s, 8s, 16s（上限 30s），最多 8 次
     ======================================================================== */

  /** 建立 WebSocket 连接，绑定事件处理。
   *  @returns {void}
   */
  function connectWebSocket() {
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
        case "telemetry": onTelemetryEvent(msg.data || msg.payload || msg); break;
        case "corner": onCornerEvent(msg.data || msg.payload || msg); break;
        case "suggestion": onSuggestionEvent(msg.data || msg.payload || msg); break;
        case "telemetry_status": setWsStatus((msg.data || msg.payload || msg).connected === true); break;
        default: break;
      }
    };
  }

  /** 安排重连：指数退避。 */
  /** 安排 WebSocket 重连：指数退避。
   *  @returns {void}
   */
  function scheduleReconnect() {
    state.wsReconnectCount += 1;
    if (state.wsReconnectCount <= WS_RECONNECT_MAX) {
      const delay = Math.min(WS_RECONNECT_BASE * Math.pow(2, state.wsReconnectCount - 1), WS_RECONNECT_CAP);
      setTimeout(connectWebSocket, delay);
    }
  }

  /** 设置 WebSocket 连接状态指示器。
   *  @param {boolean} connected — 是否已连接
   *  @returns {void}
   */
  function setWsStatus(connected) {
    if (connected) {
      dom.wsDot.className = "status-dot status-online";
      dom.wsText.textContent = "遥测已连接";
    } else {
      dom.wsDot.className = "status-dot status-offline";
      dom.wsText.textContent = "遥测未连接";
    }
  }

  /** 遥测事件 → 防抖更新显示。
   *  @param {Object} t — 遥测数据
   *  @returns {void}
   */
  function onTelemetryEvent(t) {
    state.latestTelemetry = t;
    if (state.telemetryDebounceTimer) return;
    state.telemetryDebounceTimer = setTimeout(() => {
      state.telemetryDebounceTimer = null;
      if (state.latestTelemetry) onTelemetry(state.latestTelemetry);
    }, TELEMETRY_DEBOUNCE);
  }

  /** 遥测数据 → 更新显示（数值变化添加 flash 过渡 + 仪表盘进度条）。 */
  /** 更新扇区显示（数值 + 三色编码标签）。F1 扇区三色编码：S1=紫、S2=绿、S3=黄。
   *  入参为 **1 基**扇区（1/2/3）—— 后端 `to_sector_1based` 已把 UDP 的 0/1/2 转换好，
   *  早期前端直读 0 基值导致显示 "S0/S1/S2" 且 sector-0 样式类不存在。
   *  @param {number} sector — 扇区编号（1/2/3）
   */
  function updateSectorDisplay(sector) {
    updateTelValue(dom.telSector, `S${sector}`);
    const sectorText = `扇区 ${sector}`;
    if (dom.sectorTag.textContent !== sectorText) {
      dom.sectorTag.textContent = sectorText;
    }
    const newCls = "sector-tag sector-" + sector;
    if (dom.sectorTag.className !== newCls) {
      dom.sectorTag.className = newCls;
    }
  }

  /** 遥测数据 → 更新显示（数值变化添加 flash 过渡 + 仪表盘进度条）。
   *  字段契约与后端 `api/ws.py::_push_telemetry_frame` 一致：
   *  speed / throttle / brake / steer / gear / engine_rpm / drs / lap_time_ms / sector(1 基)。
   *  @param {Object} t — 遥测数据对象
   *  @returns {void}
   */
  function onTelemetry(t) {
    if (t.speed != null) {
      updateTelValue(dom.telSpeed, String(Math.round(t.speed)));
      setGaugeBar(dom.telSpeedBar, t.speed, SPEED_MAX);
    }
    if (t.throttle != null) {
      const pct = Math.round(t.throttle * 100);
      updateTelValue(dom.telThrottle, String(pct));
      setGaugeBar(dom.telThrottleBar, pct, PERCENT_MAX);
    }
    if (t.brake != null) {
      const pct = Math.round(t.brake * 100);
      updateTelValue(dom.telBrake, String(pct));
      setGaugeBar(dom.telBrakeBar, pct, PERCENT_MAX);
    }
    if (t.gear != null) { updateTelValue(dom.telGear, t.gear < 0 ? "R" : String(t.gear)); }
    // 后端字段名为 engine_rpm（原实现误读 t.rpm → 转速恒为 "—"）
    if (t.engine_rpm != null) { updateTelValue(dom.telRpm, String(Math.round(t.engine_rpm))); }
    if (t.lap_time_ms != null) { updateTelValue(dom.telLaptime, fmtLapTime(t.lap_time_ms)); }
    // sector 由后端统一转为 1 基（S1/S2/S3）
    if (t.sector != null) updateSectorDisplay(t.sector);
  }

  /** 当前弯道事件 → 高亮弯道 + 更新扇区。
   *  @param {Object} c — 弯道事件数据
   *  @returns {void}
   */
  function onCornerEvent(c) {
    const num = c.corner_number;
    highlightCurrentCorner(num);
    updateTelValue(dom.telCorner, num != null ? `T${num}` : "—");
    if (c.sector != null) updateSectorDisplay(c.sector);
  }

  /** 建议结果事件 → 自动刷新报告。
   *  兼容字段名：后端 WS 推送用 report_json，REST 返回用 report。
   *  @param {Object} s — 建议事件数据
   *  @returns {void}
   */
  function onSuggestionEvent(s) {
    const report = s.report_json || s.report;
    if (report) renderReport(report);
    showToast("新建议已生成", "success");
  }

  /* ========================================================================
     5. 建议报告
     ======================================================================== */

  /** 生成调教建议（POST /api/v1/suggest）。
   *  @returns {Promise<void>}
   */
  async function generateSuggestion() {
    if (!state.currentTrackId) {
      showToast("请先选择赛道", "error");
      return;
    }
    dom.btnSuggest.disabled = true;
    const oldText = dom.btnSuggest.textContent;
    dom.btnSuggest.innerHTML = '<span class="spinner-inline"></span>生成中…';
    // 读取模型类型
    const modelType = dom.modelTypeSelect ? dom.modelTypeSelect.value : "hybrid";
    // 更新顶栏模式显示
    if (dom.topbarMode) dom.topbarMode.textContent = modelType.toUpperCase();
    try {
      const data = await fetchJSON("/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: state.currentTrackId, model_type: modelType }),
      });
      // ★ Bug 修复：后端 SuggestionView 字段名为 report（非 report_json）
      const report = data && data.report ? data.report : data;
      if (report) renderReport(report);
      else loadLatestSuggestion(state.currentTrackId);
      showToast("建议已生成", "success");
    } catch (e) {
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

  /** 读取最新建议并渲染。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
  async function loadLatestSuggestion(trackId) {
    try {
      const data = await fetchJSON(`/suggest/latest?track_id=${encodeURIComponent(trackId)}`);
      // ★ Bug 修复：后端 SuggestionView 字段名为 report（非 report_json）
      const report = data && data.report ? data.report : data;
      if (report && report.parameters) renderReport(report);
    } catch (_) { /* 无建议，保持空态 */ }
  }

  /** 渲染报告摘要区（summary + generated_at）。
   *  @param {Object} report — 建议报告对象
   *  @returns {void}
   */
  function renderReportSummary(report) {
    if (report.summary) {
      dom.reportSummary.innerHTML = `<div>${esc(report.summary)}</div>` +
        (report.generated_at ? `<div class="meta">生成时间：${esc(report.generated_at)}</div>` : "");
      dom.reportSummary.classList.add("visible");
    } else {
      dom.reportSummary.classList.remove("visible");
    }
  }

  /** 构建单个参数的表格行 <tr>。
   *  @param {Object} p — 单个参数报告项
   *  @returns {HTMLTableRowElement}
   */
  function buildReportRow(p) {
    const tr = document.createElement("tr");
    const delta = p.setup_delta;
    const deltaCls = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "delta-zero";
    const deltaSign = delta > 0 ? "+" : "";
    const conf = p.confidence || "medium";
    const confCls = conf === "high" ? "conf-high" : conf === "low" ? "conf-low" : "conf-medium";
    const suggestVal = p.suggested != null ? p.suggested : (p.current != null && delta != null ? p.current + delta : "—");
    const linkages = Array.isArray(p.linkages) ? p.linkages.join("、") : (p.linkages || "");
    const tradeoff = p.tradeoff ? `<span class="tradeoff-text">${esc(p.tradeoff)}</span>` : "—";
    const barFill = renderDeltaBar(delta);
    // 参数名中文显示
    const paramLabel = PARAM_LABEL_ZH[p.param] || p.param;

    tr.innerHTML = `
      <td class="col-param" title="${esc(p.param)}">${esc(paramLabel)}</td>
      <td class="col-current">${esc(p.current)}</td>
      <td class="col-delta ${deltaCls}">${deltaSign}${esc(delta)}${barFill}</td>
      <td class="col-suggest">${esc(suggestVal)}</td>
      <td class="col-link">${esc(linkages)}</td>
      <td class="col-note">${esc(p.linked_notes || "")}</td>
      <td class="col-source">${esc(p.source || "")}</td>
      <td><span class="confidence-tag ${confCls}">${esc(conf)}</span></td>
      <td>${tradeoff}</td>`;
    return tr;
  }

  /** 渲染建议报告（20 参数表格 + 可视化条形图）。
   *  使用 DocumentFragment 批量插入以优化性能。
   *  报告结构（design.md 2.7.7）：
   *  { track_id, generated_at, parameters: [{param, current, setup_delta,
   *    linkages, linked_notes, source, confidence, tradeoff}], summary }
   *  @param {Object} report — 建议报告对象
   *  @returns {void}
   */
  function renderReport(report) {
    const params = report.parameters || [];
    if (!params.length) {
      dom.reportSummary.classList.remove("visible");
      dom.reportWrap.innerHTML = '<div class="report-empty">建议报告为空</div>';
      return;
    }

    renderReportSummary(report);

    // 使用 DocumentFragment 批量插入表格行
    const table = document.createElement("table");
    table.className = "report-table";
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>
      <th>参数</th><th>当前值</th><th>调整量</th><th>建议值</th>
      <th>联动维度</th><th>联动说明</th><th>出处</th><th>置信度</th><th>权衡</th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    const frag = document.createDocumentFragment();
    params.forEach((p) => frag.appendChild(buildReportRow(p)));
    tbody.appendChild(frag);
    table.appendChild(tbody);

    dom.reportWrap.innerHTML = "";
    dom.reportWrap.appendChild(table);
  }

  /** 渲染参数调整可视化条形图。
   *  @param {number} delta — 调整量（正/负/零）
   *  @returns {string} — HTML 字符串
   */
  function renderDeltaBar(delta) {
    if (delta == null || delta === 0) return "";
    const ratio = Math.min(Math.abs(delta) / DELTA_BAR_SCALE, 1) * DELTA_BAR_HALF_PCT;
    const dir = delta > 0 ? "up" : "down";
    const width = ratio.toFixed(1);
    const left = delta > 0 ? "50%" : `${(DELTA_BAR_HALF_PCT - ratio).toFixed(1)}%`;
    return `<span class="delta-bar"><span class="delta-bar-fill ${dir}" style="left:${left};width:${width}%;"></span></span>`;
  }

  /* ========================================================================
     6. 迭代历史
     ======================================================================== */

  /** 加载迭代历史列表。
   *  @param {string} trackId — 赛道 ID
   *  @returns {Promise<void>}
   */
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

  /** 渲染迭代历史列表（DocumentFragment 批量插入）。
   *  @param {Array} list — 迭代历史数据列表
   *  @returns {void}
   */
  function renderHistory(list) {
    if (!list.length) {
      dom.historyWrap.innerHTML = '<div class="history-empty">暂无迭代记录</div>';
      return;
    }
    // 使用 DocumentFragment 批量插入
    const frag = document.createDocumentFragment();
    list.forEach((it) => {
      const round = it.round_no != null ? `第 ${it.round_no} 轮` : "迭代";
      const time = it.created_at || "";
      const before = it.before_params || (it.before_setup && it.before_setup.params) || {};
      const after = it.after_params || (it.after_setup && it.after_setup.params) || {};
      const diffs = Object.keys(after).filter((k) => before[k] != null && after[k] !== before[k]);
      const diffRows = diffs.length ? diffs.map((k) => {
        const d = after[k] - before[k];
        const sign = d > 0 ? "+" : "";
        const label = PARAM_LABEL_ZH[k] || k;
        return `<div class="diff-row"><span>${esc(label)}</span><span class="dv">${sign}${esc(d)}</span></div>`;
      }).join("") : '<div class="diff-row"><span>无参数变化</span></div>';

      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <div class="history-item-head">
          <span class="history-round">${esc(round)}</span>
          <span class="history-time">${esc(time)}</span>
        </div>
        <div class="history-diff">${diffRows}</div>`;
      frag.appendChild(item);
    });
    dom.historyWrap.innerHTML = "";
    dom.historyWrap.appendChild(frag);
  }

  /* ========================================================================
      7. 21 项调教参数输入面板
     ======================================================================== */

  /** 加载参数定义（优先 GET /api/v1/setup/fields，失败时用 fallback）。
   *  @returns {Promise<void>}
   */
  async function loadSetupFields() {
    try {
      const data = await fetchJSON("/setup/fields");
      const list = Array.isArray(data) ? data : (data && data.items) || [];
      if (list.length) {
        // 兼容字段名：min_val/min, max_val/max, label_zh/label
        state.setupFields = list.map((f) => ({
          name: f.name,
          group: f.group,
          label: f.label_zh || f.label || PARAM_LABEL_ZH[f.name] || f.name,
          min: f.min != null ? f.min : f.min_val,
          max: f.max != null ? f.max : f.max_val,
          step: f.step,
          default: f.default,
          unit: f.unit || "",
        }));
      } else {
        state.setupFields = FALLBACK_SETUP_FIELDS.slice();
      }
    } catch (_) {
      state.setupFields = FALLBACK_SETUP_FIELDS.slice();
    }
    // 初始化参数值为默认值
    state.setupValues = {};
    state.setupModified = {};
    state.setupFields.forEach((f) => {
      state.setupValues[f.name] = f.default;
    });
    renderSetupPanel();
  }

  /** 将参数定义按 group 分组。
   *  @returns {{groups: Object<string, Array>, groupKeys: string[]}}
   */
  function groupSetupFieldsByGroup() {
    const groups = {};
    state.setupFields.forEach((f) => {
      if (!groups[f.group]) groups[f.group] = [];
      groups[f.group].push(f);
    });
    return { groups, groupKeys: Object.keys(groups) };
  }

  /** 渲染 tab 栏按钮（含滑动指示器）。
   *  @param {Object} groups — 按组分类的参数定义
   *  @param {string[]} groupKeys — 组名列表
   *  @returns {void}
   */
  function renderSetupTabs(groups, groupKeys) {
    const tabsContainer = $("setup-tabs");
    tabsContainer.innerHTML = "";
    const tabsFrag = document.createDocumentFragment();
    groupKeys.forEach((group, gi) => {
      const groupLabel = GROUP_LABEL_ZH[group] || group;
      const tab = document.createElement("button");
      tab.className = "setup-tab" + (gi === 0 ? " active" : "");
      tab.dataset.group = group;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", gi === 0 ? "true" : "false");
      tab.innerHTML = `${esc(groupLabel)}<span class="setup-tab-count">${groups[group].length}</span>`;
      tab.addEventListener("click", () => switchSetupTab(group));
      tabsFrag.appendChild(tab);
    });
    tabsContainer.appendChild(tabsFrag);
    // 创建滑动指示器
    const indicator = document.createElement("div");
    indicator.className = "setup-tabs-indicator";
    tabsContainer.appendChild(indicator);
    // 初始化指示器位置（延迟一帧确保 DOM 已布局）
    requestAnimationFrame(() => updateTabIndicator());
  }

  /** 更新 tab 滑动指示器位置与宽度。
   *  @returns {void}
   */
  function updateTabIndicator() {
    const tabsContainer = $("setup-tabs");
    if (!tabsContainer) return;
    const indicator = tabsContainer.querySelector(".setup-tabs-indicator");
    if (!indicator) return;
    const activeTab = tabsContainer.querySelector(".setup-tab.active");
    if (!activeTab) return;
    indicator.style.left = activeTab.offsetLeft + "px";
    indicator.style.width = activeTab.offsetWidth + "px";
  }

  /** 创建单个参数字段的 DOM 节点（含 range + number 输入与事件绑定）。
   *  @param {Object} f — 参数定义项
   *  @returns {HTMLDivElement}
   */
  function createSetupFieldDiv(f) {
    const fieldDiv = document.createElement("div");
    fieldDiv.className = "setup-field";
    fieldDiv.dataset.name = f.name;
    const val = state.setupValues[f.name] != null ? state.setupValues[f.name] : f.default;
    fieldDiv.innerHTML = `
      <div class="setup-field-info">
        <span class="setup-field-label">${esc(f.label)}</span>
        <span class="setup-field-name">${esc(f.name)}</span>
      </div>
      <div class="setup-field-control">
        <input type="range" class="setup-field-range" min="${f.min}" max="${f.max}" step="${f.step}" value="${val}" aria-label="${esc(f.label)}" />
        <input type="number" class="setup-field-value" min="${f.min}" max="${f.max}" step="${f.step}" value="${val}" aria-label="${esc(f.label)} 数值" />
        <span class="setup-field-unit">${esc(f.unit || "")}</span>
      </div>`;
    const range = fieldDiv.querySelector(".setup-field-range");
    const num = fieldDiv.querySelector(".setup-field-value");
    // 初始化 range 渐变轨道填充
    updateRangeFill(range, f);
    range.addEventListener("input", () => {
      num.value = range.value;
      updateRangeFill(range, f);
      onSetupFieldChange(f.name, parseFloat(range.value));
    });
    num.addEventListener("input", () => {
      let v = parseFloat(num.value);
      if (isNaN(v)) return;
      v = Math.max(f.min, Math.min(f.max, v));
      range.value = v;
      updateRangeFill(range, f);
      onSetupFieldChange(f.name, v);
    });
    num.addEventListener("blur", () => {
      let v = parseFloat(num.value);
      if (isNaN(v)) { num.value = state.setupValues[f.name]; return; }
      v = Math.max(f.min, Math.min(f.max, v));
      const steps = Math.round((v - f.min) / f.step);
      v = f.min + steps * f.step;
      v = Math.max(f.min, Math.min(f.max, v));
      num.value = v;
      range.value = v;
      updateRangeFill(range, f);
      onSetupFieldChange(f.name, v);
    });
    return fieldDiv;
  }

  /** 渲染 21 参数输入面板（6 大类 tab 分组）。
   *  @returns {void}
   */
  function renderSetupPanel() {
    if (!state.setupFields.length) {
      dom.setupFieldsWrap.innerHTML = '<div class="setup-empty">参数定义加载失败</div>';
      return;
    }
    const { groups, groupKeys } = groupSetupFieldsByGroup();

    renderSetupTabs(groups, groupKeys);

    // 渲染 tab 内容容器
    dom.setupFieldsWrap.innerHTML = "";
    const contentFrag = document.createDocumentFragment();
    groupKeys.forEach((group, gi) => {
      const container = document.createElement("div");
      container.className = "setup-group-container" + (gi === 0 ? " active" : "");
      container.dataset.group = group;
      groups[group].forEach((f) => container.appendChild(createSetupFieldDiv(f)));
      contentFrag.appendChild(container);
    });
    dom.setupFieldsWrap.appendChild(contentFrag);
    updateSetupModifiedMarks();
    // 初始化所有 range 渐变轨道填充
    dom.setupFieldsWrap.querySelectorAll(".setup-field").forEach((el) => {
      const name = el.dataset.name;
      const field = state.setupFields.find((f) => f.name === name);
      const range = el.querySelector(".setup-field-range");
      if (field && range) updateRangeFill(range, field);
    });
  }

  /** 切换调教参数 tab（含滑动指示器动画）。
   *  @param {string} group — 目标组名
   *  @returns {void}
   */
  function switchSetupTab(group) {
    // 更新 tab 按钮
    $("setup-tabs").querySelectorAll(".setup-tab").forEach((tab) => {
      const isActive = tab.dataset.group === group;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    // 更新内容容器
    dom.setupFieldsWrap.querySelectorAll(".setup-group-container").forEach((container) => {
      container.classList.toggle("active", container.dataset.group === group);
    });
    // 更新滑动指示器位置
    requestAnimationFrame(() => updateTabIndicator());
  }

  /** 更新 range input 的渐变轨道填充百分比（CSS 变量 --range-fill）。
   *  @param {HTMLInputElement} rangeEl — range input 元素
   *  @param {Object} field — 参数定义项（含 min/max）
   */
  function updateRangeFill(rangeEl, field) {
    if (!rangeEl || !field) return;
    const pct = ((parseFloat(rangeEl.value) - field.min) / (field.max - field.min)) * 100;
    rangeEl.style.setProperty("--range-fill", Math.max(0, Math.min(100, pct)) + "%");
  }

  /** 参数值变化处理：更新状态、标记修改、flash 高亮。
   *  @param {string} name — 参数名
   *  @param {number} value — 新值
   *  @returns {void}
   */
  function onSetupFieldChange(name, value) {
    state.setupValues[name] = value;
    // 标记是否相对默认值已修改
    const field = state.setupFields.find((f) => f.name === name);
    if (field) {
      state.setupModified[name] = Math.abs(value - field.default) > FLOAT_EPSILON;
    }
    updateSetupModifiedMarks();
    // 数值输入 flash
    const numInput = dom.setupFieldsWrap.querySelector(`.setup-field[data-name="${name}"] .setup-field-value`);
    if (numInput) {
      numInput.classList.add("flash");
      setTimeout(() => numInput.classList.remove("flash"), FLASH_DURATION_MS);
    }
  }

  /** 更新参数已修改标记（DOM class toggle）。
   *  @returns {void}
   */
  function updateSetupModifiedMarks() {
    dom.setupFieldsWrap.querySelectorAll(".setup-field").forEach((el) => {
      const name = el.dataset.name;
      el.classList.toggle("modified", !!state.setupModified[name]);
    });
  }

  /** 保存调教 → POST /api/v1/setup/manual {track_id, params: {}}
   *  @returns {Promise<void>}
   */
  async function saveSetup() {
    if (!state.currentTrackId) {
      showToast("请先选择赛道", "error");
      return;
    }
    dom.btnSaveSetup.disabled = true;
    try {
      await fetchJSON("/setup/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_id: state.currentTrackId, params: state.setupValues }),
      });
      showToast("调教已保存", "success");
      // 清除修改标记
      state.setupModified = {};
      updateSetupModifiedMarks();
    } catch (e) {
      showToast("保存调教失败：" + e.message, "error");
    } finally {
      dom.btnSaveSetup.disabled = false;
    }
  }

  /** 将 state.setupValues 同步到所有参数字段的 range/number UI（含渐变轨道）。
   *  @returns {void}
   */
  function syncAllSetupFieldsUI() {
    dom.setupFieldsWrap.querySelectorAll(".setup-field").forEach((el) => {
      const name = el.dataset.name;
      const v = state.setupValues[name];
      if (v == null) return;
      const range = el.querySelector(".setup-field-range");
      const num = el.querySelector(".setup-field-value");
      range.value = v;
      num.value = v;
      // 同步渐变轨道填充
      const field = state.setupFields.find((f) => f.name === name);
      if (field) updateRangeFill(range, field);
    });
  }

  /** 重置所有参数到默认值。
   *  @returns {void}
   */
  function resetSetupDefault() {
    state.setupFields.forEach((f) => {
      state.setupValues[f.name] = f.default;
    });
    state.setupModified = {};
    syncAllSetupFieldsUI();
    updateSetupModifiedMarks();
    showToast("已重置为默认值", "success");
  }

  /** 导入当前调教 → GET /api/v1/setup/current
   *  @returns {Promise<void>}
   */
  async function importCurrentSetup() {
    dom.btnImportCurrent.disabled = true;
    try {
      const data = await fetchJSON("/setup/current");
      // 兼容字段：data.params / data.setup / data 直接是 dict
      const params = data.params || data.setup || (data && typeof data === "object" && !Array.isArray(data) ? data : {});
      if (!params || typeof params !== "object") {
        showToast("未找到调教数据", "error");
        return;
      }
      // 更新参数值
      state.setupFields.forEach((f) => {
        if (params[f.name] != null) {
          state.setupValues[f.name] = parseFloat(params[f.name]);
        }
      });
      // 更新 UI
      syncAllSetupFieldsUI();
      // 计算修改标记
      state.setupFields.forEach((f) => {
        const v = state.setupValues[f.name];
        if (v != null) state.setupModified[f.name] = Math.abs(v - f.default) > FLOAT_EPSILON;
      });
      updateSetupModifiedMarks();
      showToast("已导入当前调教", "success");
    } catch (e) {
      showToast("导入当前调教失败：" + e.message, "error");
    } finally {
      dom.btnImportCurrent.disabled = false;
    }
  }

  /** 从遥测导入调教 → POST /api/v1/setup/import
   *  从实时遥测 Packet 5 (CarSetups) 读取当前调教参数并导入。
   *  @returns {Promise<void>}
   */
  async function importFromTelemetry() {
    dom.btnImportTelemetry.disabled = true;
    try {
      const data = await fetchJSON("/setup/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const params = data.params || data.setup || (data && typeof data === "object" && !Array.isArray(data) ? data : {});
      if (!params || typeof params !== "object") {
        showToast("未找到调教数据", "error");
        return;
      }
      state.setupFields.forEach((f) => {
        if (params[f.name] != null) {
          state.setupValues[f.name] = parseFloat(params[f.name]);
        }
      });
      syncAllSetupFieldsUI();
      state.setupFields.forEach((f) => {
        const v = state.setupValues[f.name];
        if (v != null) state.setupModified[f.name] = Math.abs(v - f.default) > FLOAT_EPSILON;
      });
      updateSetupModifiedMarks();
      showToast("已从遥测导入调教", "success");
    } catch (e) {
      showToast("从遥测导入失败：" + e.message, "error");
    } finally {
      dom.btnImportTelemetry.disabled = false;
    }
  }

  /* ========================================================================
     8. 事件绑定 & 初始化
     ======================================================================== */

  /** 绑定赛道与建议相关事件。
   *  @returns {void}
   */
  function bindTrackAndSuggestEvents() {
    dom.trackSelect.addEventListener("change", onTrackChange);
    dom.btnSuggest.addEventListener("click", generateSuggestion);

    // 模型类型切换
    if (dom.modelTypeSelect) {
      dom.modelTypeSelect.addEventListener("change", () => {
        if (dom.topbarMode) dom.topbarMode.textContent = dom.modelTypeSelect.value.toUpperCase();
      });
    }

    // 刷新历史
    dom.btnHistory.addEventListener("click", () => loadIterationHistory(state.currentTrackId));
  }

  /** 绑定 20 参数面板事件。
   *  @returns {void}
   */
  function bindSetupEvents() {
    dom.btnSaveSetup.addEventListener("click", saveSetup);
    dom.btnResetDefault.addEventListener("click", resetSetupDefault);
    dom.btnImportCurrent.addEventListener("click", importCurrentSetup);
    dom.btnImportTelemetry.addEventListener("click", importFromTelemetry);
  }

  /** 绑定反馈面板与摘要事件。
   *  @returns {void}
   */
  function bindFeedbackEvents() {
    // 遥测录制
    if (dom.btnRecord) dom.btnRecord.addEventListener("click", toggleRecording);
    // 遥测收集开关
    if (dom.btnListenerToggle) dom.btnListenerToggle.addEventListener("click", toggleListener);

    // 反馈摘要
    dom.btnClearFeedback.addEventListener("click", clearFeedbackDisplay);

    // 赛道级反馈按钮（仅全局症状）
    if (dom.btnTrackFeedback) {
      dom.btnTrackFeedback.addEventListener("click", () => {
        if (!state.currentTrackId) {
          showToast("请先选择赛道", "error");
          return;
        }
        openFeedbackPanel(null, null, "track");
      });
    }

    // 反馈面板
    dom.fbClose.addEventListener("click", closeFeedbackPanel);
    dom.fbSubmit.addEventListener("click", submitFeedback);
    dom.fbOverlay.addEventListener("click", (e) => {
      if (e.target === dom.fbOverlay) closeFeedbackPanel();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !dom.fbOverlay.hidden) closeFeedbackPanel();
    });

    // checkbox 勾选 → 更新计数 + 渲染强度滑块
    dom.fbOverlay.querySelectorAll('input[name="symptom"]').forEach((cb) => {
      cb.addEventListener("change", () => {
        updateFbSelectedCount();
        renderStrengthSliders();
      });
    });

    // 症状分组展开/折叠
    document.querySelectorAll(".sym-group-head").forEach((head) => {
      head.addEventListener("click", () => {
        head.parentElement.classList.toggle("expanded");
      });
      head.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          head.parentElement.classList.toggle("expanded");
        }
      });
    });
  }

  /** 绑定所有事件并启动初始化。
   *  @returns {void}
   */
  function bindEvents() {
    bindTrackAndSuggestEvents();
    bindSetupEvents();
    bindFeedbackEvents();
    // 窗口 resize 时更新 tab 指示器位置
    window.addEventListener("resize", debounce(updateTabIndicator, 150));
  }

  /** 应用入口：绑定事件、加载赛道/参数、连接 WebSocket。
   *  @returns {void}
   */
  function init() {
    bindEvents();
    loadTracks();
    loadSetupFields();
    connectWebSocket();
    updateListenerUI();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
