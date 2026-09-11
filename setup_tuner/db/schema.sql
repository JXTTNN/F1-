-- F1OPT 赛车调教优化助手 SQLite 建表 DDL
-- 对照 design.md 2.3.2 节，共 6 张核心表。
-- 纯本地单文件 SQLite，写入频率低（反馈/建议为低频，遥测不落库）。

-- track：赛道主表（24 条静态数据，来自核对后的 tracks.py）
CREATE TABLE IF NOT EXISTS track (
  track_id          TEXT PRIMARY KEY,      -- 如 'suzuka'
  official_name     TEXT NOT NULL,
  circuit_name      TEXT NOT NULL,
  track_type        TEXT NOT NULL,         -- high_speed_low_downforce|street|high_downforce|medium|mixed
  length_m          REAL NOT NULL,
  corners           INTEGER NOT NULL,
  udp_track_id      INTEGER,               -- 遥测 m_trackId 数值，用于自动识别
  svg_path          TEXT NOT NULL          -- 对应 SVG 资源路径
);

-- corner：弯道表（含 SVG 热区锚点）
CREATE TABLE IF NOT EXISTS corner (
  track_id          TEXT NOT NULL REFERENCES track(track_id),
  corner_number     INTEGER NOT NULL,      -- 1-based
  name              TEXT,
  corner_type       TEXT NOT NULL,         -- slow|medium|fast
  speed_kmh         REAL,
  anchor_x          REAL NOT NULL,         -- SVG 归一化坐标 x (0~1)
  anchor_y          REAL NOT NULL,         -- SVG 归一化坐标 y (0~1)
  PRIMARY KEY (track_id, corner_number)
);

-- setup：调教快照（导入自 Car Setups 包）
CREATE TABLE IF NOT EXISTS setup (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id          TEXT NOT NULL,
  imported_at       TEXT NOT NULL,         -- ISO8601
  params_json       TEXT NOT NULL          -- 23 项参数 JSON 快照
);

-- feedback：玩家弯道反馈
CREATE TABLE IF NOT EXISTS feedback (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  setup_id          INTEGER REFERENCES setup(id),
  track_id          TEXT NOT NULL,
  corner_number     INTEGER,               -- 点击弯道；NULL 表示「全局」症状
  symptom           TEXT NOT NULL,         -- 12 症状标识之一
  category          TEXT NOT NULL,         -- entry|apex|exit|global
  strength          INTEGER NOT NULL DEFAULT 3 CHECK(strength BETWEEN 0 AND 5),
  created_at        TEXT NOT NULL
);

-- suggestion：调教建议（规则引擎输出）
CREATE TABLE IF NOT EXISTS suggestion (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  setup_id          INTEGER REFERENCES setup(id),
  track_id          TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  report_json       TEXT NOT NULL          -- 含 setupDelta/联动/出处/置信度/tradeoff
);

-- iteration：迭代闭环记录（前后对比）
CREATE TABLE IF NOT EXISTS iteration (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id          TEXT NOT NULL,
  round_no          INTEGER NOT NULL,      -- 第几轮闭环
  before_setup_id   INTEGER REFERENCES setup(id),
  after_setup_id    INTEGER REFERENCES setup(id),
  suggestion_id     INTEGER REFERENCES suggestion(id),
  created_at        TEXT NOT NULL
);

-- 索引：加速按赛道查询反馈/建议/迭代历史
CREATE INDEX IF NOT EXISTS idx_feedback_track ON feedback(track_id);
CREATE INDEX IF NOT EXISTS idx_feedback_setup ON feedback(setup_id);
CREATE INDEX IF NOT EXISTS idx_setup_track ON setup(track_id);
CREATE INDEX IF NOT EXISTS idx_suggestion_track ON suggestion(track_id);
CREATE INDEX IF NOT EXISTS idx_iteration_track ON iteration(track_id);

-- 复合索引：加速 ORDER BY ... DESC LIMIT 1 查询（性能优化 task-36）
-- get_latest_setup: WHERE track_id=? ORDER BY imported_at DESC LIMIT 1
CREATE INDEX IF NOT EXISTS idx_setup_track_imported_at
  ON setup(track_id, imported_at DESC);
-- get_latest_suggestion: WHERE track_id=? ORDER BY created_at DESC LIMIT 1
CREATE INDEX IF NOT EXISTS idx_suggestion_track_created_at
  ON suggestion(track_id, created_at DESC);
-- get_latest_round: WHERE track_id=? 聚合 MAX(round_no)
CREATE INDEX IF NOT EXISTS idx_iteration_track_round
  ON iteration(track_id, round_no DESC);