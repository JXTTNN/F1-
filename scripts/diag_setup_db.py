"""检查数据库中的setup快照，诊断KeyError 'front_suspension'根因。"""
import sqlite3
import json
import os
import sys

sys.path.insert(0, "D:/F1OPT-Test")

from setup_tuner.domain.setup import ALL_SETUP_FIELDS

ALL_FIELD_NAMES = {f.name for f in ALL_SETUP_FIELDS}
print(f"ALL_SETUP_FIELDS ({len(ALL_FIELD_NAMES)} 项): {sorted(ALL_FIELD_NAMES)}")

db = "D:/F1OPT-Test/data/f1opt.db"
print(f"\nDB路径: {db}")
print(f"DB存在: {os.path.exists(db)}")

if not os.path.exists(db):
    print("数据库不存在，退出")
    sys.exit(0)

conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# 检查setup表
rows = conn.execute("SELECT * FROM setup ORDER BY imported_at DESC").fetchall()
print(f"\nSetup表: {len(rows)} 条记录")

for r in rows:
    params = json.loads(r["params_json"])
    param_keys = set(params.keys())
    missing = ALL_FIELD_NAMES - param_keys
    extra = param_keys - ALL_FIELD_NAMES
    print(f"\n  id={r['id']} track_id={r['track_id']} imported_at={r['imported_at']}")
    print(f"  params字段数: {len(param_keys)} / 期望: {len(ALL_FIELD_NAMES)}")
    if missing:
        print(f"  缺失字段: {sorted(missing)}")
    else:
        print(f"  字段完整")
    if extra:
        print(f"  多余字段: {sorted(extra)}")
    print(f"  params: {json.dumps(params, ensure_ascii=False)[:300]}")

# 检查yas_marina的setup
print("\n--- yas_marina 的setup快照 ---")
yas_rows = conn.execute(
    "SELECT * FROM setup WHERE track_id = ? ORDER BY imported_at DESC", ("yas_marina",)
).fetchall()
print(f"yas_marina setup: {len(yas_rows)} 条")
for r in yas_rows:
    params = json.loads(r["params_json"])
    print(f"  id={r['id']} params_keys={list(params.keys())}")
    if "front_suspension" not in params:
        print(f"  *** 缺少 front_suspension ***")

conn.close()