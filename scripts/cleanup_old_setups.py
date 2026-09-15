"""清理数据库中旧格式setup快照（缺少front_suspension等新字段的记录）。"""
import sqlite3
import json
import os
import sys

sys.path.insert(0, "D:/F1OPT-Test")
from setup_tuner.domain.setup import ALL_SETUP_FIELDS

ALL_FIELD_NAMES = {f.name for f in ALL_SETUP_FIELDS}

db = "D:/F1OPT-Test/data/f1opt.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# 找出所有旧格式记录（params中缺少任意ALL_SETUP_FIELDS字段）
rows = conn.execute("SELECT id, track_id, params_json FROM setup").fetchall()
old_ids = []
for r in rows:
    params = json.loads(r["params_json"])
    missing = ALL_FIELD_NAMES - set(params.keys())
    if missing:
        old_ids.append((r["id"], r["track_id"], sorted(missing)))

print(f"总setup记录: {len(rows)}")
print(f"旧格式记录: {len(old_ids)}")

if old_ids:
    print("\n旧格式记录详情（前10条）:")
    for id_, tid, missing in old_ids[:10]:
        print(f"  id={id_} track={tid} 缺失={missing}")

    # 删除旧格式记录
    ids_to_delete = [str(r[0]) for r in old_ids]
    placeholders = ",".join("?" * len(ids_to_delete))
    cur = conn.execute(
        f"DELETE FROM setup WHERE id IN ({placeholders})",
        ids_to_delete,
    )
    conn.commit()
    print(f"\n已删除 {cur.rowcount} 条旧格式setup记录")

    # 验证
    remaining = conn.execute("SELECT COUNT(*) FROM setup").fetchone()[0]
    print(f"剩余setup记录: {remaining}")
else:
    print("无需清理")

conn.close()