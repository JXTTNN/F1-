"""检查数据库内容完整性。"""
import sqlite3
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
db_path = ROOT / "data" / "f1opt.db"
print(f"DB exists: {db_path.exists()}, size: {db_path.stat().st_size if db_path.exists() else 0}")

conn = sqlite3.connect(str(db_path))
c = conn.cursor()

# 检查表是否存在
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f"Tables: {tables}")

# 检查赛道数据
c.execute("SELECT COUNT(*) FROM track")
track_count = c.fetchone()[0]
print(f"Track rows: {track_count}")

# 检查弯道数据
c.execute("SELECT COUNT(*) FROM corner")
corner_count = c.fetchone()[0]
print(f"Corner rows: {corner_count}")

# 检查每条赛道的弯道数
c.execute("SELECT track_id, COUNT(*) as cnt FROM corner GROUP BY track_id ORDER BY track_id")
corner_per_track = c.fetchall()
for row in corner_per_track:
    print(f"  {row[0]}: {row[1]} corners")

# 检查是否有赛道缺少弯道数据
c.execute("SELECT track_id FROM track")
all_tracks = [r[0] for r in c.fetchall()]
tracks_with_corners = [r[0] for r in corner_per_track]
missing = set(all_tracks) - set(tracks_with_corners)
if missing:
    print(f"WARNING: Tracks without corners: {missing}")
else:
    print("All tracks have corner data")

# 检查 setup 数据
c.execute("SELECT COUNT(*) FROM setup")
setup_count = c.fetchone()[0]
print(f"Setup rows: {setup_count}")

conn.close()
