"""验证calibrator.py — 用最快圈数据运行校准"""
import json
from setup_tuner.domain.setup import CarSetup
from setup_tuner.domain.track import get_track_by_id
from setup_tuner.physics.calibrator import Calibrator, TelemetryBenchmark
from setup_tuner.physics.simulator import Simulator

# 加载遥测数据
with open("D:/F1TelemetryCollector/dist/data/lap_abu_dhabi_9447996851316664300_2.json", encoding="utf-8") as f:
    data = json.load(f)

# 提取基准数据
samples = data.get("samples", [])
speeds_kmh = [s.get("speed", 0) for s in samples]
speeds_ms = [v / 3.6 for v in speeds_kmh if v > 0]
top_speed = max(speeds_ms) if speeds_ms else 0.0

# 弯道速度：取速度最低25%的平均值
sorted_speeds = sorted(speeds_ms)
cutoff = max(1, int(len(sorted_speeds) * 0.25))
avg_corner_speed = sum(sorted_speeds[:cutoff]) / len(sorted_speeds[:cutoff])

# 胎温
mid_start = len(samples) // 4
mid_end = len(samples) * 3 // 4
mid_samples = samples[mid_start:mid_end]
temps_sum = [0.0, 0.0, 0.0, 0.0]
count = 0
for s in mid_samples:
    st = s.get("tyre_surface_temp", [])
    if len(st) >= 4:
        for i in range(4):
            temps_sum[i] += float(st[i])
        count += 1
tyre_temps = {"FL": temps_sum[0]/count, "FR": temps_sum[1]/count, "RL": temps_sum[2]/count, "RR": temps_sum[3]/count} if count > 0 else {"FL": 100, "FR": 100, "RL": 100, "RR": 100}

benchmark = TelemetryBenchmark(
    lap_time=data.get("lap_time_ms", 0) / 1000.0,
    sector_times=[t / 1000.0 for t in data.get("sector_times_ms", [0, 0, 0])],
    top_speed=top_speed,
    corner_speeds={},  # 空字典，用平均弯道速度替代
    tyre_temps=tyre_temps,
    track_id="yas_marina",
)

print(f"基准: 圈速={benchmark.lap_time:.3f}s, 最高速度={benchmark.top_speed:.1f}m/s, 平均弯道速度={avg_corner_speed:.1f}m/s")

# 构造CarSetup（用默认值，因为遥测中的setup值是UDP编码）
setup = CarSetup.default()

# 构造Simulator
track = get_track_by_id("yas_marina")
weather = data.get("weather", {})
tyre_data = data.get("tyre", {})
compound_name = tyre_data.get("compound_name", "medium")
compound_mapping = {"C1": "hard", "C2": "medium", "C3": "soft", "C4": "soft", "intermediate": "intermediate", "wet": "wet"}
tyre_compound = compound_mapping.get(compound_name, "medium")

sim = Simulator(
    setup=setup,
    track=track,
    tyre_type=tyre_compound,
    track_temp=float(weather.get("track_temp", 30.0)),
    ambient_temp=float(weather.get("air_temp", 20.0)),
    fuel_load=float(data.get("fuel", {}).get("load_kg", 100.0)),
)

print(f"轮胎配方: {tyre_compound} (原始: {compound_name})")
print(f"赛道: {track.track_id}, 温度: {weather.get('track_temp', 30)}C")

# 运行校准
calibrator = Calibrator(sim, benchmark)
result = calibrator.calibrate()

print()
print("=== 校准结果 ===")
print(f"校准前误差: {result.error_before:.6f}")
print(f"校准后误差: {result.error_after:.6f}")
print(f"改善百分比: {result.improvement_pct:.2f}%")
print(f"迭代次数: {result.iterations}")
print(f"是否收敛: {result.converged}")
print()
print("=== 校准后参数 ===")
for k, v in result.calibrated_params.items():
    print(f"  {k}: {v:.4f}")
