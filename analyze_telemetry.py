"""分析D盘4圈阿布扎比遥测数据，提取关键特征用于调教优化方案设计。"""
import json
import os
from statistics import mean, stdev

DATA_DIR = r"D:\F1TelemetryCollector\dist\data"

def load_lap(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_lap(lap_data, lap_num):
    samples = lap_data.get("samples", [])
    if not samples:
        print(f"圈{lap_num}: 无采样数据")
        return

    setup = lap_data.get("setup", {})
    weather = lap_data.get("weather", {})
    tyre = lap_data.get("tyre", {})
    lap_time = lap_data.get("lap_time_str", "?")
    sample_count = len(samples)

    print(f"\n{'='*70}")
    print(f"圈{lap_num}: {lap_time} | 天气={weather.get('name','?')} | 胎={tyre.get('compound_name','?')} 胎龄={tyre.get('age_laps','?')}")
    print(f"采样点数={sample_count} | 赛道={lap_data.get('track_name','?')}")

    # 调教参数
    print(f"\n--- 调教参数 ---")
    for k, v in setup.items():
        if isinstance(v, float) and (abs(v) < 1e-10 or abs(v) > 1e10):
            print(f"  {k}: {v:.6e} (异常值!)")
        else:
            print(f"  {k}: {v}")

    # 提取各维度的采样序列
    speeds = [s.get("speed", 0) for s in samples]
    throttles = [s.get("throttle", 0) for s in samples]
    brakes = [s.get("brake", 0) for s in samples]
    steers = [s.get("steer", 0) for s in samples]
    gears = [s.get("gear", 0) for s in samples]
    rpms = [s.get("rpm", 0) for s in samples]
    lat_gs = [s.get("lat_g", 0) for s in samples]
    long_gs = [s.get("long_g", 0) for s in samples]
    vert_gs = [s.get("vert_g", 0) for s in samples]
    lap_dists = [s.get("lap_distance", 0) for s in samples]

    # 胎温（4轮）
    fl_surface_temps = [s.get("tyre_surface_temp", [0,0,0,0])[0] for s in samples]
    fr_surface_temps = [s.get("tyre_surface_temp", [0,0,0,0])[1] for s in samples]
    rl_surface_temps = [s.get("tyre_surface_temp", [0,0,0,0])[2] for s in samples]
    rr_surface_temps = [s.get("tyre_surface_temp", [0,0,0,0])[3] for s in samples]

    fl_inner_temps = [s.get("tyre_inner_temp", [0,0,0,0])[0] for s in samples]
    fr_inner_temps = [s.get("tyre_inner_temp", [0,0,0,0])[1] for s in samples]
    rl_inner_temps = [s.get("tyre_inner_temp", [0,0,0,0])[2] for s in samples]
    rr_inner_temps = [s.get("tyre_inner_temp", [0,0,0,0])[3] for s in samples]

    # 胎压（4轮）
    fl_pressures = [s.get("tyre_pressure", [0,0,0,0])[0] for s in samples]
    fr_pressures = [s.get("tyre_pressure", [0,0,0,0])[1] for s in samples]
    rl_pressures = [s.get("tyre_pressure", [0,0,0,0])[2] for s in samples]
    rr_pressures = [s.get("tyre_pressure", [0,0,0,0])[3] for s in samples]

    # 刹车温度（4轮）
    fl_brake_temps = [s.get("brake_temp", [0,0,0,0])[0] for s in samples]
    fr_brake_temps = [s.get("brake_temp", [0,0,0,0])[1] for s in samples]
    rl_brake_temps = [s.get("brake_temp", [0,0,0,0])[2] for s in samples]
    rr_brake_temps = [s.get("brake_temp", [0,0,0,0])[3] for s in samples]

    # 路面类型
    surface_types = [s.get("surface_type", [0,0,0,0]) for s in samples]

    # 全圈统计
    print(f"\n--- 全圈统计 ---")
    print(f"  速度: 均值={mean(speeds):.1f} 峰值={max(speeds):.1f} 最低={min(speeds):.1f}")
    print(f"  油门: 均值={mean(throttles):.3f} >0.5占比={sum(1 for t in throttles if t>0.5)/len(throttles)*100:.1f}%")
    print(f"  刹车: 均值={mean(brakes):.3f} >0.3占比={sum(1 for b in brakes if b>0.3)/len(brakes)*100:.1f}%")
    print(f"  方向盘: 均值={mean(steers):.3f} 最大幅度={max(abs(s) for s in steers):.3f}")
    print(f"  挡位: 均值={mean(gears):.1f} 最高={max(gears)}")
    print(f"  RPM: 均值={mean(rpms):.0f} 峰值={max(rpms)}")
    print(f"  横向G力: 均值={mean(lat_gs):.2f} 峰值={max(abs(g) for g in lat_gs):.2f}")
    print(f"  纵向G力: 均值={mean(long_gs):.2f} 峰值={max(abs(g) for g in long_gs):.2f}")
    print(f"  垂直G力: 均值={mean(vert_gs):.2f} 峰值={max(abs(g) for g in vert_gs):.2f}")

    # 胎温统计
    print(f"\n--- 胎温统计 (表面) ---")
    print(f"  FL: 均值={mean(fl_surface_temps):.1f} 峰值={max(fl_surface_temps):.1f}")
    print(f"  FR: 均值={mean(fr_surface_temps):.1f} 峰值={max(fr_surface_temps):.1f}")
    print(f"  RL: 均值={mean(rl_surface_temps):.1f} 峰值={max(rl_surface_temps):.1f}")
    print(f"  RR: 均值={mean(rr_surface_temps):.1f} 峰值={max(rr_surface_temps):.1f}")
    avg_temps = [mean(fl_surface_temps), mean(fr_surface_temps), mean(rl_surface_temps), mean(rr_surface_temps)]
    print(f"  四轮均值偏差: max-min={max(avg_temps)-min(avg_temps):.1f}°C")

    print(f"\n--- 胎温统计 (内部) ---")
    print(f"  FL: 均值={mean(fl_inner_temps):.1f} 峰值={max(fl_inner_temps):.1f}")
    print(f"  FR: 均值={mean(fr_inner_temps):.1f} 峰值={max(fr_inner_temps):.1f}")
    print(f"  RL: 均值={mean(rl_inner_temps):.1f} 峰值={max(rl_inner_temps):.1f}")
    print(f"  RR: 均值={mean(rr_inner_temps):.1f} 峰值={max(rr_inner_temps):.1f}")

    # 胎压统计
    print(f"\n--- 胎压统计 ---")
    # 过滤异常值（科学记数法）
    def safe_stats(vals, name):
        clean = [v for v in vals if 15 < v < 35]
        if not clean:
            print(f"  {name}: 全部异常值，无法统计")
            return
        print(f"  {name}: 均值={mean(clean):.2f} 峰值={max(clean):.2f} 最低={min(clean):.2f} 标准差={stdev(clean) if len(clean)>1 else 0:.3f}")

    safe_stats(fl_pressures, "FL")
    safe_stats(fr_pressures, "FR")
    safe_stats(rl_pressures, "RL")
    safe_stats(rr_pressures, "RR")

    # 刹车温度统计
    print(f"\n--- 刹车温度统计 ---")
    print(f"  FL: 均值={mean(fl_brake_temps):.0f} 峰值={max(fl_brake_temps):.0f}")
    print(f"  FR: 均值={mean(fr_brake_temps):.0f} 峰值={max(fr_brake_temps):.0f}")
    print(f"  RL: 均值={mean(rl_brake_temps):.0f} 峰值={max(rl_brake_temps):.0f}")
    print(f"  RR: 均值={mean(rr_brake_temps):.0f} 峰值={max(rr_brake_temps):.0f}")

    # 分段分析：用lap_distance识别直道和弯道
    print(f"\n--- 分段分析 (按lap_distance) ---")
    if lap_dists:
        total_dist = max(lap_dists)
        print(f"  总距离: {total_dist:.1f}m")

        # 简单分段：速度<50视为弯道，速度>200视为直道
        corner_samples = [(i, samples[i]) for i in range(len(samples)) if speeds[i] < 80]
        straight_samples = [(i, samples[i]) for i in range(len(samples)) if speeds[i] > 200]

        if corner_samples:
            corner_speeds = [s[1].get("speed", 0) for s in corner_samples]
            corner_lat_gs = [abs(s[1].get("lat_g", 0)) for s in corner_samples]
            corner_brakes = [s[1].get("brake", 0) for s in corner_samples]
            print(f"  弯道段(速度<80): {len(corner_samples)}个采样点")
            print(f"    速度: 均值={mean(corner_speeds):.1f} 最低={min(corner_speeds):.1f}")
            print(f"    横向G力峰值: {max(corner_lat_gs):.2f}")
            print(f"    刹车占比: {sum(1 for b in corner_brakes if b>0.3)/len(corner_brakes)*100:.1f}%")

        if straight_samples:
            straight_speeds = [s[1].get("speed", 0) for s in straight_samples]
            straight_throttles = [s[1].get("throttle", 0) for s in straight_samples]
            print(f"  直道段(速度>200): {len(straight_samples)}个采样点")
            print(f"    速度: 均值={mean(straight_speeds):.1f} 极速={max(straight_speeds):.1f}")
            print(f"    油门均值: {mean(straight_throttles):.3f}")

        # 识别刹车点（从高油门突然切到刹车）
        brake_points = []
        for i in range(1, len(samples)):
            if throttles[i-1] > 0.5 and brakes[i] > 0.3 and throttles[i] < 0.2:
                brake_points.append({
                    "dist": lap_dists[i],
                    "speed": speeds[i],
                    "brake": brakes[i],
                })
        if brake_points:
            print(f"  识别到{len(brake_points)}个刹车点:")
            for bp in brake_points[:8]:
                print(f"    距离={bp['dist']:.0f}m 速度={bp['speed']:.0f}km/h 刹车={bp['brake']:.2f}")

        # 识别出弯点（从刹车/低油门突然切到高油门）
        exit_points = []
        for i in range(1, len(samples)):
            if throttles[i] > 0.5 and throttles[i-1] < 0.2 and brakes[i-1] < 0.1:
                exit_points.append({
                    "dist": lap_dists[i],
                    "speed": speeds[i],
                    "throttle": throttles[i],
                    "lat_g": lat_gs[i],
                })
        if exit_points:
            print(f"  识别到{len(exit_points)}个出弯油门点:")
            for ep in exit_points[:8]:
                print(f"    距离={ep['dist']:.0f}m 速度={ep['speed']:.0f}km/h 油门={ep['throttle']:.2f} 横G={ep['lat_g']:.2f}")

    # 路面类型分析
    print(f"\n--- 路面类型 ---")
    all_surface_types = set()
    for st in surface_types:
        for v in st:
            all_surface_types.add(v)
    print(f"  出现的路面类型值: {sorted(all_surface_types)}")

    # DRS使用
    drs_values = [s.get("drs", 0) for s in samples]
    drs_active = sum(1 for d in drs_values if d > 0)
    print(f"\n--- DRS ---")
    print(f"  DRS激活采样点: {drs_active}/{sample_count} ({drs_active/sample_count*100:.1f}%)")

    # 驾驶风格指标
    driver_style = lap_data.get("driver_style", {})
    print(f"\n--- 驾驶风格 ---")
    for k, v in driver_style.items():
        print(f"  {k}: {v}")


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("lap_") and f.endswith(".json")])
    print(f"找到{len(files)}个遥测文件")

    for filepath in files:
        full_path = os.path.join(DATA_DIR, filepath)
        lap_data = load_lap(full_path)
        lap_num = lap_data.get("lap_number", 0)
        analyze_lap(lap_data, lap_num)


if __name__ == "__main__":
    main()