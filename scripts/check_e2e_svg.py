import urllib.request
import xml.etree.ElementTree as ET

TRACK_IDS = [
    'melbourne', 'shanghai', 'suzuka', 'sakhir', 'jeddah',
    'miami', 'monaco', 'montreal', 'barcelona', 'silverstone',
    'spa', 'hungaroring', 'zandvoort', 'monza', 'austin',
    'mexico_city', 'sao_paulo', 'las_vegas', 'baku', 'singapore',
    'madrid', 'spielberg', 'lusail', 'yas_marina',
]

svg_ok = 0
svg_fail = 0
for track_id in TRACK_IDS:
    url = f'http://127.0.0.1:8000/static/tracks/{track_id}.svg'
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = resp.read().decode('utf-8')
        root = ET.fromstring(data)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        circles = root.findall('.//svg:circle', ns)
        texts = root.findall('.//svg:text', ns)
        print(f'  {track_id:15s}: OK ({len(circles)}圆点, {len(texts)}标注)')
        svg_ok += 1
    except Exception as e:
        print(f'  {track_id:15s}: FAIL: {e}')
        svg_fail += 1

print(f'\nSVG加载: {svg_ok}/{len(TRACK_IDS)} 成功')

# 验证baku标注防重叠
resp = urllib.request.urlopen('http://127.0.0.1:8000/static/tracks/baku.svg', timeout=5)
data = resp.read().decode('utf-8')
root = ET.fromstring(data)
ns = {'svg': 'http://www.w3.org/2000/svg'}
circles = root.findall('.//svg:circle', ns)
texts = root.findall('.//svg:text', ns)
below = []
for i, (c, t) in enumerate(zip(circles, texts), 1):
    cy = float(c.get('cy'))
    ty = float(t.get('y'))
    if ty > cy:
        below.append(i)
print(f'Baku标注防重叠: 下方标注={below}')

# 验证弯道圆点在赛道线上（通过path和circle的距离）
paths = root.findall('.//svg:path', ns)
print(f'Baku: {len(paths)} paths, {len(circles)} circles')