import urllib.request

urls = [
    'http://127.0.0.1:8000/api/v1/tracks/baku/svg',
    'http://127.0.0.1:8000/static/tracks/baku.svg',
    'http://127.0.0.1:8000/tracks/baku',
    'http://127.0.0.1:8000/api/v1/tracks/baku',
]
for url in urls:
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = resp.read()
        ct = resp.headers.get('content-type', '?')
        print(f'{url}: OK ({len(data)} bytes, content-type={ct})')
    except Exception as e:
        print(f'{url}: FAIL ({e})')