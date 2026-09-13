"""快速验证资源路径修复是否正确。"""
import subprocess
import sys
import time

import requests

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "setup_tuner.app:create_app", "--factory",
     "--host", "127.0.0.1", "--port", "8765"],
    cwd="D:/F1OPT-Test",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)

try:
    r_root = requests.get("http://127.0.0.1:8765/", timeout=5)
    r_css = requests.get("http://127.0.0.1:8765/static/style.css", timeout=5)
    r_js = requests.get("http://127.0.0.1:8765/static/app.js", timeout=5)
    r_svg = requests.get("http://127.0.0.1:8765/static/tracks/suzuka.svg", timeout=5)

    print("=== HTTP STATUS ===")
    print(f"GET /                        => {r_root.status_code} | has <html>: {'<html' in r_root.text}")
    print(f"GET /static/style.css        => {r_css.status_code} | len: {len(r_css.text)} | has body: {'body' in r_css.text}")
    print(f"GET /static/app.js           => {r_js.status_code} | len: {len(r_js.text)} | has SVG_PREFIX: {'SVG_PREFIX' in r_js.text}")
    print(f"GET /static/tracks/suzuka.svg=> {r_svg.status_code} | len: {len(r_svg.text)} | has <svg>: {'<svg' in r_svg.text}")

    print()
    print("=== PATH REFERENCES ===")
    print(f"HTML refs /static/style.css  : {'/static/style.css' in r_root.text}")
    print(f"HTML refs /static/app.js     : {'/static/app.js' in r_root.text}")
    print(f"JS SVG_PREFIX=/static/tracks/: {'/static/tracks/' in r_js.text}")
    old_prefix_marker = 'SVG_PREFIX = "/tracks/"'
    print(f"JS has OLD /tracks/          : {old_prefix_marker in r_js.text}")

    print()
    print("=== VERDICT ===")
    checks = [
        ("Root returns HTML", r_root.status_code == 200 and "<html" in r_root.text),
        ("CSS 200", r_css.status_code == 200),
        ("JS 200", r_js.status_code == 200),
        ("SVG 200", r_svg.status_code == 200),
        ("HTML refs /static/style.css", "/static/style.css" in r_root.text),
        ("HTML refs /static/app.js", "/static/app.js" in r_root.text),
        ("JS SVG_PREFIX=/static/tracks/", "/static/tracks/" in r_js.text),
        ("JS no OLD /tracks/", 'SVG_PREFIX = "/tracks/"' not in r_js.text),
    ]
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status} - {name}")
        if not ok:
            all_ok = False
    print()
    print(f"ALL CHECKS: {'PASS' if all_ok else 'FAIL'}")
finally:
    proc.terminate()
    proc.wait()