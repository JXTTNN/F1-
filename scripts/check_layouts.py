"""检查24个layoutId是否都存在于参考项目中。"""
import json
import urllib.request

url = "https://api.github.com/repos/julesr0y/f1-circuits-svg/contents/circuits/minimal/white-outline"
data = json.loads(urllib.request.urlopen(url).read())
files = [item["name"] for item in data if item["type"] == "file"]

needed = [
    "melbourne-2", "shanghai-1", "suzuka-2", "bahrain-1", "jeddah-1",
    "miami-1", "montreal-6", "monaco-6", "catalunya-6", "spielberg-3",
    "silverstone-8", "spa-francorchamps-4", "hungaroring-3", "zandvoort-5",
    "monza-7", "madring-1", "baku-1", "marina-bay-4", "austin-1",
    "mexico-city-3", "interlagos-2", "las-vegas-1", "lusail-1", "yas-marina-2",
]

missing = []
for n in needed:
    fname = n + ".svg"
    if fname in files:
        print(f"  {n}: FOUND")
    else:
        print(f"  {n}: MISSING !!!")
        missing.append(n)

print(f"\nTotal: {len(needed)} needed, {len(missing)} missing")
if missing:
    print(f"Missing: {missing}")