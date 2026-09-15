import json

with open("dist/10round_check_report.json", encoding="utf-8") as f:
    r = json.load(f)

print("=" * 70)
print("F1OPT 10轮递进式全量检查报告")
print("=" * 70)
print()
print(f"总计: {r['summary']['total_checks']}项检查, 通过{r['summary']['total_passed']}, 失败{r['summary']['total_failed']}")
print(f"最终结果: {'✅ 全部通过' if r['summary']['all_passed'] else '❌ 存在失败'}")
print()
print("-" * 70)
for rd in r["rounds"]:
    status = "✅" if rd["failed"] == 0 else "❌"
    print(f"  {status} 第{rd['round']}轮({rd['name']}): {rd['total']}项, 通过{rd['passed']}, 失败{rd['failed']}, 耗时{rd['duration']}s")
print("-" * 70)
print()

# 验证递增约束
print("递增约束验证:")
for i in range(1, len(r["rounds"])):
    prev = r["rounds"][i-1]
    curr = r["rounds"][i]
    ok = curr["total"] >= prev["total"]
    print(f"  第{prev['round']}轮({prev['total']}项) → 第{curr['round']}轮({curr['total']}项): {'✅' if ok else '❌'}")