"""边界条件测试脚本：验证21项参数的min/max/step边界值。

测试内容：
- 每项参数取min值时是否被正确接受
- 每项参数取max值时是否被正确接受
- 每项参数取min-1时是否被正确拒绝
- 每项参数取max+1时是否被正确拒绝
- 每项参数的step是否正确（非对齐值被拒绝）
"""

import sys
import traceback

sys.path.insert(0, ".")

from setup_tuner.domain.setup import (
    ALL_SETUP_FIELDS,
    CarSetup,
    validate_value,
)

passed = 0
failed = 0
errors = []

print("=" * 80)
print("边界条件测试：21项参数 min/max/step 验证")
print("=" * 80)

for field in ALL_SETUP_FIELDS:
    name = field.name
    min_val = field.min_val
    max_val = field.max_val
    step = field.step

    print(f"\n--- {name} (min={min_val}, max={max_val}, step={step}) ---")

    # 1. min值应被接受
    try:
        result = validate_value(name, min_val)
        assert abs(result - min_val) < 1e-6, f"min值返回了不同的值: {result}"
        print(f"  ✅ min={min_val} 被正确接受")
        passed += 1
    except Exception as e:
        print(f"  ❌ min={min_val} 应被接受但被拒绝: {e}")
        failed += 1
        errors.append(f"{name}: min={min_val} 被拒绝 - {e}")

    # 2. max值应被接受
    try:
        result = validate_value(name, max_val)
        assert abs(result - max_val) < 1e-6, f"max值返回了不同的值: {result}"
        print(f"  ✅ max={max_val} 被正确接受")
        passed += 1
    except Exception as e:
        print(f"  ❌ max={max_val} 应被接受但被拒绝: {e}")
        failed += 1
        errors.append(f"{name}: max={max_val} 被拒绝 - {e}")

    # 3. min-step（低于min）应被拒绝
    below_min = min_val - step
    try:
        validate_value(name, below_min)
        print(f"  ❌ min-step={below_min} 应被拒绝但被接受")
        failed += 1
        errors.append(f"{name}: min-step={below_min} 被接受（应拒绝）")
    except ValueError:
        print(f"  ✅ min-step={below_min} 被正确拒绝")
        passed += 1
    except Exception as e:
        print(f"  ❌ min-step={below_min} 拒绝时抛出了非ValueError: {e}")
        failed += 1
        errors.append(f"{name}: min-step={below_min} 异常类型错误 - {e}")

    # 4. max+step（高于max）应被拒绝
    above_max = max_val + step
    try:
        validate_value(name, above_max)
        print(f"  ❌ max+step={above_max} 应被拒绝但被接受")
        failed += 1
        errors.append(f"{name}: max+step={above_max} 被接受（应拒绝）")
    except ValueError:
        print(f"  ✅ max+step={above_max} 被正确拒绝")
        passed += 1
    except Exception as e:
        print(f"  ❌ max+step={above_max} 拒绝时抛出了非ValueError: {e}")
        failed += 1
        errors.append(f"{name}: max+step={above_max} 异常类型错误 - {e}")

    # 5. step对齐验证：min + step/2 应被拒绝（不对齐）
    misaligned = min_val + step / 2
    if step > 0 and misaligned < max_val:
        try:
            validate_value(name, misaligned)
            # 对于step=1.0的情况，min+0.5可能因为浮点精度被接受
            # 但对于step=0.1或0.01的情况，应该被拒绝
            if step < 1.0:
                print(f"  ❌ 不对齐值={misaligned} 应被拒绝但被接受")
                failed += 1
                errors.append(f"{name}: 不对齐值={misaligned} 被接受（应拒绝）")
            else:
                # step=1.0时，min+0.5不对齐，应被拒绝
                print(f"  ❌ 不对齐值={misaligned} 应被拒绝但被接受")
                failed += 1
                errors.append(f"{name}: 不对齐值={misaligned} 被接受（应拒绝）")
        except ValueError:
            print(f"  ✅ 不对齐值={misaligned} 被正确拒绝（step={step}对齐校验生效）")
            passed += 1
        except Exception as e:
            print(f"  ❌ 不对齐值={misaligned} 拒绝时异常类型错误: {e}")
            failed += 1
            errors.append(f"{name}: 不对齐值={misaligned} 异常类型错误 - {e}")

    # 6. min + step 应被接受（下一个合法档位）
    next_step = min_val + step
    if next_step <= max_val:
        try:
            result = validate_value(name, next_step)
            assert abs(result - next_step) < 1e-6
            print(f"  ✅ min+step={next_step} 被正确接受")
            passed += 1
        except Exception as e:
            print(f"  ❌ min+step={next_step} 应被接受但被拒绝: {e}")
            failed += 1
            errors.append(f"{name}: min+step={next_step} 被拒绝 - {e}")

    # 7. max - step 应被接受（上一个合法档位）
    prev_step = max_val - step
    if prev_step >= min_val:
        try:
            result = validate_value(name, prev_step)
            assert abs(result - prev_step) < 1e-6
            print(f"  ✅ max-step={prev_step} 被正确接受")
            passed += 1
        except Exception as e:
            print(f"  ❌ max-step={prev_step} 应被接受但被拒绝: {e}")
            failed += 1
            errors.append(f"{name}: max-step={prev_step} 被拒绝 - {e}")

# 额外验证：CarSetup.validate() 在min/max时通过
print("\n" + "=" * 80)
print("CarSetup 整体校验：min/max 极端配置")
print("=" * 80)

# 所有参数取min值
min_kwargs = {f.name: f.min_val for f in ALL_SETUP_FIELDS}
try:
    cs_min = CarSetup(**min_kwargs)
    cs_min.validate()
    print("✅ 全min值 CarSetup 校验通过")
    passed += 1
except Exception as e:
    print(f"❌ 全min值 CarSetup 校验失败: {e}")
    failed += 1
    errors.append(f"全min值 CarSetup 校验失败 - {e}")

# 所有参数取max值
max_kwargs = {f.name: f.max_val for f in ALL_SETUP_FIELDS}
try:
    cs_max = CarSetup(**max_kwargs)
    cs_max.validate()
    print("✅ 全max值 CarSetup 校验通过")
    passed += 1
except Exception as e:
    print(f"❌ 全max值 CarSetup 校验失败: {e}")
    failed += 1
    errors.append(f"全max值 CarSetup 校验失败 - {e}")

# 所有参数取min-1值（应失败）
min_minus_1 = {f.name: f.min_val - f.step for f in ALL_SETUP_FIELDS}
try:
    cs_below = CarSetup(**min_minus_1)
    cs_below.validate()
    print("❌ 全min-step值 CarSetup 校验应失败但通过了")
    failed += 1
    errors.append("全min-step值 CarSetup 校验应失败但通过了")
except ValueError:
    print("✅ 全min-step值 CarSetup 校验正确拒绝")
    passed += 1
except Exception as e:
    print(f"❌ 全min-step值 CarSetup 拒绝时异常类型错误: {e}")
    failed += 1
    errors.append(f"全min-step值 CarSetup 异常类型错误 - {e}")

# 所有参数取max+1值（应失败）
max_plus_1 = {f.name: f.max_val + f.step for f in ALL_SETUP_FIELDS}
try:
    cs_above = CarSetup(**max_plus_1)
    cs_above.validate()
    print("❌ 全max+step值 CarSetup 校验应失败但通过了")
    failed += 1
    errors.append("全max+step值 CarSetup 校验应失败但通过了")
except ValueError:
    print("✅ 全max+step值 CarSetup 校验正确拒绝")
    passed += 1
except Exception as e:
    print(f"❌ 全max+step值 CarSetup 拒绝时异常类型错误: {e}")
    failed += 1
    errors.append(f"全max+step值 CarSetup 异常类型错误 - {e}")

print("\n" + "=" * 80)
print(f"边界条件测试结果：{passed} passed, {failed} failed")
print("=" * 80)

if errors:
    print("\n失败详情：")
    for err in errors:
        print(f"  - {err}")

sys.exit(0 if failed == 0 else 1)