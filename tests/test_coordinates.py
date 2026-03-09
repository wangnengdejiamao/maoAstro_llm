#!/usr/bin/env python3
"""
测试其他天体坐标
"""

# 测试天体列表
TEST_TARGETS = [
    ("EV UMa", 13.1316, 53.8585, "激变变星"),
    ("Sirius", 101.2875, -16.7161, "最亮恒星"),
    ("M31", 10.6847, 41.2687, "仙女座星系"),
    ("M33", 23.4621, 30.6602, "三角座星系"),
    ("Vega", 279.2347, 38.7837, "织女星"),
]

print("=" * 70)
print("🌟 测试天体坐标列表")
print("=" * 70)
print()

for i, (name, ra, dec, desc) in enumerate(TEST_TARGETS, 1):
    print(f"{i}. {name}")
    print(f"   坐标: RA={ra}, DEC={dec}")
    print(f"   描述: {desc}")
    print(f"   测试命令: python query_ra_dec.py {ra} {dec} --no-ai")
    print()

print("=" * 70)
print("💡 提示: 使用 --no-ai 参数可以只查询数据，不调用 AI 分析")
print("=" * 70)
