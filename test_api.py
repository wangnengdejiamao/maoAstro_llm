#!/usr/bin/env python3
"""测试API可用性"""

from openai import OpenAI

API_KEYS = [
    # 请在此处填入你的API Keys
    # 可以从环境变量读取: os.getenv("MOONSHOT_API_KEYS", "").split(",")
]

BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "kimi-k2-07132-preview"

print("测试API可用性...\n")

for name, api_key in API_KEYS:
    print(f"测试 {name}...")
    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Hello, test."}
            ],
            max_tokens=10,
            timeout=30
        )
        print(f"  ✓ {name}: 可用")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
    print()
