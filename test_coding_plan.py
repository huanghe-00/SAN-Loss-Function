#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI  # 直接使用官方SDK测试

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("OPENAI_API_BASE", "https://coding.dashscope.aliyuncs.com/v1")
model = os.getenv("OPENAI_MODEL_NAME", "qwen3.5-plus")

print(f"🔍 诊断 Coding Plan 连接")
print(f"Key: {api_key[:15]}... ({len(api_key)}字符)")
print(f"URL: {base_url}")
print(f"Model: {model}")

# 检查模型是否在支持列表
supported = [
    "qwen3.5-plus", "qwen3-max-2026-01-23", "qwen3-coder-next", "qwen3-coder-plus",
    "glm-5", "glm-4.7", "kimi-k2.5", "MiniMax-M2.5"
]

if model not in supported:
    print(f"\n⚠️ 警告: {model} 可能不在你的套餐支持列表中！")
    print(f"支持的模型: {', '.join(supported)}")

print(f"\n🌐 测试 API 连接...")

try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好，测试连接"}],
        temperature=0.7
    )
    
    print(f"✅ 连接成功!")
    print(f"📄 响应内容: {resp.choices[0].message.content}")
    print(f"🔢 消耗token: {resp.usage.total_tokens if resp.usage else 'N/A'}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print(f"\n排查清单:")
    print(f"1. URL 是否为 https://coding.dashscope.aliyuncs.com/v1 ?")
    print(f"2. 模型名称是否复制正确（区分大小写）?")
    print(f"3. Key 是否以 sk-sp- 开头（Coding Plan 应用Key）?")