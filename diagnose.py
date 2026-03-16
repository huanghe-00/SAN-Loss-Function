#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model = os.getenv("OPENAI_MODEL_NAME")

print(f"模型: {model}")
print(f"端点: {base_url}")

# 根据模型推荐端点
if "kimi" in model.lower():
    expected = "api.moonshot.cn"
    if expected not in base_url:
        print(f"⚠️ 警告: 使用{model}但端点是{base_url}")
        print(f"✅ 建议: 改为 https://{expected}/v1")
elif "qwen" in model.lower() or "deepseek" in model.lower():
    expected = "bailian"
    if expected not in base_url:
        print(f"⚠️ 警告: 使用{model}但端点似乎不是百炼")

# 测试调用
try:
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    resp = llm.invoke("Hello")
    print(f"✅ 连接成功: {resp.content[:20]}...")
except Exception as e:
    print(f"❌ 错误: {e}")