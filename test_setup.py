#!/usr/bin/env python3
"""
验证阿里百炼Coding Plan连接
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def test_connection():
    print("🔍 检查百炼配置...")
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL_NAME", "qwen-max")
    
    if not api_key:
        print("❌ 未找到 DASHSCOPE_API_KEY")
        return False
    
    print(f"✅ API Key: {api_key[:8]}...")
    print(f"✅ Base URL: {base_url}")
    print(f"✅ Model: {model}")
    
    print("\n🌐 测试API连接...")
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7
        )
        
        # 简单测试
        messages = [
            SystemMessage(content="你是COC跑团主持人，简短回答。"),
            HumanMessage(content="描述一个阴森的图书馆")
        ]
        
        response = llm.invoke(messages)
        print(f"\n✅ 连接成功！")
        print(f"📝 响应示例：{response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"\n❌ 连接失败：{e}")
        print("\n排查建议：")
        print("1. 确认API Key是Coding Plan的'应用API Key'，不是个人Key")
        print("2. 确认Base URL是完整的Endpoint地址（以/v1结尾）")
        print("3. 确认模型名称在百炼中已部署（qwen-max/qwen-turbo等）")
        return False

if __name__ == "__main__":
    test_connection()