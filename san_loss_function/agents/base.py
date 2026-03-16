#!/usr/bin/env python3
"""
SAN-Loss-Function: 适配阿里百炼 Coding Plan
注意：必须使用 https://coding.dashscope.aliyuncs.com/v1 端点
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from san_loss_function.core.dice import DiceRoller, DiceResult

load_dotenv()


class SimpleKP:
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        # Coding Plan 专属端点！
        base_url = os.getenv("OPENAI_API_BASE", "https://coding.dashscope.aliyuncs.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "qwen3.5-plus")
        
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY")
        
        print(f"🌐 Coding Plan 接入")
        print(f"🤖 模型: {model_name}")
        print(f"🔗 端点: coding.dashscope.aliyuncs.com/v1")
        
        # 适配 Coding Plan 的参数
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            max_tokens=1000,
            # Coding Plan 需要显式设置 stream_options 或 model_kwargs
            model_kwargs={
                "stream": False,  # 确保非流式响应
            }
        )
        
        self.roller = DiceRoller()
        
        # 测试连接
        self._verify_connection()
    
    def _verify_connection(self):
        """验证 Coding Plan 连接"""
        try:
            # 简单测试调用
            test_resp = self.llm.invoke("测试")
            print(f"✅ 连接成功: {test_resp.content[:20]}...")
        except Exception as e:
            if "choices" in str(e):
                raise ValueError(
                    f"\n❌ Coding Plan 响应解析失败！\n"
                    f"请确认：\n"
                    f"1. Base URL 必须是: https://coding.dashscope.aliyuncs.com/v1\n"
                    f"2. 模型名必须是套餐支持的: qwen3.5-plus / qwen3-max-2026-01-23 / kimi-k2.5 / glm-5\n"
                    f"3. API Key 是 Coding Plan 的应用 Key（以 sk-sp- 开头）\n"
                    f"当前配置: {self.llm.model_name} @ {self.llm.openai_api_base}"
                ) from e
            raise
    
    def narrate_check(self, action: str, skill_name: str, skill_value: int) -> str:
        result = self.roller.skill_check(skill_value)
        
        # 中文优化 Prompt（适配 qwen/kimi/glm）
        system_prompt = f"""你是Call of Cthulhu 7th Edition守秘人。
场景：1924年阿卡姆，阴雨连绵的图书馆。
检定：{skill_name}({skill_value}%)，骰子{result.raw_roll}，结果{result.level}。
要求：150字内，中文，悬疑恐怖氛围，{'成功给线索' if result.is_success else '失败遇阻碍'}。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=action)
        ]
        
        try:
            resp = self.llm.invoke(messages)
            return f"🎲 {result}\n📜 {resp.content}"
        except Exception as e:
            return f"🎲 {result}\n❌ 生成失败: {str(e)[:80]}"


def quick_test():
    print("="*50)
    print("SAN-Loss-Function x 阿里百炼 Coding Plan")
    print("="*50)
    
    # 配置检查
    print(f"API Key: {os.getenv('DASHSCOPE_API_KEY', '未设置')[:10]}...")
    print(f"Base URL: {os.getenv('OPENAI_API_BASE', '未设置')}")
    print(f"Model: {os.getenv('OPENAI_MODEL_NAME', '未设置')}")
    print("-"*50)
    
    kp = SimpleKP()
    print(kp.narrate_check("查找《死灵之书》借阅记录", "Library Use", 60))


if __name__ == "__main__":
    quick_test()