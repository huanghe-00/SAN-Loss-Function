#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from san_loss_function.agents.base import SimpleKP

def main():
    print("🎲 SAN-Loss-Function Week 1")
    print("🌐  powered by 阿里百炼 Coding Plan")
    print("=" * 50)
    
    try:
        kp = SimpleKP()
        
        print("\n【场景】密斯卡托尼克大学图书馆")
        print("【调查员】寻找《死灵之书》线索")
        print("【技能】图书馆使用 (60%)\n")
        
        result = kp.narrate_check(
            "仔细翻阅1923年的借阅登记册，寻找可疑记录",
            "Library Use",
            60
        )
        print(result)
        
        print("\n" + "=" * 50)
        print("✅ Week 1验证完成！")
        print("📝 接下来：尝试修改 .env 中的模型为 'qwen-turbo' 对比速度")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查：")
        print("1. .env 文件是否配置正确")
        print("2. 是否在 (san-loss) conda 环境中")

if __name__ == "__main__":
    main()
