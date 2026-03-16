# examples/week1_optimized_demo.py
#!/usr/bin/env python3
"""
Week 1 按次计费优化版Demo
特点：每回合仅1次API调用，最大化利用qwen3.5-plus长上下文
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from san_loss_function.agents.batch_kp import BatchKPAgent

def main():
    print("🎲 SAN-Loss-Function Week 1 (Coding Plan优化版)")
    print("💰 计费模式：按调用次数（每回合1次API调用）")
    print("=" * 60)
    
    try:
        kp = BatchKPAgent()
        
        print("\n【开场】")
        print(f"场景：{kp.world_state['scene']['location']}")
        print(f"氛围：{kp.world_state['scene']['atmosphere']}")
        print(f"你的技能：{', '.join(kp.world_state['investigator']['skills'].keys())}")
        print(f"初始物品：{', '.join(kp.world_state['investigator']['inventory'])}")
        print("\n【剧情开始】")
        print("你站在档案室门口，手中握着父亲留下的神秘钥匙。")
        print("远处传来雷声，档案柜在闪电中投下扭曲的影子...")
        
        # 交互循环
        while True:
            try:
                user_input = input(f"\n[第{kp.turn_count+1}轮] 调查员 > ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit', '结束']:
                    print(f"\n{kp.get_session_summary()}")
                    print("\nKeeper：调查中止，但真相仍在黑暗中等待...")
                    break
                
                if user_input.lower() in ['status', '状态']:
                    print(f"\n当前状态：SAN {kp.world_state['investigator']['san']}")
                    print(f"已发现线索：{kp.world_state['investigator']['clues_found']}")
                    continue
                
                # 处理回合（单次API调用）
                result = kp.process_turn(user_input)
                
                # 展示结果
                print(f"\n{'─' * 60}")
                print(f"🎲 检定：{result['dice']}")
                print(f"📖 {result['narrative']}")
                print(f"{'─' * 60}")
                
                if result.get('clue'):
                    print(f"🔍 发现线索：{result['clue']}")
                if result.get('san_check'):
                    print(f"⚠️  需要进行理智检定！（SAN Loss: {result.get('san_loss', '??')}）")
                
                print(f"\n💡 你可以：")
                for i, opt in enumerate(result['options'], 1):
                    print(f"   {i}. {opt}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{kp.get_session_summary()}")
                break
            except Exception as e:
                print(f"\n❌ 错误：{e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        print("\n请检查：")
        print("1. .env 文件中的 DASHSCOPE_API_KEY 和 OPENAI_API_BASE")
        print("2. 网络连接（Coding Plan端点：coding.dashscope.aliyuncs.com）")

if __name__ == "__main__":
    main()