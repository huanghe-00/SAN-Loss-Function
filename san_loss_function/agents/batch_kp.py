# san_loss_function/agents/batch_kp.py
"""
SAN-Loss-Function: Batch-Optimized KP Agent
针对按调用次数计费优化：单次调用完成检定判定+叙事生成+状态更新
充分利用qwen3.5-plus长上下文能力（支持32K+ tokens）
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from san_loss_function.core.dice import DiceRoller, DiceResult

load_dotenv()


class BatchKPAgent:
    """
    Week 1 优化版KP：单次API调用完成完整回合处理
    - 输入：玩家行动 + 完整游戏状态（长上下文）
    - 输出：检定结果 + 叙事 + 状态更新建议
    - 计费：每回合仅1次调用
    """
    
    def __init__(self, economy_mode=False):
        self.economy_mode = economy_mode
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "qwen3.5-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://coding.dashscope.aliyuncs.com/v1"),
            temperature=0.8,  # 略高增强叙事多样性
            max_tokens=2000,  # 按次计费，可以放心用长输出
        )
        self.roller = DiceRoller()
        
        # 初始化世界状态（长上下文基础）
        self.world_state = {
            "scene": {
                "location": "密斯卡托尼克大学图书馆地下档案室",
                "time": "1924年10月15日，下午3点47分",
                "weather": "暴雨，雷声轰鸣",
                "atmosphere": "霉味混合着古老纸张的腐朽气息，灯光闪烁不定",
                "danger_level": 0  # 动态追踪
            },
            "investigator": {
                "name": "未命名调查员",
                "skills": {
                    "图书馆使用": 60,
                    "侦查": 50,
                    "话术": 45,
                    "心理学": 40,
                    "格斗": 30
                },
                "san": 60,
                "hp": 12,
                "inventory": ["手电筒", "笔记本", "父亲留下的神秘钥匙"],
                "mental_state": "稳定",
                "clues_found": []  # 已发现线索追踪
            },
            "npcs": {
                "librarian_william": {
                    "name": "威廉·阿米蒂奇",
                    "role": "图书馆管理员",
                    "attitude": "怀疑",
                    "secret": "知道地下保险库的真正用途",
                    "location": "一楼服务台"
                }
            },
            "plot": {
                "current_objective": "找到《死灵之书》的借阅记录",
                "key_clues": ["V-17保险库", "三位失踪教授", "蛇形徽章"],
                "timeline": []  # 事件时间线
            }
        }
        
        self.history: List[Dict] = []  # 完整历史记录（长上下文）
        self.turn_count = 0
    
    def _build_system_context(self) -> str:
        """构建超上下文（充分利用按次计费的长文本优势）"""
        return f"""【SAN-Loss-Function 守秘人系统】
你是Call of Cthulhu 7th Edition的资深守秘人(Keeper)，主持这场1920年代的恐怖调查。

=== 当前世界状态 ===
地点：{self.world_state['scene']['location']}
时间：{self.world_state['scene']['time']}
环境：{self.world_state['scene']['weather']}，{self.world_state['scene']['atmosphere']}

调查员状态：
- SAN值：{self.world_state['investigator']['san']}/60
- 精神状态：{self.world_state['investigator']['mental_state']}
- 持有物品：{', '.join(self.world_state['investigator']['inventory'])}
- 已发现线索：{self.world_state['investigator']['clues_found'] if self.world_state['investigator']['clues_found'] else '无'}

NPC状态：
{json.dumps(self.world_state['npcs'], ensure_ascii=False, indent=2)}

主线进展：
- 当前目标：{self.world_state['plot']['current_objective']}
- 关键线索池：{', '.join(self.world_state['plot']['key_clues'])}

=== COC 7th版规则速查 ===
检定机制：1d100，≤技能值成功
成功等级：1（大成功），≤技能/5（极限），≤技能/2（困难），≤技能（通常），96+且失败（大失败）
SAN损失：目睹恐怖场景时损失，0/1d3/1d6/1d10/1d20/1d100

=== 叙事指南 ===
1. 保持1920年代美国新英格兰地区氛围（阿卡姆、波士顿、普罗维登斯）
2. 根据检定成功等级差异化信息：
   - Extreme（≤1/5）：发现隐藏细节或深层联系
   - Hard（≤1/2）：高效完成，获得额外提示
   - Regular：标准信息
   - Fail：遭遇阻碍或错误信息
   - Fumble（96+）：厄运、引来注意、物品损坏或轻微身体伤害
3. 大失败时必须引入压力：NPC敌意增加、环境恶化、或暗示神话存在
4. 每回合结尾提出2-3个开放式选择，推动玩家主动决策
5. 严格控制：200-300字，避免替玩家做决定

=== 输出格式要求（JSON）===
你必须输出严格JSON格式：
{{
    "dice_result": {{"roll": 数字, "skill": "技能名", "level": "成功等级", "success": true/false}},
    "narrative": "中文叙事文本（200-300字）",
    "world_changes": {{"npcs": {{}}, "scene": {{}}, "investigator": {{}}}},  // 建议的状态更新
    "san_check_required": false,  // 是否需要理智检定
    "san_loss": "0/1d3",  // 如果有SAN损失，标注骰子
    "next_options": ["选项1", "选项2", "选项3"],  // 给玩家的选择
    "clue_revealed": null  // 本回合发现的新线索，如果没有填null
}}
注意：仅输出JSON，不要任何markdown代码块标记，不要任何解释性文字。"""

    def process_turn(self, player_action: str, skill_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        核心回合处理：1次API调用完成全部逻辑
        - 本地执行骰子（不消耗API）
        - 单次LLM调用生成：叙事+状态更新+下一步建议
        """
        self.turn_count += 1
        
        # 1. 本地判定技能（不调用API）
        if skill_hint and skill_hint in self.world_state['investigator']['skills']:
            skill_name = skill_hint
            skill_value = self.world_state['investigator']['skills'][skill_hint]
        else:
            # 从行动文本推断技能（简化版，Week 2用LLM意图识别）
            skill_name = self._infer_skill(player_action)
            skill_value = self.world_state['investigator']['skills'].get(skill_name, 25)
        
        dice_result = self.roller.skill_check(skill_value)
        
        # 2. 单次LLM调用：携带完整上下文，生成全部内容
        system_ctx = self._build_system_context()
        user_prompt = f"""第{self.turn_count}轮
        
玩家行动：{player_action}
意图技能：{skill_name}（{skill_value}%）
骰子结果：{dice_result.raw_roll}
检定等级：{dice_result.level}
是否成功：{'成功' if dice_result.is_success else '失败'}

请根据上述检定结果，生成JSON格式回复。"""

        try:
            # 单次API调用（计费点）
            messages = [
                SystemMessage(content=system_ctx),
                HumanMessage(content=user_prompt)
            ]
            
            print(f"🎲 本地检定：{dice_result}（技能：{skill_name}）")
            print(f"🌐 调用Coding Plan生成叙事（第{self.turn_count}次API调用）...")
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # 解析JSON（处理可能的格式问题）
            # 清理可能的markdown代码块
            clean_content = content.replace("```json", "").replace("```", "").strip()
            result_data = json.loads(clean_content)
            
            # 3. 更新本地状态（基于LLM建议）
            self._update_state(result_data, player_action, dice_result)
            
            return {
                "turn": self.turn_count,
                "dice": dice_result,
                "narrative": result_data.get("narrative", "叙事生成失败"),
                "options": result_data.get("next_options", ["继续调查"]),
                "clue": result_data.get("clue_revealed"),
                "san_check": result_data.get("san_check_required", False),
                "raw_json": result_data
            }
            
        except json.JSONDecodeError as e:
            # 容错：如果JSON解析失败，返回原始文本
            return {
                "turn": self.turn_count,
                "dice": dice_result,
                "narrative": f"【原始响应】{content[:200]}...",
                "options": ["重试该行动", "改变策略"],
                "error": str(e)
            }
        except Exception as e:
            return {
                "turn": self.turn_count,
                "error": str(e),
                "narrative": "系统遭遇神话干扰..."
            }
    
    def _infer_skill(self, action: str) -> str:
        """简易技能推断（本地处理，0 API成本）"""
        action_lower = action.lower()
        keywords = {
            "图书馆使用": ["找", "查", "书", "档案", "记录", "目录", "搜索"],
            "侦查": ["看", "观察", "找", "注意", "检查", "发现", "细节"],
            "话术": ["说", "谈", "问", "骗", "说服", "聊", "打听"],
            "心理学": ["读", "心理", "想", "动机", "分析", "判断"],
            "格斗": ["打", "攻击", "防御", "踢", "枪", "刀", "战斗"]
        }
        
        for skill, words in keywords.items():
            if any(w in action_lower for w in words):
                return skill
        return "侦查"  # 默认
    
    def _update_state(self, result_data: Dict, action: str, dice: DiceResult):
        """根据回合结果更新世界状态（本地操作，0成本）"""
        changes = result_data.get("world_changes", {})
        
        # 更新NPC状态
        if "npcs" in changes:
            self.world_state["npcs"].update(changes["npcs"])
        
        # 更新场景
        if "scene" in changes:
            self.world_state["scene"].update(changes["scene"])
        
        # 更新调查员
        if "investigator" in changes:
            self.world_state["investigator"].update(changes["investigator"])
        
        # 记录线索
        clue = result_data.get("clue_revealed")
        if clue:
            self.world_state["investigator"]["clues_found"].append(clue)
        
        # 记录历史（用于长上下文）
        self.history.append({
            "turn": self.turn_count,
            "action": action,
            "dice": dice.dict(),
            "narrative": result_data.get("narrative", "")[:100],  # 摘要存储
            "clue": clue
        })
        
        # 如果超过20轮，清理早期历史但保留关键线索（上下文窗口管理）
        if len(self.history) > 20:
            # 保留线索发现记录，清理详细叙事
            self.history = self.history[-10:]  # 保留最近10轮
    
    def get_session_summary(self) -> str:
        """生成会话摘要"""
        return f"""
【SAN-Loss-Function 会话摘要】
总轮次：{self.turn_count}
API调用次数：{self.turn_count}（按次计费）
当前地点：{self.world_state['scene']['location']}
SAN值：{self.world_state['investigator']['san']}/60
发现线索：{len(self.world_state['investigator']['clues_found'])}条
关键NPC：{', '.join(self.world_state['npcs'].keys())}
"""


# 便捷入口
KPAgent = BatchKPAgent