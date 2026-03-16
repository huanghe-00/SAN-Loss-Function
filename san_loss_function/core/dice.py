# san_loss_function/core/dice.py（无需修改）
import random
from pydantic import BaseModel, Field
from typing import Optional

class DiceResult(BaseModel):
    raw_roll: int = Field(..., ge=1, le=100)
    skill_value: Optional[int] = None
    is_success: bool = False
    level: str = "unknown"  # critical/extreme/hard/regular/failure/fumble
    
    def __str__(self):
        status = "✅" if self.is_success else "❌"
        return f"{status} {self.raw_roll}/100 [{self.level}]"

class DiceRoller:
    @staticmethod
    def skill_check(skill_value: int) -> DiceResult:
        roll = random.randint(1, 100)
        is_success = roll <= skill_value
        
        if roll == 1:
            level = "critical"
        elif roll <= skill_value // 5:
            level = "extreme"
        elif roll <= skill_value // 2:
            level = "hard"
        elif is_success:
            level = "regular"
        elif roll >= 96:
            level = "fumble"
        else:
            level = "failure"
            
        return DiceResult(raw_roll=roll, skill_value=skill_value, 
                         is_success=is_success, level=level)