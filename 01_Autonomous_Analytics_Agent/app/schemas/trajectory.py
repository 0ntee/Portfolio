from pydantic import BaseModel, Field
from typing import List
from app.schemas.agent import Action, Observation

class TrajectoryStep(BaseModel):
    step_number: int = Field(description="Порядковый номер шага.")
    action: Action
    observation: Observation

class AgentTrajectory(BaseModel):
    task_description: str = Field(description="Формулировка задачи от пользователя.")
    steps: List[TrajectoryStep] = Field(default_factory=list, description="Цепочка шагов.")
    is_success: bool = Field(description="Удалось ли решить задачу.")
