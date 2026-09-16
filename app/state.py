from typing import TypedDict, List, Annotated
from operator import add
from app.schemas import TrajectoryStep

class AgentState(TypedDict):
    task_description: str
    current_thought: str
    current_code: str
    steps: Annotated[List[TrajectoryStep], add]
    retry_count: int
    is_success: bool
