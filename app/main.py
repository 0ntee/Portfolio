import json
import os
from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.nodes.planner import planner_node
from app.nodes.executor import executor_node
from app.schemas import AgentTrajectory

def reflexion_routing_edge(state: AgentState):
    last_step = state["steps"][-1]
    obs = last_step.observation
    
    if obs.exit_code == 0:
        return "complete"
        
    state["retry_count"] += 1
    
    if state["retry_count"] >= 3:
        print(f"[Graph] Превышен лимит рефлексии ({state['retry_count']}/3). Завершение сессии.")
        return "complete"
        
    return "retry"


def main():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        reflexion_routing_edge,
        {"retry": "planner", "complete": END}
    )
    
    agent_app = workflow.compile()
    
    task = (
        "Проведи глубокое исследование файла `data/ecommerce_transactions.csv`:\n"
        "1. Проверь данные на пропуски (NaN). Если они есть в колонках Age или Purchase_Amount, заполни их медианными значениями.\n"
        "2. Разбей пользователей на 3 возрастные группы (Группы: 'Молодежь' [0-30], 'Средний возраст' [31-55], 'Пожилые' [56-100]).\n"
        "3. Посчитай средний чек (Purchase_Amount) для каждой группы и выведи на экран.\n"
        "4. Построй графики плотности распределения трат (KDE plot или гистограмму) для этих трех групп с помощью matplotlib/seaborn. "
        "Поскольку графического интерфейса нет, ОБЯЗАТЕЛЬНО сохрани этот график в файл `data/outputs/age_spending_distribution.png`. "
        "Выведи текстовые отчеты через print."
    )

    
    initial_state = {
        "task_description": task,
        "current_thought": "",
        "current_code": "",
        "steps": [],
        "retry_count": 0,
        "is_success": False
    }
    
    final_state = agent_app.invoke(initial_state)
    
    success_flag = final_state["steps"][-1].observation.exit_code == 0
    trajectory = AgentTrajectory(
        task_description=task,
        steps=final_state["steps"],
        is_success=success_flag
    )
    
    os.makedirs("data/outputs", exist_ok=True)
    output_path = "data/outputs/trajectory_sample.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(trajectory.model_dump_json(indent=2))

        success_flag = final_state["steps"][-1].observation.exit_code == 0
    trajectory = AgentTrajectory(
        task_description=task,
        steps=final_state["steps"],
        is_success=success_flag
    )
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("data/outputs", exist_ok=True)
    
    output_path = f"data/outputs/trajectory_{timestamp}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(trajectory.model_dump_json(indent=2))
        
    print(f"\n[Graph] Траектория сессии успешно сохранена в уникальный файл: {output_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
