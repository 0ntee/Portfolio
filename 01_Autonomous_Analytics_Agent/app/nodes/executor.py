import os
from app.state import AgentState
from app.runtime import execute_file_safely
from app.schemas import TrajectoryStep, Action

def executor_node(state: AgentState) -> dict:
    current_step_num = len(state["steps"]) + 1
    print(f"[Executor] Шаг №{current_step_num}. Подготовка к записи скрипта на диск...")
    
    code_to_run = state["current_code"]
    
    os.makedirs("data/agent_scripts", exist_ok=True)
    
    file_path = f"data/agent_scripts/step_{current_step_num}.py"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_to_run)
    print(f"[Executor] Скрипт успешно сохранен по пути: {file_path}")
    
    print("[Executor] Запуск файла в изолированном подпроцессе ОС...")
    observation = execute_file_safely(file_path, timeout_seconds=7)
    
    executed_action = Action(thought=state["current_thought"], code=code_to_run)
    
    new_step = TrajectoryStep(
        step_number=current_step_num,
        action=executed_action,
        observation=observation
    )
    
    new_retry_count = state["retry_count"]
    if observation.exit_code != 0:
        new_retry_count += 1
        print(f"[Executor] Файл завершился со сбоем (Exit Code: {observation.exit_code})")
        print(f"[Executor] Лог ошибки из файла:\n{observation.stderr.strip()}")
    else:
        print("[Executor]  Файл успешно выполнен! (Exit Code: 0)")
        if observation.stdout:
            print(f"[Executor] Вывод, напечатанный файлом:\n{observation.stdout.strip()}")
            
    return {"steps": [new_step], "retry_count": new_retry_count}
