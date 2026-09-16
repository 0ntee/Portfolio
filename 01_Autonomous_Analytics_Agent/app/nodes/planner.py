from llama_cpp import Llama
from app.state import AgentState

llm = Llama(model_path="/tmp/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", n_ctx=2048, n_threads=1, verbose=False)

SYSTEM_PROMPT = (
    "Ты — Senior Data Scientist. Напиши самодостаточный Python-скрипт для продвинутого EDA анализа данных.\n"
    "Файл данных: `data/real_transactions.csv`.\n"
    "Колонки: Transaction_ID, User_Name, Age, Country, Product_Category, Purchase_Amount, Payment_Method, Transaction_Date\n\n"
    "ЖЕСТКИЕ ПРАВИЛА ГЕНЕРАЦИИ КОДА:\n"
    "1. ВСЕГДА импортируй библиотеки: import pandas as pd, import matplotlib.pyplot as plt, import seaborn as sns\n"
    "2. Для заполнения пропусков (NaN) в Pandas 3.0 КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать inplace=True.\n"
    "   Используй строгий современный синтаксис: df['Age'] = df['Age'].fillna(df['Age'].median())\n"
    "3. Для разбивки на группы используй pd.cut() с bins= и labels=['Молодежь', 'Средний возраст', 'Пожилые'].\n"
    "4. НИКОГДА не вызывай plt.show(). Обязательно в конце пиши:\n"
    "   plt.savefig('data/outputs/spending_distribution_{iteration}.png', bbox_inches='tight')\n"
    "   plt.close()\n"
    "5. Выводи ТОЛЬКО чистый код без markdown-разметки (без ```python), вводных слов и вежливых фраз."
)




def planner_node(state: AgentState) -> dict:
    current_iteration = state['retry_count'] + 1
    print(f"\n[Planner] Запуск итерации {current_iteration}...")
    
    error_context = ""
    if state["steps"]:
        for step in state["steps"]:
            if step.observation.exit_code != 0:
                error_context += (
                    f"\nТвой прошлый код упал с ошибкой!\n"
                    f"--- ОШИБОЧНЫЙ КОД ---\n{step.action.code}\n"
                    f"--- ЛОГ ОШИБКИ (stderr) ---\n{step.observation.stderr}\n"
                    f"Исправь эту ошибку в новом коде!\n"
                )

    formatted_system_prompt = SYSTEM_PROMPT.format(iteration=current_iteration)
                
    prompt = (
        f"<|im_start|>system\n{formatted_system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nЗадача: {state['task_description']}\n{error_context}\n"
        f"Напиши чистый Python-код решения.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    print("[Planner] Модель генерирует Python-код решения датасета Kaggle...")
    response = llm(prompt, max_tokens=1024, stop=["<|im_end|>"])
    generated_code = response["choices"][0]["text"].strip()
    
    generated_code = generated_code.replace("```python", "").replace("```", "").strip() #На всякий случай очищаем от мета-тегов, если модель их таки добавила
    
    return {
        "current_thought": "Анализ реального Kaggle-датасета транзакций.",
        "current_code": generated_code
    }
