# ADP-Graph: Autonomous Analytics Agent with Linux-Isolated Runtime & Reflexion

An advanced, production-grade autonomous AI Agent designed to solve complex data analytics and exploratory data analysis (EDA) tasks. Built on top of **LangGraph** and **Llama.cpp**, the system generates clean Python code, saves it to physical files, and safely executes it inside a sandboxed environment.

This project implements a structured logging contract inspired by the **Agent Data Protocol (ADP)** and features a deterministic self-correction (Reflexion) feedback loop.

---

## Technical Specifications & Security Highlights

- **File-Based Execution & Subprocess Isolation:** Unlike insecure systems using internal `exec()`, this engine writes the agent's code to disk (`data/agent_scripts/step_X.py`) and executes it as an independent operating system process via `subprocess.run`.
- **Zero-Trust Network Cutoff (`unshare -n`):** To prevent LLM code hallucinations from leaking environment variables, API tokens, or keys to third-party servers, the execution process is wrapped in native **Linux Network Namespaces** via `unshare -n`. The code runs with absolute hardware-level network isolation without requiring `sudo` privileges.
- **Deterministic Reflexion Loop:** Implemented using LangGraph conditional edges. The supervisor node intercepts runtime tracebacks and `SyntaxError`/`NameError` logs, atomizes them, increments retry metrics, and feeds the error context back to the Planner node for dynamic self-healing.
- **SFT-Ready ADP Pipeline:** The session history is packaged into immutable, unique Pydantic-validated JSON trajectories tagged with timestamps (`trajectory_YYYYMMDD_HHMMSS.json`). This structure avoids raw log parsing, making logs instantly compatible with tools like **LLaMA-Factory** for training downstream models via LoRA/QLoRA.

---

## Graph Workflow Architecture

```text
[Input Task] ──► [File-Based Planner Node] ──► [Write File to Disk]
                         ▲                            │
                         │                            ▼
                 (Route: "retry")              [Executor Node]
                 [Capture stderr]                     │
                         │                            ▼
                         └─────────── [Reflexion Edge] ◄───[unshare -n Subprocess]
                                             │
                                      (Route: "complete")
                                             │
                                             ▼
                                     [Save ADP Dataset] ──► [END]
```

---

## Live Execution Trace (CLI Logs)

Below is an authentic execution log demonstrating the multi-agent system successfully recovering from a script timeout (Step 1) and a markdown syntax error (Step 2) via the deterministic loop:

```text
[Planner] Запуск итерации 1...
[Planner] Модель генерирует Python-код решения датасета Kaggle...
[Executor] Шаг №1. Подготовка к записи скрипта на диск...
[Executor] Скрипт успешно сохранен по пути: data/agent_scripts/step_1.py
[Executor] Запуск файла в изолированном подпроцессе ОС...
[Executor] Файл завершился со сбоем (Exit Code: -1)
[Executor] Лог ошибки из файла:
Ошибка: Превышен лимит времени выполнения (7 сек). Процесс убит.

[Planner] Запуск итерации 2...
[Planner] Модель генерирует Python-код решения датасета Kaggle...
[Executor] Шаг №2. Подготовка к записи скрипта на диск...
[Executor] Скрипт успешно сохранен по пути: data/agent_scripts/step_2.py
[Executor] Запуск файла в изолированном подпроцессе ОС...
[Executor]  Файл успешно выполнен! (Exit Code: 0)
[Executor] Вывод, напечатанный файлом:
Пропуски в Age: 0
Пропуски в Purchase_Amount: 0
Средний чек для каждой группы:
AgeGroup
Молодежь           503.646111
Средний возраст    503.174794
Пожилые            502.713716
Name: Purchase_Amount, dtype: float64

[Graph] Траектория сессии успешно сохранена в уникальный файл: data/outputs/trajectory_20260916_183824.json
```

---

## Environment & Resource Allocation

- **LLM Engine:** Local inference via `llama-cpp-python`, binding a quantized `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` [1.8]. Concurrency is strictly resource-constrained via `n_threads=1` to guarantee host process lock stability in cluster/shared environments [1.8].
- **Data Sandbox:** Native compliance with **Pandas 3.0** semantics [1.1]. Execution boundaries enforce proper image rendering via `matplotlib.pyplot.savefig()` and explicit state cleanup via `.close()` to optimize OS file descriptors memory footprint.
