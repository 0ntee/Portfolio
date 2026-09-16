import subprocess
from app.schemas import Observation

def execute_file_safely(file_path: str, timeout_seconds: int = 7) -> Observation:
    """
    Безопасно запускает готовый файл скрипта как независимый процесс ОС.
    """
    try:
        result = subprocess.run(
            ["python3", file_path],
            capture_output=True, # Принудительно перехватываем stdout и stderr
            text=True,           # Декодируем байты в обычный текст автоматически
            timeout=timeout_seconds # Защита от бесконечных циклов на уровне ОС
        )
        
        return Observation(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode
        )
        
    except subprocess.TimeoutExpired:
        return Observation(
            stdout="",
            stderr=f"Ошибка: Превышен лимит времени выполнения ({timeout_seconds} сек). Процесс убит.",
            exit_code=-1
        )
    except Exception as e:
        return Observation(
            stdout="",
            stderr=f"Критическая ошибка вызова процесса: {str(e)}",
            exit_code=2
        )
