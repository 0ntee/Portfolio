from pydantic import BaseModel, Field

class Action(BaseModel):
    thought: str = Field(description="Внутреннее рассуждение агента.")
    code: str = Field(description="Чистый Python-код для запуска.")

class Observation(BaseModel):
    stdout: str = Field(description="Стандартный вывод из процесса.")
    stderr: str = Field(description="Лог ошибок (Traceback), если код упал.")
    exit_code: int = Field(description="Код завершения процесса.")
